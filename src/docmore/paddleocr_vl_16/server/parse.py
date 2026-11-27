"""Convert engine ``BatchOutputItem`` results to unified-parse/v1 output schema.

Maps the internal engine output (chunks with labels + pixel-space bboxes,
markdown, HTML) into the /v1/parse output format (docmore 统一接口约定).
"""

from __future__ import annotations

import io
import base64
import re
import time
import uuid
from typing import Any

from PIL import Image

from docmore.paddleocr_vl_16.mode import default_model_alias
from docmore.paddleocr_vl_16.schema import BatchOutputItem
from docmore.paddleocr_vl_16.settings import settings

_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# 引擎 label（PP-DocLayoutV3 25 类）-> unified block type
_BLOCK_TYPE_MAP: dict[str, str] = {
    "text": "paragraph",
    "ocr": "paragraph",
    "spotting": "paragraph",
    "vertical_text": "paragraph",
    "reference_content": "reference",
    "vision_footnote": "footnote",
    "content": "paragraph",
    "doc_title": "title",
    "paragraph_title": "title",
    "figure_title": "title",
    "table": "table",
    "image": "image",
    "header_image": "image",
    "footer_image": "image",
    "chart": "chart",
    "seal": "seal",
    "formula": "formula",
    "display_formula": "formula",
    "inline_formula": "formula",
    "formula_number": "formula_number",
    "algorithm": "algorithm",
    "code": "code",
    "footnote": "footnote",
    "number": "page_number",
    "header": "page_header",
    "footer": "page_footer",
    "aside_text": "aside_text",
    "abstract": "abstract",
    "reference": "reference",
    # 无版面回退路径的整页 text 区域
    "Text": "paragraph",
}

_IMAGE_LABELS = {"image", "header_image", "footer_image"}


def _label_to_block_type(label: str) -> str:
    return _BLOCK_TYPE_MAP.get(label, "paragraph")


def _title_level(label: str) -> int | None:
    if label == "doc_title" or label == "Title":
        return 1
    if label in ("paragraph_title", "figure_title"):
        return 2
    return None


def _resolve_chunk_image(result: BatchOutputItem, chunk: dict) -> Image.Image | None:
    """Return the exact crop for a block, preferring its ``asset`` key.

    New engine results carry ``chunk["asset"]`` (a key into
    ``BatchOutputItem.images``). Older results only have a bbox, so fall back
    to the historical same-size heuristic.
    """
    key = chunk.get("asset")
    if key:
        image = (result.images or {}).get(key)
        if isinstance(image, Image.Image):
            return image

    bbox_raw = chunk.get("bbox") or []
    if bbox_raw and result.images:
        width = max(1, bbox_raw[2] - bbox_raw[0])
        height = max(1, bbox_raw[3] - bbox_raw[1])
        for image in result.images.values():
            if isinstance(image, Image.Image) and image.size == (width, height):
                return image
    return None


def _build_page(
    result: BatchOutputItem,
    page_idx: int,
    include_images: bool,
    include_bounding_boxes: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert a single BatchOutputItem into a unified-parse/v1 page dict + assets."""
    w, h = result.page_box[2], result.page_box[3]
    assets: list[dict[str, Any]] = []

    chunks = result.chunks if result.chunks else []
    blocks_out: list[dict[str, Any]] = []

    for blk_idx, chunk in enumerate(chunks):
        label = chunk.get("label", "Text")
        btype = _label_to_block_type(label)
        bbox_raw = chunk.get("bbox", [])
        bbox: list[float] | None = (
            [float(v) for v in bbox_raw] if bbox_raw else None
        )
        level = _title_level(label)

        content_dict: dict[str, Any] = {}
        raw_content = chunk.get("content", "")
        image_payload: tuple[str, str] | None = None

        if btype == "table" and raw_content:
            content_dict["html"] = raw_content.strip()
        elif btype == "chart" and raw_content:
            # Chart Recognition may return an HTML/OTSL table or a caption.
            if "<table" in raw_content.lower():
                content_dict["html"] = raw_content.strip()
            else:
                content_dict["text"] = raw_content.strip()
        elif btype == "formula" and raw_content:
            content_dict["latex"] = raw_content.strip()
            content_dict["format"] = "latex"
        elif btype == "code" and raw_content:
            content_dict["code"] = raw_content.strip()
        elif btype in _IMAGE_LABELS:
            chunk_image = _resolve_chunk_image(result, chunk)
            if include_images and chunk_image is not None:
                buffer = io.BytesIO()
                chunk_image.save(buffer, format="JPEG", quality=85)
                image_payload = (
                    base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "image/jpeg",
                )
                content_dict["base64"] = image_payload[0]
                content_dict["mime_type"] = image_payload[1]
            if raw_content:
                content_dict["caption"] = raw_content.strip()
        else:
            if raw_content:
                content_dict["text"] = raw_content.strip()

        block_out: dict[str, Any] = {
            "id": f"p{page_idx}-b{blk_idx}",
            "type": btype,
            "reading_order": blk_idx,
            "content": content_dict,
        }
        if bbox is not None and include_bounding_boxes:
            block_out["bbox"] = bbox
        if level is not None:
            block_out["level"] = level
        if label != btype:
            block_out["meta"] = {"source_label": label}

        # Image blocks produce asset entries
        if btype in _IMAGE_LABELS:
            asset: dict[str, Any] = {
                "id": block_out["id"],
                "type": btype,
                "page_index": page_idx,
            }
            if bbox is not None and include_bounding_boxes:
                asset["bbox"] = bbox
            if image_payload is not None:
                asset["base64"] = image_payload[0]
                asset["mime_type"] = image_payload[1]
            else:
                asset["base64"] = None
            assets.append(asset)

        blocks_out.append(block_out)

    page_markdown = result.markdown
    if not include_images:
        page_markdown = _IMAGE_LINK_RE.sub("", page_markdown).strip()

    page_out = {
        "index": page_idx,
        "number": page_idx + 1,
        "size": {
            "width": w,
            "height": h,
            "unit": "px",
            "dpi": settings.IMAGE_DPI,
        },
        "markdown": page_markdown,
        "blocks": blocks_out,
    }
    return page_out, assets


def build_parse_result(
    results: list[BatchOutputItem],
    *,
    model_id: str = default_model_alias(),
    include_images: bool = True,
    include_bounding_boxes: bool = True,
    page_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Build a complete /v1/parse response from BatchOutputItem objects.

    Args:
        results: one BatchOutputItem per page, in order.
        model_id: the model identifier to report.
        include_images: whether to inline image crops into assets[] and keep
            image links in the page Markdown.
        include_bounding_boxes: whether to include bbox fields in blocks/assets.
        page_indices: original 0-based page numbers when the request selected
            a subset of pages; defaults to 0..len(results)-1.

    Returns:
        A dict conforming to the unified-parse/v1 output schema.
    """
    pages: list[dict[str, Any]] = []
    all_assets: list[dict[str, Any]] = []
    total_tokens = 0
    warnings: list[str] = []

    for pos, result in enumerate(results):
        page_idx = (
            page_indices[pos]
            if page_indices is not None and pos < len(page_indices)
            else pos
        )
        if result.error:
            warnings.append(f"page {page_idx + 1}: inference error")
            continue

        page_data, page_assets = _build_page(
            result,
            page_idx,
            include_images,
            include_bounding_boxes,
        )
        all_assets.extend(page_assets)
        total_tokens += result.token_count
        pages.append(page_data)

    content = "\n\n---\n\n".join(p["markdown"] for p in pages)

    return {
        "id": f"ocr-{uuid.uuid4().hex[:12]}",
        "object": "parse.result",
        "created": int(time.time()),
        "model": model_id,
        "schema": "unified-parse/v1",
        "finish_reason": "stop",
        "pages": pages,
        "assets": all_assets,
        "content": content,
        "annotation": None,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": total_tokens,
            "total_tokens": total_tokens,
            "pages_processed": len(pages),
        },
        "warnings": warnings,
    }
