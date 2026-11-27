"""Bridge between engine output and docling-core's ``DoclingDocument``.

Engine ``BatchOutputItem.chunks`` carry structured layout blocks (label,
pixel bbox, content). docling-core provides the unified ``DoclingDocument``
model that the wider Docling ecosystem (chunkers, exporters, LangChain /
LlamaIndex loaders) consumes; this module converts one output per page into
a single document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from bs4 import BeautifulSoup
from PIL import Image

if TYPE_CHECKING:  # avoid importing the package at runtime
    from docmore.paddleocr_vl_16.schema import BatchOutputItem

from docling_core.types.doc import (
    BoundingBox,
    DescriptionMetaField,
    DocItemLabel,
    DoclingDocument,
    ImageRef,
    PictureMeta,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

# 引擎 label（PP-DocLayoutV3 25 类）-> docling DocItemLabel。需要专用 API 的
# （doc_title/paragraph_title -> add_heading, table/chart HTML -> add_table,
# image -> add_picture）在下方单独处理。
TEXT_LABEL_MAP: dict[str, DocItemLabel] = {
    "text": DocItemLabel.TEXT,
    "vertical_text": DocItemLabel.TEXT,
    "content": DocItemLabel.TEXT,
    "abstract": DocItemLabel.TEXT,
    "aside_text": DocItemLabel.TEXT,
    "reference_content": DocItemLabel.TEXT,
    "reference": DocItemLabel.TEXT,
    "seal": DocItemLabel.TEXT,
    "formula_number": DocItemLabel.TEXT,
    "vision_footnote": DocItemLabel.FOOTNOTE,
    "footnote": DocItemLabel.FOOTNOTE,
    "header": DocItemLabel.PAGE_HEADER,
    "footer": DocItemLabel.PAGE_FOOTER,
    "algorithm": DocItemLabel.CODE,
    "code": DocItemLabel.CODE,
    "formula": DocItemLabel.FORMULA,
    "display_formula": DocItemLabel.FORMULA,
    "inline_formula": DocItemLabel.FORMULA,
}

PICTURE_LABELS = {"image", "header_image", "footer_image"}


def _picture_for_chunk(page_images: dict, chunk: dict, bbox: list[float]):
    """Resolve the exact crop for an image block.

    Prefer the engine's ``asset`` key; fall back to the historical same-size
    heuristic for results produced before asset keys existed.
    """
    key = chunk.get("asset")
    if key:
        image = page_images.get(key)
        if isinstance(image, Image.Image):
            return image

    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    return next(
        (
            image
            for image in page_images.values()
            if isinstance(image, Image.Image) and image.size == (width, height)
        ),
        None,
    )


def _provenance(bbox: list[float], page_no: int) -> ProvenanceItem:
    """Build a ProvenanceItem from a pixel-space bbox [x0,y0,x1,y1]."""
    x0, y0, x1, y1 = bbox
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(
            l=x0,
            t=y0,
            r=x1,
            b=y1,
            coord_origin="TOPLEFT",
        ),
        charspan=(0, 0),
    )


def _html_table_to_data(html: str) -> TableData:
    """Parse an HTML <table> into docling ``TableData``."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return TableData()

    rows = table.find_all("tr")
    cells = []
    num_rows = len(rows)
    num_cols = 0
    for r_idx, tr in enumerate(rows):
        col = 0
        # Account for cells already filled by row/col spans from previous rows.
        for td in tr.find_all(["td", "th"]):
            while any(
                c.start_row_offset_idx <= r_idx < c.end_row_offset_idx
                and c.start_col_offset_idx <= col < c.end_col_offset_idx
                for c in cells
            ):
                col += 1

            rs = int(td.get("rowspan", 1) or 1)
            cs = int(td.get("colspan", 1) or 1)
            text = td.get_text(separator=" ", strip=True)
            cells.append(
                TableCell(
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + rs,
                    start_col_offset_idx=col,
                    end_col_offset_idx=col + cs,
                    text=text,
                    row_span=rs,
                    col_span=cs,
                    column_header=td.name == "th",
                )
            )
            col += cs
            num_cols = max(num_cols, col)

    return TableData(
        table_cells=cells,
        num_rows=num_rows,
        num_cols=num_cols,
    )


def _block_text(content: str) -> str:
    """Strip HTML from a block, keep readable text."""
    return BeautifulSoup(content, "html.parser").get_text(separator=" ", strip=True)


def _add_text(
    doc: DoclingDocument, label: DocItemLabel, text: str, prov: ProvenanceItem
):
    """add_text() with a fallback to TEXT for labels the docling version rejects."""
    try:
        return doc.add_text(label=label, text=text, prov=prov)
    except AssertionError:
        return doc.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)


def to_docling_document(
    outputs: Iterable["BatchOutputItem"],
    *,
    name: str = "paddleocr_vl_16",
    include_images: bool = True,
) -> DoclingDocument:
    """Convert page outputs into a single ``DoclingDocument``.

    Parameters
    ----------
    outputs:
        One ``BatchOutputItem`` per page, in page order.
    name:
        Document name recorded on the ``DoclingDocument``.
    include_images:
        Embed cropped figure bitmaps (from ``BatchOutputItem.images``) as
        base64 data URIs on the picture items. When ``False``, pictures are
        recorded as placeholders with a description.
    """
    doc = DoclingDocument(name=name)

    for page_no, out in enumerate(outputs, start=1):
        # page_box is [0, 0, width, height] in pixels.
        _, _, width, height = out.page_box
        doc.add_page(page_no=page_no, size=Size(width=width, height=height))

        page_images = out.images or {}

        for chunk in out.chunks:
            label = chunk.get("label", "Text")
            bbox = chunk.get("bbox") or [0, 0, width, height]
            content = chunk.get("content", "")
            prov = _provenance(bbox, page_no)

            # 图片区域（image/header_image/footer_image，content 为空）→ add_picture，按 bbox 尺寸
            # 匹配版面裁剪图。
            if label in PICTURE_LABELS:
                img_tag = BeautifulSoup(content or "", "html.parser").find("img")
                alt = img_tag.get("alt", "") if img_tag else ""

                pil = None
                if include_images and page_images:
                    pil = _picture_for_chunk(page_images, chunk, bbox)

                if pil is not None:
                    image_ref = ImageRef.from_pil(image=pil, dpi=96)
                    pic = doc.add_picture(image=image_ref, prov=prov)
                else:
                    pic = doc.add_picture(prov=prov)

                if alt:
                    pic.meta = PictureMeta(
                        description=DescriptionMetaField(
                            text=alt, created_by="paddleocr_vl_16"
                        )
                    )
                continue

            if not content:
                continue

            # 表格与图表：Table Recognition 产出 HTML 表；Chart Recognition 可能产出
            # HTML 表，也可能只是一段图表说明文字，后者走普通文本分支。
            if label in ("table", "Table", "chart") and "<table" in content:
                doc.add_table(
                    data=_html_table_to_data(content),
                    prov=prov,
                )
                continue

            if label in ("doc_title", "paragraph_title", "figure_title"):
                text = _block_text(content)
                if text:
                    level = 1 if label == "doc_title" else 2
                    doc.add_heading(text=text, level=level, prov=prov)
                continue

            # Default: text-like block.
            text = _block_text(content)
            if not text:
                continue
            _add_text(
                doc,
                TEXT_LABEL_MAP.get(label, DocItemLabel.TEXT),
                text,
                prov,
            )

    return doc
