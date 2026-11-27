"""High-level conversion API shared by the CLI and library users.

``convert()`` turns a PDF/image file (or a directory of them) into
Markdown / HTML (and optionally a docling-core ``DoclingDocument`` JSON).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Sequence

from docmore.paddleocr_vl_16.input import load_file
from docmore.paddleocr_vl_16.schema import BatchInputItem
from docmore.paddleocr_vl_16.settings import settings

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".bmp",
}


def get_supported_files(input_path: Path) -> List[Path]:
    """Get list of supported image/PDF files from a file or directory path."""
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    if input_path.is_dir():
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(set(files))

    raise FileNotFoundError(f"Path does not exist: {input_path}")


def _dedupe_image_key(key: str, page_num: int, used: set[str]) -> str:
    """Return a key unique within one merged output directory.

    Engine-produced keys are already page-prefixed; this is a safety net for
    callers that pass results generated without ``page_index``.
    """
    if key not in used:
        return key
    name = Path(key).name
    candidate = f"imgs/page_{page_num + 1:04d}/{name}"
    if candidate not in used:
        return candidate
    stem, suffix = Path(name).stem, Path(name).suffix
    counter = 2
    while f"imgs/page_{page_num + 1:04d}/{stem}_{counter}{suffix}" in used:
        counter += 1
    return f"imgs/page_{page_num + 1:04d}/{stem}_{counter}{suffix}"


def save_merged_output(
    output_dir: Path,
    file_name: str,
    results: Sequence,
    *,
    save_images: bool = True,
    save_html: bool = True,
    save_json: bool = True,
    paginate_output: bool = False,
    docling: bool = False,
    progress: Callable[[str], None] = print,
) -> Path:
    """Save merged OCR results for all pages; returns the per-file output dir."""
    output_dir = Path(output_dir)
    safe_name = Path(file_name).stem
    file_output_dir = output_dir / safe_name
    file_output_dir.mkdir(parents=True, exist_ok=True)

    all_markdown: list[str] = []
    all_html: list[str] = []
    all_metadata: list[dict] = []
    document_pages: list[dict] = []
    total_tokens = 0
    total_chunks = 0
    total_images = 0
    used_image_keys: set[str] = set()

    for page_num, result in enumerate(results):
        if page_num > 0 and paginate_output:
            all_markdown.append(f"\n\n{page_num}" + "-" * 48 + "\n\n")
            all_html.append(f"\n\n<!-- Page {page_num + 1} -->\n\n")

        page_markdown = result.markdown
        page_html = result.html
        image_key_map: dict[str, str] = {}

        if save_images and result.images:
            for img_name, pil_image in result.images.items():
                new_name = _dedupe_image_key(
                    img_name, page_num, used_image_keys
                )
                used_image_keys.add(new_name)
                image_key_map[img_name] = new_name
                if new_name != img_name:
                    page_markdown = page_markdown.replace(
                        f"]({img_name})", f"]({new_name})"
                    )
                    page_html = page_html.replace(
                        f"]({img_name})", f"]({new_name})"
                    )
                img_path = file_output_dir / new_name
                img_path.parent.mkdir(parents=True, exist_ok=True)
                pil_image.save(img_path)

        all_markdown.append(page_markdown)
        all_html.append(page_html)

        if not paginate_output and page_num < len(results) - 1:
            all_markdown.append("\n\n")
            all_html.append("\n\n")

        total_tokens += result.token_count
        total_chunks += len(result.chunks)
        total_images += len(result.images or {})

        all_metadata.append(
            {
                "page_num": page_num,
                "page_box": result.page_box,
                "token_count": result.token_count,
                "num_chunks": len(result.chunks),
                "num_images": len(result.images or {}),
            }
        )

        blocks: list[dict] = []
        for block_index, chunk in enumerate(result.chunks):
            block = {
                "id": f"p{page_num}-b{block_index}",
                "reading_order": block_index,
                "label": chunk.get("label", "text"),
                "bbox": chunk.get("bbox"),
                "content": chunk.get("content", ""),
            }
            asset = chunk.get("asset")
            if asset:
                block["asset"] = image_key_map.get(asset, asset)
            blocks.append(block)

        document_pages.append(
            {
                "page_num": page_num,
                "page_box": result.page_box,
                "markdown": page_markdown,
                "blocks": blocks,
            }
        )

    if docling:
        try:
            from docling_core.types.doc import ImageRefMode

            from docmore.paddleocr_vl_16.docling_bridge import to_docling_document

            doc = to_docling_document(results, name=safe_name)
            docling_json_path = file_output_dir / f"{safe_name}.docling.json"
            doc.save_as_json(
                docling_json_path,
                artifacts_dir=Path("pictures"),
                image_mode=ImageRefMode.REFERENCED,
            )
            docling_md_path = file_output_dir / f"{safe_name}.docling.md"
            doc.save_as_markdown(
                docling_md_path,
                artifacts_dir=Path("pictures"),
                image_mode=ImageRefMode.REFERENCED,
            )
            progress(f"  Saved docling: {docling_json_path}")
        except ImportError:
            progress(
                "  Warning: docling-core not installed. "
                "Install with: pip install docling-core"
            )

    markdown_path = file_output_dir / f"{safe_name}.md"
    markdown_path.write_text("".join(all_markdown), encoding="utf-8")

    if save_html:
        html_path = file_output_dir / f"{safe_name}.html"
        html_path.write_text("".join(all_html), encoding="utf-8")

    if save_json:
        document = {
            "file_name": file_name,
            "model": settings.OPENAI_MODEL_NAME,
            "num_pages": len(results),
            "total_token_count": total_tokens,
            "pages": document_pages,
        }
        json_path = file_output_dir / f"{safe_name}.json"
        json_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    metadata = {
        "file_name": file_name,
        "num_pages": len(results),
        "total_token_count": total_tokens,
        "total_chunks": total_chunks,
        "total_images": total_images,
        "pages": all_metadata,
    }
    metadata_path = file_output_dir / f"{safe_name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    progress(f"  Saved: {markdown_path} ({len(results)} page(s))")
    return file_output_dir


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    page_range: str | None = None,
    max_output_tokens: int | None = None,
    max_workers: int | None = None,
    max_retries: int | None = None,
    save_html: bool = True,
    save_json: bool = True,
    batch_size: int | None = None,
    paginate_output: bool = False,
    output_format: str = "default",
    model=None,
    progress: Callable[[str], None] = print,
) -> Path:
    """Convert a file (or every supported file in a directory) to Markdown/HTML.

    Args:
        input_path: PDF / image file, or a directory containing them.
        output_path: Output directory (created if missing). One sub-folder
            per input file is written, containing merged markdown, optional
            html, metadata json and -- with ``output_format="docling"`` --
            the DoclingDocument json/markdown.
        page_range: 1-based page-range string for PDFs, e.g. ``"1-5,7"``.
        output_format: ``"default"`` or ``"docling"`` (also emits
            docling-core JSON + markdown).
        model: Pre-built ``InferenceManager`` to reuse across calls;
            one is created when omitted.
        progress: Callable used for status output.

    Returns:
        The output directory path.

    Raises:
        ValueError: unknown output_format.
    """
    if output_format not in ("default", "docling"):
        raise ValueError(
            f"output_format must be 'default' or 'docling': {output_format!r}"
        )

    if batch_size is None:
        batch_size = 28

    if model is None:
        from docmore.paddleocr_vl_16.model import InferenceManager

        progress("Loading engine...")
        kwargs: dict = {}
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        if max_workers is not None:
            kwargs["max_workers"] = max_workers
        model = InferenceManager(**kwargs)

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    files_to_process = get_supported_files(input_path)
    progress(f"Found {len(files_to_process)} file(s) to process.")
    if not files_to_process:
        return output_path

    for file_idx, file_path in enumerate(files_to_process, 1):
        progress(f"[{file_idx}/{len(files_to_process)}] Processing: {file_path.name}")
        try:
            images = load_file(str(file_path), {"page_range": page_range} if page_range else {})
            progress(f"  Loaded {len(images)} page(s)")

            all_results = []
            for batch_start in range(0, len(images), batch_size):
                batch_end = min(batch_start + batch_size, len(images))
                batch = [
                    BatchInputItem(
                        image=img,
                        page_index=batch_start + offset,
                    )
                    for offset, img in enumerate(images[batch_start:batch_end])
                ]
                progress(f"  Processing pages {batch_start + 1}-{batch_end}...")

                all_results.extend(model.generate(batch))

            save_merged_output(
                output_path,
                file_path.name,
                all_results,
                save_html=save_html,
                save_json=save_json,
                paginate_output=paginate_output,
                docling=(output_format == "docling"),
                progress=progress,
            )
            progress(f"  Completed: {file_path.name}")
        except Exception as e:
            progress(f"  Error processing {file_path.name}: {e}")
            continue

    return output_path
