"""Load input files (PDF / image) into PIL images for OCR.

PDF rendering and content-type detection are delegated to
``docmore_utility`` (PyMuPDF + Magika); this module only keeps
package-specific policy (DPI, minimum dimension, page ranges).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

from PIL import Image
from docmore_utility import pdf_to_images

from docmore.paddleocr_vl_16.settings import settings

# 读取文件头部用于内容识别的字节数
_SNIFF_BYTES = 64 * 1024


def load_image(
    filepath: str, min_image_dim: int = settings.MIN_IMAGE_DIM
) -> Image.Image:
    """Load a raster image, upscaling so its short side meets ``min_image_dim``."""
    return load_image_bytes(Path(filepath).read_bytes(), min_image_dim)


def load_image_bytes(
    data: bytes, min_image_dim: int = settings.MIN_IMAGE_DIM
) -> Image.Image:
    """Load a raster image from raw bytes (same policy as :func:`load_image`)."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if image.width < min_image_dim or image.height < min_image_dim:
        scale = min_image_dim / min(image.width, image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_pdf_images(
    filepath: str,
    page_range: List[int] | None,
    image_dpi: int = settings.IMAGE_DPI,
    min_pdf_image_dim: int = settings.MIN_PDF_IMAGE_DIM,
) -> List[Image.Image]:
    """Render selected PDF pages to RGB PIL images (``page_range`` is 1-based)."""
    return load_pdf_bytes(
        Path(filepath).read_bytes(), page_range, image_dpi, min_pdf_image_dim
    )


def load_pdf_bytes(
    data: bytes,
    page_range: List[int] | None,
    image_dpi: int = settings.IMAGE_DPI,
    min_pdf_image_dim: int = settings.MIN_PDF_IMAGE_DIM,
) -> List[Image.Image]:
    """Render selected PDF pages from raw bytes (same policy as :func:`load_pdf_images`)."""
    pages = [p - 1 for p in page_range] if page_range else None
    return pdf_to_images(
        data,
        dpi=image_dpi,
        pages=pages,
        min_dim=min_pdf_image_dim,
        flatten=True,
        return_pil=True,
    )


def parse_range_str(range_str: str) -> List[int]:
    """Parse a 1-based page-range string like ``"1-5,7,9-12"``.

    Returns deduplicated, ascending page numbers.
    """
    range_lst = range_str.split(",")
    page_lst = []
    for i in range_lst:
        if "-" in i:
            start, end = i.split("-")
            page_lst += list(range(int(start), int(end) + 1))
        else:
            page_lst.append(int(i))
    page_lst = sorted(list(set(page_lst)))  # Deduplicate page numbers and sort in order
    return page_lst


def _sniff_is_pdf(data: bytes) -> bool:
    """Decide whether ``data`` is a PDF by content (no filename involved).

    Uses :func:`docmore_utility.identify` (Magika) when the optional
    ``detect`` extra is installed, and falls back to the ``%PDF-`` magic
    bytes otherwise.
    """
    try:
        from docmore_utility import identify
    except ImportError:
        return data[:5] == b"%PDF-"
    try:
        return identify(data[:_SNIFF_BYTES]).label == "pdf"
    except Exception:
        return data[:5] == b"%PDF-"


def load_file(filepath: str, config: dict) -> List[Image.Image]:
    """Load a PDF or image file into a list of RGB PIL images (one per page)."""
    return load_bytes(Path(filepath).read_bytes(), config)


def load_bytes(data: bytes, config: dict) -> List[Image.Image]:
    """Load PDF or image bytes into a list of RGB PIL images (one per page).

    ``config`` may carry ``page_range`` (a 1-based range string like
    ``"1-5,7"``); content type is detected from the bytes themselves.
    """
    page_range = config.get("page_range")
    if page_range:
        page_range = parse_range_str(page_range)

    if _sniff_is_pdf(data):
        images = load_pdf_bytes(data, page_range)
    else:
        images = [load_image_bytes(data)]
    return images
