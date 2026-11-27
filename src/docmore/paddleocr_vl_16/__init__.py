"""docmore.paddleocr_vl_16: OCR document conversion.

Convert PDF / image documents to Markdown / HTML, or to a unified
``DoclingDocument`` (docling-core) via the ``docling`` output format.

Usage (library)::

    from docmore.paddleocr_vl_16 import convert
    convert("doc.pdf", "out/", output_format="docling")

Usage (CLI)::

    pip install docmore-paddleocr-vl-16
    docmore-paddleocr-vl-16 doc.pdf out/ --format docling
    # or: python -m docmore.paddleocr_vl_16 doc.pdf out/
"""

__version__ = "0.1.0"


def __getattr__(name):
    """Lazily expose the public API without importing heavy deps up front."""
    if name == "convert":
        from docmore.paddleocr_vl_16.api import convert

        return convert
    if name == "to_docling_document":
        from docmore.paddleocr_vl_16.docling_bridge import to_docling_document

        return to_docling_document
    if name == "main":
        from docmore.paddleocr_vl_16.cli import main

        return main
    if name == "InferenceManager":
        from docmore.paddleocr_vl_16.model import InferenceManager

        return InferenceManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "convert",
    "to_docling_document",
    "InferenceManager",
    "main",
    "__version__",
]
