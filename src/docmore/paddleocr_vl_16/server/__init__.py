"""FastAPI application factory for the PaddleOCR-VL 1.6 document understanding service.

Exposes two families of routes:

* ``POST /v1/chat/completions`` — OpenAI-compatible chat-completion API
  (multimodal image + text → markdown).
* ``POST /v1/parse``             — unified-parse/v1 document-parsing API
  (single endpoint, multi-image, blocks + assets output).

Both families share the same ``InferenceManager`` pipeline and the same
auth / semaphore guard. The ``server`` package exposes
:func:`create_app` for direct use and :func:`serve` via the CLI script
``docmore-paddleocr-vl-16-serve``.
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazily import so the ``[server]`` extra stays optional."""
    if name == "create_app":
        from docmore.paddleocr_vl_16.server.router import create_app
        return create_app
    if name == "ServeSettings":
        from docmore.paddleocr_vl_16.server.config import ServeSettings
        return ServeSettings
    if name == "serve":
        from docmore.paddleocr_vl_16.server.cli import serve
        return serve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["create_app", "ServeSettings", "serve"]
