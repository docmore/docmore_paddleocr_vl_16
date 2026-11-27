"""Pydantic request models for the /v1/parse endpoint.

Supports both the unified-parse input schema (``content[]`` array) and the
legacy OpenAI chat-completions shape (``messages[]``) for backwards
compatibility. ``model`` must be one of the six ``paddleocr-vl-1.6-*``
aliases served by :mod:`docmore.paddleocr_vl_16.mode`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from docmore.paddleocr_vl_16.mode import default_model_alias


class ImageSource(BaseModel):
    type: Literal["base64", "url"]
    data: str | None = None
    url: str | None = None


class ContentBlock(BaseModel):
    type: Literal["text", "image", "file"]
    text: str | None = None
    source: ImageSource | None = None
    mime_type: str | None = None


class ParseOptions(BaseModel):
    output_format: str = "markdown"
    # Kept True by default for backwards compatibility: the endpoint has
    # always returned bboxes. Set False to omit them from blocks and assets.
    include_bounding_boxes: bool = True
    include_lines: bool = False
    include_images: bool = True
    language: str = "auto"
    pages: str | None = None
    table_format: str = "html"
    include_headers_footers: bool = False


class ParseRequest(BaseModel):
    model: str = default_model_alias()
    content: list[ContentBlock] = Field(default_factory=list)
    options: ParseOptions = Field(default_factory=ParseOptions)
    stream: bool = False
    response_schema: dict | None = None

    # Legacy OpenAI-compatible fields (accepted but ignored by /v1/parse)
    messages: list[dict] | None = None
    ocr_options: dict | None = None
