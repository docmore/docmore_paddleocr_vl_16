"""OpenAI-compatible client for the PaddleOCR-VL 1.6 endpoint.

A single :class:`VLMClient` wraps an ``openai.OpenAI`` client configured
against ``OPENAI_API_BASE`` (default ``http://127.0.0.1:8113/v1``, model
``paddleocr-vl-1.6-doc``, typically served by llama.cpp). Each request is one cropped
layout region plus its task prompt (image first, then text); PaddleOCR-VL 1.6 does
not use a system prompt.

Sampling is greedy (temperature 0.0). Transient errors and degenerate
repetition are retried with an escalating temperature, mirroring the
chandra / navidc / paddleocr-vl retry convention.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from openai import OpenAI
from PIL import Image

from docmore_utility import pil_to_base64

from docmore.paddleocr_vl_16.model.util import detect_repeat_token
from docmore.paddleocr_vl_16.prompts import TASK_PROMPTS
from docmore.paddleocr_vl_16.schema import GenerationResult
from docmore.paddleocr_vl_16.settings import settings


def normalize_base_url(base_url: str) -> str:
    """Append ``/v1`` when ``OPENAI_API_BASE`` is a bare server root.

    ``http://127.0.0.1:8113`` and ``http://127.0.0.1:8113/v1`` both work;
    URLs that already carry a path (proxy prefixes, other API versions) are
    left untouched.
    """
    base = base_url.rstrip("/")
    if not urlparse(base).path:
        base += "/v1"
    return base


def _build_content(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    """Build a multimodal user-content list (image first, then the prompt)."""
    data_url = pil_to_base64(
        image, format=settings.IMAGE_FORMAT, with_data_uri=True
    )
    return [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": prompt},
    ]


class VLMClient:
    """Thin wrapper over ``openai.OpenAI`` for PaddleOCR-VL 1.6.

    Args:
        api_key: OpenAI API key (default from settings; empty is fine for a
            local llama.cpp endpoint).
        base_url: OpenAI-compatible base URL (default from settings).
        model_name: model name served at the endpoint (default from settings).
        max_output_tokens: ``max_tokens`` per request.
        temperature: base sampling temperature (0.0 for deterministic OCR).
        top_p: nucleus sampling probability.
        max_retries: per-request retries on empty / repetitive / transient
            output.
        timeout: per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = normalize_base_url(base_url or settings.OPENAI_API_BASE)
        self.model_name = model_name or settings.OPENAI_MODEL_NAME
        self.max_output_tokens = max_output_tokens or settings.MAX_OUTPUT_TOKENS
        self.temperature = (
            temperature if temperature is not None else settings.TEMPERATURE
        )
        self.top_p = top_p if top_p is not None else settings.TOP_P
        self.max_retries = (
            max_retries if max_retries is not None else settings.MAX_RETRIES
        )
        self.timeout = timeout if timeout is not None else settings.TIMEOUT
        self._client = OpenAI(
            api_key=self.api_key or "EMPTY",
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def complete(
        self,
        image: Image.Image,
        prompt: str | None = None,
    ) -> GenerationResult:
        """Run one region-recognition completion.

        Args:
            image: cropped region (RGB PIL image).
            prompt: task prompt (``OCR:`` by default).

        Returns:
            A :class:`GenerationResult` (``error=True`` if all retries fail).
        """
        user_prompt = prompt or TASK_PROMPTS["ocr"]

        for attempt in range(self.max_retries + 1):
            temperature = min(self.temperature + 0.2 * attempt, 0.8)
            top_p = self.top_p if attempt == 0 else 0.95
            try:
                completion = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": _build_content(image, user_prompt),
                        },
                    ],
                    max_tokens=self.max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                msg = completion.choices[0].message
                raw = msg.content or ""
                token_count = (
                    completion.usage.completion_tokens if completion.usage else 0
                )
                if not raw.strip():
                    continue  # empty OCR output -> retry
                if detect_repeat_token(raw):
                    continue  # degenerate repetition -> retry hotter
                return GenerationResult(raw=raw, token_count=token_count, error=False)
            except Exception as exc:  # noqa: BLE001 - retry any transient error
                if attempt >= self.max_retries:
                    print(
                        f"[paddleocr-vl-16] completion failed "
                        f"after {attempt + 1} tries: {exc}"
                    )
                    return GenerationResult(raw="", token_count=0, error=True)
        return GenerationResult(raw="", token_count=0, error=True)