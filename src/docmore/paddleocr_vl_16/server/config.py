"""Server-specific settings for the PaddleOCR-VL 1.6 OpenAI-compatible service.

Kept separate from the pipeline :class:`~docmore.paddleocr_vl_16.settings.Settings`
singleton: the server reads (never mutates) pipeline settings for upstream
VLM configuration, and carries its own HTTP-facing knobs here. All fields can
be overridden via environment variables prefixed
``PADDLEOCR_VL_16_SERVER_`` or a ``local.env`` file.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServeSettings(BaseSettings):
    """HTTP service configuration.

    Attributes:
        HOST / PORT: Bind address for uvicorn.
        API_KEYS: Inbound Bearer tokens; empty list disables authentication.
        MAX_PAGES: Maximum number of images per /parse request.
        MAX_IMAGE_BYTES: Maximum decoded size per image (default 25 MiB).
        MAX_CONCURRENT: Semaphore bound on concurrent /parse in-flight requests.
        PAGE_SEPARATOR: Markdown joiner between per-page results.
        ALLOW_URL_FETCH: Accept ``{"type": "url"}`` image sources.
        ALLOW_PRIVATE_URLS: Permit URLs resolving to loopback / private /
            link-local addresses (SSRF guard; off by default).
        URL_FETCH_TIMEOUT: Per-request timeout for remote image fetches.
        URL_FETCH_MAX_REDIRECTS: Maximum redirect hops when fetching images.
    """

    HOST: str = "0.0.0.0"
    PORT: int = 18910

    # Authentication (empty list = open, deploy behind an intranet/proxy)
    API_KEYS: list[str] = []

    # Request limits
    MAX_PAGES: int = 32
    MAX_IMAGE_BYTES: int = 25 * 1024 * 1024
    MAX_CONCURRENT: int = 4
    PAGE_SEPARATOR: str = "\n\n"

    # Remote image sources
    ALLOW_URL_FETCH: bool = True
    ALLOW_PRIVATE_URLS: bool = False
    URL_FETCH_TIMEOUT: float = 30.0
    URL_FETCH_MAX_REDIRECTS: int = 5

    model_config = SettingsConfigDict(
        env_prefix="PADDLEOCR_VL_16_SERVER_",
        env_file="local.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
