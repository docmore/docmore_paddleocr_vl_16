"""Concurrency tests for the FastAPI service.

The OCR pipeline is synchronous. These tests pin down that it runs in worker
threads (``asyncio.to_thread``) so it never blocks the event loop: two slow
requests overlap, and ``/health`` still answers while one is in flight.
"""

from __future__ import annotations

import asyncio
import base64
import io
import threading
import time

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from docmore.paddleocr_vl_16.schema import GenerationResult
from docmore.paddleocr_vl_16.server import create_app
from docmore.paddleocr_vl_16.server.config import ServeSettings
from docmore.paddleocr_vl_16.settings import settings


class SlowClient:
    """VLM client that sleeps to make blocking behaviour observable."""

    model_name = "paddleocr-vl-1.6-doc"

    def __init__(self, delay: float = 0.5) -> None:
        self.delay = delay
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, image, prompt=None):
        with self._lock:
            self.calls += 1
        time.sleep(self.delay)
        return GenerationResult(raw="slow text", token_count=1)


@pytest.fixture(autouse=True)
def _disable_layout(monkeypatch):
    monkeypatch.setattr(settings, "USE_LAYOUT_DETECTION", False)


def _png_b64() -> str:
    image = Image.new("RGB", (200, 150), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_body() -> dict:
    return {
        "content": [
            {"type": "image", "source": {"type": "base64", "data": _png_b64()}}
        ]
    }


def test_slow_requests_overlap_and_health_stays_responsive():
    app = create_app(
        ServeSettings(MAX_CONCURRENT=2),
        client=SlowClient(delay=0.5),
    )

    async def run():
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            started = time.perf_counter()
            first = asyncio.create_task(client.post("/v1/parse", json=_parse_body()))
            second = asyncio.create_task(client.post("/v1/parse", json=_parse_body()))

            # While both pipelines sleep in worker threads, the event loop
            # must still be able to serve /health.
            await asyncio.sleep(0.1)
            health = await asyncio.wait_for(client.get("/health"), timeout=0.3)
            assert health.status_code == 200

            responses = await asyncio.gather(first, second)
            elapsed = time.perf_counter() - started
        return responses, elapsed

    responses, elapsed = asyncio.run(run())

    assert all(response.status_code == 200 for response in responses)
    # Serial execution would take >= 1.0s; overlapping takes ~0.5s.
    assert elapsed < 0.9