"""Tests for URL image sources (fetch, SSRF guard, /v1/parse integration)."""

from __future__ import annotations

import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from mock_server import start_server as start_mock_server  # noqa: E402

from docmore.paddleocr_vl_16.server import images as image_sources  # noqa: E402
from docmore.paddleocr_vl_16.server.images import (  # noqa: E402
    ImageSourceError,
    UrlFetchError,
    fetch_image_url,
    load_image_source,
)


def _png_bytes(width: int = 40, height: int = 30) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class _Handler(BaseHTTPRequestHandler):
    png = _png_bytes()

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/page.png":
            self._respond(200, self.png, "image/png")
        elif self.path == "/big.png":
            self._respond(200, b"\x00" * 10_000, "application/octet-stream")
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/page.png")
            self.end_headers()
        elif self.path == "/redirect-loop":
            self.send_response(302)
            self.send_header("Location", "/redirect-loop")
            self.end_headers()
        else:
            self._respond(404, b"not found", "text/plain")

    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence test output
        pass


@pytest.fixture(scope="module")
def image_base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


# ---------------------------------------------------------------------------
# fetch_image_url unit tests
# ---------------------------------------------------------------------------


def test_fetch_rejects_private_address_by_default(image_base_url):
    with pytest.raises(UrlFetchError, match="non-public"):
        fetch_image_url(
            f"{image_base_url}/page.png", max_bytes=10_000_000, allow_private=False
        )


def test_fetch_allows_private_when_configured(image_base_url):
    raw = fetch_image_url(
        f"{image_base_url}/page.png", max_bytes=10_000_000, allow_private=True
    )
    assert raw == _Handler.png


def test_fetch_follows_redirect(image_base_url):
    raw = fetch_image_url(
        f"{image_base_url}/redirect", max_bytes=10_000_000, allow_private=True
    )
    assert raw == _Handler.png


def test_fetch_rejects_oversized_body(image_base_url):
    with pytest.raises(UrlFetchError, match="exceeds"):
        fetch_image_url(
            f"{image_base_url}/big.png", max_bytes=1000, allow_private=True
        )


def test_fetch_rejects_too_many_redirects(image_base_url):
    with pytest.raises(UrlFetchError, match="too many redirects"):
        fetch_image_url(
            f"{image_base_url}/redirect-loop",
            max_bytes=10_000_000,
            allow_private=True,
            max_redirects=2,
        )


def test_fetch_rejects_non_http_scheme():
    with pytest.raises(UrlFetchError, match="scheme"):
        fetch_image_url("file:///etc/passwd", max_bytes=1000, allow_private=True)


def test_fetch_revalidates_each_redirect_hop(image_base_url, monkeypatch):
    calls: list[str] = []
    real_check = image_sources._check_host

    def guarded(host: str, *, allow_private: bool) -> None:
        calls.append(host)
        if len(calls) > 1:
            raise UrlFetchError("blocked after redirect")
        real_check(host, allow_private=True)

    monkeypatch.setattr(image_sources, "_check_host", guarded)

    with pytest.raises(UrlFetchError, match="blocked after redirect"):
        fetch_image_url(
            f"{image_base_url}/redirect", max_bytes=10_000_000, allow_private=False
        )
    assert len(calls) == 2


def test_load_image_source_from_url(image_base_url):
    image = load_image_source(
        url=f"{image_base_url}/page.png",
        max_bytes=10_000_000,
        allow_private=True,
    )
    assert image.mode == "RGB"
    assert image.size == (40, 30)


def test_load_image_source_respects_allow_url_flag(image_base_url):
    with pytest.raises(UrlFetchError, match="disabled"):
        load_image_source(
            url=f"{image_base_url}/page.png",
            max_bytes=10_000_000,
            allow_private=True,
            allow_url=False,
        )


def test_load_image_source_decodes_data_uri():
    import base64

    payload = "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()
    image = load_image_source(data=payload, max_bytes=10_000_000)
    assert image.size == (40, 30)


def test_load_image_source_rejects_empty_source():
    with pytest.raises(ImageSourceError, match="requires"):
        load_image_source(max_bytes=10_000_000)


# ---------------------------------------------------------------------------
# /v1/parse integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_vlm_url():
    server, port = start_mock_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


@pytest.fixture(scope="module")
def _disable_layout():
    from docmore.paddleocr_vl_16.settings import settings

    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "USE_LAYOUT_DETECTION", False)
    yield
    mp.undo()


def _client(mock_vlm_url, **serve_overrides):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from docmore.paddleocr_vl_16.model.client import VLMClient
    from docmore.paddleocr_vl_16.server import create_app
    from docmore.paddleocr_vl_16.server.config import ServeSettings

    app = create_app(
        ServeSettings(MAX_CONCURRENT=2, **serve_overrides),
        client=VLMClient(base_url=mock_vlm_url),
    )
    return TestClient(app)


def test_parse_accepts_url_source(image_base_url, mock_vlm_url, _disable_layout):
    with _client(
        mock_vlm_url, ALLOW_PRIVATE_URLS=True, URL_FETCH_TIMEOUT=5
    ) as client:
        resp = client.post(
            "/v1/parse",
            json={
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": f"{image_base_url}/page.png"},
                    }
                ]
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["pages"][0]["size"]["width"] == 40
    assert "Mock text line" in data["pages"][0]["markdown"]


def test_parse_rejects_private_url_by_default(image_base_url, mock_vlm_url, _disable_layout):
    with _client(mock_vlm_url) as client:
        resp = client.post(
            "/v1/parse",
            json={
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": f"{image_base_url}/page.png"},
                    }
                ]
            },
        )

    assert resp.status_code == 400
    assert "non-public" in resp.json()["error"]["message"]


def test_parse_rejects_url_when_fetch_disabled(image_base_url, mock_vlm_url, _disable_layout):
    with _client(
        mock_vlm_url, ALLOW_URL_FETCH=False, ALLOW_PRIVATE_URLS=True
    ) as client:
        resp = client.post(
            "/v1/parse",
            json={
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": f"{image_base_url}/page.png"},
                    }
                ]
            },
        )

    assert resp.status_code == 400
    assert "disabled" in resp.json()["error"]["message"]


def test_parse_rejects_unreachable_url(image_base_url, mock_vlm_url, _disable_layout):
    with _client(
        mock_vlm_url, ALLOW_PRIVATE_URLS=True, URL_FETCH_TIMEOUT=5
    ) as client:
        resp = client.post(
            "/v1/parse",
            json={
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": f"{image_base_url}/missing.png"},
                    }
                ]
            },
        )

    assert resp.status_code == 400
    assert "404" in resp.json()["error"]["message"]