"""Tests for the /v1/parse endpoint in docmore.paddleocr_vl_16.server.

Unit tests exercise :func:`build_parse_result` directly; integration tests
run against the FastAPI app wired to :mod:`tests.mock_server`.
"""

from __future__ import annotations

import base64
import io
import threading

import pytest
from PIL import Image

from mock_server import start_server as start_mock_server  # noqa: E402

pytest.importorskip("fastapi")

from docmore.paddleocr_vl_16.model.client import VLMClient  # noqa: E402
from docmore.paddleocr_vl_16.schema import BatchOutputItem  # noqa: E402
from docmore.paddleocr_vl_16.server import create_app  # noqa: E402
from docmore.paddleocr_vl_16.server.config import ServeSettings  # noqa: E402
from docmore.paddleocr_vl_16.server.parse import build_parse_result  # noqa: E402
from docmore.paddleocr_vl_16.settings import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _disable_layout():
    """Layout detection is irrelevant here; keep tests model-free."""
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "USE_LAYOUT_DETECTION", False)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def mock_base_url():
    """Start the mock VLM endpoint for integration tests."""
    server, port = start_mock_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


@pytest.fixture(scope="module")
def parse_client(mock_base_url):
    """Build a FastAPI TestClient wired to the mock upstream."""
    serve_settings = ServeSettings(API_KEYS=["secret"], MAX_CONCURRENT=2)
    app = create_app(serve_settings, client=VLMClient(base_url=mock_base_url))
    from fastapi.testclient import TestClient

    with TestClient(app) as tc:
        yield tc


def _headers(token: str = "secret"):
    return {"Authorization": f"Bearer {token}"}


def _make_batch_output(
    w: int = 1000,
    h: int = 1200,
    error: bool = False,
    chunks: list[dict] | None = None,
) -> BatchOutputItem:
    """Create a minimal BatchOutputItem for testing the converter."""
    if chunks is None:
        chunks = [
            {
                "label": "doc_title",
                "bbox": [50, 36, 950, 96],
                "content": "Title",
            },
            {
                "label": "text",
                "bbox": [50, 120, 950, 300],
                "content": "Body text here.",
            },
            {
                "label": "table",
                "bbox": [50, 320, 950, 480],
                "content": "<table><tr><th>H1</th><th>H2</th></tr><tr><td>A</td><td>1</td></tr></table>",
            },
            {
                "label": "display_formula",
                "bbox": [50, 500, 950, 560],
                "content": r"E=mc^2",
            },
            {
                "label": "chart",
                "bbox": [50, 600, 950, 760],
                "content": "<table><tr><td>Q1</td><td>10</td></tr></table>",
            },
        ]
    return BatchOutputItem(
        markdown="\n\n".join(c["content"] for c in chunks),
        html="<div>" + "</div><div>".join(c["content"] for c in chunks) + "</div>",
        chunks=chunks,
        raw="\n\n".join(c["content"] for c in chunks),
        page_box=[0, 0, w, h],
        token_count=150,
        images={},
        error=error,
    )


def _png_b64(width: int = 400, height: int = 300) -> str:
    img = Image.new("RGB", (width, height), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Unit tests for build_parse_result
# ---------------------------------------------------------------------------


def test_build_parse_result_single_page():
    """build_parse_result produces a valid unified-parse/v1 structure."""
    result = _make_batch_output()
    output = build_parse_result([result], model_id="paddleocr-vl-1.6-doc")

    assert output["object"] == "parse.result"
    assert output["schema"] == "unified-parse/v1"
    assert output["model"] == "paddleocr-vl-1.6-doc"
    assert output["finish_reason"] == "stop"
    assert len(output["pages"]) == 1

    page = output["pages"][0]
    assert page["index"] == 0
    assert page["number"] == 1
    assert page["size"]["width"] == 1000
    assert page["size"]["height"] == 1200
    assert page["size"]["unit"] == "px"
    assert "Title" in page["markdown"]

    blocks = page["blocks"]
    assert len(blocks) >= 4

    block_types = {b["type"] for b in blocks}
    assert "title" in block_types
    assert "paragraph" in block_types
    assert "table" in block_types
    assert "formula" in block_types

    title_block = next(b for b in blocks if b["type"] == "title")
    assert title_block["level"] == 1
    assert title_block["content"]["text"] == "Title"

    table_block = next(b for b in blocks if b["type"] == "table")
    assert "<table>" in table_block["content"]["html"]

    formula_block = next(b for b in blocks if b["type"] == "formula")
    assert formula_block["content"]["latex"] == "E=mc^2"

    chart_block = next(b for b in blocks if b["type"] == "chart")
    assert "<table>" in chart_block["content"]["html"]


def test_build_parse_result_skips_failed_pages():
    ok = _make_batch_output()
    bad = _make_batch_output(error=True)
    output = build_parse_result([ok, bad])

    assert len(output["pages"]) == 1
    assert len(output["warnings"]) == 1
    assert output["usage"]["pages_processed"] == 1


def test_build_parse_result_matches_assets_by_asset_key():
    red = Image.new("RGB", (20, 20), "red")
    blue = Image.new("RGB", (20, 20), "blue")
    chunks = [
        {
            "label": "image",
            "bbox": [0, 0, 20, 20],
            "content": "",
            "asset": "red.png",
        },
        {
            "label": "image",
            "bbox": [0, 0, 20, 20],
            "content": "",
            "asset": "blue.png",
        },
    ]
    result = BatchOutputItem(
        markdown="",
        html="",
        chunks=chunks,
        page_box=[0, 0, 100, 100],
        images={"red.png": red, "blue.png": blue},
    )

    output = build_parse_result([result], include_images=True)

    assert len(output["assets"]) == 2
    colors = []
    for asset in output["assets"]:
        raw = base64.b64decode(asset["base64"])
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        colors.append(image.getpixel((10, 10)))

    assert colors[0][0] > 200 and colors[0][2] < 100
    assert colors[1][2] > 200 and colors[1][0] < 100


def test_build_parse_result_can_omit_bounding_boxes():
    image = Image.new("RGB", (20, 20), "red")
    result = BatchOutputItem(
        markdown="",
        html="",
        chunks=[
            {
                "label": "image",
                "bbox": [0, 0, 20, 20],
                "content": "",
                "asset": "x.png",
            }
        ],
        page_box=[0, 0, 100, 100],
        images={"x.png": image},
    )
    output = build_parse_result([result], include_bounding_boxes=False)

    for block in output["pages"][0]["blocks"]:
        assert "bbox" not in block
    assert output["assets"]
    for asset in output["assets"]:
        assert "bbox" not in asset


def test_build_parse_result_can_omit_image_links_and_payloads():
    image = Image.new("RGB", (20, 20), "red")
    result = BatchOutputItem(
        markdown="before ![](imgs/x.png) after",
        html="",
        chunks=[
            {
                "label": "image",
                "bbox": [0, 0, 20, 20],
                "content": "",
                "asset": "imgs/x.png",
            }
        ],
        page_box=[0, 0, 20, 20],
        images={"imgs/x.png": image},
    )

    output = build_parse_result([result], include_images=False)

    assert "![](" not in output["pages"][0]["markdown"]
    assert output["assets"][0]["base64"] is None


# ---------------------------------------------------------------------------
# Integration tests against the FastAPI app + mock upstream
# ---------------------------------------------------------------------------


def test_parse_returns_unified_schema(parse_client):
    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "model": "paddleocr-vl-1.6-doc",
            "content": [
                {"type": "text", "text": "识别图片"},
                {"type": "image", "source": {"type": "base64", "data": _png_b64()}},
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "parse.result"
    assert data["schema"] == "unified-parse/v1"
    assert len(data["pages"]) == 1
    page = data["pages"][0]
    assert page["size"]["unit"] == "px"
    assert page["size"]["width"] == 400
    assert "blocks" in page
    assert "Mock text line" in page["markdown"]
    assert data["usage"]["pages_processed"] == 1


def test_parse_get_returns_405(parse_client):
    resp = parse_client.get("/v1/parse")
    assert resp.status_code == 405


def test_parse_missing_image(parse_client):
    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "model": "paddleocr-vl-1.6-doc",
            "content": [{"type": "text", "text": "hi"}],
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "missing_content"


def test_parse_multi_image(parse_client):
    b64 = _png_b64()
    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "model": "paddleocr-vl-1.6-doc",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": b64}},
                {"type": "image", "source": {"type": "base64", "data": b64}},
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["usage"]["pages_processed"] == 2
    assert len(data["pages"]) == 2


def test_parse_streaming_rejected(parse_client):
    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "model": "paddleocr-vl-1.6-doc",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": _png_b64()}},
            ],
            "stream": True,
        },
    )
    assert resp.status_code == 400


def test_parse_pages_selection_preserves_original_number(parse_client):
    b64 = _png_b64()
    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "content": [
                {"type": "image", "source": {"type": "base64", "data": b64}},
                {"type": "image", "source": {"type": "base64", "data": b64}},
            ],
            "options": {"pages": "2"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pages"]) == 1
    assert data["pages"][0]["index"] == 1
    assert data["pages"][0]["number"] == 2
    assert data["usage"]["pages_processed"] == 1


def test_parse_include_bounding_boxes_false(parse_client):
    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "content": [
                {"type": "image", "source": {"type": "base64", "data": _png_b64()}}
            ],
            "options": {"include_bounding_boxes": False},
        },
    )
    assert resp.status_code == 200
    blocks = resp.json()["pages"][0]["blocks"]
    assert blocks
    assert all("bbox" not in block for block in blocks)


@pytest.mark.parametrize(
    "options",
    [
        {"output_format": "html"},
        {"table_format": "markdown"},
        {"language": "zh"},
        {"include_lines": True},
        {"pages": "0"},
        {"pages": "2"},
    ],
)
def test_parse_rejects_unsupported_options(parse_client, options):
    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "content": [
                {"type": "image", "source": {"type": "base64", "data": _png_b64()}}
            ],
            "options": options,
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_option"


def test_parse_rejects_response_schema(parse_client):
    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "content": [
                {"type": "image", "source": {"type": "base64", "data": _png_b64()}}
            ],
            "response_schema": {"type": "object"},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_option"


def test_parse_warns_for_legacy_ocr_options(parse_client):
    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "content": [
                {"type": "image", "source": {"type": "base64", "data": _png_b64()}}
            ],
            "ocr_options": {"legacy": True},
        },
    )
    assert resp.status_code == 200
    warnings = resp.json()["warnings"]
    assert any("ocr_options" in warning for warning in warnings)


def test_parse_forwards_include_headers_footers(parse_client, monkeypatch):
    from docmore.paddleocr_vl_16.model import InferenceManager

    captured: dict = {}
    original = InferenceManager.generate

    def spy(self, batch):
        captured["batch"] = batch
        return original(self, batch)

    monkeypatch.setattr(InferenceManager, "generate", spy)

    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "content": [
                {"type": "image", "source": {"type": "base64", "data": _png_b64()}}
            ],
            "options": {"include_headers_footers": True},
        },
    )
    assert resp.status_code == 200
    assert captured["batch"][0].include_headers_footers is True


def test_parse_rejects_unknown_model(parse_client):
    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "model": "paddleocr_vl_1.6",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": _png_b64()}}
            ],
        },
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "model_not_found"
    assert error["param"] == "model"


@pytest.mark.parametrize(
    ("model", "block_type", "content_key"),
    [
        ("paddleocr-vl-1.6-table", "table", "html"),
        ("paddleocr-vl-1.6-text", "paragraph", "text"),
        ("paddleocr-vl-1.6-formula", "formula", "latex"),
        ("paddleocr-vl-1.6-seal", "seal", "text"),
        ("paddleocr-vl-1.6-chart", "chart", "html"),
    ],
)
def test_parse_model_alias_selects_single_block_mode(
    parse_client, model, block_type, content_key
):
    """Each non-doc alias bypasses layout and forces its own whole-page task."""
    resp = parse_client.post(
        "/v1/parse",
        headers=_headers(),
        json={
            "model": model,
            "content": [
                {"type": "image", "source": {"type": "base64", "data": _png_b64()}}
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == model
    blocks = data["pages"][0]["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == block_type
    assert blocks[0]["content"][content_key]


def test_build_parse_result_chart_caption_uses_text_content():
    result = _make_batch_output(
        chunks=[
            {
                "label": "chart",
                "bbox": [50, 600, 950, 760],
                "content": "A bar chart showing revenue growth.",
            }
        ]
    )
    output = build_parse_result([result])

    block = output["pages"][0]["blocks"][0]
    assert block["type"] == "chart"
    assert block["content"]["text"] == "A bar chart showing revenue growth."
    assert "html" not in block["content"]


def test_build_parse_result_ocr_label_maps_to_paragraph():
    result = _make_batch_output(
        chunks=[{"label": "ocr", "bbox": [0, 0, 100, 40], "content": "Body"}]
    )
    output = build_parse_result([result])

    block = output["pages"][0]["blocks"][0]
    assert block["type"] == "paragraph"
    assert block["content"]["text"] == "Body"
    assert block["meta"] == {"source_label": "ocr"}
