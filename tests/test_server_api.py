"""End-to-end tests for the FastAPI server routes against the mock VLM endpoint.

Starts :mod:`tests.mock_server` as the upstream VLM, builds the FastAPI app
with a client pointed at it, and exercises ``/v1/models``, ``/health``,
auth and the main POST route with fastapi's TestClient.
"""

from __future__ import annotations

import threading

import pytest
from mock_server import start_server  # noqa: E402

pytest.importorskip("fastapi")

from docmore.paddleocr_vl_16.model.client import VLMClient  # noqa: E402
from docmore.paddleocr_vl_16.server import create_app  # noqa: E402
from docmore.paddleocr_vl_16.server.config import ServeSettings  # noqa: E402
from docmore.paddleocr_vl_16.settings import settings  # noqa: E402
from docmore_utility import pil_to_base64  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_ID = "paddleocr-vl-1.6-doc"

DATA_URI = pil_to_base64(Image.new("RGB", (400, 300), "white"), with_data_uri=True)

CHAT_PATH = "/v1/chat/completions"

def _chat_body(model: str | None = None, url: str = DATA_URI, **extra):
    body: dict = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    }
    if model is not None:
        body["model"] = model
    body.update(extra)
    return body


@pytest.fixture(scope="module", autouse=True)
def _disable_layout():
    """Layout detection is irrelevant here; keep tests model-free."""
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "USE_LAYOUT_DETECTION", False)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def base_url():
    """Start the mock upstream on a free port for the module's lifetime."""
    server, port = start_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


@pytest.fixture(scope="module")
def client(base_url):
    serve_settings = ServeSettings(API_KEYS=["secret"], MAX_CONCURRENT=2)
    app = create_app(serve_settings, client=VLMClient(base_url=base_url))
    from fastapi.testclient import TestClient

    with TestClient(app) as tc:
        yield tc


def _headers(token: str = "secret"):
    return {"Authorization": f"Bearer {token}"}


# -- metadata routes ----------------------------------------------------------


def test_models_lists_aliases(client):
    resp = client.get("/v1/models", headers=_headers())
    assert resp.status_code == 200
    ids = [item["id"] for item in resp.json()["data"]]
    assert ids == [
        "paddleocr-vl-1.6-doc",
        "paddleocr-vl-1.6-table",
        "paddleocr-vl-1.6-text",
        "paddleocr-vl-1.6-formula",
        "paddleocr-vl-1.6-seal",
        "paddleocr-vl-1.6-chart",
    ]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["backend"] == MODEL_ID
    assert "layout_available" in payload


# -- auth ---------------------------------------------------------------------


def test_missing_api_key_rejected(client):
    resp = client.post(CHAT_PATH, json=_chat_body())
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_wrong_api_key_rejected(client):
    resp = client.post(
        CHAT_PATH, json=_chat_body(), headers=_headers("nope")
    )
    assert resp.status_code == 401


# -- chat completions -----------------------------------------------------------


def test_missing_image_rejected(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": MODEL_ID, "messages": [{"role": "user", "content": "text only"}]},
        headers=_headers(),
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_chat_completions_returns_markdown(client):
    resp = client.post(
        CHAT_PATH, json=_chat_body(), headers=_headers()
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == MODEL_ID
    content = data["choices"][0]["message"]["content"]
    assert "Mock text line" in content
    assert "second line" in content
    assert data["usage"]["total_tokens"] > 0


def test_chat_rejects_unknown_model(client):
    resp = client.post(
        CHAT_PATH,
        json=_chat_body(model="paddleocr_vl_1.6"),
        headers=_headers(),
    )
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "model_not_found"
    assert error["param"] == "model"


def test_chat_table_mode_forces_table_prompt(client):
    resp = client.post(
        CHAT_PATH,
        json=_chat_body(model="paddleocr-vl-1.6-table"),
        headers=_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "paddleocr-vl-1.6-table"
    assert "<table>" in data["choices"][0]["message"]["content"]
