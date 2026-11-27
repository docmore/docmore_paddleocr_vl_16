"""Tests for the six model aliases and their pipeline modes."""

from __future__ import annotations

import pytest
from PIL import Image

from docmore.paddleocr_vl_16.mode import (
    DOC,
    MODES_BY_NAME,
    default_model_alias,
    model_aliases,
    resolve_model_or_doc,
    resolve_pipeline_mode,
)
from docmore.paddleocr_vl_16.model import InferenceManager
from docmore.paddleocr_vl_16.schema import BatchInputItem, GenerationResult
from docmore.paddleocr_vl_16.settings import settings


class RecordingClient:
    """Return the mock response matching the prompt and record the calls."""

    model_name = "paddleocr-vl-1.6-doc"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, image: Image.Image, prompt: str | None = None):
        prompt = prompt or ""
        self.calls.append(prompt)
        if prompt == "Table Recognition:":
            raw = "<otsl><fcel>Head<fcel>Value<nl><fcel>A<fcel>1<nl></otsl>"
        elif prompt == "Chart Recognition:":
            raw = "<otsl><fcel>Q1<fcel>10<nl></otsl>"
        elif prompt == "Formula Recognition:":
            raw = "\\[\nE=mc^2\n\\]"
        elif prompt == "Seal Recognition:":
            raw = "SEAL123"
        else:
            raw = "Mock text"
        return GenerationResult(raw=raw, token_count=1)


def _page() -> Image.Image:
    return Image.new("RGB", (400, 200), "white")


def test_model_aliases_cover_six_modes_in_order():
    assert list(model_aliases()) == [
        "paddleocr-vl-1.6-doc",
        "paddleocr-vl-1.6-table",
        "paddleocr-vl-1.6-text",
        "paddleocr-vl-1.6-formula",
        "paddleocr-vl-1.6-seal",
        "paddleocr-vl-1.6-chart",
    ]
    assert default_model_alias() == "paddleocr-vl-1.6-doc"
    assert set(MODES_BY_NAME) == {"doc", "table", "text", "formula", "seal", "chart"}


@pytest.mark.parametrize(
    ("model", "mode_name", "prompt"),
    [
        ("paddleocr-vl-1.6-doc", "doc", None),
        ("paddleocr-vl-1.6-table", "table", "Table Recognition:"),
        ("paddleocr-vl-1.6-text", "text", "OCR:"),
        ("paddleocr-vl-1.6-formula", "formula", "Formula Recognition:"),
        ("paddleocr-vl-1.6-seal", "seal", "Seal Recognition:"),
        ("paddleocr-vl-1.6-chart", "chart", "Chart Recognition:"),
    ],
)
def test_each_alias_selects_its_mode(model, mode_name, prompt):
    mode = resolve_pipeline_mode(model_name=model)
    assert mode.name == mode_name
    assert mode.prompt == prompt
    assert mode.use_layout is (mode_name == "doc")


def test_unknown_model_and_mode_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_pipeline_mode(model_name="paddleocr_vl_1.6")
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_pipeline_mode(model_name="paddleocr-vl-1.6")
    with pytest.raises(ValueError, match="Unknown pipeline mode"):
        resolve_pipeline_mode(override="bogus")


def test_non_alias_library_model_falls_back_to_doc():
    assert resolve_model_or_doc("some-gateway-model") is DOC
    assert resolve_model_or_doc(None) is DOC


def test_manager_derives_mode_from_client_model_name():
    client = RecordingClient()
    client.model_name = "paddleocr-vl-1.6-chart"
    manager = InferenceManager(client=client)

    out = manager.generate([BatchInputItem(image=_page())])[0]

    assert manager.mode.name == "chart"
    assert client.calls == ["Chart Recognition:"]
    assert out.chunks[0]["label"] == "chart"
    assert out.chunks[0]["content"].startswith("<table>")


def test_settings_pipeline_mode_overrides_client_model(monkeypatch):
    monkeypatch.setattr(settings, "PIPELINE_MODE", "table")
    manager = InferenceManager(client=RecordingClient())

    assert manager.mode.name == "table"


def test_explicit_mode_argument_wins_over_settings(monkeypatch):
    monkeypatch.setattr(settings, "PIPELINE_MODE", "chart")
    manager = InferenceManager(client=RecordingClient(), mode="table")

    assert manager.mode.name == "table"


def test_manager_rejects_unknown_explicit_mode():
    with pytest.raises(ValueError, match="Unknown pipeline mode"):
        InferenceManager(client=RecordingClient(), mode="bogus")


def test_manager_single_block_mode_bypasses_layout():
    client = RecordingClient()
    manager = InferenceManager(client=client, mode="table")
    # Layout must never be consulted in single-block mode.
    manager._detect_boxes = lambda image: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("layout called")
    )

    out = manager.generate([BatchInputItem(image=_page())])[0]

    assert manager.mode.name == "table"
    assert client.calls == ["Table Recognition:"]
    assert out.chunks[0]["label"] == "table"
    assert "<td>A</td>" in out.chunks[0]["content"]


def test_mode_seal_overrides_disabled_recognition_setting(monkeypatch):
    monkeypatch.setattr(settings, "USE_SEAL_RECOGNITION", False)
    client = RecordingClient()
    manager = InferenceManager(client=client, mode="seal")

    out = manager.generate([BatchInputItem(image=_page())])[0]

    assert client.calls == ["Seal Recognition:"]
    assert out.chunks[0]["label"] == "seal"
    assert out.chunks[0]["content"] == "SEAL123"


def test_mode_chart_overrides_disabled_recognition_setting(monkeypatch):
    monkeypatch.setattr(settings, "USE_CHART_RECOGNITION", False)
    client = RecordingClient()
    manager = InferenceManager(client=client, mode="chart")

    out = manager.generate([BatchInputItem(image=_page())])[0]

    assert client.calls == ["Chart Recognition:"]
    assert out.chunks[0]["label"] == "chart"
    assert out.chunks[0]["content"].startswith("<table>")


def test_explicit_item_prompt_wins_over_mode():
    client = RecordingClient()
    manager = InferenceManager(client=client, mode="table")

    out = manager.generate(
        [
            BatchInputItem(
                image=_page(),
                prompt="Seal Recognition:",
                prompt_type="seal",
            )
        ]
    )[0]

    assert client.calls == ["Seal Recognition:"]
    assert out.chunks[0]["label"] == "seal"
    assert out.chunks[0]["content"] == "SEAL123"


def test_doc_mode_keeps_layout_path(monkeypatch):
    client = RecordingClient()
    manager = InferenceManager(client=client, mode="doc")
    assert manager.mode.name == "doc"

    # With layout disabled it falls back to the whole-page OCR label.
    monkeypatch.setattr(settings, "USE_LAYOUT_DETECTION", False)
    out = manager.generate([BatchInputItem(image=_page())])[0]

    assert client.calls == ["OCR:"]
    assert out.chunks[0]["label"] == "ocr"