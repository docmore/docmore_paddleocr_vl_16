"""Tests for the two-stage layout + region-routing pipeline.

Layout detection is stubbed (no ONNX / model dir needed); the VLM client is a
fake that returns canned PaddleOCR-VL 1.6 task outputs, so the test checks routing,
post-processing and reading-order assembly deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from docmore.paddleocr_vl_16.model import InferenceManager
from docmore.paddleocr_vl_16.layout import _box_widths, normalize_reading_order
from docmore.paddleocr_vl_16.prompts import prompt_for_label, task_for_label
from docmore.paddleocr_vl_16.schema import (
    BatchInputItem,
    GenerationResult,
    LayoutBox,
)
from docmore.paddleocr_vl_16.settings import settings

TABLE_OTSL = "<otsl><fcel>Head<fcel>Value<nl><fcel>A<fcel>1<nl></otsl>"
CHART_OTSL = "<otsl><fcel>Q1<fcel>10<nl><fcel>Q2<fcel>20<nl></otsl>"


class FakeClient:
    """Records prompts and returns canned task-specific outputs."""

    model_name = "paddleocr-vl-1.6-doc"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, int]]] = []

    def complete(self, image: Image.Image, prompt: str | None = None):
        self.calls.append((prompt or "", image.size))
        if prompt == "Table Recognition:":
            raw = TABLE_OTSL
        elif prompt == "Chart Recognition:":
            raw = CHART_OTSL
        elif prompt == "Formula Recognition:":
            raw = "\\(x^2\\)" if image.size == (250, 40) else "\\[\nE=mc^2\n\\]"
        elif prompt == "Seal Recognition:":
            raw = "SEAL123"
        elif prompt == "Spotting:":
            raw = "SPOT: alpha"
        else:
            raw = "Mock text"
        return GenerationResult(raw=raw, token_count=1)


def _page() -> Image.Image:
    return Image.new("RGB", (1000, 1400), "white")


def _boxes() -> list[LayoutBox]:
    return [
        LayoutBox(6, "doc_title", 0.9, [50, 20, 950, 80]),
        LayoutBox(17, "paragraph_title", 0.9, [50, 100, 500, 140]),
        LayoutBox(22, "text", 0.9, [50, 160, 950, 300]),
        LayoutBox(21, "table", 0.9, [50, 320, 950, 500]),
        LayoutBox(5, "display_formula", 0.9, [50, 520, 950, 580]),
        LayoutBox(15, "inline_formula", 0.9, [50, 600, 300, 640]),
        LayoutBox(3, "chart", 0.9, [50, 700, 600, 880]),
        LayoutBox(14, "image", 0.9, [650, 700, 950, 880]),
        LayoutBox(20, "seal", 0.9, [700, 900, 900, 1000]),
        LayoutBox(12, "header", 0.9, [50, 0, 950, 18]),
    ]


def _manager(boxes, include_headers_footers=False):
    client = FakeClient()
    manager = InferenceManager(
        client=client,
        use_layout_detection=True,
        include_headers_footers=include_headers_footers,
    )
    manager._detect_boxes = lambda image: (boxes, True)  # type: ignore[method-assign]
    return manager, client


def test_two_stage_pipeline_routes_each_region():
    manager, client = _manager(_boxes())
    out = manager.generate([BatchInputItem(image=_page())])[0]

    prompts = [prompt for prompt, _ in client.calls]
    assert prompts.count("OCR:") == 3
    assert prompts.count("Table Recognition:") == 1
    assert prompts.count("Formula Recognition:") == 2
    assert prompts.count("Chart Recognition:") == 1
    assert prompts.count("Seal Recognition:") == 1
    assert len(prompts) == 8  # header abandoned, image never sent

    labels = [chunk["label"] for chunk in out.chunks]
    assert labels == [
        "doc_title",
        "paragraph_title",
        "text",
        "table",
        "display_formula",
        "inline_formula",
        "chart",
        "image",
        "seal",
    ]
    assert out.error is False
    assert out.page_box == [0, 0, 1000, 1400]
    assert out.token_count == 8

    chunks = {chunk["label"]: chunk for chunk in out.chunks}
    assert chunks["doc_title"]["content"] == "Mock text"
    assert chunks["table"]["content"].startswith("<table>")
    assert "<td>A</td>" in chunks["table"]["content"]
    assert chunks["chart"]["content"].startswith("<table>")
    assert "<td>Q1</td>" in chunks["chart"]["content"]
    assert chunks["display_formula"]["content"] == "E=mc^2"
    assert chunks["inline_formula"]["content"] == "x^2"
    assert chunks["seal"]["content"] == "SEAL123"
    assert chunks["image"]["content"] == ""


def test_reading_order_markdown_assembly():
    manager, _ = _manager(_boxes())
    out = manager.generate([BatchInputItem(image=_page())])[0]

    md = out.markdown
    assert md.startswith("# Mock text")
    assert "## Mock text" in md
    assert "<td>A</td>" in md
    assert "$$\nE=mc^2\n$$" in md
    assert "$x^2$" in md
    assert "<td>Q1</td>" in md
    assert "SEAL123" in md
    assert "![](imgs/img_in_image_box_650_700_950_880.png)" in md
    assert "header" not in md

    # Reading order: table -> formula -> chart -> image -> seal.
    order = [
        md.index("<td>A</td>"),
        md.index("E=mc^2"),
        md.index("<td>Q1</td>"),
        md.index("img_in_image_box"),
        md.index("SEAL123"),
    ]
    assert order == sorted(order)


def test_image_region_saved_as_asset():
    manager, _ = _manager(_boxes())
    out = manager.generate([BatchInputItem(image=_page())])[0]

    key = "imgs/img_in_image_box_650_700_950_880.png"
    assert key in out.images
    assert out.images[key].size == (300, 180)


def test_page_index_prefixes_assets_and_chunks():
    manager, _ = _manager(_boxes())
    out = manager.generate(
        [BatchInputItem(image=_page(), page_index=2)]
    )[0]

    key = "imgs/page_0003/img_in_image_box_650_700_950_880.png"
    assert key in out.images
    assert f"![]({key})" in out.markdown
    image_chunk = next(chunk for chunk in out.chunks if chunk["label"] == "image")
    assert image_chunk["asset"] == key


def test_per_item_headers_override_manager():
    manager, _ = _manager(_boxes(), include_headers_footers=False)
    out = manager.generate(
        [BatchInputItem(image=_page(), include_headers_footers=True)]
    )[0]

    assert "header" in [chunk["label"] for chunk in out.chunks]


def test_headers_recognized_when_enabled():
    manager, client = _manager(_boxes(), include_headers_footers=True)
    out = manager.generate([BatchInputItem(image=_page())])[0]

    prompts = [prompt for prompt, _ in client.calls]
    assert prompts.count("OCR:") == 4
    assert "header" in [chunk["label"] for chunk in out.chunks]


def test_seal_recognition_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "USE_SEAL_RECOGNITION", False)
    boxes = [LayoutBox(20, "seal", 0.9, [700, 900, 900, 1000])]
    manager, client = _manager(boxes)
    out = manager.generate([BatchInputItem(image=_page())])[0]

    assert client.calls == []
    assert out.images["imgs/img_in_seal_box_700_900_900_1000.png"].size == (200, 100)
    assert "![](" in out.markdown


def test_forced_prompt_bypasses_layout():
    manager, client = _manager(_boxes())
    out = manager.generate(
        [BatchInputItem(image=_page(), prompt="Table Recognition:", prompt_type="table")]
    )[0]

    assert [prompt for prompt, _ in client.calls] == ["Table Recognition:"]
    assert out.chunks[0]["label"] == "table"
    assert out.chunks[0]["content"].startswith("<table>")


def test_normalize_reading_order_fixes_stacked_text():
    boxes = [
        LayoutBox(22, "text", 0.78, [115, 412, 344, 441], order=1),
        LayoutBox(22, "text", 0.80, [117, 363, 283, 395], order=2),
        LayoutBox(22, "text", 0.78, [115, 456, 400, 487], order=3),
        LayoutBox(14, "image", 0.90, [475, 367, 682, 596], order=4),
    ]
    ordered = normalize_reading_order(boxes)

    assert [box.coordinate[1] for box in ordered[:3]] == [363, 412, 456]
    assert ordered[3].label == "image"  # non-text keeps its model position
    assert [box.order for box in ordered] == [1, 2, 3, 4]


def test_normalize_reading_order_keeps_columns():
    boxes = [
        LayoutBox(22, "text", 0.9, [50, 500, 300, 560]),   # left column, lower
        LayoutBox(22, "text", 0.9, [400, 100, 650, 160]),  # right column, upper
    ]
    assert normalize_reading_order(boxes) == boxes


def test_box_widths_use_x_coordinates():
    boxes = np.array(
        [
            [22, 0.9, 10, 100, 110, 140],
            [22, 0.9, 200, 0, 300, 50],
        ],
        dtype=np.float32,
    )
    assert _box_widths(boxes).tolist() == [100.0, 100.0]


def test_mask_polygon_distance_uses_max_box_width(monkeypatch):
    """Regression: ``max_box_w`` must be ``x2 - x1``, not ``x2 - y1``."""
    pytest.importorskip("cv2")
    from docmore.paddleocr_vl_16 import layout

    seen: list[float] = []
    monkeypatch.setattr(
        layout,
        "_mask_to_polygon",
        lambda mask, max_allowed_dist: seen.append(max_allowed_dist) or None,
    )

    boxes = np.array(
        [
            [22, 0.9, 10, 100, 110, 140],
            [22, 0.9, 200, 0, 300, 50],
        ],
        dtype=np.float32,
    )
    masks = np.ones((2, 140, 300), dtype=np.float32)

    layout._extract_polygon_points_by_masks(boxes, masks, (300, 140))

    # Both boxes are 100 px wide, so max_box_w = 100 and each box keeps its
    # own width as the allowed vertex distance. The old formula computed
    # max(110 - 100, 300 - 0) = 300 here.
    assert seen == [100.0, 100.0]


def test_prompt_routing_table():
    assert task_for_label("table") == "table"
    assert task_for_label("chart") == "chart"
    assert task_for_label("display_formula") == "formula"
    assert task_for_label("inline_formula") == "inline_formula"
    assert task_for_label("seal") == "seal"
    assert task_for_label("text") == "ocr"
    assert task_for_label("spotting") == "spotting"
    assert task_for_label("unknown-label") == "ocr"

    assert prompt_for_label("table") == "Table Recognition:"
    assert prompt_for_label("chart") == "Chart Recognition:"
    assert prompt_for_label("display_formula") == "Formula Recognition:"
    assert prompt_for_label("inline_formula") == "Formula Recognition:"
    assert prompt_for_label("seal") == "Seal Recognition:"
    assert prompt_for_label("spotting") == "Spotting:"
    assert prompt_for_label("text") == "OCR:"
    assert prompt_for_label("image") is None
    assert prompt_for_label("header") is None
    assert prompt_for_label("header", include_headers_footers=True) == "OCR:"

def test_forced_spotting_prompt():
    manager, client = _manager(_boxes())
    out = manager.generate(
        [
            BatchInputItem(
                image=_page(),
                prompt="Spotting:",
                prompt_type="spotting",
            )
        ]
    )[0]

    assert [prompt for prompt, _ in client.calls] == ["Spotting:"]
    assert out.chunks[0]["label"] == "spotting"
    assert out.chunks[0]["content"] == "SPOT: alpha"
    assert "SPOT: alpha" in out.markdown


def test_layout_fallback_uses_ocr_label():
    manager = InferenceManager(
        client=FakeClient(),
        use_layout_detection=False,
        include_headers_footers=False,
    )
    out = manager.generate([BatchInputItem(image=_page())])[0]

    assert out.chunks[0]["label"] == settings.PROMPT_LABEL == "ocr"
    assert out.chunks[0]["content"] == "Mock text"
