"""End-to-end tests for the convert() pipeline against the mock VLM endpoint.

Starts :mod:`tests.mock_server` as the upstream VLM, points the pipeline
``settings`` singleton at it, and checks the files convert() writes.
"""

from __future__ import annotations

import json
import threading

import pytest
from PIL import Image

from mock_server import start_server  # noqa: E402

from docmore.paddleocr_vl_16 import convert  # noqa: E402
from docmore.paddleocr_vl_16.api import save_merged_output  # noqa: E402
from docmore.paddleocr_vl_16.schema import BatchOutputItem  # noqa: E402
from docmore.paddleocr_vl_16.settings import settings  # noqa: E402


@pytest.fixture(scope="module")
def mock_base_url():
    """Start the mock upstream on a free port for the module's lifetime."""
    server, port = start_server(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


@pytest.fixture()
def pointed_at_mock(monkeypatch, mock_base_url):
    """Point the settings singleton at the mock upstream (layout disabled)."""
    monkeypatch.setattr(settings, "OPENAI_API_BASE", mock_base_url)
    monkeypatch.setattr(settings, "OPENAI_MODEL_NAME", "paddleocr-vl-1.6-doc")
    monkeypatch.setattr(settings, "USE_LAYOUT_DETECTION", False)
    return mock_base_url


def _make_png(path, width=200, height=150):
    Image.new("RGB", (width, height), "white").save(path, format="PNG")
    return path


def test_convert_image_writes_markdown(tmp_path, pointed_at_mock):
    src = _make_png(tmp_path / "page.png")
    out = tmp_path / "out"

    result = convert(src, out)

    md = (result / "page" / "page.md").read_text(encoding="utf-8")
    assert "Mock text line" in md
    assert "second line" in md
    # metadata json + structured json
    assert (result / "page" / "page_metadata.json").exists()
    structured = json.loads((result / "page" / "page.json").read_text(encoding="utf-8"))
    assert structured["model"] == "paddleocr-vl-1.6-doc"
    assert structured["pages"][0]["blocks"][0]["label"] == "ocr"
    assert "Mock text line" in structured["pages"][0]["markdown"]
    assert (result / "page" / "page.html").exists()


def test_convert_docling_format(tmp_path, pointed_at_mock):
    pytest.importorskip("docling_core")
    src = _make_png(tmp_path / "page.png")
    out = tmp_path / "out"

    convert(src, out, output_format="docling", save_html=False)

    assert (out / "page" / "page.docling.json").exists()
    assert (out / "page" / "page.docling.md").exists()


def test_docling_picture_prefers_asset_key():
    pytest.importorskip("docling_core")
    from docmore.paddleocr_vl_16.docling_bridge import _picture_for_chunk

    red = Image.new("RGB", (20, 20), "red")
    blue = Image.new("RGB", (20, 20), "blue")
    chunk = {"asset": "blue.png", "bbox": [0, 0, 20, 20]}

    picture = _picture_for_chunk({"red.png": red, "blue.png": blue}, chunk, [0, 0, 20, 20])
    assert picture.getpixel((0, 0)) == (0, 0, 255)


def test_save_merged_output_dedupes_legacy_asset_keys(tmp_path):
    red = Image.new("RGB", (10, 10), "red")
    blue = Image.new("RGB", (10, 10), "blue")
    results = [
        BatchOutputItem(
            markdown="![](imgs/x.png)",
            html="![](imgs/x.png)",
            chunks=[
                {
                    "label": "image",
                    "bbox": [0, 0, 10, 10],
                    "content": "",
                    "asset": "imgs/x.png",
                }
            ],
            page_box=[0, 0, 10, 10],
            images={"imgs/x.png": red},
        ),
        BatchOutputItem(
            markdown="![](imgs/x.png)",
            html="![](imgs/x.png)",
            chunks=[
                {
                    "label": "image",
                    "bbox": [0, 0, 10, 10],
                    "content": "",
                    "asset": "imgs/x.png",
                }
            ],
            page_box=[0, 0, 10, 10],
            images={"imgs/x.png": blue},
        ),
    ]

    out = save_merged_output(
        tmp_path, "doc.pdf", results, save_json=False, save_html=False
    )

    markdown = (out / "doc.md").read_text(encoding="utf-8")
    assert "](imgs/x.png)" in markdown
    assert "](imgs/page_0002/x.png)" in markdown
    assert (out / "imgs" / "x.png").exists()
    assert (out / "imgs" / "page_0002" / "x.png").exists()


def test_convert_directory(tmp_path, pointed_at_mock):
    src_dir = tmp_path / "docs"
    src_dir.mkdir()
    for name in ("a.png", "b.png"):
        _make_png(src_dir / name)
    out = tmp_path / "out"

    convert(src_dir, out)

    assert (out / "a" / "a.md").exists()
    assert (out / "b" / "b.md").exists()


def test_convert_rejects_unknown_format(tmp_path, pointed_at_mock):
    src = _make_png(tmp_path / "page.png")
    with pytest.raises(ValueError):
        convert(src, tmp_path / "out", output_format="bogus")


def test_convert_rejects_unsupported_file(tmp_path, pointed_at_mock):
    bad = tmp_path / "file.xyz"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        convert(bad, tmp_path / "out")
