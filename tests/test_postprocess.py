"""Unit tests for PaddleOCR-VL 1.6 output post-processing."""

from __future__ import annotations

import pytest

from docmore.paddleocr_vl_16.postprocess import (
    format_markdown_block,
    html_table_from_raw,
    image_filename,
    normalize_region_content,
    normalize_text,
    strip_formula_delimiters,
)


def test_otsl_wrapper_converted_to_html_table():
    html = html_table_from_raw(
        "<otsl><fcel>姓名<fcel>年龄<nl><fcel>张三<fcel>28<nl></otsl>"
    )
    assert html.startswith("<table>")
    assert "<td>姓名</td><td>年龄</td>" in html
    assert "<td>张三</td><td>28</td>" in html
    assert "otsl" not in html


def test_otsl_without_wrapper_converted():
    html = html_table_from_raw("<fcel>A<fcel>B<nl><fcel>1<fcel>2<nl>")
    assert html == (
        "<table><tr><td>A</td><td>B</td></tr>"
        "<tr><td>1</td><td>2</td></tr></table>"
    )


def test_existing_html_table_kept_verbatim():
    raw = '<table><tr><td>A</td></tr></table>'
    assert html_table_from_raw(raw) == raw


def test_fenced_html_table_unfenced():
    raw = "```html\n<table><tr><td>A</td></tr></table>\n```"
    assert html_table_from_raw(raw) == "<table><tr><td>A</td></tr></table>"


def test_html_document_wrappers_stripped():
    raw = "<html><body><table><tr><td>A</td></tr></table></body></html>"
    assert html_table_from_raw(raw) == "<table><tr><td>A</td></tr></table>"


def test_chart_caption_is_preserved_as_text():
    raw = "A bar chart showing quarterly revenue growth."
    assert html_table_from_raw(raw) == raw


def test_markdown_pipe_table_fallback():
    raw = "姓名 | 年龄\n--- | ---\n张三 | 28"
    html = html_table_from_raw(raw)
    assert html == (
        "<table><tr><td>姓名</td><td>年龄</td></tr>"
        "<tr><td>张三</td><td>28</td></tr></table>"
    )


def test_pipe_table_without_separator_accepted():
    # PaddleOCR-VL 1.6 Chart Recognition returns this separator-less form.
    raw = "Quarter | Revenue\nQ1 | 140\nQ2 | 220"
    html = html_table_from_raw(raw)
    assert html == (
        "<table><tr><td>Quarter</td><td>Revenue</td></tr>"
        "<tr><td>Q1</td><td>140</td></tr>"
        "<tr><td>Q2</td><td>220</td></tr></table>"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\\[\nE=mc^2\n\\]", "E=mc^2"),
        ("$$E=mc^2$$", "E=mc^2"),
        ("\\(x^{2}\\)", "x^{2}"),
        ("$x$", "x"),
        ("```latex\n\\frac{1}{2}\n```", "\\frac{1}{2}"),
        ("\\begin{equation}\\alpha\\end{equation}", "\\alpha"),
    ],
)
def test_strip_formula_delimiters(raw, expected):
    assert strip_formula_delimiters(raw) == expected


def test_normalize_text_crlf():
    assert normalize_text("a\r\nb\r\n") == "a\nb"


def test_normalize_region_content_by_task():
    assert normalize_region_content("table", "table", "<otsl><fcel>A<nl></otsl>") == (
        "<table><tr><td>A</td></tr></table>"
    )
    assert normalize_region_content("display_formula", "formula", "\\[x^2\\]") == "x^2"
    assert normalize_region_content(
        "inline_formula", "inline_formula", "\\(x^2\\)"
    ) == "x^2"
    assert normalize_region_content("text", "ocr", "a\r\nb") == "a\nb"
    assert normalize_region_content("spotting", "spotting", "a\r\nb") == "a\nb"
    assert normalize_region_content(
        "chart", "chart", "A bar chart showing revenue."
    ) == "A bar chart showing revenue."


def test_markdown_rendering():
    assert format_markdown_block("doc_title", " Annual Report ") == "# Annual Report"
    assert format_markdown_block("paragraph_title", "1. Intro") == "## 1. Intro"
    assert format_markdown_block("display_formula", "E=mc^2") == "$$\nE=mc^2\n$$"
    assert format_markdown_block("inline_formula", "x^2") == "$x^2$"
    assert format_markdown_block("table", "<table></table>") == "<table></table>"
    assert format_markdown_block("chart", "<table></table>") == "<table></table>"
    assert format_markdown_block("chart", "Revenue grew.") == "Revenue grew."
    # Single newlines become paragraph breaks; blank lines are preserved.
    assert format_markdown_block("text", "line1\nline2") == "line1\n\nline2"
    assert format_markdown_block("ocr", "line1\nline2") == "line1\n\nline2"
    assert format_markdown_block("spotting", "alpha") == "alpha"
    assert format_markdown_block("text", "para1\n\npara2") == "para1\n\npara2"


def test_image_filename():
    assert image_filename("image", [10, 20, 30, 40]) == (
        "imgs/img_in_image_box_10_20_30_40.png"
    )