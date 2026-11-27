"""Post-processing for PaddleOCR-VL 1.6 region outputs.

PaddleOCR-VL 1.6 returns task-specific formats that need light normalization
before assembly:

* ``OCR:`` / ``Spotting:`` / ``Seal Recognition:`` -> plain text (CRLF on
  llama.cpp; normalized to LF).
* ``Formula Recognition:`` -> LaTeX, usually wrapped in ``\\[...\\]`` or
  ``\\(...\\)``.
* ``Table Recognition:`` -> OTSL wrapped in ``<otsl>...</otsl>`` or HTML;
  converted to an HTML ``<table>`` here (with a Markdown pipe-table fallback
  when the model ignores the OTSL convention).
* ``Chart Recognition:`` -> a table in OTSL/HTML form on some endpoints, a
  caption on others; both are accepted and normalized accordingly.

The same functions produce the ``chunks[].content`` payload (plain text,
LaTeX or HTML) and the per-label Markdown rendering.
"""

from __future__ import annotations

import html
import re

from docmore.paddleocr_vl_16.table import convert_otsl_to_html

_OTSL_TOKEN_RE = re.compile(r"<(?:fcel|ecel|lcel|ucel|xcel|nl)>", re.IGNORECASE)
_OTSL_WRAPPER_RE = re.compile(r"</?otsl[^>]*>", re.IGNORECASE)
_TABLE_TAG_RE = re.compile(r"<table\b", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)\n?```$", re.DOTALL)
_PIPE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")

_FORMULA_WRAPPERS: tuple[tuple[str, str], ...] = (
    ("$$", "$$"),
    ("\\[", "\\]"),
    ("\\(", "\\)"),
    ("$", "$"),
)
_FORMULA_ENVIRONMENTS = ("equation", "equation*", "displaymath", "math")


def strip_code_fence(text: str) -> str:
    """Remove a single wrapping Markdown code fence (```` ```html ... ````)."""
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def normalize_text(raw: str) -> str:
    """Normalize plain-text OCR output: LF newlines, no outer whitespace."""
    text = strip_code_fence(raw.replace("\r\n", "\n").replace("\r", "\n"))
    return text.strip()


def strip_formula_delimiters(raw: str) -> str:
    """Return bare LaTeX (``\\\\[...\\\\]`` / ``$$...$$`` / ``$...$`` removed)."""
    text = strip_code_fence(raw.strip())
    for environment in _FORMULA_ENVIRONMENTS:
        begin = f"\\begin{{{environment}}}"
        end = f"\\end{{{environment}}}"
        if text.startswith(begin) and text.endswith(end):
            text = text[len(begin) : -len(end)].strip()

    changed = True
    while changed:
        changed = False
        for opening, closing in _FORMULA_WRAPPERS:
            if (
                len(text) >= len(opening) + len(closing)
                and text.startswith(opening)
                and text.endswith(closing)
            ):
                text = text[len(opening) : len(text) - len(closing)].strip()
                changed = True
    return text


def _pipe_table_to_html(text: str) -> str:
    """Fallback: convert a GitHub-style pipe table to an HTML ``<table>``."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows = [line for line in lines if "|" in line]
    if len(rows) < 2:
        return ""

    parsed: list[list[str]] = []
    for line in rows:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(_PIPE_SEPARATOR_RE.match(cell) for cell in cells):
            continue
        parsed.append(cells)
    if len(parsed) < 2:
        return ""

    width = len(parsed[0])
    if any(len(row) != width for row in parsed):
        return ""

    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        + "</tr>"
        for row in parsed
    )
    return f"<table>{body}</table>"


def _strip_html_document_wrappers(text: str) -> str:
    """Drop ``<html>`` / ``<body>`` wrappers some endpoints add to tables."""
    text = re.sub(r"^\s*<html[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</html>\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*<body[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</body>\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def html_table_from_raw(raw: str) -> str:
    """Convert a table/chart recognition result to an HTML ``<table>``.

    Handles, in order: already-HTML tables (including ``<html><body>``
    wrappers), OTSL streams (with or without the ``<otsl>`` wrapper), and
    Markdown pipe tables. Anything else is returned as normalized text so no
    information is lost -- this is how a chart caption is preserved.
    """
    text = strip_code_fence(raw.strip())
    text = _OTSL_WRAPPER_RE.sub("", text).strip()

    if _TABLE_TAG_RE.search(text):
        return _strip_html_document_wrappers(text)
    if _OTSL_TOKEN_RE.search(text):
        table_html = convert_otsl_to_html(text)
        if table_html:
            return table_html
    pipe_html = _pipe_table_to_html(text)
    if pipe_html:
        return pipe_html
    return normalize_text(text)


def normalize_region_content(label: str, task: str, raw: str) -> str:
    """Normalize one region's raw model output for ``chunks[].content``.

    Args:
        label: PP-DocLayoutV3 layout label.
        task: routing task from :func:`docmore.paddleocr_vl_16.prompts.task_for_label`.
        raw: raw model completion.

    Returns:
        Plain text, bare LaTeX, or an HTML ``<table>``.
    """
    if task in ("table", "chart"):
        return html_table_from_raw(raw)
    if task in ("formula", "inline_formula"):
        return strip_formula_delimiters(raw)
    return normalize_text(raw)


def _single_line(text: str) -> str:
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _paragraph(text: str) -> str:
    """Render text as Markdown paragraphs (single newline -> blank line)."""
    return re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text.strip())


def format_markdown_block(label: str, content: str) -> str:
    """Render one region's normalized content to its Markdown form."""
    content = content.strip()
    if not content:
        return ""
    if label == "doc_title":
        return f"# {_single_line(content)}"
    if label == "paragraph_title":
        return f"## {_single_line(content)}"
    if label == "display_formula" or label == "formula":
        return f"$$\n{content}\n$$"
    if label == "inline_formula":
        return f"${_single_line(content)}$"
    if label == "table":
        return content
    if label == "chart":
        # The model may return a table or a caption; a table is emitted raw,
        # a caption as a normal paragraph.
        return content if _TABLE_TAG_RE.search(content) else _paragraph(content)
    if label == "image":
        return content  # caller passes the already-built ![](...) link
    return _paragraph(content)


def image_filename(
    label: str,
    bbox: list[int] | tuple[int, int, int, int],
    *,
    page_index: int | None = None,
) -> str:
    """Stable asset filename for a picture region.

    With ``page_index`` the asset lives in ``imgs/page_0001/`` (1-based), which
    keeps identical bboxes on different pages from overwriting each other.
    Without it the historical flat ``imgs/`` path is kept for direct
    ``generate()`` callers.
    """
    x1, y1, x2, y2 = (int(value) for value in bbox)
    name = f"img_in_{label}_box_{x1}_{y1}_{x2}_{y2}.png"
    if page_index is None:
        return f"imgs/{name}"
    return f"imgs/page_{page_index + 1:04d}/{name}"


def image_markdown(
    label: str,
    bbox: list[int] | tuple[int, int, int, int],
    *,
    page_index: int | None = None,
) -> str:
    """Markdown image link for a picture region."""
    return f"![]({image_filename(label, bbox, page_index=page_index)})"


def prettify_markdown(text: str) -> str:
    """Collapse runs of blank lines and strip trailing whitespace (LF only)."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n") + "\n" if text.strip() else ""