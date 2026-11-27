"""Core dataclasses shared by the pipeline, server and docling bridge.

Kept at the package top level (docmore 统一形式) so neither ``model`` nor
``server`` subpackages need to import each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from PIL import Image


@dataclass(frozen=True)
class LayoutBox:
    """A detected layout element (PP-DocLayoutV3 output)."""

    cls_id: int
    label: str
    score: float
    coordinate: list[int]
    order: int | None = None
    polygon_points: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.coordinate
        polygon_points = self.polygon_points or [
            [float(x1), float(y1)],
            [float(x2), float(y1)],
            [float(x2), float(y2)],
            [float(x1), float(y2)],
        ]
        return {
            "cls_id": self.cls_id,
            "label": self.label,
            "score": self.score,
            "coordinate": self.coordinate,
            "order": self.order,
            "polygon_points": polygon_points,
        }


@dataclass
class TableCell:
    """A single cell decoded from an OTSL table stream."""

    text: str
    row_span: int
    col_span: int
    start_row: int
    end_row: int
    start_col: int
    end_col: int


@dataclass
class GenerationResult:
    """Raw completion result from the VLM endpoint."""

    raw: str
    token_count: int
    error: bool = False


@dataclass
class BatchInputItem:
    """One page to process: a PIL image plus optional per-page overrides.

    ``page_index`` (0-based) gives image assets a page-aware filename so a
    multi-page merge cannot overwrite two crops with identical bboxes.
    ``include_headers_footers`` overrides the manager-wide setting for this
    page (used by the HTTP ``/v1/parse`` options).
    """

    image: Image.Image
    prompt: str | None = None
    prompt_type: str | None = None
    page_index: int | None = None
    include_headers_footers: bool | None = None


@dataclass
class BatchOutputItem:
    """One page of engine output.

    ``chunks`` carries structured blocks (docmore 统一形式):
    ``[{"label": <layout label>, "bbox": [x0, y0, x1, y1], "content": <str>}, ...]``
    with pixel-space bboxes. Labels are the PP-DocLayoutV3 classes (``text``,
    ``table``, ``display_formula``, ...); ``content`` is plain text for text
    regions, LaTeX for formula regions and HTML ``<table>`` for table/chart
    regions.
    """

    markdown: str
    html: str
    chunks: List[dict] = field(default_factory=list)
    raw: str = ""
    page_box: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    token_count: int = 0
    images: dict = field(default_factory=dict)
    error: bool = False
