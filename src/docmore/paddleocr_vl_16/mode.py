"""Pipeline-mode resolution for docmore.paddleocr_vl_16.

The PaddleOCR-VL 1.6 endpoint serves six model aliases, each selecting a
different pipeline behaviour:

========================  ==============================  ======================
model                     pipeline behaviour              final prompt
========================  ==============================  ======================
paddleocr-vl-1.6-doc      full two-stage layout + VLM     derived per block label
paddleocr-vl-1.6-table    whole-page single block         Table Recognition:
paddleocr-vl-1.6-text     whole-page single block         OCR:
paddleocr-vl-1.6-formula  whole-page single block         Formula Recognition:
paddleocr-vl-1.6-seal     whole-page single block         Seal Recognition:
paddleocr-vl-1.6-chart    whole-page single block         Chart Recognition:
========================  ==============================  ======================

The mode is derived from ``OPENAI_MODEL_NAME`` by default. An explicit
``PIPELINE_MODE`` / ``mode=`` override wins over the model-name inference.
Only the six aliases are accepted by :func:`resolve_pipeline_mode`; a server
request with any other model is rejected with ``model_not_found``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineMode:
    """A pipeline configuration resolved from a model alias.

    Attributes:
        name: canonical mode name (``"doc"`` / ``"table"`` / ...).
        block_label: layout label applied to the whole page in single-block
            modes (``None`` for ``"doc"``, which runs per-block layout).
        use_layout: run the PP-DocLayoutV3 two-stage pipeline.
        prompt: forced whole-page prompt for single-block modes
            (``None`` for ``"doc"``).
    """

    name: str
    block_label: str | None
    use_layout: bool
    prompt: str | None = None


DOC = PipelineMode(name="doc", block_label=None, use_layout=True)
TABLE = PipelineMode(
    name="table",
    block_label="table",
    use_layout=False,
    prompt="Table Recognition:",
)
TEXT = PipelineMode(
    name="text",
    block_label="ocr",
    use_layout=False,
    prompt="OCR:",
)
FORMULA = PipelineMode(
    name="formula",
    block_label="display_formula",
    use_layout=False,
    prompt="Formula Recognition:",
)
SEAL = PipelineMode(
    name="seal",
    block_label="seal",
    use_layout=False,
    prompt="Seal Recognition:",
)
CHART = PipelineMode(
    name="chart",
    block_label="chart",
    use_layout=False,
    prompt="Chart Recognition:",
)

MODES_BY_NAME: dict[str, PipelineMode] = {
    mode.name: mode for mode in (DOC, TABLE, TEXT, FORMULA, SEAL, CHART)
}

#: Served model-alias prefix; the six names are ``<prefix>-<mode>``.
DEFAULT_MODEL_PREFIX = "paddleocr-vl-1.6"


def model_aliases(prefix: str = DEFAULT_MODEL_PREFIX) -> dict[str, PipelineMode]:
    """Return the six supported model aliases in mode order."""
    return {f"{prefix}-{name}": mode for name, mode in MODES_BY_NAME.items()}


def default_model_alias(prefix: str = DEFAULT_MODEL_PREFIX) -> str:
    """Return the default model alias, ``<prefix>-doc``."""
    return f"{prefix}-doc"


def resolve_pipeline_mode(
    model_name: str | None = None,
    override: str | None = None,
    *,
    prefix: str = DEFAULT_MODEL_PREFIX,
) -> PipelineMode:
    """Resolve the pipeline mode for a model name / explicit override.

    Args:
        model_name: served model alias. ``None`` / blank defaults to doc mode.
        override: explicit mode name (``"doc"``/``"table"``/...) which wins
            over model-name inference.
        prefix: model-alias prefix (default ``paddleocr-vl-1.6``).

    Returns:
        The matching :class:`PipelineMode`.

    Raises:
        ValueError: if ``override`` is not a known mode name, or
            ``model_name`` is not one of the six supported aliases.
    """
    if override:
        name = override.strip().lower()
        if name not in MODES_BY_NAME:
            raise ValueError(
                f"Unknown pipeline mode {override!r}; expected one of "
                f"{sorted(MODES_BY_NAME)}"
            )
        return MODES_BY_NAME[name]
    if model_name is None or not model_name.strip():
        return DOC
    aliases = model_aliases(prefix)
    mode = aliases.get(model_name.strip())
    if mode is None:
        raise ValueError(
            f"Unknown model {model_name!r}; supported models: "
            f"{', '.join(aliases)}"
        )
    return mode


def resolve_model_or_doc(model_name: str | None) -> PipelineMode:
    """Resolve a model alias, falling back to doc for non-alias names.

    Library callers may point ``VLMClient`` at a gateway with its own model
    naming. Such names keep the full two-stage behaviour instead of failing;
    the HTTP layer still enforces the six-alias contract.
    """
    try:
        return resolve_pipeline_mode(model_name=model_name)
    except ValueError:
        return DOC