"""Runtime settings for docmore.paddleocr_vl_16.

All defaults can be overridden via environment variables, a ``.env`` file
(shared defaults) or a ``local.env`` file (machine-local overrides), loaded
by pydantic-settings relative to the current working directory. Precedence:
real environment variables > ``local.env`` > ``.env``. The ``settings``
singleton is imported everywhere and used as default-argument values
(docmore 统一形式).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# docmore_paddleocr_vl_16/src/docmore/paddleocr_vl_16/settings.py -> docmore_paddleocr_vl_16/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def layout_model_candidates() -> list[Path]:
    """Directories searched for the PP-DocLayoutV3 ONNX model.

    Order: project-local ``models/`` first, then the sibling
    ``docmore_paddleocr_vl`` / ``docmore_glmocr`` checkouts that ship the same
    model (the ONNX file is ~124 MB, so it is not copied between projects).
    """
    return [
        _PROJECT_ROOT / "models" / "PP-DocLayoutV3-onnx",
        _PROJECT_ROOT.parent / "docmore_paddleocr_vl" / "models" / "PP-DocLayoutV3-onnx",
        _PROJECT_ROOT.parent / "docmore_glmocr" / "models" / "PP-DocLayoutV3-onnx",
    ]


def default_layout_model_dir() -> str:
    """First existing PP-DocLayoutV3 model dir, else the project-local path."""
    for candidate in layout_model_candidates():
        if (candidate / "inference.onnx").is_file():
            return str(candidate)
    return str(layout_model_candidates()[0])


class Settings(BaseSettings):
    """Package-wide configuration.

    Attributes:
        IMAGE_DPI: PDF rasterization resolution (delegated to docmore_utility).
        MIN_PDF_IMAGE_DIM: Minimum rendered short-side pixels per PDF page.
        MIN_IMAGE_DIM: Minimum short-side pixels for standalone images.
        OPENAI_API_KEY: API key for the OpenAI-compatible endpoint.
        OPENAI_API_BASE: Base URL of the PaddleOCR-VL 1.6 endpoint.
        OPENAI_MODEL_NAME: Served model alias; one of the six
            ``paddleocr-vl-1.6-<mode>`` names. It also selects the pipeline
            mode unless ``PIPELINE_MODE`` is set.
        PIPELINE_MODE: Explicit mode override (``doc`` / ``table`` / ``text``
            / ``formula`` / ``seal`` / ``chart``); empty means infer from the
            model name.
        MAX_OUTPUT_TOKENS: ``max_tokens`` for each generation.
        TEMPERATURE: Sampling temperature (0 for deterministic OCR).
        TOP_P: Nucleus sampling probability.
        IMAGE_FORMAT: Image encoding sent to the endpoint (PNG/JPEG).
        MAX_RETRIES: Per-request retries on degenerate / transient output.
        MAX_WORKERS: Thread pool size for concurrent region recognition.
        TIMEOUT: Per-request HTTP timeout in seconds.
        USE_LAYOUT_DETECTION: Run the PP-DocLayoutV3 ONNX stage; False treats
            the whole page as one OCR region.
        PROMPT_LABEL: Layout label used for the whole-page fallback region.
        LAYOUT_MODEL_DIR: Directory containing ``inference.onnx`` (defaults to
            the first existing candidate from :func:`layout_model_candidates`).
        LAYOUT_TARGET_SIZE: Square edge the layout model resizes inputs to.
        LAYOUT_THRESHOLD: Score threshold for layout detection boxes.
        LAYOUT_NMS: Apply per-class NMS to layout boxes.
        LAYOUT_UNCLIP_RATIO: Box expansion ratios applied after detection.
        LAYOUT_PROVIDER: ONNX Runtime EP ("auto" / "directml" / "cuda" / "cpu").
        INCLUDE_HEADERS_FOOTERS: Recognize and emit header/footer/aside/
            footnote/reference regions (skipped by default).
        USE_SEAL_RECOGNITION: Send seal regions to the model (plain text).
        USE_CHART_RECOGNITION: Send chart regions to the model (table HTML or
            caption text, depending on the model output).
        SHOW_FORMULA_NUMBER: Emit formula_number blocks in markdown.
        PRETTIFY_MARKDOWN: Collapse blank lines / strip trailing whitespace.
    """

    BASE_DIR: str = str(_PROJECT_ROOT)

    # Input rendering (delegated to docmore_utility)
    IMAGE_DPI: int = 200
    MIN_PDF_IMAGE_DIM: int = 1024
    MIN_IMAGE_DIM: int = 1536

    # OpenAI-compatible PaddleOCR-VL 1.6 endpoint。
    # 注意：这是"上游推理端点"，不是本服务自身的端口（本服务端口见
    # server/config.py 的 PADDLEOCR_VL_16_SERVER_PORT，默认 18910）。
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "http://127.0.0.1:8113/v1"
    OPENAI_MODEL_NAME: str = "paddleocr-vl-1.6-doc"
    MAX_OUTPUT_TOKENS: int = 4096
    TEMPERATURE: float = 0.0
    TOP_P: float = 1.0
    IMAGE_FORMAT: str = "PNG"
    MAX_RETRIES: int = 4
    MAX_WORKERS: int = 8
    TIMEOUT: float = 300.0

    # Pipeline mode: empty = infer from OPENAI_MODEL_NAME (six aliases);
    # explicit doc/table/text/formula/seal/chart overrides the inference.
    PIPELINE_MODE: str = ""

    # 第一阶段：PP-DocLayoutV3 ONNX 版面检测（25 类）。
    USE_LAYOUT_DETECTION: bool = True
    PROMPT_LABEL: str = "ocr"
    LAYOUT_MODEL_DIR: str = default_layout_model_dir()
    LAYOUT_TARGET_SIZE: int = 800
    LAYOUT_THRESHOLD: float = 0.3
    LAYOUT_NMS: bool = True
    LAYOUT_UNCLIP_RATIO: tuple = (1.0, 1.0)
    LAYOUT_PROVIDER: str = "auto"

    # 输出策略
    INCLUDE_HEADERS_FOOTERS: bool = False
    USE_SEAL_RECOGNITION: bool = True
    USE_CHART_RECOGNITION: bool = True
    SHOW_FORMULA_NUMBER: bool = True
    PRETTIFY_MARKDOWN: bool = False

    model_config = SettingsConfigDict(
        env_file=(".env", "local.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("LAYOUT_MODEL_DIR")
    @classmethod
    def _resolve_layout_dir(cls, value: str) -> str:
        """Treat an empty override as "auto-detect" instead of ``""``."""
        return value or default_layout_model_dir()


settings = Settings()