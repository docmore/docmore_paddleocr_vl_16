"""Prompts and layout-label routing for the PaddleOCR-VL 1.6 engine.

PaddleOCR-VL 1.6 is a task-prompted OCR VLM: each cropped layout region is
sent with a short task prompt that selects the output modality. The routing
follows the upstream prompt table:

============================  ============================  ===============
Layout region (PP-DocLayoutV3)  Prompt                       Output
============================  ============================  ===============
text / paragraph / title / ...  ``OCR:``                      plain text
table                           ``Table Recognition:``        OTSL / HTML
display_formula                 ``Formula Recognition:``      LaTeX (display)
inline_formula                  ``Formula Recognition:``      LaTeX (inline)
chart                           ``Chart Recognition:``        table HTML / text
seal                            ``Seal Recognition:``         plain text
============================  ============================  ===============

``Spotting:`` is part of the PaddleOCR-VL prompt set but PP-DocLayoutV3 has no
matching layout class. It is available to callers that force a whole-page or
region prompt through ``BatchInputItem(prompt="Spotting:",
prompt_type="spotting")``.

Table streams are normally returned as OTSL (``<fcel>`` / ``<nl>`` ...) and
converted to HTML during post-processing (:mod:`docmore.paddleocr_vl_16.
postprocess`); newer endpoints may return HTML directly, which is passed
through unchanged. Chart recognition may return either a table or a caption;
both are accepted.

Header / footer / page-number / footnote / aside / reference regions are
abandoned by default (``INCLUDE_HEADERS_FOOTERS=true`` to recognize them).
Image regions are kept as picture assets and never sent to the model.
"""

from __future__ import annotations

SYSTEM_PROMPT = ""  # PaddleOCR-VL 1.6 is driven by the task prompt only.

# 上游 task -> prompt 映射（与 PaddleOCR-VL 1.6 的任务提示词一致）
TASK_PROMPTS: dict[str, str] = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "inline_formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}

# PP-DocLayoutV3 25 类 -> 任务类型。``ocr`` / ``formula`` / ``spotting``
# 同时充当合成 label，供无版面回退和调用方强制 prompt 使用。
LABEL_TASK_MAPPING: dict[str, list[str]] = {
    "ocr": [
        "abstract",
        "algorithm",
        "content",
        "doc_title",
        "figure_title",
        "formula_number",
        "ocr",
        "paragraph_title",
        "reference_content",
        "text",
        "vertical_text",
        "vision_footnote",
    ],
    "table": ["table"],
    "formula": ["display_formula", "formula"],
    "inline_formula": ["inline_formula"],
    "seal": ["seal"],
    "chart": ["chart"],
    "spotting": ["spotting"],
    "skip": ["image", "header_image", "footer_image"],
    "abandon": [
        "aside_text",
        "footnote",
        "footer",
        "header",
        "number",
        "reference",
    ],
}

# 反查表：native label -> task type
_LABEL_TO_TASK: dict[str, str] = {
    label: task
    for task, labels in LABEL_TASK_MAPPING.items()
    for label in labels
}

# 显式 prompt -> 任务类型（batch 调用方直接指定 prompt 时用）
_PROMPT_TO_TASK: dict[str, str] = {
    "OCR:": "ocr",
    "Table Recognition:": "table",
    "Formula Recognition:": "formula",
    "Chart Recognition:": "chart",
    "Spotting:": "spotting",
    "Seal Recognition:": "seal",
}

# 保留为图片资产（识别结果不作为文本输出）的 label
IMAGE_LABELS = {"image", "header_image", "footer_image"}

# 默认丢弃、INCLUDE_HEADERS_FOOTERS=true 时按文本识别的 label
ABANDON_LABELS = set(LABEL_TASK_MAPPING["abandon"])


def task_for_label(label: str) -> str:
    """Return the task type for a layout label.

    Unknown labels (including the synthetic whole-page ``ocr`` label) fall
    back to ``"ocr"``.
    """
    return _LABEL_TO_TASK.get(label, "ocr")


def task_for_prompt(prompt: str) -> str:
    """Infer the task type from an explicit prompt override.

    ``Formula Recognition:`` is used by both display and inline formula
    tasks; callers that need the inline rendering should set
    ``BatchInputItem.prompt_type`` to ``"inline_formula"``.
    """
    return _PROMPT_TO_TASK.get(prompt, "ocr")


def prompt_for_label(
    label: str,
    *,
    include_headers_footers: bool = False,
    use_seal_recognition: bool = True,
    use_chart_recognition: bool = True,
) -> str | None:
    """Return the PaddleOCR-VL 1.6 task prompt for a layout label, or ``None``.

    Args:
        label: PP-DocLayoutV3 layout label.
        include_headers_footers: recognize header/footer/aside/footnote/
            reference regions instead of abandoning them.
        use_seal_recognition: send seal regions to the model.
        use_chart_recognition: send chart regions to the model.

    Returns:
        A prompt string, or ``None`` when the region should not be sent to
        the model (image asset / abandoned label / disabled task).
    """
    task = task_for_label(label)
    if task == "skip":
        return None
    if task == "abandon":
        if not include_headers_footers:
            return None
        task = "ocr"
    if task == "seal" and not use_seal_recognition:
        return None
    if task == "chart" and not use_chart_recognition:
        return None
    return TASK_PROMPTS[task]