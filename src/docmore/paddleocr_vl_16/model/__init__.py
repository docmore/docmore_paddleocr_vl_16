"""Inference manager -- the PaddleOCR-VL 1.6 two-stage pipeline.

For each page the manager runs:

1. **Layout detection** -- PP-DocLayoutV3 ONNX (25 classes) produces regions
   in reading order; when ONNX / the model dir is unavailable, or detection
   returns nothing, the page falls back to a single whole-page OCR region.
2. **Region recognition** -- every region is cropped and sent to the
   PaddleOCR-VL 1.6 OpenAI endpoint with the task prompt routed by its label
   (``OCR:`` / ``Table Recognition:`` / ``Formula Recognition:`` /
   ``Seal Recognition:`` / ``Chart Recognition:``).
3. **Post-processing & assembly** -- OTSL -> HTML tables, LaTeX normalization,
   and reading-order Markdown (titles, formulas, tables, image links).

The public entry point is :meth:`InferenceManager.generate`, which preserves
the ``generate(batch) -> list[BatchOutputItem]`` contract used by the CLI and
the FastAPI server.
"""

from __future__ import annotations

import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from PIL import Image

from docmore.paddleocr_vl_16.geometry import _filter_overlap_boxes
from docmore.paddleocr_vl_16.layout import (
    LAYOUT_MERGE_MODE,
    PPDocLayoutV3Onnx,
    normalize_reading_order,
    resolve_layout_providers,
)
from docmore.paddleocr_vl_16.model.client import VLMClient
from docmore.paddleocr_vl_16.mode import (
    PipelineMode,
    resolve_model_or_doc,
    resolve_pipeline_mode,
)
from docmore.paddleocr_vl_16.postprocess import (
    format_markdown_block,
    image_filename,
    image_markdown,
    normalize_region_content,
    prettify_markdown,
)
from docmore.paddleocr_vl_16.prompts import (
    IMAGE_LABELS,
    prompt_for_label,
    task_for_label,
    task_for_prompt,
)
from docmore.paddleocr_vl_16.schema import BatchInputItem, BatchOutputItem, LayoutBox
from docmore.paddleocr_vl_16.settings import settings

__all__ = ["InferenceManager", "PipelineMode", "VLMClient", "layout_available"]

_HEADER_FOOTER_IMAGE_LABELS = {"header_image", "footer_image"}


def layout_available() -> bool:
    """Whether the ONNX layout stage can run (onnxruntime + inference.onnx)."""
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return (Path(settings.LAYOUT_MODEL_DIR) / "inference.onnx").is_file()


class InferenceManager:
    """Two-stage layout-then-recognize manager backed by a VLMClient.

    Args:
        client: pre-built :class:`VLMClient` (tests point this at a mock
            endpoint); one is built from settings when omitted.
        max_workers: thread pool size for concurrent page / region
            recognition.
        use_layout_detection: override :attr:`settings.USE_LAYOUT_DETECTION`.
        include_headers_footers: override
            :attr:`settings.INCLUDE_HEADERS_FOOTERS`.
        max_output_tokens / max_retries: forwarded to a fresh VLMClient
            (ignored when ``client`` is given; api.py passes these).
        mode: explicit pipeline mode name (``doc`` / ``table`` / ``text`` /
            ``formula`` / ``seal`` / ``chart``) or a :class:`PipelineMode`.
            When omitted, ``settings.PIPELINE_MODE`` wins, then the client's
            model alias (the six ``paddleocr-vl-1.6-*`` names); other model
            names keep the full doc pipeline.
    """

    def __init__(
        self,
        client: VLMClient | None = None,
        max_workers: int | None = None,
        use_layout_detection: bool | None = None,
        include_headers_footers: bool | None = None,
        max_output_tokens: int | None = None,
        max_retries: int | None = None,
        mode: PipelineMode | str | None = None,
    ) -> None:
        if client is None:
            client_kwargs: dict = {}
            if max_output_tokens is not None:
                client_kwargs["max_output_tokens"] = max_output_tokens
            if max_retries is not None:
                client_kwargs["max_retries"] = max_retries
            client = VLMClient(**client_kwargs)
        self.client = client
        self.max_workers = max_workers or settings.MAX_WORKERS
        if isinstance(mode, PipelineMode):
            self.mode = mode
        elif mode:
            self.mode = resolve_pipeline_mode(override=mode)
        elif settings.PIPELINE_MODE:
            self.mode = resolve_pipeline_mode(override=settings.PIPELINE_MODE)
        else:
            self.mode = resolve_model_or_doc(self.client.model_name)
        self.use_layout_detection = (
            settings.USE_LAYOUT_DETECTION
            if use_layout_detection is None
            else use_layout_detection
        )
        self.include_headers_footers = (
            settings.INCLUDE_HEADERS_FOOTERS
            if include_headers_footers is None
            else include_headers_footers
        )
        self._layout_model: PPDocLayoutV3Onnx | None = None
        self._layout_failed = False
        self._layout_cpu_fallback = False
        self._layout_lock = threading.Lock()

    # -- layout backend ----------------------------------------------------

    def _build_layout(
        self, providers: list[str], requested_provider: str
    ) -> PPDocLayoutV3Onnx:
        """Build one ONNX session with the given execution providers."""
        return PPDocLayoutV3Onnx(
            model_dir=settings.LAYOUT_MODEL_DIR,
            providers=providers,
            requested_provider=requested_provider,
            target_size=settings.LAYOUT_TARGET_SIZE,
        )

    def _layout(self) -> PPDocLayoutV3Onnx:
        """Lazily build (and cache) the ONNX layout detector.

        The double-checked lock keeps concurrent first requests from building
        duplicate ONNX sessions once the HTTP layer starts running pipelines
        in parallel.
        """
        if self._layout_model is None:
            with self._layout_lock:
                if self._layout_model is None:
                    import onnxruntime

                    available = onnxruntime.get_available_providers()
                    providers = resolve_layout_providers(
                        available, settings.LAYOUT_PROVIDER, platform.system()
                    )
                    self._layout_model = self._build_layout(
                        providers, settings.LAYOUT_PROVIDER
                    )
        return self._layout_model

    def _fallback_layout(self) -> PPDocLayoutV3Onnx | None:
        """Rebuild the layout session on CPU once, under the layout lock.

        Returns ``None`` when the CPU retry is unavailable or already failed.
        """
        with self._layout_lock:
            if self._layout_failed:
                return None
            if self._layout_cpu_fallback and self._layout_model is not None:
                return self._layout_model
            self._layout_cpu_fallback = True
            try:
                self._layout_model = self._build_layout(
                    ["CPUExecutionProvider"], "cpu"
                )
            except Exception as cpu_exc:  # noqa: BLE001
                self._layout_failed = True
                print(
                    f"[paddleocr-vl-16] CPU layout fallback failed ({cpu_exc}); "
                    "using whole-page OCR"
                )
                return None
            return self._layout_model

    def _predict_layout(self, image: Image.Image) -> list[LayoutBox]:
        """Run one layout inference and drop overlapping boxes."""
        boxes, _ = self._layout().predict_image(
            image,
            threshold=settings.LAYOUT_THRESHOLD,
            layout_nms=settings.LAYOUT_NMS,
            layout_unclip_ratio=tuple(settings.LAYOUT_UNCLIP_RATIO),
            layout_merge_bboxes_mode=LAYOUT_MERGE_MODE,
        )
        return normalize_reading_order(_filter_overlap_boxes(boxes))

    def _detect_boxes(self, image: Image.Image) -> tuple[list[LayoutBox], bool]:
        """Run layout detection; fall back to one whole-page text region."""
        width, height = image.size
        if self.use_layout_detection and not self._layout_failed:
            try:
                boxes = self._predict_layout(image)
                if boxes:
                    return boxes, True
            except Exception as exc:  # noqa: BLE001 - retry on CPU, then fall back
                # A CUDA-enabled onnxruntime without cuDNN fails at run time;
                # retry the same model on CPU once before giving up.
                fallback = self._fallback_layout()
                if fallback is not None:
                    try:
                        boxes = self._predict_layout(image)
                        if boxes:
                            return boxes, True
                    except Exception as cpu_exc:  # noqa: BLE001
                        exc = cpu_exc
                self._layout_failed = True
                print(
                    f"[paddleocr-vl-16] layout detection failed ({exc}); "
                    "using whole-page OCR"
                )
        return [
            LayoutBox(
                cls_id=0,
                label=settings.PROMPT_LABEL,
                score=1.0,
                coordinate=[0, 0, width, height],
                order=0,
            )
        ], False

    # -- per-page pipeline --------------------------------------------------

    def _generate_one(self, item: BatchInputItem) -> BatchOutputItem:
        image = item.image.convert("RGB")
        width, height = image.size
        page_box = [0, 0, width, height]

        # Stage 1: mode-level whole-page task, caller prompt, or layout.
        forced_prompt: str | None
        forced_label: str | None
        if item.prompt:
            forced_prompt = item.prompt
            forced_label = item.prompt_type
        elif not self.mode.use_layout:
            forced_prompt = self.mode.prompt
            forced_label = item.prompt_type or self.mode.block_label
        else:
            forced_prompt = None
            forced_label = None

        if forced_prompt:
            label = forced_label or settings.PROMPT_LABEL
            boxes = [
                LayoutBox(
                    cls_id=0,
                    label=label,
                    score=1.0,
                    coordinate=[0, 0, width, height],
                    order=0,
                )
            ]
        else:
            boxes, _ = self._detect_boxes(image)

        # Stage 2: route + crop regions, then recognize concurrently.
        include_headers_footers = (
            self.include_headers_footers
            if item.include_headers_footers is None
            else item.include_headers_footers
        )
        tasks: list[tuple[int, LayoutBox, str, str, Image.Image]] = []
        images: dict[str, Image.Image] = {}
        image_links: dict[int, str] = {}
        asset_keys: dict[int, str] = {}
        for index, box in enumerate(boxes):
            label = box.label
            x1, y1, x2, y2 = box.coordinate
            if x2 <= x1 or y2 <= y1:
                continue

            if label in IMAGE_LABELS:
                if label in _HEADER_FOOTER_IMAGE_LABELS and not include_headers_footers:
                    continue
                crop = image.crop((x1, y1, x2, y2))
                key = image_filename(
                    label, box.coordinate, page_index=item.page_index
                )
                images[key] = crop
                asset_keys[index] = key
                image_links[index] = image_markdown(
                    label, box.coordinate, page_index=item.page_index
                )
                continue

            if label == "formula_number" and not settings.SHOW_FORMULA_NUMBER:
                continue

            prompt = forced_prompt or prompt_for_label(
                label,
                include_headers_footers=include_headers_footers,
                use_seal_recognition=settings.USE_SEAL_RECOGNITION,
                use_chart_recognition=settings.USE_CHART_RECOGNITION,
            )
            if forced_prompt:
                task = (
                    task_for_label(forced_label)
                    if forced_label
                    else task_for_prompt(forced_prompt)
                )
            else:
                task = task_for_label(label)
            if prompt is None:
                # Recognition disabled (seal/chart) -> keep the region as an
                # image asset so no information is lost.
                if task in ("seal", "chart"):
                    crop = image.crop((x1, y1, x2, y2))
                    key = image_filename(
                        label, box.coordinate, page_index=item.page_index
                    )
                    images[key] = crop
                    asset_keys[index] = key
                    image_links[index] = image_markdown(
                        label, box.coordinate, page_index=item.page_index
                    )
                continue

            crop = image.crop((x1, y1, x2, y2))
            tasks.append((index, box, prompt, task, crop))

        contents: dict[int, str] = {}
        token_count = 0
        failures = 0
        if tasks:

            def _do(task: tuple[int, LayoutBox, str, str, Image.Image]):
                idx, box, prompt, region_task, crop = task
                return idx, box, region_task, self.client.complete(crop, prompt=prompt)

            worker_count = min(max(1, self.max_workers), len(tasks))
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                for idx, box, region_task, result in pool.map(_do, tasks):
                    if result.error:
                        failures += 1
                        continue
                    token_count += result.token_count
                    content = normalize_region_content(box.label, region_task, result.raw)
                    if content.strip():
                        contents[idx] = content

        # Stage 3: reading-order chunks + Markdown.
        chunks: list[dict] = []
        markdown_parts: list[str] = []
        raw_parts: list[str] = []
        for index, box in enumerate(boxes):
            if index in image_links:
                chunks.append(
                    {
                        "label": box.label,
                        "bbox": box.coordinate,
                        "content": "",
                        "asset": asset_keys[index],
                    }
                )
                markdown_parts.append(image_links[index])
                continue

            content = contents.get(index)
            if content is None:
                continue
            chunks.append(
                {"label": box.label, "bbox": box.coordinate, "content": content}
            )
            markdown = format_markdown_block(box.label, content)
            if markdown:
                markdown_parts.append(markdown)
            raw_parts.append(f"<{box.label}>\n{content}")

        markdown = "\n\n".join(part for part in markdown_parts if part)
        if settings.PRETTIFY_MARKDOWN:
            markdown = prettify_markdown(markdown)

        return BatchOutputItem(
            markdown=markdown,
            html=markdown,
            chunks=chunks,
            raw="\n\n".join(raw_parts),
            page_box=page_box,
            token_count=token_count,
            images=images,
            error=not markdown and failures > 0,
        )

    # 注意：勿改此契约 —— server/router.py 与 api.py/cli.py 全部依赖
    # generate(batch) -> list[BatchOutputItem]。
    def generate(self, batch: List[BatchInputItem]) -> List[BatchOutputItem]:
        """Run the batch through the thread pool, preserving input order."""
        if not batch:
            return []
        if len(batch) == 1:
            return [self._generate_one(batch[0])]
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as pool:
            return list(pool.map(self._generate_one, batch))