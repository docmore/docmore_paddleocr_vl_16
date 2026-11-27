"""FastAPI application and route handlers for the PaddleOCR-VL 1.6 service.

The server is a thin protocol-translation layer: request validation, image
decoding, and response shaping live here, while all recognition work is
delegated to the existing
:class:`~docmore.paddleocr_vl_16.model.InferenceManager` pipeline.

Handlers are async but never run the pipeline or remote fetches on the event
loop: both are offloaded with :func:`asyncio.to_thread`. The ``/v1/parse``
semaphore bounds concurrent in-flight requests so a nested block-extraction
thread pool cannot multiply out of control.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from docmore.paddleocr_vl_16 import __version__
from docmore.paddleocr_vl_16.input import parse_range_str
from docmore.paddleocr_vl_16.mode import (
    PipelineMode,
    default_model_alias,
    model_aliases,
    resolve_pipeline_mode,
)
from docmore.paddleocr_vl_16.model import InferenceManager, layout_available
from docmore.paddleocr_vl_16.model.client import VLMClient
from docmore.paddleocr_vl_16.schema import BatchInputItem
from docmore.paddleocr_vl_16.server.config import ServeSettings
from docmore.paddleocr_vl_16.server.images import ImageSourceError, load_image_source
from docmore.paddleocr_vl_16.server.parse import build_parse_result
from docmore.paddleocr_vl_16.server.schemas import ParseRequest
from docmore.paddleocr_vl_16.settings import settings

MODEL_ID = default_model_alias()
MODELS = model_aliases()
_PARTIAL_HEADER = "X-PaddleocrVl16-Partial"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_body(
    message: str,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """Shape any error as the OpenAI error envelope."""
    return {
        "error": {"message": message, "type": error_type, "param": param, "code": code}
    }


def _unsupported_option(param: str, message: str) -> JSONResponse:
    """400 for a ParseOptions value this service cannot honour."""
    return JSONResponse(
        status_code=400,
        content=_error_body(
            message,
            "invalid_request_error",
            param=param,
            code="unsupported_option",
        ),
    )


def _resolve_request_model(
    model: str | None,
) -> tuple[str, PipelineMode] | JSONResponse:
    """Resolve a request model to its pipeline mode, or a 404 envelope.

    The endpoint serves exactly the six ``paddleocr-vl-1.6-*`` aliases; any
    other model name is rejected as ``model_not_found`` instead of silently
    falling back to the doc pipeline.
    """
    try:
        model_id = (model or MODEL_ID).strip() or MODEL_ID
        return model_id, resolve_pipeline_mode(model_name=model_id)
    except (AttributeError, ValueError) as exc:
        return JSONResponse(
            status_code=404,
            content=_error_body(
                str(exc) or f"Unknown model {model!r}",
                "invalid_request_error",
                param="model",
                code="model_not_found",
            ),
        )


def _batch_item(
    image: Any,
    page_index: int,
    mode: PipelineMode,
    **kwargs: Any,
) -> BatchInputItem:
    """Build a batch item for the selected mode.

    Doc mode takes the normal per-block layout path. Every other mode is a
    whole-page single block: the mode's prompt and label are injected as
    per-item overrides, which take precedence over layout detection and over
    the seal/chart recognition switches.
    """
    if mode.use_layout:
        return BatchInputItem(image=image, page_index=page_index, **kwargs)
    return BatchInputItem(
        image=image,
        page_index=page_index,
        prompt=mode.prompt,
        prompt_type=mode.block_label,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    serve_settings: ServeSettings | None = None,
    *,
    client: VLMClient | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        serve_settings: HTTP-facing configuration (defaults from env / local.env).
        client: an optional pre-built :class:`VLMClient` shared by every
            request (used by tests to point at a mock endpoint). When
            omitted, one is built from the pipeline settings.
    """
    serve_settings = serve_settings or ServeSettings()

    # One InferenceManager instance shared across all requests. generate() is
    # synchronous and thread-safe via its internal ThreadPoolExecutor. The
    # manager itself stays in doc mode; request-selected single-block modes
    # are injected per item by _batch_item().
    _manager: InferenceManager | None = None
    _manager_lock = threading.Lock()

    def _get_manager() -> InferenceManager:
        nonlocal _manager
        with _manager_lock:
            if _manager is None:
                _manager = InferenceManager(client=client, mode="doc")
            return _manager

    _pipeline_semaphore = threading.Semaphore(max(1, serve_settings.MAX_CONCURRENT))

    async def _load_image(
        *, data: str | None = None, url: str | None = None
    ) -> Any:
        """Decode or download one image without blocking the event loop."""
        return await asyncio.to_thread(
            load_image_source,
            data=data,
            url=url,
            max_bytes=serve_settings.MAX_IMAGE_BYTES,
            timeout=serve_settings.URL_FETCH_TIMEOUT,
            allow_private=serve_settings.ALLOW_PRIVATE_URLS,
            max_redirects=serve_settings.URL_FETCH_MAX_REDIRECTS,
            allow_url=serve_settings.ALLOW_URL_FETCH,
        )

    def _run_pipeline(manager: InferenceManager, batch: list[BatchInputItem]):
        """Run the synchronous pipeline off the event loop.

        The semaphore is acquired inside the worker thread so waiting for a
        free slot never blocks FastAPI's event loop.
        """
        with _pipeline_semaphore:
            return manager.generate(batch)

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        yield

    app = FastAPI(
        title="docmore-paddleocr-vl-16",
        version=__version__,
        lifespan=_lifespan,
    )

    # -- error shaping -------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=400,
            content=_error_body(
                f"invalid request body: {exc.errors()[:3]}",
                "invalid_request_error",
                param="body",
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_handler(_: Request, exc: StarletteHTTPException):
        error_type = (
            "authentication_error" if exc.status_code == 401 else "invalid_request_error"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(str(exc.detail), error_type),
        )

    # -- auth ----------------------------------------------------------------

    async def _verify_auth(
        authorization: str | None = Header(default=None),
    ) -> None:
        if not serve_settings.API_KEYS:
            return
        for key in serve_settings.API_KEYS:
            if authorization == f"Bearer {key}":
                return
        raise StarletteHTTPException(status_code=401, detail="invalid API key")

    # -- routes --------------------------------------------------------------

    @app.get("/v1/models")
    @app.get("/models")
    def _list_models(_: None = Depends(_verify_auth)):
        return {
            "object": "list",
            "data": [
                {"id": alias, "object": "model", "owned_by": MODEL_ID}
                for alias in MODELS
            ],
        }

    @app.get("/health")
    def _health():
        return {
            "status": "ok",
            "backend": MODEL_ID,
            "upstream": settings.OPENAI_API_BASE,
            "layout_available": layout_available(),
        }

    # -- /v1/chat/completions -----------------------------------------------

    def _extract_images_from_body(body: dict) -> list[str]:
        """Pull every image_url data URI from the last user message."""
        urls: list[str] = []
        for message in reversed(body.get("messages", [])):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                break
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url" and part.get("image_url"):
                        urls.append(part["image_url"].get("url", ""))
                break
        return urls

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def _chat_completions(
        request: Request, _: None = Depends(_verify_auth)
    ) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content=_error_body("invalid JSON body", "invalid_request_error", param="body"),
            )

        resolved = _resolve_request_model(body.get("model"))
        if isinstance(resolved, JSONResponse):
            return resolved
        model, mode = resolved

        image_urls = _extract_images_from_body(body)
        if not image_urls:
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    "chat completion requires at least one image_url part",
                    "invalid_request_error",
                    param="messages",
                ),
            )
        if len(image_urls) > serve_settings.MAX_PAGES:
            return JSONResponse(
                status_code=413,
                content=_error_body(
                    f"too many images ({len(image_urls)} > {serve_settings.MAX_PAGES})",
                    "invalid_request_error",
                    param="messages",
                    code="request_too_large",
                ),
            )

        # Decode image URIs (data: URIs or http(s) URLs) to PIL Images
        images: list[Any] = []
        for url in image_urls:
            try:
                images.append(await _load_image(url=url))
            except ImageSourceError as exc:
                return JSONResponse(
                    status_code=400,
                    content=_error_body(str(exc), "invalid_request_error", param="messages"),
                )

        manager = _get_manager()

        batch = [
            _batch_item(img, index, mode)
            for index, img in enumerate(images)
        ]

        try:
            results = await asyncio.to_thread(_run_pipeline, manager, batch)
        except Exception as exc:
            return JSONResponse(
                status_code=502,
                content=_error_body(
                    f"pipeline failed: {exc}",
                    "upstream_error",
                    code="pipeline_failed",
                ),
            )

        succeeded = [r for r in results if not r.error]
        failed_count = len(results) - len(succeeded)
        total_tokens = sum(r.token_count for r in succeeded)

        if not succeeded:
            return JSONResponse(
                status_code=502,
                content=_error_body(
                    "all pages failed",
                    "upstream_error",
                    code="pipeline_failed",
                ),
            )
        if total_tokens == 0 and all(not r.markdown.strip() for r in succeeded):
            return JSONResponse(
                status_code=502,
                content=_error_body(
                    "upstream produced no recognitions for any page",
                    "upstream_error",
                    code="empty_output",
                ),
            )

        content = serve_settings.PAGE_SEPARATOR.join(r.markdown for r in succeeded)
        prompt_tokens = sum(
            max(1, (img.size[0] * img.size[1]) // 784) for img in images
        )

        resp: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_tokens,
                "total_tokens": prompt_tokens + total_tokens,
            },
        }
        if failed_count > 0:
            resp["docmore"] = {"pages": len(results), "errors": failed_count}
        headers = {_PARTIAL_HEADER: "1"} if failed_count else None
        return JSONResponse(status_code=200, content=resp, headers=headers)

    # -- /v1/parse (GET -> 405, POST -> unified-parse/v1) --------------------

    @app.get("/v1/parse")
    @app.get("/parse")
    def _parse_get() -> JSONResponse:
        return JSONResponse(
            status_code=405,
            content=_error_body(
                "use POST for /v1/parse",
                "invalid_request_error",
                param="method",
                code="method_not_allowed",
            ),
        )

    @app.post("/v1/parse")
    @app.post("/parse")
    async def _parse_endpoint(
        request: ParseRequest, _: None = Depends(_verify_auth)
    ) -> JSONResponse:
        resolved = _resolve_request_model(request.model)
        if isinstance(resolved, JSONResponse):
            return resolved
        model_id, mode = resolved

        if request.stream:
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    "streaming is not supported",
                    "invalid_request_error",
                    param="stream",
                    code="streaming_not_supported",
                ),
            )

        options = request.options
        if options.output_format != "markdown":
            return _unsupported_option(
                "options.output_format",
                f"unsupported output_format: {options.output_format!r}",
            )
        if options.table_format != "html":
            return _unsupported_option(
                "options.table_format",
                f"unsupported table_format: {options.table_format!r}",
            )
        if options.language != "auto":
            return _unsupported_option(
                "options.language",
                f"unsupported language: {options.language!r}",
            )
        if options.include_lines:
            return _unsupported_option(
                "options.include_lines",
                "line-level structure is not available",
            )
        if request.response_schema is not None:
            return _unsupported_option(
                "response_schema",
                "response_schema is not supported",
            )

        option_warnings: list[str] = []
        if request.ocr_options:
            option_warnings.append(
                "ocr_options is a legacy field and is ignored; use options instead"
            )

        manager = _get_manager()

        # Collect image sources from unified content[] (or legacy messages[])
        # before loading any of them, so MAX_PAGES can reject oversized
        # batches without downloading anything.
        sources: list[tuple[str | None, str | None]] = []
        for block in request.content:
            if block.type == "image" and block.source:
                src = block.source
                sources.append(
                    (src.data if src.type == "base64" else None,
                     src.url if src.type == "url" else None)
                )

        # Fallback: try legacy OpenAI messages format
        if not sources and request.messages:
            for msg in request.messages:
                if msg.get("role") != "user":
                    continue
                msg_content = msg.get("content")
                if isinstance(msg_content, list):
                    for part in msg_content:
                        if part.get("type") == "image_url" and part.get("image_url"):
                            sources.append((None, part["image_url"].get("url", "")))
                            break

        if not sources:
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    "no image content found; provide image blocks with base64 or url source",
                    "missing_content",
                    param="content",
                    code="missing_content",
                ),
            )

        if len(sources) > serve_settings.MAX_PAGES:
            return JSONResponse(
                status_code=413,
                content=_error_body(
                    f"too many images ({len(sources)} > {serve_settings.MAX_PAGES})",
                    "invalid_request_error",
                    param="content",
                    code="request_too_large",
                ),
            )

        # Decode base64 payloads / download URLs to PIL Images
        images: list[Any] = []
        for data, url in sources:
            try:
                images.append(await _load_image(data=data, url=url))
            except ImageSourceError as exc:
                return JSONResponse(
                    status_code=400,
                    content=_error_body(str(exc), "invalid_request_error", param="content"),
                )

        page_numbers: list[int] | None = None
        if options.pages:
            try:
                page_numbers = parse_range_str(options.pages)
            except (TypeError, ValueError):
                return _unsupported_option(
                    "options.pages",
                    f"invalid page range: {options.pages!r}",
                )
            if not page_numbers or any(
                page < 1 or page > len(images) for page in page_numbers
            ):
                return _unsupported_option(
                    "options.pages",
                    f"page range out of bounds: {options.pages!r}",
                )

        selected = [
            (page_idx, image)
            for page_idx, image in enumerate(images)
            if page_numbers is None or (page_idx + 1) in page_numbers
        ]
        if not selected:
            return _unsupported_option(
                "options.pages", "page selection is empty"
            )

        selected_indices = [page_idx for page_idx, _ in selected]
        batch = [
            _batch_item(
                image,
                page_idx,
                mode,
                include_headers_footers=options.include_headers_footers,
            )
            for page_idx, image in selected
        ]

        try:
            results = await asyncio.to_thread(_run_pipeline, manager, batch)
        except Exception as exc:
            return JSONResponse(
                status_code=502,
                content=_error_body(
                    f"pipeline failed: {exc}",
                    "upstream_error",
                    code="pipeline_failed",
                ),
            )

        succeeded_pairs = [
            (page_idx, result)
            for page_idx, result in zip(selected_indices, results)
            if not result.error
        ]
        succeeded = [result for _, result in succeeded_pairs]
        succeeded_indices = [page_idx for page_idx, _ in succeeded_pairs]
        failed_count = len(results) - len(succeeded)

        if failed_count == len(results):
            return JSONResponse(
                status_code=502,
                content=_error_body(
                    "all pages failed",
                    "upstream_error",
                    code="pipeline_failed",
                ),
            )
        total_tokens = sum(r.token_count for r in succeeded)
        if total_tokens == 0 and all(not r.markdown.strip() for r in succeeded):
            return JSONResponse(
                status_code=502,
                content=_error_body(
                    "upstream produced no recognitions for any page",
                    "upstream_error",
                    code="empty_output",
                ),
            )

        parse_result = build_parse_result(
            succeeded,
            model_id=model_id,
            include_images=options.include_images,
            include_bounding_boxes=options.include_bounding_boxes,
            page_indices=succeeded_indices,
        )
        parse_result["warnings"].extend(option_warnings)

        headers = {_PARTIAL_HEADER: "1"} if failed_count else None
        return JSONResponse(
            status_code=200,
            content=parse_result,
            headers=headers,
        )

    return app
