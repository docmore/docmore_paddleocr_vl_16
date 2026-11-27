"""``docmore-paddleocr-vl-16-serve`` command — run the FastAPI/Uvicorn server."""

from __future__ import annotations

import click


@click.command(name="serve")
@click.option("--host", default=None, help="Bind address (default from settings).")
@click.option("--port", type=int, default=None, help="Bind port (default from settings).")
@click.option(
    "--api-key",
    "api_keys",
    multiple=True,
    help="Enable Bearer auth (repeatable; empty = open).",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=None,
    help="Max concurrent /parse requests inside the pipeline.",
)
def serve(
    host: str | None,
    port: int | None,
    api_keys: tuple[str, ...],
    max_concurrent: int | None,
) -> None:
    """Serve the PaddleOCR-VL 1.6 OpenAI-compatible API via Uvicorn."""
    overrides = {}
    if host is not None:
        overrides["HOST"] = host
    if port is not None:
        overrides["PORT"] = port
    if api_keys:
        overrides["API_KEYS"] = list(api_keys)
    if max_concurrent is not None:
        overrides["MAX_CONCURRENT"] = max_concurrent

    from docmore.paddleocr_vl_16.server.config import ServeSettings
    from docmore.paddleocr_vl_16.server.router import create_app

    serve_settings = ServeSettings(**overrides)
    app = create_app(serve_settings)

    try:
        import uvicorn
    except ImportError as exc:
        raise click.UsageError(
            "uvicorn is required for the serve subcommand. "
            "Install it with: pip install 'docmore-paddleocr-vl-16[server]'"
        ) from exc

    uvicorn.run(
        app,
        host=serve_settings.HOST,
        port=serve_settings.PORT,
        log_level="info",
    )
