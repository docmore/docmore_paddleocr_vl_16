"""``docmore-paddleocr-vl-16`` command-line interface (thin wrapper over convert)."""

from pathlib import Path

import click

from docmore.paddleocr_vl_16.api import convert as _convert
from docmore.paddleocr_vl_16.api import get_supported_files, save_merged_output  # re-exported for backwards compat

__all__ = ["main", "get_supported_files", "save_merged_output"]


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--page-range",
    type=str,
    default=None,
    help="Page range for PDFs (1-based, e.g., '1-5,7,9-12'). Only applicable to PDF files.",
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=None,
    help="Maximum number of output tokens per page.",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Maximum number of parallel inference workers.",
)
@click.option(
    "--max-retries",
    type=int,
    default=None,
    help="Maximum number of retries per request.",
)
@click.option(
    "--save-html/--no-html",
    default=True,
    help="Save HTML output files.",
)
@click.option(
    "--save-json/--no-json",
    default=True,
    help="Save the structured JSON output (pages + reading-order blocks).",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Number of pages to process in a batch.",
)
@click.option(
    "--paginate-output",
    "--paginate_output",
    "paginate_output",
    is_flag=True,
    default=False,
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["default", "docling"], case_sensitive=False),
    default="default",
    help="Output format: 'default' (markdown+html+json), 'docling' (docling-core JSON+markdown).",
)
def main(
    input_path: Path,
    output_path: Path,
    page_range: str,
    max_output_tokens: int,
    max_workers: int,
    max_retries: int,
    save_html: bool,
    save_json: bool,
    batch_size: int,
    paginate_output: bool,
    output_format: str,
):
    """Convert documents (PDF/images) to Markdown/HTML via PaddleOCR-VL 1.6.

    INPUT_PATH: a PDF/image file or a directory of them.

    OUTPUT_PATH: output directory (created if missing).
    """
    click.echo("PaddleOCR-VL 1.6 CLI - Starting OCR processing")
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")

    try:
        _convert(
            input_path,
            output_path,
            page_range=page_range,
            max_output_tokens=max_output_tokens,
            max_workers=max_workers,
            max_retries=max_retries,
            save_html=save_html,
            save_json=save_json,
            batch_size=batch_size,
            paginate_output=paginate_output,
            output_format=output_format,
            progress=click.echo,
        )
    except ValueError as e:
        raise click.UsageError(str(e))

    click.echo(f"\nProcessing complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
