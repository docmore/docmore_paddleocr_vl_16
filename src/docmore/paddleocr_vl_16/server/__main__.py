"""Allow running the server via ``python -m docmore.paddleocr_vl_16.server``."""

from docmore.paddleocr_vl_16.server.cli import serve

if __name__ == "__main__":
    serve()
