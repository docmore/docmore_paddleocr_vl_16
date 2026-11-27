"""A minimal OpenAI-compatible mock of the upstream VLM endpoint.

Serves ``POST /v1/chat/completions`` and returns a canned markdown response
for any OCR prompt; also answers ``GET /v1/models``.

Used by the test suite to exercise the full pipeline (and the FastAPI
server) without a GPU or the real upstream server.

Run standalone::

    python tests/mock_server.py 18999
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_IDS = (
    "paddleocr-vl-1.6-doc",
    "paddleocr-vl-1.6-table",
    "paddleocr-vl-1.6-text",
    "paddleocr-vl-1.6-formula",
    "paddleocr-vl-1.6-seal",
    "paddleocr-vl-1.6-chart",
)
TEXT_RESPONSE = "Mock text line\nsecond line"
TABLE_RESPONSE = "<otsl><fcel>Head<fcel>Value<nl><fcel>A<fcel>1<nl></otsl>"
CHART_RESPONSE = "<otsl><fcel>Q1<fcel>10<nl><fcel>Q2<fcel>20<nl></otsl>"
FORMULA_RESPONSE = "\\[\nE = mc^2\n\\]"
SEAL_RESPONSE = "Mock Seal"
SPOTTING_RESPONSE = "Mock spotting: alpha, beta"


def response_for(prompt: str) -> str:
    """Canned response keyed by the PaddleOCR-VL task prompt."""
    if prompt.startswith("Table Recognition:"):
        return TABLE_RESPONSE
    if prompt.startswith("Chart Recognition:"):
        return CHART_RESPONSE
    if prompt.startswith("Formula Recognition:"):
        return FORMULA_RESPONSE
    if prompt.startswith("Seal Recognition:"):
        return SEAL_RESPONSE
    if prompt.startswith("Spotting:"):
        return SPOTTING_RESPONSE
    return TEXT_RESPONSE


def extract_user_text(body: dict) -> str:
    """Pull the text part out of an OpenAI chat-completion request body."""
    for message in body.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        for part in content or []:
            if part.get("type") == "text":
                return part.get("text", "")
    return ""


class MockHandler(BaseHTTPRequestHandler):
    """Serve the mock chat-completions API."""

    def log_message(self, *args):  # noqa: D102 - silence stderr access logs
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path in ("/v1/models", "/models"):
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "owned_by": "mock",
                        }
                        for model_id in MODEL_IDS
                    ],
                }
            )
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):  # noqa: N802 - http.server API
        if "chat/completions" not in self.path:
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        reply = response_for(extract_user_text(body))
        self._send_json(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", MODEL_IDS[0]),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": max(1, len(reply) // 4),
                    "total_tokens": 10 + max(1, len(reply) // 4),
                },
            }
        )


def start_server(port: int = 0) -> tuple[ThreadingHTTPServer, int]:
    """Start the mock server on ``port`` (0 = auto-assign a free port).

    Returns:
        ``(server, bound_port)`` — call ``server.serve_forever()`` in a thread
        (or ``server.shutdown()`` to stop).
    """
    server = ThreadingHTTPServer(("127.0.0.1", port), MockHandler)
    return server, server.server_address[1]


if __name__ == "__main__":
    listen_port = int(sys.argv[1]) if len(sys.argv) > 1 else 18999
    httpd, bound = start_server(listen_port)
    print(f"mock endpoint on http://127.0.0.1:{bound}/v1", flush=True)
    httpd.serve_forever()
