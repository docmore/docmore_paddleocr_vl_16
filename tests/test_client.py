"""Unit tests for the OpenAI client helpers (no HTTP endpoint needed)."""

from __future__ import annotations

import pytest

from docmore.paddleocr_vl_16.model.client import normalize_base_url
from docmore.paddleocr_vl_16.model.util import detect_repeat_token


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://127.0.0.1:8113", "http://127.0.0.1:8113/v1"),
        ("http://127.0.0.1:8113/", "http://127.0.0.1:8113/v1"),
        ("http://127.0.0.1:8113/v1", "http://127.0.0.1:8113/v1"),
        ("http://127.0.0.1:8113/v1/", "http://127.0.0.1:8113/v1"),
        ("https://gateway.example.com/openai/v1", "https://gateway.example.com/openai/v1"),
    ],
)
def test_normalize_base_url(raw, expected):
    assert normalize_base_url(raw) == expected


def test_detect_repeat_token():
    assert detect_repeat_token("ok " * 100) is True
    assert detect_repeat_token("a normal OCR result with varied words") is False
    assert detect_repeat_token("") is False