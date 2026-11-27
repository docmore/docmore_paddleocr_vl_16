"""Degenerate-repetition detection for PaddleOCR-VL 1.6 completions.

A long run of the same trailing token/word means the model has fallen into a
loop; the client retries those requests with a higher temperature. Kept
separate from the client so it can be unit-tested without an HTTP endpoint.
Ported from the navidc / paddleocr-vl convention.
"""

from __future__ import annotations


def detect_repeat_token(
    predicted_tokens: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    scaling_factor: float = 3.0,
) -> bool:
    """Detect degenerate single-token repetition in a generation.

    Shorter sequences require more repeats to count as degenerate:
    ``max_repeats = base_max_repeats * (1 + scaling_factor / seq_len)``.

    Args:
        predicted_tokens: the model's raw text output.
        base_max_repeats: baseline repeat threshold for long sequences.
        window_size: only the last ``window_size`` chars are inspected.
        scaling_factor: extra strictness for short sequences.

    Returns:
        True if a degenerate repetition is detected (worth retrying).
    """
    text = predicted_tokens[-window_size:]
    seq_len = len(text)
    if seq_len == 0:
        return False
    tokens = text.split()
    stream = tokens if tokens else list(text)
    if len(stream) < 2:
        return False
    max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))
    last = stream[-1]
    count = 0
    for token in reversed(stream):
        if token == last:
            count += 1
        else:
            break
    return count >= max_repeats