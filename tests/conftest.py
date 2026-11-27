"""Pytest bootstrap: make ``src/`` and the tests dir importable without install."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_TESTS = Path(__file__).resolve().parent

for _p in (_SRC, _TESTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
