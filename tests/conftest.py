"""Shared test fixtures.

Deliberately small. Most tests construct their own events via the helpers in
``tests/helpers.py`` so that a failure points at the test rather than at shared
mutable state (Phase 1 code-quality rule: no hidden global state).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def run_id() -> str:
    return "run-test-0001"


@pytest.fixture()
def producer() -> str:
    return "pytorch-adapter:r0:pid4242"
