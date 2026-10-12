"""LLM-Training-Debugger -- Phase 1 core training incident pipeline.

Layer boundaries (Phase 1 brief section 19). Dependencies point downward only::

    Adapter  ->  Observer  ->  TrainingEvent  ->  Storage  ->  Detector  ->  Incident

Nothing in this package may import a UI module, and no detector may contain
agent logic. ``tests/unit/test_layering.py`` enforces the direction.
"""

from __future__ import annotations

__version__ = "0.1.7a1"
#: The Python package version, PEP 440. The GitHub pre-release tag is the
#: human-readable counterpart: ``v0.1.7-alpha`` <-> ``0.1.7a1``.
GITHUB_TAG = "v0.1.7-alpha"

__all__ = ["__version__", "GITHUB_TAG"]
