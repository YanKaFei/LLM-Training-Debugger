"""Command-line interface. Loads, calls the library, formats -- nothing else."""

from __future__ import annotations

from .main import build_parser, main

__all__ = ["main", "build_parser"]
