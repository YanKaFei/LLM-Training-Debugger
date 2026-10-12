"""Storage: Parquet is the source of truth, DuckDB is the query engine.

Trackio is deliberately absent here. Phase 0 established that its metric table
stores names as unindexed JSON keys and that every chart read is an unbounded
``fetchall()``, which makes it unsuitable as a telemetry store; it remains the
*experiment presentation* layer (runs, configs, artefacts, alerts) and is
therefore W7 work, not W3.
"""

from __future__ import annotations

from .duckdb_reader import DuckDBReader, ReaderCoverage
from .layout import (
    ARROW_SCHEMA,
    INDEXED_COLUMNS,
    events_dir,
    list_part_files,
    run_dir,
)
from .parquet_writer import ParquetEventWriter, ParquetWriteStats

__all__ = [
    "ParquetEventWriter",
    "ParquetWriteStats",
    "DuckDBReader",
    "ReaderCoverage",
    "ARROW_SCHEMA",
    "INDEXED_COLUMNS",
    "run_dir",
    "events_dir",
    "list_part_files",
]
