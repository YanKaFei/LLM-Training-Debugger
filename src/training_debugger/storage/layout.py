"""Parquet dataset layout and the Arrow schema.

Two decisions here are load-bearing and were made from Phase 0's findings rather
than by default:

**Parquet is the source of truth; DuckDB is only the query engine.** Trackio is
not used for telemetry at all, because its ``metrics`` table stores metric names
as unindexed JSON keys and every chart read does an unbounded ``fetchall()``.

**Files are large and few, not small and many.** A file-per-event layout would
produce millions of tiny objects, destroy scan performance and make the directory
unusable. The writer buffers to a row threshold and emits one file per batch per
partition.

**The Arrow schema is declared explicitly** rather than inferred. Inference makes
the on-disk type of a column depend on which batch happened to be written first
(an all-null column infers as ``null``), which then breaks a later read. An
explicit schema is the difference between "this data is queryable forever" and
"this data is queryable until the first unusual batch".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa

from ..schema.enums import SCHEMA_VERSION

__all__ = [
    "ARROW_SCHEMA",
    "METADATA_COLUMN",
    "run_dir",
    "events_dir",
    "partition_dir",
    "part_filename",
    "temp_filename",
    "list_part_files",
    "new_run_manifest",
]

#: ``metadata`` is stored as a JSON string. Parquet can nest, but a map column
#: makes every query pay for a structure almost no query reads, and DuckDB's
#: ``json_extract`` covers the cases that need it.
METADATA_COLUMN = "metadata"

#: Explicit Arrow schema for one event row. Field order matches
#: ``TrainingEvent.to_dict()`` for readability, not for correctness.
ARROW_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("idempotency_key", pa.string(), nullable=True),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("producer", pa.string(), nullable=False),
        pa.field("producer_version", pa.string(), nullable=True),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("rank", pa.int32(), nullable=True),
        pa.field("local_rank", pa.int32(), nullable=True),
        pa.field("world_size", pa.int32(), nullable=True),
        pa.field("step", pa.int64(), nullable=True),
        pa.field("microstep", pa.int32(), nullable=True),
        pa.field("source_seq", pa.int64(), nullable=True),
        pa.field("ingest_seq", pa.int64(), nullable=True),
        pa.field("parent_event_id", pa.string(), nullable=True),
        pa.field("span_kind", pa.string(), nullable=True),
        pa.field("duration_ns", pa.int64(), nullable=True),
        pa.field("wall_time_ns", pa.int64(), nullable=False),
        pa.field("monotonic_time_ns", pa.int64(), nullable=True),
        pa.field("model_id", pa.string(), nullable=True),
        pa.field("module_path", pa.string(), nullable=True),
        pa.field("layer_index", pa.int32(), nullable=True),
        pa.field("metric_name", pa.string(), nullable=True),
        pa.field("metric_value", pa.float64(), nullable=True),
        pa.field("metric_unit", pa.string(), nullable=True),
        pa.field("device_type", pa.string(), nullable=True),
        pa.field("device_id", pa.string(), nullable=True),
        pa.field("dataset_id", pa.string(), nullable=True),
        pa.field("shard_id", pa.string(), nullable=True),
        pa.field("batch_id", pa.string(), nullable=True),
        pa.field("severity", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=True),
        pa.field("priority", pa.string(), nullable=False),
        pa.field("trace_id", pa.string(), nullable=True),
        pa.field("span_id", pa.string(), nullable=True),
        pa.field("message", pa.string(), nullable=True),
        pa.field(METADATA_COLUMN, pa.string(), nullable=True),
    ]
)

#: Columns a DuckDB query may safely filter on without parsing JSON.
INDEXED_COLUMNS = (
    "run_id",
    "event_type",
    "metric_name",
    "step",
    "rank",
    "layer_index",
    "module_path",
    "priority",
    "severity",
    "source_seq",
    "ingest_seq",
)


def run_dir(root: os.PathLike, run_id: str) -> Path:
    return Path(root) / "runs" / run_id


def events_dir(root: os.PathLike, run_id: str, event_type: Optional[str] = None) -> Path:
    base = run_dir(root, run_id) / "events"
    if event_type:
        return base / "event_type={}".format(event_type)
    return base


def partition_dir(root: os.PathLike, run_id: str, event_type: str, partitions: bool) -> Path:
    return events_dir(root, run_id, event_type if partitions else None)


def part_filename(index: int, first_source_seq: Optional[int], unique: str) -> str:
    """``part-<index>-<seq>-<unique>.parquet``.

    The zero-padded index makes lexical order equal to write order, so a human
    (or ``ls``) sees the dataset in the order it was produced. The unique suffix
    prevents a collision between two writers that happen to pick the same index
    in different processes.
    """
    seq = "none" if first_source_seq is None else str(first_source_seq)
    return "part-{:08d}-{}-{}.parquet".format(index, seq, unique)


def temp_filename(index: int, unique: str) -> str:
    """Temp name for the write-then-rename dance.

    The ``.tmp`` suffix is what makes crash safety work: readers glob
    ``*.parquet`` only, so a half-written temp file is invisible rather than
    being read as a truncated batch.
    """
    return ".part-{:08d}-{}.parquet.tmp".format(index, unique)


def list_part_files(root: os.PathLike, run_id: str) -> List[Path]:
    """Completed Parquet files for a run, sorted. Ignores temp files."""
    base = events_dir(root, run_id)
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*.parquet") if p.is_file())


def new_run_manifest(run_id: str, **extra) -> Dict[str, object]:
    """Metadata written alongside a run so a reader can explain what it holds."""
    manifest: Dict[str, object] = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "format": "parquet",
        "indexed_columns": list(INDEXED_COLUMNS),
    }
    manifest.update(extra)
    return manifest
