"""Parquet event writer.

Implements the observer's ``EventWriter`` protocol, so the observation hot path
does not know or care that Parquet exists.

Design notes, each tied to a Phase 1 requirement:

**Batched, never one file per event.** The observer already hands over batches;
the writer additionally buffers to a row threshold so a short run produces one
file rather than hundreds. Brief section 5.2 requires the small-file problem to
be tested, and ``tests/performance`` measures file count against row count.

**Write-then-rename.** A batch is written to a ``.parquet.tmp`` file and then
atomically renamed. Readers glob ``*.parquet``, so a process killed mid-write
leaves a temp file that is ignored rather than a truncated Parquet that makes the
whole run unreadable. This is the whole of the crash-safety story: brief section
5.3 asks for "completed files readable", not transactional semantics, and an
atomic rename is exactly that guarantee.

**Ingest sequence is assigned here.** ``ingest_seq`` means *storage order*, and
storage order is what the writer did -- so the writer is the only component that
can honestly stamp it.

**Partitioning is optional and off by default.** Partitioning by ``event_type``
makes type-filtered scans cheaper but multiplies file count by the number of
event types in each batch. The default favours few files; the flag exists so the
trade-off is measurable rather than assumed.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..schema.event import TrainingEvent
from .layout import (
    ARROW_SCHEMA,
    METADATA_COLUMN,
    events_dir,
    list_part_files,
    new_run_manifest,
    part_filename,
    run_dir,
    temp_filename,
)

__all__ = ["ParquetEventWriter", "ParquetWriteStats"]


class ParquetWriteStats:
    def __init__(self) -> None:
        self.files_written = 0
        self.rows_written = 0
        self.batches = 0
        self.bytes_written = 0
        self.failures = 0
        self.last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "files_written": self.files_written,
            "rows_written": self.rows_written,
            "batches": self.batches,
            "bytes_written": self.bytes_written,
            "failures": self.failures,
            "last_error": self.last_error,
        }


class ParquetEventWriter:
    """Buffered, crash-safe Parquet writer for one run."""

    name = "parquet"

    def __init__(
        self,
        root: os.PathLike,
        run_id: str,
        *,
        rows_per_file: int = 50_000,
        compression: str = "zstd",
        partition_by_event_type: bool = False,
        write_manifest: bool = True,
    ) -> None:
        if rows_per_file <= 0:
            raise ValueError("rows_per_file must be positive")
        self.root = Path(root)
        self.run_id = run_id
        self.rows_per_file = rows_per_file
        self.compression = compression
        self.partition_by_event_type = partition_by_event_type
        self.write_manifest = write_manifest

        self.stats = ParquetWriteStats()
        self._buffer: List[TrainingEvent] = []
        self._lock = threading.Lock()
        self._closed = False
        self._next_ingest_seq = 0
        self._file_index = 0
        self._unique = uuid.uuid4().hex[:8]

        self.root.mkdir(parents=True, exist_ok=True)
        if self.write_manifest:
            self._write_manifest()

    # ------------------------------------------------------------------
    # EventWriter protocol
    # ------------------------------------------------------------------

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        if not events:
            return
        with self._lock:
            if self._closed:
                raise RuntimeError("ParquetEventWriter is closed")
            self._buffer.extend(events)
            self.stats.batches += 1
            if len(self._buffer) >= self.rows_per_file:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._flush_locked()
            finally:
                self._closed = True
                if self.write_manifest:
                    self._write_manifest(final=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush_locked(self) -> None:
        """Write the buffer out. Caller holds the lock.

        On failure the buffer is kept so the next flush retries, but the error is
        re-raised: the observer's circuit breaker is the component that decides
        whether to keep trying, and a writer that silently swallowed a failure
        would hide it from that decision.
        """
        if not self._buffer:
            return
        pending = self._buffer
        try:
            # Chunk to the configured row bound. Without this, `rows_per_file`
            # only bounded the *buffer trigger*, so one large batch (the
            # observer's default batch is 256 events) produced a single
            # arbitrarily large file and the file-size bound was fiction.
            if self.partition_by_event_type:
                grouped: Dict[str, List[TrainingEvent]] = {}
                for event in pending:
                    grouped.setdefault(str(event.event_type), []).append(event)
                for event_type, group in sorted(grouped.items()):
                    for chunk in self._chunks(group):
                        self._write_group(chunk, event_type=event_type)
            else:
                for chunk in self._chunks(pending):
                    self._write_group(chunk, event_type=None)
        except Exception as exc:
            self.stats.failures += 1
            self.stats.last_error = "{}: {}".format(type(exc).__name__, exc)
            raise
        self._buffer = []
        self._write_manifest()

    def _chunks(self, events: List[TrainingEvent]) -> List[List[TrainingEvent]]:
        """Split into pieces of at most ``rows_per_file`` rows."""
        size = self.rows_per_file
        return [events[i:i + size] for i in range(0, len(events), size)]

    def _write_group(self, events: List[TrainingEvent], *, event_type: Optional[str]) -> None:
        directory = events_dir(self.root, self.run_id, event_type)
        directory.mkdir(parents=True, exist_ok=True)

        rows = []
        for event in events:
            event = event.copy_with(ingest_seq=self._next_ingest_seq)
            self._next_ingest_seq += 1
            rows.append(self._to_row(event))

        table = pa.Table.from_pylist(rows, schema=ARROW_SCHEMA)
        index = self._file_index
        self._file_index += 1
        first_seq = rows[0]["source_seq"] if rows else None

        temp_path = directory / temp_filename(index, self._unique)
        final_path = directory / part_filename(index, first_seq, self._unique)

        pq.write_table(
            table,
            temp_path,
            compression=self.compression,
            use_dictionary=["event_type", "metric_name", "metric_unit", "priority", "severity",
                            "source", "producer", "model_id", "module_path", "device_type"],
        )
        # Atomic rename: readers only ever see complete files.
        os.replace(str(temp_path), str(final_path))

        size = final_path.stat().st_size
        self.stats.files_written += 1
        self.stats.rows_written += len(rows)
        self.stats.bytes_written += size

    @staticmethod
    def _to_row(event: TrainingEvent) -> Dict[str, object]:
        payload = event.to_dict()
        payload[METADATA_COLUMN] = json.dumps(payload.get(METADATA_COLUMN) or {}, sort_keys=True)
        return payload

    def _write_manifest(self, final: bool = False) -> None:
        """Write the run manifest atomically.

        Cheap, and it is what lets a reader answer "what is in this directory and
        is it complete?" without scanning every file.
        """
        try:
            manifest = new_run_manifest(
                self.run_id,
                rows_written=self.stats.rows_written,
                files_written=self.stats.files_written,
                bytes_written=self.stats.bytes_written,
                partition_by_event_type=self.partition_by_event_type,
                rows_per_file=self.rows_per_file,
                compression=self.compression,
                closed=final,
            )
            target_dir = run_dir(self.root, self.run_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            temp = target_dir / ".manifest.json.tmp"
            final_path = target_dir / "manifest.json"
            temp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(str(temp), str(final_path))
        except Exception:
            # A manifest failure must not fail a successful data write. The data
            # is the product; the manifest is an optimisation.
            self.stats.last_error = "manifest write failed"

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def part_files(self) -> List[Path]:
        return list_part_files(self.root, self.run_id)

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.part_files())
