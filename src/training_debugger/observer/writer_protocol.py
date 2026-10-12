"""Writer abstraction: what the observer hands events to.

The observer owns *policy* (sampling, priority, drop accounting, failure
isolation). The writer owns *destination* (Parquet, JSONL, memory). Keeping them
apart is what lets W1 ship and be fault-injected before W3's Parquet layer
exists, and it is why the storage layer is swappable without touching the
training hot path.

A writer contract is deliberately small -- ``write_batch``, ``flush``, ``close``
-- because every method added here becomes a method every writer implementation
must get right under failure.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from ..schema.event import TrainingEvent

__all__ = [
    "EventWriter",
    "NullWriter",
    "MemoryWriter",
    "JsonlWriter",
    "MultiWriter",
    "FailingWriter",
]


@runtime_checkable
class EventWriter(Protocol):
    """Destination for batches of events.

    Implementations must be safe to call from a single writer thread. They may
    raise; the observer treats any exception as a writer failure, counts it, and
    -- after a threshold -- trips a circuit breaker so a persistently broken
    destination cannot become a spin loop inside the training process.
    """

    name: str

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        """Persist a batch. May raise; the observer isolates the failure."""

    def flush(self) -> None:
        """Make previously accepted batches durable. May raise."""

    def close(self) -> None:
        """Release resources. Must be idempotent and must not raise."""


class NullWriter:
    """Discards everything. The default when no destination is configured.

    Having an explicit null writer rather than ``writer=None`` everywhere keeps
    the observer's code path identical whether or not a destination exists --
    which is exactly the property the fault-injection tests want to exercise.
    """

    name = "null"

    def __init__(self) -> None:
        self.batches = 0
        self.events = 0

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        self.batches += 1
        self.events += len(events)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class MemoryWriter:
    """Keeps events in a list. For tests and for the in-process query path."""

    name = "memory"

    def __init__(self) -> None:
        self._events: List[TrainingEvent] = []
        self._lock = threading.Lock()
        self.batches = 0

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        with self._lock:
            self._events.extend(events)
            self.batches += 1

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    @property
    def events(self) -> List[TrainingEvent]:
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class JsonlWriter:
    """One JSON line per event, per producer.

    Not a Phase 1 storage format -- Parquet is the storage of record -- but it is
    the format a human reads when debugging the observer itself, and it is the
    crash-safe fallback a Phase 2 hardened writer would use.

    Appends are done with a single ``write`` of a pre-joined string so that a
    batch is one syscall rather than N.
    """

    name = "jsonl"

    def __init__(self, directory: os.PathLike, *, filename: str = "events.jsonl") -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / filename
        self._handle = open(self._path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False
        self.events_written = 0

    @property
    def path(self) -> Path:
        return self._path

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        if not events:
            return
        payload = "".join(event.to_json_line() + "\n" for event in events)
        with self._lock:
            if self._closed:
                raise RuntimeError("JsonlWriter is closed")
            self._handle.write(payload)
            self.events_written += len(events)

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._handle.flush()
                os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._handle.flush()
            finally:
                self._handle.close()


class MultiWriter:
    """Fan out to several writers.

    A failure in one child is recorded and does not stop the others, so a
    secondary destination (a debug sink, a mirror) cannot take down the primary.
    """

    name = "multi"

    def __init__(self, writers: Sequence[EventWriter], *, strict: bool = False) -> None:
        self._writers = list(writers)
        self._strict = strict
        self.child_failures: Dict[str, int] = {}

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        failures: List[BaseException] = []
        for writer in self._writers:
            try:
                writer.write_batch(events)
            except Exception as exc:  # isolated per child by design
                self.child_failures[writer.name] = self.child_failures.get(writer.name, 0) + 1
                failures.append(exc)
        if failures and self._strict:
            raise failures[0]

    def flush(self) -> None:
        for writer in self._writers:
            try:
                writer.flush()
            except Exception:
                self.child_failures[writer.name] = self.child_failures.get(writer.name, 0) + 1

    def close(self) -> None:
        for writer in self._writers:
            try:
                writer.close()
            except Exception:
                # A child that cannot close cleanly is counted rather than
                # ignored, so a partially-closed destination is visible.
                self.child_failures[writer.name] = self.child_failures.get(writer.name, 0) + 1


class FailingWriter:
    """Always raises. Exists so the fault-injection suite can drive the
    circuit-breaker path without mocking internals."""

    name = "failing"

    def __init__(self, message: str = "synthetic writer failure") -> None:
        self.message = message
        self.attempts = 0

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        self.attempts += 1
        raise IOError(self.message)

    def flush(self) -> None:
        raise IOError(self.message)

    def close(self) -> None:
        return None
