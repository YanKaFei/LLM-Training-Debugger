"""Observer health: counters and the emergency sink.

The counters here are not decoration. Phase 1's fault-injection gate asserts on
them, and Phase 0's observer-overhead rule ("a debugger that hides its own cost
is untrustworthy") requires that drops and internal errors be *observable* rather
than inferred from missing data.

The emergency sink exists for the one unacceptable loss: a P0 event (an incident
or an error) that could not be written anywhere. It writes to stderr and keeps a
bounded in-memory copy so that a test, or a human, can retrieve what training
could not persist.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TextIO

from ..schema.event import TrainingEvent

__all__ = ["ObserverStats", "EmergencySink"]


@dataclass
class ObserverStats:
    """Cumulative observer counters. Cheap to read, cheap to increment."""

    # lifecycle
    started: int = 0
    closed: int = 0

    # emission
    emitted_total: int = 0
    written_total: int = 0
    flush_count: int = 0

    # the four counters the brief names explicitly
    observer_internal_errors_total: int = 0
    events_dropped_total: int = 0
    queue_overflow_total: int = 0
    writer_failures_total: int = 0

    # sampling
    sampled_out_total: int = 0

    # validation
    validation_failures_total: int = 0

    # emergency
    emergency_writes_total: int = 0
    emergency_buffer_depth: int = 0

    # timing (monotonic ns)
    emit_time_total_ns: int = 0
    writer_time_total_ns: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started": self.started,
            "closed": self.closed,
            "emitted_total": self.emitted_total,
            "written_total": self.written_total,
            "flush_count": self.flush_count,
            "observer_internal_errors_total": self.observer_internal_errors_total,
            "events_dropped_total": self.events_dropped_total,
            "queue_overflow_total": self.queue_overflow_total,
            "writer_failures_total": self.writer_failures_total,
            "sampled_out_total": self.sampled_out_total,
            "validation_failures_total": self.validation_failures_total,
            "emergency_writes_total": self.emergency_writes_total,
            "emergency_buffer_depth": self.emergency_buffer_depth,
            "emit_time_total_ns": self.emit_time_total_ns,
            "writer_time_total_ns": self.writer_time_total_ns,
            "last_error": self.last_error,
        }


class EmergencySink:
    """Last-resort destination for events that could not be queued or written.

    Two responsibilities:

    1. Keep a bounded in-memory copy so the data is recoverable at all.
    2. Emit a rate-limited, human-readable line on stderr so an operator notices
       without the observer becoming a log flood.

    Rate limiting matters: a queue-full condition is exactly the situation where
    an unbounded stderr write would make things worse by consuming the GIL.
    """

    def __init__(
        self,
        maxlen: int = 1000,
        *,
        stream: Optional[TextIO] = None,
        rate_limit_per_sec: float = 5.0,
    ) -> None:
        self._items: List[TrainingEvent] = []
        self._reasons: List[str] = []
        self._maxlen = max(1, maxlen)
        self._lock = threading.Lock()
        self._stream = stream
        self._rate_limit = rate_limit_per_sec
        self._window_start = 0.0
        self._window_count = 0
        self._suppressed = 0
        #: Counts failures of the last-resort path itself. Non-zero means even
        #: stderr did not accept the emergency record.
        self._stream_failures = 0

    @property
    def stream_failures(self) -> int:
        return self._stream_failures

    def __call__(self, event: TrainingEvent, reason: str) -> None:
        self.record(event, reason)

    def record(self, event: TrainingEvent, reason: str) -> None:
        with self._lock:
            if len(self._items) >= self._maxlen:
                # Drop the oldest emergency record; the newest are the ones an
                # operator investigating a live failure needs.
                self._items.pop(0)
                self._reasons.pop(0)
            self._items.append(event)
            self._reasons.append(reason)
            should_log = self._should_log_locked()
            suppressed = self._suppressed
            self._suppressed = 0
            stream = self._stream if self._stream is not None else sys.stderr

        if should_log:
            try:
                line = (
                    "training-debugger: EMERGENCY event not persisted ({reason}): "
                    "type={etype} step={step} producer={producer} metric={metric}={value}"
                ).format(
                    reason=reason,
                    etype=event.event_type,
                    step=event.step,
                    producer=event.producer_id,
                    metric=event.metric_name,
                    value=event.metric_value,
                )
                if suppressed:
                    line += " (+{} suppressed)".format(suppressed)
                stream.write(line + "\n")
                stream.flush()
            except Exception:
                # Writing to stderr failing must never propagate into training,
                # but it must not be silent either: the count is the only
                # evidence that the last-resort path also failed.
                self._stream_failures += 1

    def _should_log_locked(self) -> bool:
        now = time.monotonic()
        if now - self._window_start >= 1.0:
            self._window_start = now
            self._window_count = 0
        if self._window_count < self._rate_limit:
            self._window_count += 1
            return True
        self._suppressed += 1
        return False

    def drain(self) -> List[TrainingEvent]:
        with self._lock:
            out = list(self._items)
            self._items.clear()
            self._reasons.clear()
            return out

    def drain_with_reasons(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = [
                {"reason": reason, "event": event.to_dict()}
                for event, reason in zip(self._items, self._reasons)
            ]
            self._items.clear()
            self._reasons.clear()
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def stats_to_json(stats: ObserverStats) -> str:
    return json.dumps(stats.to_dict(), sort_keys=True)
