"""Bounded, priority-aware event queue.

Phase 1 brief section 3.3 is unambiguous: no unbounded queue, and a full queue
must never block training. That leaves exactly one option -- drop -- so the only
real design question is *what* to drop, and the answer has to be data rather
than a coin flip.

The priority ladder (``EventPriority``):

===================  ==========================================================
``P0``               incident / error / non-finite. **Never silently dropped.**
``P1``               core training metrics (loss, lr, grad norm, step time)
``P2``               diagnostic metrics (per-layer metrics, memory)
``P3``               verbose / debug telemetry
===================  ==========================================================

Eviction rule: a full queue accepts an incoming event only by evicting an
existing event of *strictly lower* importance. An incoming P3 is therefore
dropped rather than displacing a queued P1, and an incoming P0 evicts anything
non-P0. When the queue is entirely P0 and still full, the event goes to the
emergency sink -- it is still not dropped silently.

Deliberately not implemented: a heap. Ordering within the queue is insertion
order, because fragments of a training step should reach the writer in the order
they happened; reordering by priority would scramble the very sequence the
partial-order model depends on.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional

from ..schema.enums import EventPriority
from ..schema.event import TrainingEvent

__all__ = ["BoundedEventQueue", "QueueStats", "PRIORITY_RANK"]

#: Lower rank == more important. Used for eviction comparisons.
PRIORITY_RANK = {
    EventPriority.P0: 0,
    EventPriority.P1: 1,
    EventPriority.P2: 2,
    EventPriority.P3: 3,
}


@dataclass
class QueueStats:
    """Counters a health endpoint and the fault-injection tests assert on."""

    accepted: int = 0
    evicted: int = 0
    rejected: int = 0
    emergency_sink_failures: int = 0
    depth: int = 0
    max_depth: int = 0
    evicted_by_priority: dict = field(default_factory=dict)
    rejected_by_priority: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "evicted": self.evicted,
            "rejected": self.rejected,
            "emergency_sink_failures": self.emergency_sink_failures,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "evicted_by_priority": dict(self.evicted_by_priority),
            "rejected_by_priority": dict(self.rejected_by_priority),
        }


class BoundedEventQueue:
    """Thread-safe bounded FIFO with priority-aware eviction.

    ``emergency_sink`` receives events that could not be enqueued at all. It is
    optional so that tests and non-critical paths can run without one, but the
    observer always supplies one: an unwritten P0 incident is the one loss this
    system cannot tolerate quietly.
    """

    def __init__(
        self,
        maxlen: int = 10_000,
        *,
        emergency_sink: Optional[Callable[[TrainingEvent, str], None]] = None,
    ) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive, got %r" % (maxlen,))
        self._maxlen = maxlen
        self._items: Deque[TrainingEvent] = deque()
        self._lock = threading.Lock()
        self._emergency_sink = emergency_sink
        self._stats = QueueStats()

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def is_full(self) -> bool:
        return len(self) >= self._maxlen

    def stats(self) -> QueueStats:
        with self._lock:
            snapshot = QueueStats(
                accepted=self._stats.accepted,
                evicted=self._stats.evicted,
                rejected=self._stats.rejected,
                emergency_sink_failures=self._stats.emergency_sink_failures,
                depth=len(self._items),
                max_depth=self._stats.max_depth,
                evicted_by_priority=dict(self._stats.evicted_by_priority),
                rejected_by_priority=dict(self._stats.rejected_by_priority),
            )
        return snapshot

    # -- the queue ----------------------------------------------------------

    def put(self, event: TrainingEvent) -> bool:
        """Enqueue ``event``. Returns True if it was accepted.

        Never blocks and never raises for a full queue. Raises only for
        programmer error (a non-event argument).
        """
        if not isinstance(event, TrainingEvent):
            raise TypeError("BoundedEventQueue.put expects a TrainingEvent")

        with self._lock:
            if len(self._items) < self._maxlen:
                self._items.append(event)
                self._stats.accepted += 1
                depth = len(self._items)
                if depth > self._stats.max_depth:
                    self._stats.max_depth = depth
                return True

            rank = PRIORITY_RANK.get(event.priority, PRIORITY_RANK[EventPriority.P1])
            victim_index = self._find_victim(rank)
            if victim_index is not None:
                victim = self._items[victim_index]
                del self._items[victim_index]
                self._stats.evicted += 1
                key = str(victim.priority)
                self._stats.evicted_by_priority[key] = (
                    self._stats.evicted_by_priority.get(key, 0) + 1
                )
                self._items.append(event)
                self._stats.accepted += 1
                return True

            # Nothing less important is queued: the incoming event loses.
            self._stats.rejected += 1
            key = str(event.priority)
            self._stats.rejected_by_priority[key] = (
                self._stats.rejected_by_priority.get(key, 0) + 1
            )
            sink = self._emergency_sink

        # Call the sink outside the lock: it may write to stderr, and holding
        # the queue lock across I/O would reintroduce the blocking we removed.
        if sink is not None:
            try:
                sink(event, "queue_full")
            except Exception:
                # The emergency sink failing must not propagate into training,
                # but it must be counted: a silent sink failure would make the
                # queue look like it dropped nothing.
                with self._lock:
                    self._stats.emergency_sink_failures += 1
        return False

    def _find_victim(self, incoming_rank: int) -> Optional[int]:
        """Index of the least-important evictable item, scanning from the back.

        Caller holds the lock. Scanning from the back evicts newer diagnostics
        first, which keeps a step's earliest events (the ones the ordering model
        needs) preferentially.
        """
        best_index: Optional[int] = None
        best_rank = incoming_rank
        for index in range(len(self._items) - 1, -1, -1):
            candidate_rank = PRIORITY_RANK.get(
                self._items[index].priority, PRIORITY_RANK[EventPriority.P1]
            )
            if candidate_rank > best_rank:
                best_rank = candidate_rank
                best_index = index
                if best_rank == 3:  # cannot do better than evicting P3
                    break
        return best_index

    def drain(self, max_items: int) -> List[TrainingEvent]:
        """Remove and return up to ``max_items`` events, oldest first."""
        if max_items <= 0:
            return []
        with self._lock:
            count = min(max_items, len(self._items))
            out = [self._items.popleft() for _ in range(count)]
            return out

    def clear(self) -> List[TrainingEvent]:
        with self._lock:
            out = list(self._items)
            self._items.clear()
            return out

    def peek_all(self) -> List[TrainingEvent]:
        with self._lock:
            return list(self._items)
