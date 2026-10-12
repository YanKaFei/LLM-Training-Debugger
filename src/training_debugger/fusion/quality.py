"""A1: evidence completeness.

The brief's rule, and the reason this module exists:

> 禁止在证据完整性未知或不足时输出一个没有限定说明的 NO_ANOMALY
> 不要将观测丢失直接解释为模型正常

Until now a `NO_ANOMALY` verdict meant two different things that a user could not tell
apart:

```
no anomaly was found in a complete stream      <- reassurance
no anomaly was found in the events that arrived <- not the same claim
```

The observer is fail-open by design: it drops events rather than blocking training. That
is the correct trade, but it means the analysis layer can be handed a stream with holes
in it and has no way to know. **A dropped event is indistinguishable from a step where
nothing happened**, which is exactly how a lost NaN becomes a clean bill of health.

Three statuses, and ``UNKNOWN`` is the default rather than ``COMPLETE``: a caller that
does not supply quality information has not demonstrated completeness, and the absence
of a warning must not be mistaken for evidence of one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

__all__ = ["DataQualityStatus", "DataQuality", "UNKNOWN_QUALITY"]


class DataQualityStatus(str, Enum):
    #: Completeness was demonstrated: no drops, no persistence failures, no overflow.
    COMPLETE = "COMPLETE"
    #: Events were lost or a write failed. Any conclusion is qualified.
    DEGRADED = "DEGRADED"
    #: Completeness was not reported. **Not** the same as complete.
    UNKNOWN = "UNKNOWN"


@dataclass
class DataQuality:
    """Completeness of the evidence a verdict was derived from."""

    status: str = DataQualityStatus.UNKNOWN.value
    dropped_events: int = 0
    overflow_events: int = 0
    persistence_failures: int = 0
    unreadable_parts: int = 0
    events_persisted: int = 0
    events_emitted: int = 0
    warnings: list = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.status == DataQualityStatus.COMPLETE.value

    @property
    def warns(self) -> bool:
        return not self.is_complete

    def warning_text(self) -> Optional[str]:
        if self.is_complete:
            return None
        if self.status == DataQualityStatus.UNKNOWN.value:
            return (
                "evidence completeness was NOT reported for this run, so this verdict "
                "describes only the events that arrived. It is not a statement that the "
                "run was healthy."
            )
        parts = []
        if self.dropped_events:
            parts.append("{} dropped".format(self.dropped_events))
        if self.overflow_events:
            parts.append("{} overflow evictions".format(self.overflow_events))
        if self.persistence_failures:
            parts.append("{} persistence failures".format(self.persistence_failures))
        if self.unreadable_parts:
            parts.append("{} unreadable store parts".format(self.unreadable_parts))
        detail = ", ".join(parts) or "an unspecified loss"
        return (
            "evidence is DEGRADED ({}), so a negative verdict is a statement about the "
            "events that arrived, NOT that the run was healthy.".format(detail)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "dropped_events": self.dropped_events,
            "overflow_events": self.overflow_events,
            "persistence_failures": self.persistence_failures,
            "unreadable_parts": self.unreadable_parts,
            "events_persisted": self.events_persisted,
            "events_emitted": self.events_emitted,
            "warnings": list(self.warnings),
            "warning_text": self.warning_text(),
        }

    @classmethod
    def from_observer_health(cls, health: Dict[str, Any]) -> "DataQuality":
        """Build from an observer's ``health()`` snapshot.

        The snapshot is **nested** -- ``observer.events_dropped_total``,
        ``queue.evicted`` -- and an earlier version of this method looked for flat
        keys that never existed. Every lookup therefore missed, every value defaulted
        to zero, and the method returned ``COMPLETE`` for a run whose completeness had
        never been read. That silently disabled the entire completeness check, which
        is the one failure this module exists to prevent.

        So the rule is now explicit: **if none of the expected fields are present, the
        answer is UNKNOWN, never COMPLETE.** Absence of a reading is not a reading of
        zero.
        """
        health = health or {}
        observer = health.get("observer") or {}
        queue = health.get("queue") or {}

        def pick(*candidates: Any) -> Optional[int]:
            for source in candidates:
                if isinstance(source, dict):
                    for key in ("events_dropped_total", "dropped", "evicted", "rejected",
                                "writer_failures_total", "written_total", "emitted_total",
                                "queue_overflow_total"):
                        if key in source:
                            return int(source.get(key) or 0)
                elif source is not None:
                    return int(source)
            return None

        dropped = pick(observer.get("events_dropped_total"), observer.get("dropped"))
        overflow = pick(observer.get("queue_overflow_total"),
                        queue.get("evicted"), queue.get("rejected"))
        failures = pick(observer.get("writer_failures_total"))
        written = pick(observer.get("written_total"))
        emitted = pick(observer.get("emitted_total"))

        if all(value is None for value in (dropped, overflow, failures, written, emitted)):
            # No recognisable completeness field: report UNKNOWN rather than COMPLETE.
            return cls(status=DataQualityStatus.UNKNOWN.value)

        dropped = dropped or 0
        overflow = overflow or 0
        failures = failures or 0
        degraded = (dropped + overflow + failures) > 0
        return cls(
            status=(DataQualityStatus.DEGRADED.value if degraded
                    else DataQualityStatus.COMPLETE.value),
            dropped_events=dropped,
            overflow_events=overflow,
            persistence_failures=failures,
            events_persisted=written or 0,
            events_emitted=emitted or 0,
        )


#: The default when a caller supplies nothing: not demonstrated complete.
UNKNOWN_QUALITY = DataQuality()
