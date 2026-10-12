"""The ``TrainingEvent`` envelope: flat, typed, versioned, serializable.

Design rules inherited from Phase 0 (``research/TRAINING_EVENT_SCHEMA_PROPOSAL.md``):

A1  one envelope, many event types
A2  the envelope is flat and strictly typed; ``metadata`` is an escape hatch and
    must never carry a correlation key
A3  ordering is explicit and partial -- see ``ordering.py``
A5  emission must be non-blocking and lossy under pressure
A7  every event carries its provenance

Phase 1 additions over the Phase 0 proposal, each with a reason:

* ``priority``      -- the W1 drop policy is defined over it (P0 never silently
                       dropped). It has to live on the event, not in the queue,
                       because the queue must not parse metadata to decide.
* ``parent_event_id`` -- the only *provable* cross-producer happens-before
                       witness besides a shared producer sequence.
* ``span_kind``     -- span pairing without inferring it from a name suffix.
* ``idempotency_key`` -- derived by default from (run_id, producer, source_seq),
                       so a duplicated flush is detectable without relying on
                       ``event_id`` (which is random and legitimately differs
                       across a retry of the *same* logical event).
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from .enums import (
    SCHEMA_VERSION,
    DeviceType,
    EventPriority,
    EventType,
    MetricUnit,
    Severity,
    SourceKind,
    SpanKind,
)

__all__ = [
    "TrainingEvent",
    "new_event_id",
    "make_producer_id",
    "RESERVED_METADATA_KEYS",
]

#: ``metadata`` keys this module or the storage layer owns. User metadata must
#: not collide with these; the validator rejects collisions rather than letting
#: a user silently shadow the schema's own bookkeeping.
RESERVED_METADATA_KEYS = frozenset(
    {
        "_unrecognized",
        "_schema_from_future",
        "_observer_error",
        "_drop_reason",
    }
)

_METADATA_MAX_BYTES = 16 * 1024

#: Phase 0's message budget. Middle truncation preserves both the beginning
#: (what failed) and the end (the most recent cause in a chained message).
MAX_MESSAGE_CHARS = 1000


def _truncate_message(message: Optional[str]) -> Optional[str]:
    if message is None:
        return None
    if len(message) <= MAX_MESSAGE_CHARS:
        return message
    half = MAX_MESSAGE_CHARS // 2
    return message[:half] + "..." + message[-half:]

# ---------------------------------------------------------------------------
# Identifier generation
# ---------------------------------------------------------------------------

# Crockford base32, no I/L/O/U -- avoids transcription ambiguity in logs.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

_id_lock = threading.Lock()
_id_last_ms = 0
_id_last_rand = 0


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def new_event_id(now_ms: Optional[int] = None) -> str:
    """Return a ULID-shaped, time-sortable, process-unique event id.

    Shape: 48-bit millisecond timestamp + 80 bits of monotonic randomness,
    Crockford-base32 encoded to 26 characters.

    IMPORTANT: lexical sortability of an id is *not* evidence of causal order.
    Two ids generated in the same millisecond on different ranks are
    independent. ``ordering.py`` never uses ``event_id`` as an ordering
    witness; it is used only as a stable tiebreak for deterministic output.
    """
    global _id_last_ms, _id_last_rand
    ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    with _id_lock:
        if ms <= _id_last_ms:
            # Keep the timestamp monotonic within a process and bump the
            # random component so ids stay unique and non-decreasing.
            ms = _id_last_ms
            _id_last_rand += 1
            if _id_last_rand >= (1 << 80):
                ms += 1
                _id_last_rand = 0
        else:
            _id_last_ms = ms
            _id_last_rand = random.getrandbits(80)
        rand = _id_last_rand
        _id_last_ms = ms
    return _encode(ms, 10) + _encode(rand, 16)


def make_producer_id(
    source: str,
    rank: Optional[int] = None,
    instance: Optional[str] = None,
) -> str:
    """Build the concrete producer identity for a single emitting stream.

    A *producer* is one sequentially-emitting component. Ordering is total
    within a producer and undefined across producers, so this string is the
    key that ``ordering.py`` groups by. It must be stable for the lifetime of
    the emitting stream and distinct between concurrent streams.
    """
    parts = [source]
    if rank is not None:
        parts.append("r{}".format(rank))
    else:
        parts.append("r?")
    if instance is None:
        instance = "pid{}".format(os.getpid())
    parts.append(instance)
    return ":".join(parts)


# ---------------------------------------------------------------------------
# The event
# ---------------------------------------------------------------------------


@dataclass
class TrainingEvent:
    """One observation. Flat, typed, append-only.

    ``step``/``microstep`` are the *logical training position*.
    ``source_seq`` is *producer-local* ordering.
    ``wall_time_ns``/``monotonic_time_ns`` are *observation time*.
    ``ingest_seq`` is *storage order*, assigned later and never by a producer.

    These four concepts are deliberately separate fields and must never be
    conflated: see the module docstring of ``ordering.py``.
    """

    # --- schema / identity -------------------------------------------------
    schema_version: str = SCHEMA_VERSION
    event_id: str = field(default_factory=new_event_id)
    idempotency_key: Optional[str] = None
    #: ``None`` means "absent, please fill from context"; ``""`` means "the
    #: producer explicitly supplied an empty value", which is an error. The
    #: observer relies on that distinction, so these default to ``None``.
    run_id: Optional[str] = None

    # --- provenance --------------------------------------------------------
    producer: Optional[str] = None
    producer_version: str = "0.1"
    event_type: EventType = EventType.WARNING
    source: Optional[str] = None

    # --- distributed context ----------------------------------------------
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    world_size: Optional[int] = None

    # --- logical training position ----------------------------------------
    step: Optional[int] = None
    microstep: Optional[int] = None

    # --- ordering ----------------------------------------------------------
    source_seq: Optional[int] = None
    ingest_seq: Optional[int] = None
    parent_event_id: Optional[str] = None
    span_kind: SpanKind = SpanKind.EVENT
    duration_ns: Optional[int] = None

    # --- observation time --------------------------------------------------
    wall_time_ns: int = field(default_factory=lambda: time.time_ns())
    monotonic_time_ns: int = field(default_factory=time.monotonic_ns)

    # --- model / module ----------------------------------------------------
    model_id: Optional[str] = None
    module_path: Optional[str] = None
    layer_index: Optional[int] = None

    # --- metric payload ----------------------------------------------------
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    metric_unit: Optional[MetricUnit] = None

    # --- device ------------------------------------------------------------
    device_type: Optional[DeviceType] = None
    device_id: Optional[str] = None

    # --- data provenance ---------------------------------------------------
    dataset_id: Optional[str] = None
    shard_id: Optional[str] = None
    batch_id: Optional[str] = None

    # --- classification ----------------------------------------------------
    severity: Severity = Severity.INFO
    confidence: Optional[float] = None
    priority: EventPriority = EventPriority.P1

    # --- trace context -----------------------------------------------------
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # --- human-readable text ----------------------------------------------
    #: Free text for WARNING/ERROR and for diagnostics. Bounded to
    #: ``MAX_MESSAGE_CHARS`` with middle truncation, matching Phase 0's budget
    #: (``research/TRAINING_EVENT_SCHEMA_PROPOSAL.md`` section 5.4). Never used
    #: as a correlation key.
    message: Optional[str] = None

    # --- escape hatch ------------------------------------------------------
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def producer_id(self) -> str:
        """Key used for producer-local total ordering."""
        return self.producer or "{}:r?:unknown".format(self.source or "unknown")

    @property
    def logical_position(self) -> Optional[Tuple[int, int]]:
        """``(step, microstep)`` or ``None`` when the event is not step-scoped."""
        if self.step is None:
            return None
        return (self.step, self.microstep if self.microstep is not None else 0)

    @property
    def is_metric(self) -> bool:
        return self.metric_name is not None and self.metric_value is not None

    @property
    def is_finite(self) -> bool:
        """``False`` when the metric value is NaN or infinite.

        NaN is a *legitimate* value here -- ``NAN_DETECTED`` is one of the four
        required Phase 1 incidents -- so this reports rather than rejects.
        """
        if self.metric_value is None:
            return True
        return math.isfinite(self.metric_value)

    def computed_idempotency_key(self) -> str:
        """Default idempotency key.

        ``(run_id, producer, source_seq)`` identifies a logical emission
        uniquely within a run. It deliberately excludes ``event_id`` (random,
        and correctly different across a retry of the same logical event) and
        ``wall_time_ns`` (never an identity).
        """
        if self.idempotency_key is not None:
            return self.idempotency_key
        if self.source_seq is None:
            # No producer sequence: fall back to the event id so the key is at
            # least unique. Such events cannot be deduplicated on retry, which
            # is a real limitation and is reported by the validator.
            return "noidem:{}".format(self.event_id)
        return "{}|{}|{}".format(self.run_id, self.producer_id, self.source_seq)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict. Enums become their string values; no nesting added."""
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "idempotency_key": self.computed_idempotency_key(),
            "run_id": self.run_id,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "event_type": str(self.event_type),
            "source": self.source,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "step": self.step,
            "microstep": self.microstep,
            "source_seq": self.source_seq,
            "ingest_seq": self.ingest_seq,
            "parent_event_id": self.parent_event_id,
            "span_kind": str(self.span_kind),
            "duration_ns": self.duration_ns,
            "wall_time_ns": self.wall_time_ns,
            "monotonic_time_ns": self.monotonic_time_ns,
            "model_id": self.model_id,
            "module_path": self.module_path,
            "layer_index": self.layer_index,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_unit": None if self.metric_unit is None else str(self.metric_unit),
            "device_type": None if self.device_type is None else str(self.device_type),
            "device_id": self.device_id,
            "dataset_id": self.dataset_id,
            "shard_id": self.shard_id,
            "batch_id": self.batch_id,
            "severity": str(self.severity),
            "confidence": self.confidence,
            "priority": str(self.priority),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "message": _truncate_message(self.message),
            "metadata": dict(self.metadata),
        }

    #: Fields a reader at this version understands. Anything else in an input
    #: dict is an *unknown field* and is handled per the policy in
    #: ``docs/SCHEMA_EVOLUTION.md``.
    #:
    #: ``ClassVar`` is load-bearing: without it the annotation makes this a
    #: dataclass field and it leaks into ``to_dict()``.
    KNOWN_FIELDS: ClassVar[Tuple[str, ...]] = (
        (
            "schema_version",
            "event_id",
            "idempotency_key",
            "run_id",
            "producer",
            "producer_version",
            "event_type",
            "source",
            "rank",
            "local_rank",
            "world_size",
            "step",
            "microstep",
            "source_seq",
            "ingest_seq",
            "parent_event_id",
            "span_kind",
            "duration_ns",
            "wall_time_ns",
            "monotonic_time_ns",
            "model_id",
            "module_path",
            "layer_index",
            "metric_name",
            "metric_value",
            "metric_unit",
            "device_type",
            "device_id",
            "dataset_id",
            "shard_id",
            "batch_id",
            "severity",
            "confidence",
            "priority",
            "trace_id",
            "span_id",
            "message",
            "metadata",
        )
    )

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        *,
        on_unknown: str = "preserve",
    ) -> "TrainingEvent":
        """Rebuild an event from a flat dict.

        ``on_unknown`` controls unknown-field handling:

        * ``"preserve"`` (default) -- keep them under
          ``metadata["_unrecognized"]`` so a newer producer's data survives a
          round trip through an older reader. Forward compatibility.
        * ``"drop"`` -- discard them (lossy, but keeps the event clean).
        * ``"error"`` -- raise ``ValueError``. Useful in tests and for strict
          ingest of an untrusted stream.

        Raises ``ValueError`` for structurally invalid input; never mutates the
        input dict.
        """
        if not isinstance(payload, dict):
            raise ValueError("event payload must be a dict, got %r" % type(payload).__name__)
        if on_unknown not in ("preserve", "drop", "error"):
            raise ValueError("on_unknown must be preserve|drop|error")

        payload = dict(payload)  # defensive copy: we never mutate the caller's dict
        unknown = {k: payload.pop(k) for k in list(payload) if k not in cls.KNOWN_FIELDS}

        if unknown and on_unknown == "error":
            raise ValueError("unknown fields: %s" % sorted(unknown))
        if unknown and on_unknown == "preserve":
            meta = dict(payload.get("metadata") or {})
            prior = meta.get("_unrecognized")
            if isinstance(prior, dict):
                merged = dict(prior)
                merged.update(unknown)
                unknown = merged
            meta["_unrecognized"] = unknown
            payload["metadata"] = meta

        kwargs: Dict[str, Any] = {}
        for name in cls.KNOWN_FIELDS:
            if name in payload:
                kwargs[name] = payload[name]

        # Enum coercion. Accept the string value, the member, or None.
        for fname, enum_cls, default in (
            ("event_type", EventType, EventType.WARNING),
            ("span_kind", SpanKind, SpanKind.EVENT),
            ("severity", Severity, Severity.INFO),
            ("priority", EventPriority, EventPriority.P1),
            ("metric_unit", MetricUnit, None),
            ("device_type", DeviceType, None),
        ):
            if fname in kwargs:
                raw = kwargs[fname]
                if raw is None:
                    kwargs[fname] = default
                elif isinstance(raw, enum_cls):
                    pass
                else:
                    try:
                        kwargs[fname] = enum_cls(raw)
                    except ValueError:
                        raise ValueError(
                            "invalid %s: %r (allowed: %s)" % (fname, raw, list(enum_cls.values()))
                        )

        if "metadata" in kwargs and kwargs["metadata"] is None:
            kwargs["metadata"] = {}
        if "idempotency_key" in kwargs:
            # Stored key is derived, not authoritative; recompute on write.
            kwargs.pop("idempotency_key")

        event = cls(**kwargs)
        if unknown and on_unknown == "preserve":
            event.metadata["_unrecognized"] = unknown
        return event

    def to_json_line(self) -> str:
        """One-line JSON. ``allow_nan=True`` so NaN/Inf survive.

        IEEE-754 non-finite values are meaningful signals for this product
        (``NAN_DETECTED`` is a required Phase 1 incident), so they are encoded
        as the JSON5-ish ``NaN``/``Infinity`` tokens rather than being nulled.
        A strict RFC-8259 consumer will reject them; Parquet, which is the
        storage of record, does not. See ``docs/SCHEMA_EVOLUTION.md``.
        """
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=True, sort_keys=True)

    @classmethod
    def from_json_line(cls, line: str, **kwargs: Any) -> "TrainingEvent":
        return cls.from_dict(json.loads(line), **kwargs)

    def copy_with(self, **changes: Any) -> "TrainingEvent":
        """Return a shallow copy with fields replaced. Never mutates self."""
        return replace(self, **changes)

    def metadata_size_bytes(self) -> int:
        try:
            return len(json.dumps(self.metadata, allow_nan=True).encode("utf-8"))
        except (TypeError, ValueError):
            return _METADATA_MAX_BYTES + 1


def assign_ingest_seq(events: List[TrainingEvent], start: int = 0) -> List[TrainingEvent]:
    """Assign ``ingest_seq`` in the given storage order.

    Storage order is whatever order the writer actually appended. It is *not*
    causal order and ``ordering.py`` never treats it as such; it exists so a
    later reader can reconstruct physical layout and detect gaps.
    """
    out = []
    for offset, event in enumerate(events):
        out.append(event.copy_with(ingest_seq=start + offset))
    return out
