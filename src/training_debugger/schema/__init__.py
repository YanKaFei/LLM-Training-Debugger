"""TrainingEvent v0.1: the frozen data contract for Phase 1.

Public surface:

* :class:`TrainingEvent`      -- the envelope
* :class:`EventOrdering`      -- the partial-order algebra
* :class:`Incident`           -- the product's primary durable object
* :class:`Evidence`           -- atomic, auditable claims
* :func:`validate_event`      -- strict validation that never raises for data
"""

from __future__ import annotations

from .enums import (
    SCHEMA_VERSION,
    DetectorId,
    DeviceType,
    EventPriority,
    EventType,
    EvidenceLevel,
    IncidentState,
    IncidentType,
    MetricUnit,
    OrderingConfidence,
    OrderingRelation,
    Severity,
    SourceKind,
    SpanKind,
)
from .event import TrainingEvent, make_producer_id, new_event_id
from .evidence import Evidence, EvidenceBuckets, Multiplicity
from .incident import Hypothesis, Incident, IncidentValidationError
from .ordering import EarliestEvidence, EventOrdering, OrderingResult
from .validation import (
    NONFINITE_ALLOWED_EVENT_TYPES,
    IssueCode,
    ProducerSequenceTracker,
    ValidationContext,
    ValidationError,
    ValidationIssue,
    ValidationResult,
    normalize_unit,
    validate_event,
    validate_event_or_raise,
)

__all__ = [
    "SCHEMA_VERSION",
    "TrainingEvent",
    "new_event_id",
    "make_producer_id",
    "EventOrdering",
    "OrderingResult",
    "EarliestEvidence",
    "OrderingRelation",
    "OrderingConfidence",
    "Evidence",
    "EvidenceBuckets",
    "Multiplicity",
    "EvidenceLevel",
    "Incident",
    "Hypothesis",
    "IncidentValidationError",
    "IncidentState",
    "IncidentType",
    "EventType",
    "EventPriority",
    "Severity",
    "MetricUnit",
    "DeviceType",
    "SourceKind",
    "SpanKind",
    "DetectorId",
    "validate_event",
    "validate_event_or_raise",
    "ValidationResult",
    "ValidationIssue",
    "ValidationContext",
    "ValidationError",
    "ProducerSequenceTracker",
    "IssueCode",
    "normalize_unit",
    "NONFINITE_ALLOWED_EVENT_TYPES",
]
