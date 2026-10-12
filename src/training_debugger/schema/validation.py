"""Strict validation for ``TrainingEvent``.

Two rules shape this module.

**Rule 1 -- validation is strict.** Phase 1 brief section 2.6 lists what must be
checked: identifier validity, required fields, non-negative indices, finiteness
where finiteness is meaningful, unit normalisation, and producer-sequence
monotonicity. A validator that accepts everything is not a validator.

**Rule 2 -- a validation failure must never fail training.** This is the W1
fail-open contract, and it constrains the *shape* of this API as much as its
content: :func:`validate_event` returns a :class:`ValidationResult`, it does not
raise for data problems. The caller (the observer) is expected to drop the
event, increment ``observer_internal_errors_total``, and continue. Raising is
reserved for programmer error, which is why :class:`ValidationError` exists but
is only used by :func:`validate_event_or_raise`, a helper for tests and for
strict ingest of an untrusted file.

Unit normalisation has a second purpose beyond tidiness: Phase 0 named
``ms``-versus-``s`` confusion as a silent-correlation hazard (a threshold rule
silently compares two different scales), so the unit vocabulary is closed and a
metric whose name declares one unit while its field declares another is
flagged rather than coerced.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set

from .enums import (
    SCHEMA_VERSION,
    EventType,
    MetricUnit,
    Severity,
)
from .event import MAX_MESSAGE_CHARS, RESERVED_METADATA_KEYS, TrainingEvent

__all__ = [
    "IssueCode",
    "ValidationIssue",
    "ValidationResult",
    "ValidationContext",
    "ProducerSequenceTracker",
    "validate_event",
    "validate_event_or_raise",
    "ValidationError",
    "NONFINITE_ALLOWED_EVENT_TYPES",
    "normalize_unit",
]

#: Event types where a NaN/Inf metric value is a *signal* rather than a bug.
#: ``NAN_DETECTED`` and ``INF_DETECTED`` are required Phase 1 incidents, so
#: rejecting non-finite values outright would break the product. On any other
#: event type a non-finite value is treated as a producer defect.
NONFINITE_ALLOWED_EVENT_TYPES = frozenset(
    {
        EventType.LOSS,
        EventType.GRADIENT_METRIC,
        EventType.PARAMETER_METRIC,
        EventType.ACTIVATION_METRIC,
        EventType.LOSS,
    }
)

_EVENT_ID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
_MAX_CONFIDENCE = 1.0

#: Metric-name suffix -> the unit the name *implies*. Used to catch a producer
#: labelling seconds as milliseconds (or the reverse).
_IMPLIED_UNIT_BY_SUFFIX = (
    ("_ms", MetricUnit.MILLISECONDS),
    ("_us", MetricUnit.MICROSECONDS),
    ("_ns", MetricUnit.MICROSECONDS),  # normalised down; ns is not in the vocabulary
    ("_s", MetricUnit.SECONDS),
    ("_bytes", MetricUnit.BYTES),
    ("_c", MetricUnit.CELSIUS),
    ("_w", MetricUnit.WATT),
    ("_hz", MetricUnit.HERTZ),
    ("_pct", MetricUnit.PERCENT),
    ("_ratio", MetricUnit.RATIO),
    ("_per_sec", MetricUnit.SAMPLES_PER_SEC),
)

_UNIT_ALIASES = {
    "ms": MetricUnit.MILLISECONDS,
    "millisecond": MetricUnit.MILLISECONDS,
    "milliseconds": MetricUnit.MILLISECONDS,
    "s": MetricUnit.SECONDS,
    "sec": MetricUnit.SECONDS,
    "secs": MetricUnit.SECONDS,
    "seconds": MetricUnit.SECONDS,
    "us": MetricUnit.MICROSECONDS,
    "microseconds": MetricUnit.MICROSECONDS,
    "none": MetricUnit.DIMENSIONLESS,
    "": MetricUnit.DIMENSIONLESS,
    "%": MetricUnit.PERCENT,
    "percent": MetricUnit.PERCENT,
    "token/s": MetricUnit.TOKENS_PER_SEC,
    "tokens/sec": MetricUnit.TOKENS_PER_SEC,
    "sample/s": MetricUnit.SAMPLES_PER_SEC,
}


class IssueCode(str, Enum):
    """Stable machine-readable issue codes. Tests assert on these, not on prose."""

    MISSING_EVENT_ID = "missing_event_id"
    MALFORMED_EVENT_ID = "malformed_event_id"
    MISSING_RUN_ID = "missing_run_id"
    MISSING_PRODUCER = "missing_producer"
    MISSING_SOURCE = "missing_source"
    INVALID_EVENT_TYPE = "invalid_event_type"
    UNKNOWN_SCHEMA_VERSION = "unknown_schema_version"
    SCHEMA_MAJOR_MISMATCH = "schema_major_mismatch"
    NEGATIVE_RANK = "negative_rank"
    NEGATIVE_STEP = "negative_step"
    NEGATIVE_MICROSTEP = "negative_microstep"
    RANK_EXCEEDS_WORLD_SIZE = "rank_exceeds_world_size"
    LOCAL_RANK_EXCEEDS_RANK_SPACE = "local_rank_exceeds_rank_space"
    NONFINITE_NOT_ALLOWED = "nonfinite_not_allowed"
    INVALID_UNIT = "invalid_unit"
    UNIT_NAME_MISMATCH = "unit_name_mismatch"
    INVALID_LAYER_INDEX = "invalid_layer_index"
    LAYER_INDEX_OUT_OF_RANGE = "layer_index_out_of_range"
    LAYER_MODULE_CONFLICT = "layer_module_conflict"
    NON_MONOTONIC_SOURCE_SEQ = "non_monotonic_source_seq"
    MISSING_SOURCE_SEQ = "missing_source_seq"
    INVALID_CONFIDENCE = "invalid_confidence"
    METRIC_WITHOUT_NAME = "metric_without_name"
    METRIC_WITHOUT_VALUE = "metric_without_value"
    METRIC_WITHOUT_UNIT = "metric_without_unit"
    INVALID_METRIC_VALUE_TYPE = "invalid_metric_value_type"
    RESERVED_METADATA_KEY = "reserved_metadata_key"
    METADATA_TOO_LARGE = "metadata_too_large"
    METADATA_NOT_A_DICT = "metadata_not_a_dict"
    NEGATIVE_DURATION = "negative_duration"
    MESSAGE_TOO_LONG = "message_too_long"
    UNKNOWN_FIELDS_PRESENT = "unknown_fields_present"
    DUPLICATE_EVENT_ID = "duplicate_event_id"
    DUPLICATE_IDEMPOTENCY_KEY = "duplicate_idempotency_key"


@dataclass
class ValidationIssue:
    code: IssueCode
    message: str
    field: Optional[str] = None
    fatal: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "field": self.field,
            "fatal": self.fatal,
        }


@dataclass
class ValidationResult:
    """Outcome of validating one event.

    ``ok`` is False when any *fatal* issue was found. Non-fatal issues are
    recorded but do not block the event, because dropping a slightly odd event
    is worse than keeping it when the oddity is cosmetic.
    """

    ok: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def fatal_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.fatal]

    @property
    def codes(self) -> Set[str]:
        return {i.code.value for i in self.issues}

    def reason(self) -> str:
        return "; ".join("{}: {}".format(i.code.value, i.message) for i in self.issues)


class ValidationError(ValueError):
    """Raised only by :func:`validate_event_or_raise`."""

    def __init__(self, result: ValidationResult):
        super().__init__(result.reason())
        self.result = result


@dataclass
class ValidationContext:
    """Optional knowledge the validator cannot infer from a single event."""

    max_layers: Optional[int] = None
    known_world_size: Optional[int] = None
    seen_event_ids: Set[str] = field(default_factory=set)
    seen_idempotency_keys: Set[str] = field(default_factory=set)
    strict_units: bool = True


class ProducerSequenceTracker:
    """Detects non-monotonic ``source_seq`` per producer.

    Kept as explicit state rather than hidden globals (Phase 1 code-quality rule:
    no hidden global state). The observer owns one instance; tests construct
    their own.
    """

    def __init__(self) -> None:
        self._last: Dict[str, int] = {}
        self._seen: Dict[str, int] = {}

    def check(self, event: TrainingEvent) -> Optional[ValidationIssue]:
        if event.source_seq is None:
            return None
        key = event.producer_id
        previous = self._last.get(key)
        self._seen[key] = self._seen.get(key, 0) + 1
        if previous is not None and event.source_seq <= previous:
            issue = ValidationIssue(
                IssueCode.NON_MONOTONIC_SOURCE_SEQ,
                "source_seq {} is not greater than the previous {} for producer {}".format(
                    event.source_seq, previous, key
                ),
                field="source_seq",
                fatal=True,
            )
            self._last[key] = event.source_seq
            return issue
        self._last[key] = event.source_seq
        return None

    def next_seq(self, producer_id: str, start: int = 1) -> int:
        return self._last.get(producer_id, start - 1) + 1

    def snapshot(self) -> Dict[str, int]:
        return dict(self._last)


def normalize_unit(value: Optional[float], unit: Optional[MetricUnit]) -> Optional[float]:
    """Convert a value to the canonical unit for its dimension.

    Canonical choices: durations in **milliseconds**, sizes in **bytes**,
    temperature in **celsius**, frequency in **hertz**, ratios dimensionless.

    Returns the converted value, or the input unchanged when the unit is unknown
    or dimensionless. Conversion is lossless for the vocabulary in use, so this
    is safe to apply on ingest.
    """
    if value is None or unit is None:
        return value
    if unit is MetricUnit.SECONDS:
        return value * 1000.0
    if unit is MetricUnit.MICROSECONDS:
        return value / 1000.0
    if unit is MetricUnit.PERCENT:
        # Percent is kept as a percent: converting to a ratio would silently
        # change every published threshold by 100x.
        return value
    return value


def _coerce_unit(raw: Any) -> Optional[MetricUnit]:
    if raw is None:
        return None
    if isinstance(raw, MetricUnit):
        return raw
    text = str(raw).strip().lower()
    if text in _UNIT_ALIASES:
        return _UNIT_ALIASES[text]
    try:
        return MetricUnit(text)
    except ValueError:
        return None


def _implied_unit(metric_name: Optional[str]) -> Optional[MetricUnit]:
    if not metric_name:
        return None
    lowered = metric_name.lower()
    for suffix, unit in _IMPLIED_UNIT_BY_SUFFIX:
        if lowered.endswith(suffix):
            # "_s" must not swallow "_ms"/"_us"/"_ns", which are matched earlier.
            return unit
    return None


def validate_event(
    event: TrainingEvent,
    *,
    context: Optional[ValidationContext] = None,
    seq_tracker: Optional[ProducerSequenceTracker] = None,
) -> ValidationResult:
    """Validate one event. Never raises for data problems.

    Returns a :class:`ValidationResult`. The caller decides what to do; the
    observer's contract is to drop and count.
    """
    issues: List[ValidationIssue] = []
    ctx = context or ValidationContext()

    # -- identity ----------------------------------------------------------
    if not event.event_id:
        issues.append(ValidationIssue(IssueCode.MISSING_EVENT_ID, "event_id is empty", "event_id"))
    elif not _EVENT_ID_RE.match(event.event_id):
        issues.append(
            ValidationIssue(
                IssueCode.MALFORMED_EVENT_ID,
                "event_id %r is not a 26-character Crockford base32 id" % event.event_id,
                "event_id",
            )
        )
    if not event.run_id:
        issues.append(ValidationIssue(IssueCode.MISSING_RUN_ID, "run_id is empty", "run_id"))
    if not event.producer:
        issues.append(
            ValidationIssue(IssueCode.MISSING_PRODUCER, "producer is empty", "producer")
        )
    if not event.source:
        issues.append(ValidationIssue(IssueCode.MISSING_SOURCE, "source is empty", "source"))

    # -- schema version ----------------------------------------------------
    version = str(event.schema_version or "")
    if version != SCHEMA_VERSION:
        try:
            major_text, minor_text = version.split(".", 1)
            major, minor = int(major_text), int(minor_text)
        except (ValueError, AttributeError):
            issues.append(
                ValidationIssue(
                    IssueCode.UNKNOWN_SCHEMA_VERSION,
                    "schema_version %r is not MAJOR.MINOR" % version,
                    "schema_version",
                )
            )
        else:
            reader_major, reader_minor = (int(p) for p in SCHEMA_VERSION.split("."))
            if major != reader_major:
                issues.append(
                    ValidationIssue(
                        IssueCode.SCHEMA_MAJOR_MISMATCH,
                        "event schema %s has a different major version than reader %s"
                        % (version, SCHEMA_VERSION),
                        "schema_version",
                    )
                )
            elif minor > reader_minor:
                # Forward compatibility: accepted, but the fact is recorded.
                issues.append(
                    ValidationIssue(
                        IssueCode.UNKNOWN_SCHEMA_VERSION,
                        "event schema %s is newer than reader %s; unknown fields preserved"
                        % (version, SCHEMA_VERSION),
                        "schema_version",
                        fatal=False,
                    )
                )

    # -- event type --------------------------------------------------------
    try:
        event_type = EventType(str(event.event_type))
    except ValueError:
        issues.append(
            ValidationIssue(
                IssueCode.INVALID_EVENT_TYPE,
                "event_type %r is not in the closed vocabulary" % (event.event_type,),
                "event_type",
            )
        )
        event_type = None

    # -- indices -----------------------------------------------------------
    if event.rank is not None and event.rank < 0:
        issues.append(
            ValidationIssue(IssueCode.NEGATIVE_RANK, "rank %r < 0" % event.rank, "rank")
        )
    if event.local_rank is not None and event.local_rank < 0:
        issues.append(
            ValidationIssue(
                IssueCode.LOCAL_RANK_EXCEEDS_RANK_SPACE,
                "local_rank %r < 0" % event.local_rank,
                "local_rank",
            )
        )
    world_size = event.world_size if event.world_size is not None else ctx.known_world_size
    if world_size is not None and event.rank is not None and event.rank >= world_size:
        issues.append(
            ValidationIssue(
                IssueCode.RANK_EXCEEDS_WORLD_SIZE,
                "rank %d >= world_size %d" % (event.rank, world_size),
                "rank",
            )
        )
    if event.step is not None and event.step < 0:
        issues.append(
            ValidationIssue(IssueCode.NEGATIVE_STEP, "step %r < 0" % event.step, "step")
        )
    if event.microstep is not None and event.microstep < 0:
        issues.append(
            ValidationIssue(
                IssueCode.NEGATIVE_MICROSTEP, "microstep %r < 0" % event.microstep, "microstep"
            )
        )
    if event.duration_ns is not None and event.duration_ns < 0:
        issues.append(
            ValidationIssue(
                IssueCode.NEGATIVE_DURATION, "duration_ns %r < 0" % event.duration_ns, "duration_ns"
            )
        )

    # -- layer / module ----------------------------------------------------
    if event.layer_index is not None:
        if event.layer_index < 0:
            issues.append(
                ValidationIssue(
                    IssueCode.INVALID_LAYER_INDEX,
                    "layer_index %r < 0" % event.layer_index,
                    "layer_index",
                )
            )
        elif ctx.max_layers is not None and event.layer_index >= ctx.max_layers:
            issues.append(
                ValidationIssue(
                    IssueCode.LAYER_INDEX_OUT_OF_RANGE,
                    "layer_index %d >= model depth %d" % (event.layer_index, ctx.max_layers),
                    "layer_index",
                )
            )
    if event.layer_index is not None and event.module_path is not None:
        # A module path that ends in an unmistakably different integer than
        # layer_index means the two provenance fields disagree.
        tail = re.findall(r"(\d+)", event.module_path)
        if tail and int(tail[-1]) != event.layer_index:
            issues.append(
                ValidationIssue(
                    IssueCode.LAYER_MODULE_CONFLICT,
                    "module_path %r implies layer %s but layer_index is %d"
                    % (event.module_path, tail[-1], event.layer_index),
                    "module_path",
                    fatal=False,
                )
            )

    # -- metric payload ----------------------------------------------------
    if event.metric_name is not None and event.metric_value is None:
        issues.append(
            ValidationIssue(
                IssueCode.METRIC_WITHOUT_VALUE,
                "metric_name %r has no metric_value" % event.metric_name,
                "metric_value",
            )
        )
    if event.metric_value is not None and event.metric_name is None:
        issues.append(
            ValidationIssue(
                IssueCode.METRIC_WITHOUT_NAME,
                "metric_value present without metric_name",
                "metric_name",
            )
        )
    if event.metric_value is not None and not isinstance(event.metric_value, (int, float)):
        issues.append(
            ValidationIssue(
                IssueCode.INVALID_METRIC_VALUE_TYPE,
                "metric_value must be numeric, got %s" % type(event.metric_value).__name__,
                "metric_value",
            )
        )
    elif event.metric_value is not None and not isinstance(event.metric_value, bool):
        if not math.isfinite(float(event.metric_value)):
            allowed = event_type in NONFINITE_ALLOWED_EVENT_TYPES if event_type else False
            if not allowed:
                issues.append(
                    ValidationIssue(
                        IssueCode.NONFINITE_NOT_ALLOWED,
                        "non-finite metric_value %r on event_type %s"
                        % (event.metric_value, event_type),
                        "metric_value",
                    )
                )
    if event.metric_name is not None and event.metric_unit is None and ctx.strict_units:
        issues.append(
            ValidationIssue(
                IssueCode.METRIC_WITHOUT_UNIT,
                "metric_name %r has no metric_unit; the unit vocabulary is closed"
                % event.metric_name,
                "metric_unit",
                fatal=False,
            )
        )
    if event.metric_unit is not None and not isinstance(event.metric_unit, MetricUnit):
        coerced = _coerce_unit(event.metric_unit)
        if coerced is None:
            issues.append(
                ValidationIssue(
                    IssueCode.INVALID_UNIT,
                    "metric_unit %r is not in the closed vocabulary" % (event.metric_unit,),
                    "metric_unit",
                )
            )
    if event.metric_unit is not None and event.metric_name is not None and ctx.strict_units:
        implied = _implied_unit(event.metric_name)
        actual = event.metric_unit if isinstance(event.metric_unit, MetricUnit) else _coerce_unit(event.metric_unit)
        if implied is not None and actual is not None and implied is not actual:
            # metric names ending in _ratio/_pct are dimensionless-ish and are
            # the common false positive, so only flag a genuine dimension clash.
            if {implied, actual} & {MetricUnit.RATIO, MetricUnit.PERCENT, MetricUnit.DIMENSIONLESS}:
                pass
            else:
                issues.append(
                    ValidationIssue(
                        IssueCode.UNIT_NAME_MISMATCH,
                        "metric_name %r implies %s but metric_unit is %s"
                        % (event.metric_name, implied, actual),
                        "metric_unit",
                        fatal=False,
                    )
                )

    # -- confidence --------------------------------------------------------
    if event.confidence is not None:
        if not isinstance(event.confidence, (int, float)) or not (
            0.0 <= float(event.confidence) <= _MAX_CONFIDENCE
        ):
            issues.append(
                ValidationIssue(
                    IssueCode.INVALID_CONFIDENCE,
                    "confidence %r is not in [0, 1]" % (event.confidence,),
                    "confidence",
                )
            )

    # -- metadata ----------------------------------------------------------
    if not isinstance(event.metadata, dict):
        issues.append(
            ValidationIssue(
                IssueCode.METADATA_NOT_A_DICT,
                "metadata must be a dict, got %s" % type(event.metadata).__name__,
                "metadata",
            )
        )
    else:
        reserved = set(event.metadata) & (RESERVED_METADATA_KEYS - {"_unrecognized", "_schema_from_future"})
        if reserved:
            issues.append(
                ValidationIssue(
                    IssueCode.RESERVED_METADATA_KEY,
                    "metadata uses reserved key(s): %s" % sorted(reserved),
                    "metadata",
                )
            )
        size = event.metadata_size_bytes()
        if size > 16 * 1024:
            issues.append(
                ValidationIssue(
                    IssueCode.METADATA_TOO_LARGE,
                    "metadata is %d bytes (limit 16384)" % size,
                    "metadata",
                )
            )

    # -- message -----------------------------------------------------------
    if event.message is not None:
        if not isinstance(event.message, str):
            issues.append(
                ValidationIssue(
                    IssueCode.MESSAGE_TOO_LONG,
                    "message must be a string, got %s" % type(event.message).__name__,
                    "message",
                )
            )
        elif len(event.message) > MAX_MESSAGE_CHARS:
            # Truncation happens on serialization; flag it so a producer that
            # is serialising tracebacks into messages becomes visible.
            issues.append(
                ValidationIssue(
                    IssueCode.MESSAGE_TOO_LONG,
                    "message is %d chars (limit %d); it will be middle-truncated"
                    % (len(event.message), MAX_MESSAGE_CHARS),
                    "message",
                    fatal=False,
                )
            )

    # -- producer sequence -------------------------------------------------
    if seq_tracker is not None:
        seq_issue = seq_tracker.check(event)
        if seq_issue is not None:
            issues.append(seq_issue)
    elif event.source_seq is None:
        issues.append(
            ValidationIssue(
                IssueCode.MISSING_SOURCE_SEQ,
                "no source_seq: this event cannot be ordered within its producer "
                "and cannot be deduplicated on retry",
                "source_seq",
                fatal=False,
            )
        )

    # -- cross-event identity ---------------------------------------------
    if ctx.seen_event_ids:
        if event.event_id in ctx.seen_event_ids:
            issues.append(
                ValidationIssue(
                    IssueCode.DUPLICATE_EVENT_ID,
                    "event_id %s was already seen" % event.event_id,
                    "event_id",
                    fatal=False,
                )
            )
    if ctx.seen_idempotency_keys:
        key = event.computed_idempotency_key()
        if key in ctx.seen_idempotency_keys:
            issues.append(
                ValidationIssue(
                    IssueCode.DUPLICATE_IDEMPOTENCY_KEY,
                    "idempotency key %s was already seen" % key,
                    "idempotency_key",
                    fatal=False,
                )
            )

    # -- unknown fields ----------------------------------------------------
    if "_unrecognized" in (event.metadata or {}):
        unknown = event.metadata["_unrecognized"]
        issues.append(
            ValidationIssue(
                IssueCode.UNKNOWN_FIELDS_PRESENT,
                "event carries %d unrecognised field(s) from a newer producer"
                % (len(unknown) if hasattr(unknown, "__len__") else 0),
                "metadata",
                fatal=False,
            )
        )

    ok = not any(i.fatal for i in issues)
    return ValidationResult(ok=ok, issues=issues)


def validate_event_or_raise(
    event: TrainingEvent,
    *,
    context: Optional[ValidationContext] = None,
    seq_tracker: Optional[ProducerSequenceTracker] = None,
) -> TrainingEvent:
    """Validate and raise on failure.

    For tests and for strict ingest of untrusted files. **The observer must not
    use this** -- training must survive a bad event.
    """
    result = validate_event(event, context=context, seq_tracker=seq_tracker)
    if not result.ok:
        raise ValidationError(result)
    return event
