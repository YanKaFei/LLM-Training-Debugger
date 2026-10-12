"""Closed vocabularies for the TrainingEvent v0.1 schema.

Every enum here is *closed*: adding a member is a schema change and requires a
``SCHEMA_VERSION`` bump plus an entry in ``docs/SCHEMA_EVOLUTION.md``. This is
deliberate. Phase 0 established that free-text event names make correlation
impossible (``research/TRAINING_EVENT_SCHEMA_PROPOSAL.md`` section 3), so the
vocabulary is a contract rather than a convention.

Python 3.9 compatible: ``enum.StrEnum`` is 3.11+, so each enum subclasses
``(str, Enum)`` instead.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "SCHEMA_VERSION",
    "EventType",
    "SourceKind",
    "Severity",
    "MetricUnit",
    "EvidenceLevel",
    "IncidentState",
    "EventPriority",
    "OrderingRelation",
    "OrderingConfidence",
    "DeviceType",
    "SpanKind",
    "IncidentType",
    "DetectorId",
]

#: Current schema version. Bumping this requires a migration note; see
#: ``docs/SCHEMA_EVOLUTION.md``.
SCHEMA_VERSION = "0.1"


class _StrEnum(str, Enum):
    """Python 3.9-compatible stand-in for ``enum.StrEnum``."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)

    @classmethod
    def values(cls) -> tuple:
        return tuple(m.value for m in cls)


class EventType(_StrEnum):
    """The closed event vocabulary.

    Phase 0's proposal listed lifecycle, training, data, checkpoint, profiling,
    distributed, warning/error and incident events. This enum is that list,
    trimmed to what Phase 1 can actually emit or consume. Members marked
    ``v0.2`` are reserved: they parse and validate, but no Phase 1 producer
    emits them, and Phase 1 detectors must not depend on them.
    """

    # --- run / process lifecycle -------------------------------------------
    RUN_START = "RUN_START"
    RUN_END = "RUN_END"
    PROCESS_START = "PROCESS_START"
    PROCESS_EXIT = "PROCESS_EXIT"
    ENV_SNAPSHOT = "ENV_SNAPSHOT"
    CONFIG_CHANGE = "CONFIG_CHANGE"

    # --- step lifecycle ----------------------------------------------------
    STEP_START = "STEP_START"
    STEP_END = "STEP_END"
    MICROSTEP_START = "MICROSTEP_START"
    MICROSTEP_END = "MICROSTEP_END"
    OPTIMIZER_STEP = "OPTIMIZER_STEP"

    # --- optimisation signals ---------------------------------------------
    LOSS = "LOSS"
    LR_CHANGE = "LR_CHANGE"
    GRADIENT_METRIC = "GRADIENT_METRIC"
    PARAMETER_METRIC = "PARAMETER_METRIC"
    ACTIVATION_METRIC = "ACTIVATION_METRIC"  # v0.2 (T2 / NNsight)
    GRAD_SCALER_STATE = "GRAD_SCALER_STATE"

    # --- host / device telemetry ------------------------------------------
    GPU_METRIC = "GPU_METRIC"  # v0.2 (DCGM adapter)
    SYSTEM_METRIC = "SYSTEM_METRIC"

    # --- data provenance ---------------------------------------------------
    DATA_BATCH = "DATA_BATCH"
    DATA_SHARD_CHANGE = "DATA_SHARD_CHANGE"
    DATALOADER_STATE = "DATALOADER_STATE"

    # --- durable state -----------------------------------------------------
    CHECKPOINT_SAVE = "CHECKPOINT_SAVE"
    CHECKPOINT_LOAD = "CHECKPOINT_LOAD"
    ARTIFACT_REF = "ARTIFACT_REF"

    # --- distributed -------------------------------------------------------
    DISTRIBUTED_EVENT = "DISTRIBUTED_EVENT"
    COLLECTIVE_EVENT = "COLLECTIVE_EVENT"  # v0.2
    HEALTH_CHECK = "HEALTH_CHECK"

    # --- profiling ---------------------------------------------------------
    PROFILE_EVENT = "PROFILE_EVENT"  # v0.2 (HTA)

    # --- diagnostics -------------------------------------------------------
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    WARNING = "WARNING"
    ERROR = "ERROR"

    # --- incident lifecycle ------------------------------------------------
    INCIDENT_OPEN = "INCIDENT_OPEN"
    INCIDENT_UPDATE = "INCIDENT_UPDATE"
    INCIDENT_CLOSE = "INCIDENT_CLOSE"


#: Event types no Phase 1 producer emits. Kept in the vocabulary so that the
#: schema is stable across phases and so that a Phase 2 producer needs no bump.
V02_RESERVED_EVENT_TYPES = frozenset(
    {
        EventType.ACTIVATION_METRIC,
        EventType.GPU_METRIC,
        EventType.COLLECTIVE_EVENT,
        EventType.PROFILE_EVENT,
    }
)


class SourceKind(_StrEnum):
    """Which component produced an event.

    ``source`` is the *kind*; ``producer`` is the concrete instance identity
    (see ``TrainingEvent.producer``). Two ranks running the same adapter share
    a ``source`` but differ in ``producer``.
    """

    PYTORCH_ADAPTER = "pytorch-adapter"
    OBSERVER = "observer"
    DETECTOR = "detector"
    INCIDENT_ENGINE = "incident-engine"
    CLI = "cli"
    SYSTEM = "system"


class DeviceType(_StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    UNKNOWN = "unknown"


class SpanKind(_StrEnum):
    """Matches the Phase 0 envelope: spans are pairs with a guaranteed close."""

    START = "start"
    END = "end"
    EVENT = "event"


class Severity(_StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"


class EventPriority(_StrEnum):
    """Drop policy is defined over these. P0 is never silently dropped.

    Phase 0's observer contract (``TRAINING_EVENT_SCHEMA_PROPOSAL.md`` 4.2)
    specifies a bounded queue with drop accounting. The priority ladder is what
    makes the drop policy principled rather than arbitrary.
    """

    P0 = "P0"  # incident / error / non-finite: never silently dropped
    P1 = "P1"  # core training metrics: loss, lr, grad norm, step time
    P2 = "P2"  # diagnostic metrics: layer metrics, memory
    P3 = "P3"  # verbose / debug telemetry


class MetricUnit(_StrEnum):
    """Closed unit vocabulary.

    Phase 0 named unit confusion (``ms`` vs ``s``) as a silent-correlation
    hazard, so the vocabulary is closed and normalised on ingest.
    """

    DIMENSIONLESS = "dimensionless"
    RATIO = "ratio"
    PERCENT = "percent"
    COUNT = "count"
    BYTES = "bytes"
    BYTES_PER_SEC = "bytes_per_sec"
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    CELSIUS = "celsius"
    WATT = "watt"
    HERTZ = "hertz"
    FLOPS = "flops"
    TFLOPS = "tflops"
    TOKENS_PER_SEC = "tokens_per_sec"
    SAMPLES_PER_SEC = "samples_per_sec"


class EvidenceLevel(_StrEnum):
    """How strongly a piece of evidence is established.

    Phase 0's rule (``ROOT_CAUSE_ARCHITECTURE.md`` section 0.1) is that an LLM
    may never promote a claim; only Observed evidence may. These four levels
    are that rule expressed as data.
    """

    OBSERVED = "OBSERVED"  # directly measured, no inference
    INFERRED = "INFERRED"  # derived by a stated statistical procedure
    HYPOTHESIS = "HYPOTHESIS"  # a causal claim awaiting a probe
    UNKNOWN = "UNKNOWN"  # the evidence is absent or insufficient


class IncidentState(_StrEnum):
    OPEN = "OPEN"
    UPDATED = "UPDATED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class OrderingRelation(_StrEnum):
    """The partial order over events.

    Deliberately *not* called ``BEFORE``/``AFTER``/``EQUAL``: the fourth member
    is the honest default. Phase 0 section 2.2 of the Phase 1 brief forbids
    claiming cross-rank absolute order, so ``CONCURRENT_OR_UNKNOWN`` exists to
    make the unprovable case representable rather than approximated.
    """

    BEFORE = "BEFORE"
    AFTER = "AFTER"
    SAME_POSITION = "SAME_POSITION"
    CONCURRENT_OR_UNKNOWN = "CONCURRENT_OR_UNKNOWN"


class OrderingConfidence(_StrEnum):
    """Why an ordering claim holds, or why it does not.

    ``PROVEN`` requires a happens-before witness: a shared producer sequence or
    an explicit causal link. Anything weaker is not an ordering claim.
    """

    PROVEN = "PROVEN"
    HIGH = "HIGH"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class IncidentType(_StrEnum):
    """Phase 1 incident taxonomy subset.

    Phase 0 defined 28 incident types. Phase 1 implements the four the release
    gate requires plus ``UNKNOWN``, which is mandatory as the honest fallback.
    The remaining Phase 0 types are Phase 2+ and must not appear here.
    """

    LOSS_SPIKE = "LOSS_SPIKE"
    LOSS_DIVERGENCE = "LOSS_DIVERGENCE"
    GRADIENT_EXPLOSION = "GRADIENT_EXPLOSION"
    NAN_DETECTED = "NAN_DETECTED"
    INF_DETECTED = "INF_DETECTED"
    THROUGHPUT_REGRESSION = "THROUGHPUT_REGRESSION"
    UNKNOWN = "UNKNOWN"


class DetectorId(_StrEnum):
    """Stable detector identifiers. Recorded on every anomaly for auditability."""

    LOSS_ROBUST_ZSCORE = "loss.robust_zscore"
    LOSS_TREND = "loss.trend_slope"
    GRAD_ROBUST_ZSCORE = "grad.robust_zscore"
    GRAD_ABSOLUTE_GUARD = "grad.absolute_guard"
    NUMERICS_FINITENESS = "numerics.finiteness"
    ACTIVATION_HEALTH = "activation.health"
    THROUGHPUT_REGRESSION = "throughput.regression"
