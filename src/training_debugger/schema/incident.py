"""The incident model: the product's primary durable object.

Phase 1's governing sentence is *the product is not the graph, the product is
the incident*. Concretely, that means this dataclass -- not the metric store and
not any UI -- is the thing the whole pipeline exists to produce.

Three design commitments are encoded here and enforced in :meth:`Incident.validate`:

1. **The earliest abnormal evidence is a set, not an id.** ``earliest`` is an
   :class:`~training_debugger.schema.ordering.EarliestEvidence`; the singular
   ``first_abnormal_event_id`` is only populated when the partial order proves
   uniqueness (Phase 1 brief sections 2.4 and 8.4).
2. **Negative evidence is mandatory.** ``not_correlated`` must be non-empty for
   any incident where candidate signals were tested, and ``missing_evidence``
   must state what is absent. Phase 0 anti-pattern 3 forbids reporting
   correlations without the tested set.
3. **Every hypothesis carries its score decomposition.** A confidence number
   with no visible decomposition is unauditable (Phase 0 anti-pattern 2).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import (
    SCHEMA_VERSION,
    DetectorId,
    EventType,
    EvidenceLevel,
    IncidentState,
    IncidentType,
    OrderingConfidence,
    Severity,
)
from .evidence import Evidence, EvidenceBuckets, Multiplicity
from .ordering import EarliestEvidence

__all__ = ["Hypothesis", "Incident", "IncidentValidationError"]


class IncidentValidationError(ValueError):
    """Raised when an incident violates a Phase 0 admissibility rule."""


def _rebuild_candidate_events(rows: List[Any]) -> List[Any]:
    """Rebuild the minimal event views used for earliest-evidence rendering.

    Only the fields the renderer reads are restored. A full ``TrainingEvent``
    would require the complete stored event, and duplicating the event stream
    inside the incident store would create a second source of truth for events.
    """
    from .event import TrainingEvent

    rebuilt = []
    for row in rows:
        rebuilt.append(
            TrainingEvent(
                event_id=str(row.get("event_id") or ""),
                run_id=str(row.get("run_id") or ""),
                producer=str(row.get("producer") or "unknown"),
                source="incident-store",
                event_type=EventType.GRADIENT_METRIC,
                step=row.get("step"),
                metric_name=row.get("metric_name"),
                metric_value=row.get("metric_value"),
                layer_index=row.get("layer_index"),
                module_path=row.get("module_path"),
                rank=row.get("rank"),
            )
        )
    return rebuilt


def _new_incident_id() -> str:
    return "inc_{}".format(uuid.uuid4().hex[:16])


@dataclass
class Hypothesis:
    """A candidate cause with an explicit, inspectable score.

    Phase 1 does not learn anything and does not ask a model. A hypothesis is a
    template from a fixed library, scored by arithmetic over evidence, and
    ``score_decomposition`` records every factor so a reviewer can recompute the
    number by hand. Phase 0 anti-pattern 2 makes this non-optional.
    """

    template_id: str
    statement: str
    score: float
    status: EvidenceLevel
    score_decomposition: Dict[str, float] = field(default_factory=dict)
    supporting: List[str] = field(default_factory=list)  # evidence_ids
    refuting: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    hypothesis_id: str = field(default_factory=lambda: "hyp_{}".format(uuid.uuid4().hex[:12]))

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise IncidentValidationError("hypothesis score must be in [0, 1]: %r" % self.score)
        if not self.score_decomposition:
            raise IncidentValidationError(
                "hypothesis %r has no score decomposition; a bare score is unauditable"
                % self.template_id
            )

    #: ``to_dict`` uses descriptive key names; the dataclass uses short ones.
    #: The mapping lives here so a rename in one place cannot silently break
    #: reconstruction.
    _FIELD_ALIASES = {
        "supporting_evidence_ids": "supporting",
        "refuting_evidence_ids": "refuting",
        "missing_evidence": "missing",
    }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Hypothesis":
        """Reconstruct from :meth:`to_dict`, preserving the stored hypothesis id."""
        data = dict(payload)
        for stored, field_name in cls._FIELD_ALIASES.items():
            if stored in data:
                data[field_name] = data.pop(stored)
        hypothesis_id = data.pop("hypothesis_id", None)
        status = data.get("status")
        if status is not None and not isinstance(status, EvidenceLevel):
            data["status"] = EvidenceLevel(status)
        if data.get("score_decomposition") is None:
            data["score_decomposition"] = {}
        for key in ("supporting", "refuting", "missing"):
            if data.get(key) is None:
                data[key] = []
        built = cls(**data)
        if hypothesis_id:
            built.hypothesis_id = str(hypothesis_id)
        return built

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "template_id": self.template_id,
            "statement": self.statement,
            "score": self.score,
            "status": str(self.status),
            "score_decomposition": dict(self.score_decomposition),
            "supporting_evidence_ids": list(self.supporting),
            "refuting_evidence_ids": list(self.refuting),
            "missing_evidence": list(self.missing),
        }


@dataclass
class Incident:
    """A detected training incident, with its evidence and ordering verdict."""

    run_id: str
    incident_type: IncidentType
    symptom: Optional[Evidence] = None
    earliest: Optional[EarliestEvidence] = None
    evidence: EvidenceBuckets = field(default_factory=EvidenceBuckets)
    multiplicity: Multiplicity = field(default_factory=Multiplicity)
    hypotheses: List[Hypothesis] = field(default_factory=list)

    incident_id: str = field(default_factory=_new_incident_id)
    schema_version: str = SCHEMA_VERSION
    state: IncidentState = IncidentState.OPEN
    severity: Severity = Severity.WARN

    detected_step: Optional[int] = None
    opened_wall_time_ns: int = field(default_factory=time.time_ns)
    updated_wall_time_ns: int = field(default_factory=time.time_ns)

    detector_ids: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    ordering_confidence: Optional[str] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors used by the renderer and the golden tests
    # ------------------------------------------------------------------

    @property
    def first_abnormal_event_id(self) -> Optional[str]:
        """Populated **only** when the partial order proves a unique earliest.

        Otherwise ``None``, and callers must use
        :attr:`earliest_abnormal_candidate_ids`. This property exists so that a
        consumer cannot accidentally read a singular answer where only a set is
        defensible.
        """
        if self.earliest is None:
            return None
        return self.earliest.first_abnormal_event_id

    @property
    def earliest_abnormal_candidate_ids(self) -> List[str]:
        if self.earliest is None:
            return []
        return self.earliest.candidate_ids()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def touch(self, state: Optional[IncidentState] = None) -> None:
        """Mark the incident updated. Phase 1 keeps workflow trivial on purpose."""
        if state is not None:
            self.state = state
        elif self.state is IncidentState.OPEN:
            self.state = IncidentState.UPDATED
        self.updated_wall_time_ns = time.time_ns()

    def resolve(self) -> None:
        self.touch(IncidentState.RESOLVED)

    def close(self) -> None:
        self.touch(IncidentState.CLOSED)

    # ------------------------------------------------------------------
    # Admissibility
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Enforce Phase 0's report contract. Raises ``IncidentValidationError``.

        Called by the engine before an incident leaves the process, and by the
        golden-case tests. It is intentionally strict: an incident that cannot
        state what it tested and what it could not establish is not a usable
        artifact.
        """
        if not self.run_id:
            raise IncidentValidationError("incident must carry a run_id")
        if self.symptom is None:
            raise IncidentValidationError(
                "incident %s has no symptom evidence; an incident without an observed "
                "symptom is not admissible" % self.incident_id
            )
        if self.symptom.level is not EvidenceLevel.OBSERVED:
            raise IncidentValidationError(
                "the symptom of incident %s must be OBSERVED evidence, got %s"
                % (self.incident_id, self.symptom.level)
            )
        if self.earliest is None or not self.earliest.candidates:
            raise IncidentValidationError(
                "incident %s has no earliest abnormal evidence" % self.incident_id
            )
        if not self.evidence.supporting:
            raise IncidentValidationError(
                "incident %s has no supporting evidence" % self.incident_id
            )
        # Phase 0: not_correlated is mandatory. It may be empty ONLY if the
        # engine explicitly recorded that it tested nothing, which it must say
        # by naming the reason in missing_evidence.
        if not self.evidence.not_correlated and not self.missing_evidence:
            raise IncidentValidationError(
                "incident %s reports no negative evidence and no missing evidence; "
                "state what was tested and found stable, or state why nothing was tested"
                % self.incident_id
            )
        for hypothesis in self.hypotheses:
            if hypothesis.status is EvidenceLevel.OBSERVED and not hypothesis.supporting:
                raise IncidentValidationError(
                    "hypothesis %s claims OBSERVED status with no supporting evidence"
                    % hypothesis.template_id
                )
        if self.earliest.first_abnormal_event_id is None and not self.earliest.explanation:
            raise IncidentValidationError(
                "incident %s has no unique first abnormal event and no explanation why"
                % self.incident_id
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Incident":
        """Reconstruct an incident from :meth:`to_dict`.

        The earliest-evidence block is rebuilt into an
        :class:`~training_debugger.schema.ordering.EarliestEvidence` so that a
        reconstructed incident renders identically to the original; a
        reconstruction that cannot do that would make the persisted audit trail
        decorative.
        """
        from .ordering import EarliestEvidence

        data = dict(payload)
        earliest_block = data.pop("earliest_abnormal", None) or {}
        symptom_block = data.pop("symptom", None)
        evidence_block = data.pop("evidence", None) or {}
        multiplicity_block = data.pop("multiplicity", None) or {}
        hypotheses_block = data.pop("hypotheses", None) or []
        # Derived / convenience keys produced by to_dict but not constructor args.
        for derived in (
            "first_abnormal_event_id",
            "earliest_abnormal_candidate_ids",
        ):
            data.pop(derived, None)

        candidates = _rebuild_candidate_events(earliest_block.get("candidates") or [])
        earliest = EarliestEvidence(
            candidates=candidates,
            ordering_confidence=OrderingConfidence(
                earliest_block.get("ordering_confidence") or "UNKNOWN"
            ),
            first_abnormal_event_id=earliest_block.get("first_abnormal_event_id"),
            explanation=earliest_block.get("explanation") or "",
        )

        incident_type = data.get("incident_type")
        if incident_type is not None and not isinstance(incident_type, IncidentType):
            data["incident_type"] = IncidentType(incident_type)
        state = data.get("state")
        if state is not None and not isinstance(state, IncidentState):
            data["state"] = IncidentState(state)
        severity = data.get("severity")
        if severity is not None and not isinstance(severity, Severity):
            data["severity"] = Severity(severity)

        incident = cls(
            symptom=None if symptom_block is None else Evidence.from_dict(symptom_block),
            earliest=earliest,
            evidence=EvidenceBuckets.from_dict(evidence_block),
            multiplicity=Multiplicity.from_dict(multiplicity_block),
            hypotheses=[Hypothesis.from_dict(h) for h in hypotheses_block],
            **data
        )
        return incident

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "incident_id": self.incident_id,
            "run_id": self.run_id,
            "incident_type": str(self.incident_type),
            "state": str(self.state),
            "severity": str(self.severity),
            "detected_step": self.detected_step,
            "opened_wall_time_ns": self.opened_wall_time_ns,
            "updated_wall_time_ns": self.updated_wall_time_ns,
            "detector_ids": list(self.detector_ids),
            "confidence": self.confidence,
            "ordering_confidence": self.ordering_confidence,
            "summary": self.summary,
            "symptom": None if self.symptom is None else self.symptom.to_dict(),
            "earliest_abnormal": {
                "candidates": [
                    {
                        "event_id": c.event_id,
                        "run_id": c.run_id,
                        "producer": c.producer,
                        "metric_name": c.metric_name,
                        "metric_value": c.metric_value,
                        "step": c.step,
                        "rank": c.rank,
                        "layer_index": c.layer_index,
                        "module_path": c.module_path,
                    }
                    for c in (self.earliest.candidates if self.earliest else [])
                ],
                "first_abnormal_event_id": self.first_abnormal_event_id,
                "earliest_abnormal_candidate_ids": self.earliest_abnormal_candidate_ids,
                "ordering_confidence": str(self.earliest.ordering_confidence)
                if self.earliest
                else None,
                "explanation": self.earliest.explanation if self.earliest else None,
            },
            "evidence": self.evidence.to_dict(),
            "multiplicity": self.multiplicity.to_dict(),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "missing_evidence": list(self.missing_evidence),
            "metadata": dict(self.metadata),
        }
