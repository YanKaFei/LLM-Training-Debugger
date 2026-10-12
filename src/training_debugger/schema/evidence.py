"""Evidence and the evidence buckets an incident is built from.

Phase 0's hardest-won rule (``research/ROOT_CAUSE_ARCHITECTURE.md`` section 3.5)
is that **negative evidence is evidence**. The brief's own worked example lists
"learning rate unchanged / parameter norm within baseline" alongside the
positive finding, because that list is what makes a diagnosis credible.

So an incident does not have "evidence"; it has four buckets, and a bucket
being empty is a fact the report must state:

``supporting``       evidence that the incident is real and where it started
``contradicting``    evidence against the leading hypothesis
``not_correlated``   signals that were *tested and found stable* -- mandatory
``unknown``          things we could not test, with the reason

Multiplicity (Phase 1 brief section 7.6) is a first-class field, not a footnote:
"1 / 32 layers anomalous" and "24 / 32 layers anomalous" are different
diagnoses, and a system that reports only "layer 12 anomalous" is lying by
omission.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .enums import EvidenceLevel, MetricUnit

__all__ = ["Evidence", "Multiplicity", "EvidenceBuckets", "EVIDENCE_BUCKETS"]


def _new_id(prefix: str) -> str:
    return "{}_{}".format(prefix, uuid.uuid4().hex[:16])


@dataclass
class Evidence:
    """One atomic, auditable claim backed by events.

    ``level`` is the crux. ``OBSERVED`` means the value was measured directly.
    ``INFERRED`` means a named statistical procedure produced it, and
    ``detector``/``threshold``/``baseline``/``window`` must therefore be
    populated -- an inferred claim with no stated method is not admissible.
    ``HYPOTHESIS`` is a causal claim; Phase 1 caps data-causation at this level
    by construction (no replay exists yet).
    """

    claim: str
    level: EvidenceLevel
    event_ids: List[str] = field(default_factory=list)

    metric_name: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[MetricUnit] = None
    baseline: Optional[float] = None
    threshold: Optional[float] = None
    window: Optional[List[int]] = None  # [start_step, end_step] inclusive

    detector: Optional[str] = None
    detector_version: Optional[str] = None
    confidence: Optional[float] = None

    rank: Optional[int] = None
    layer_index: Optional[int] = None
    module_path: Optional[str] = None

    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_id: str = field(default_factory=lambda: _new_id("ev"))

    def __post_init__(self) -> None:
        """Enforce the admissibility rule at construction time.

        An ``INFERRED`` claim without a detector is the exact failure mode Phase
        0 warned about (a confidence number with no decomposition), so it is
        rejected here rather than caught in review.
        """
        if self.level is EvidenceLevel.INFERRED and not self.detector:
            raise ValueError(
                "INFERRED evidence must name its detector (claim=%r)" % (self.claim,)
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1], got %r" % (self.confidence,))

    @property
    def is_observed(self) -> bool:
        return self.level is EvidenceLevel.OBSERVED

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Evidence":
        """Reconstruct from :meth:`to_dict`. Lossless.

        Reconstruction is required for incident persistence: an incident read back
        from Parquet must render identically to the one that was written, or the
        audit trail is worthless.
        """
        data = dict(payload)
        data.pop("evidence_id", None)  # regenerated; the stored id is authoritative
        unit = data.get("unit")
        if unit is not None and not isinstance(unit, MetricUnit):
            data["unit"] = MetricUnit(unit)
        level = data.get("level")
        if level is not None and not isinstance(level, EvidenceLevel):
            data["level"] = EvidenceLevel(level)
        window = data.get("window")
        if window is not None:
            data["window"] = list(window)
        event_ids = data.get("event_ids")
        if event_ids is not None:
            data["event_ids"] = list(event_ids)
        metadata = data.get("metadata")
        if metadata is not None:
            data["metadata"] = dict(metadata)
        for key in ("claim",):
            data.setdefault(key, "")
        built = cls(**data)
        if payload.get("evidence_id"):
            built.evidence_id = str(payload["evidence_id"])
        return built

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "claim": self.claim,
            "level": str(self.level),
            "event_ids": list(self.event_ids),
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": None if self.unit is None else str(self.unit),
            "baseline": self.baseline,
            "threshold": self.threshold,
            "window": list(self.window) if self.window else None,
            "detector": self.detector,
            "detector_version": self.detector_version,
            "confidence": self.confidence,
            "rank": self.rank,
            "layer_index": self.layer_index,
            "module_path": self.module_path,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


#: The four buckets, in a fixed order. Persistence and reconstruction iterate
#: this rather than hard-coding the names, so adding a bucket is one edit.
EVIDENCE_BUCKETS = ("supporting", "contradicting", "not_correlated", "unknown")


@dataclass
class Multiplicity:
    """How many of how many. Required on every incident.

    A detector that fires on one layer of a hundred and a detector that fires on
    ninety-nine of a hundred produce very different diagnoses; the ratio is the
    diagnosis. Every count carries its population so the ratio is always
    computable and never implied.
    """

    affected_ranks: List[int] = field(default_factory=list)
    population_ranks: int = 0

    affected_layers: List[int] = field(default_factory=list)
    population_layers: int = 0

    affected_steps: List[int] = field(default_factory=list)
    population_steps: int = 0

    event_count: int = 0
    population_count: int = 0

    def _ratio(self, affected: Sequence[Any], population: int) -> Optional[float]:
        if population <= 0:
            return None
        return len(affected) / float(population)

    @property
    def rank_ratio(self) -> Optional[float]:
        return self._ratio(self.affected_ranks, self.population_ranks)

    @property
    def layer_ratio(self) -> Optional[float]:
        return self._ratio(self.affected_layers, self.population_layers)

    @property
    def is_systemic_layers(self) -> bool:
        """True when a majority of layers are affected (a global cause, not local)."""
        ratio = self.layer_ratio
        return ratio is not None and ratio >= 0.5

    def summary(self) -> str:
        """Deterministic plain-English multiplicity line.

        Deliberately verbose about the denominator. "layer 12 anomalous" is not
        a diagnosis; "1 / 32 layers anomalous" is.
        """
        parts: List[str] = []
        if self.population_layers:
            parts.append("%d / %d layers" % (len(self.affected_layers), self.population_layers))
        if self.population_ranks:
            parts.append("%d / %d ranks" % (len(self.affected_ranks), self.population_ranks))
        if self.population_steps:
            parts.append("%d / %d steps" % (len(self.affected_steps), self.population_steps))
        if not parts:
            return "multiplicity not established"
        return "affected: " + ", ".join(parts)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Multiplicity":
        """Reconstruct from :meth:`to_dict`. Derived keys are recomputed."""
        return cls(
            affected_ranks=list(payload.get("affected_ranks") or []),
            population_ranks=int(payload.get("population_ranks") or 0),
            affected_layers=list(payload.get("affected_layers") or []),
            population_layers=int(payload.get("population_layers") or 0),
            affected_steps=list(payload.get("affected_steps") or []),
            population_steps=int(payload.get("population_steps") or 0),
            event_count=int(payload.get("event_count") or 0),
            population_count=int(payload.get("population_count") or 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "affected_ranks": list(self.affected_ranks),
            "population_ranks": self.population_ranks,
            "affected_layers": list(self.affected_layers),
            "population_layers": self.population_layers,
            "affected_steps": list(self.affected_steps),
            "population_steps": self.population_steps,
            "event_count": self.event_count,
            "population_count": self.population_count,
            "rank_ratio": self.rank_ratio,
            "layer_ratio": self.layer_ratio,
            "summary": self.summary(),
        }


@dataclass
class EvidenceBuckets:
    """The four evidence buckets. All four exist even when empty.

    ``not_correlated`` is validated at report time: if candidate signals were
    tested and none were found stable, the report must say so explicitly rather
    than omit the section. See ``Incident.validate`` and the Phase 0 rule
    "``not_correlated`` is mandatory".
    """

    supporting: List[Evidence] = field(default_factory=list)
    contradicting: List[Evidence] = field(default_factory=list)
    not_correlated: List[Evidence] = field(default_factory=list)
    unknown: List[Evidence] = field(default_factory=list)

    def all(self) -> List[Evidence]:
        return list(self.supporting) + list(self.contradicting) + list(self.not_correlated) + list(self.unknown)

    def by_level(self, level: EvidenceLevel) -> List[Evidence]:
        return [e for e in self.all() if e.level is level]

    def observed_supporting(self) -> List[Evidence]:
        return [e for e in self.supporting if e.level is EvidenceLevel.OBSERVED]

    def add_supporting(self, evidence: Evidence) -> None:
        self.supporting.append(evidence)

    def add_contradicting(self, evidence: Evidence) -> None:
        self.contradicting.append(evidence)

    def add_not_correlated(self, evidence: Evidence) -> None:
        self.not_correlated.append(evidence)

    def add_unknown(self, evidence: Evidence) -> None:
        self.unknown.append(evidence)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvidenceBuckets":
        """Reconstruct from :meth:`to_dict`, preserving empty buckets."""
        return cls(
            supporting=[Evidence.from_dict(e) for e in payload.get("supporting") or []],
            contradicting=[Evidence.from_dict(e) for e in payload.get("contradicting") or []],
            not_correlated=[Evidence.from_dict(e) for e in payload.get("not_correlated") or []],
            unknown=[Evidence.from_dict(e) for e in payload.get("unknown") or []],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supporting": [e.to_dict() for e in self.supporting],
            "contradicting": [e.to_dict() for e in self.contradicting],
            "not_correlated": [e.to_dict() for e in self.not_correlated],
            "unknown": [e.to_dict() for e in self.unknown],
        }
