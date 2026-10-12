"""The incident engine.

This is Phase 1's centre of gravity. It turns a pile of detector findings into a
single durable object that can answer:

* what kind of incident is this, and how severe;
* what was the *observed symptom* (what the user noticed);
* what **earlier** abnormal evidence exists, and is its ordering defensible;
* what was tested and found stable (negative evidence);
* how many of how many (multiplicity);
* which hypotheses the evidence supports, with decomposable scores;
* what evidence is missing.

The ordering question is handled by :class:`~training_debugger.schema.ordering.EventOrdering`,
which returns a single ``first_abnormal_event_id`` **only** when the partial order
proves uniqueness. Everything else yields a candidate set plus an explicit
statement that the ordering cannot be established -- because a plausible invented
precedence is the most damaging thing this system could produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..detectors.base import Anomaly, DetectorConfig, robust_stats
from ..detectors.runner import DetectionResult, suggest_incident_type
from ..schema.enums import (
    EvidenceLevel,
    IncidentState,
    IncidentType,
    OrderingConfidence,
    Severity,
)
from ..schema.event import TrainingEvent
from ..schema.evidence import Evidence, EvidenceBuckets, Multiplicity
from ..schema.incident import Hypothesis, Incident
from ..schema.ordering import EarliestEvidence, EventOrdering
from .evidence_builder import build_evidence
from .ranking import rank_hypotheses

__all__ = ["IncidentEngine", "IncidentEngineConfig"]

Series = Sequence[Tuple[int, float]]


@dataclass
class IncidentEngineConfig:
    """Tunables. Everything here is configuration, nothing is learned."""

    #: Incident types whose *symptom* is a training-level signal rather than a
    #: per-layer diagnostic. Used only to break ties when choosing the symptom.
    training_level_types: Tuple[IncidentType, ...] = (
        IncidentType.LOSS_SPIKE,
        IncidentType.LOSS_DIVERGENCE,
    )
    baseline_window: int = 50
    during_window: int = 30
    #: Above this score an ambiguity note is added rather than a verdict.
    ambiguity_margin: float = 0.10


class IncidentEngine:
    """Builds incidents from detection results."""

    def __init__(
        self,
        config: Optional[IncidentEngineConfig] = None,
        detector_config: Optional[DetectorConfig] = None,
    ) -> None:
        self.config = config or IncidentEngineConfig()
        self.detector_config = detector_config or DetectorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        run_id: str,
        detection: DetectionResult,
        events: Sequence[TrainingEvent],
        series_by_metric: Dict[str, Series],
        *,
        layer_series: Optional[Dict[int, Series]] = None,
    ) -> List[Incident]:
        """Build one incident per distinct incident type found.

        Phase 1 emits at most one incident per type per run: the detector layer
        already coalesces repeated firings via cooldowns, and splitting a single
        sustained anomaly into many incidents is the failure mode Phase 0's
        suppression rules exist to prevent.
        """
        if not detection.anomalies:
            return []

        by_type: Dict[IncidentType, List[Anomaly]] = {}
        for anomaly in detection.anomalies:
            incident_type = suggest_incident_type(anomaly)
            if incident_type is None:
                continue
            by_type.setdefault(incident_type, []).append(anomaly)

        incidents: List[Incident] = []
        for incident_type, anomalies in sorted(by_type.items(), key=lambda kv: str(kv[0])):
            incidents.append(
                self._build_one(
                    run_id, incident_type, anomalies, detection, events, series_by_metric,
                    layer_series or {},
                )
            )
        return incidents

    # ------------------------------------------------------------------

    def _build_one(
        self,
        run_id: str,
        incident_type: IncidentType,
        anomalies: List[Anomaly],
        detection: DetectionResult,
        events: Sequence[TrainingEvent],
        series_by_metric: Dict[str, Series],
        layer_series: Dict[int, Series],
    ) -> Incident:
        symptom = self._select_symptom(anomalies)
        onset_step = min(a.first_abnormal_step for a in anomalies)

        # Map anomalies onto the events that produced them, so the partial order
        # can be applied. An anomaly the events do not contain is a detector bug
        # and is reported as unknown ordering rather than guessed at.
        mapping = self._map_to_events(anomalies, events)
        ordering_candidates = [event for event, _ in mapping]
        earliest = EventOrdering.earliest(ordering_candidates) if ordering_candidates else EarliestEvidence(
            candidates=[],
            ordering_confidence=OrderingConfidence.UNKNOWN,
            first_abnormal_event_id=None,
            explanation=(
                "No abnormal event could be matched to the stored event stream, so "
                "no ordering claim is possible."
            ),
        )

        event_ids_by_metric: Dict[str, List[str]] = {}
        for event, anomaly in mapping:
            event_ids_by_metric.setdefault(anomaly.metric_name, []).append(event.event_id)

        buckets, assessments = build_evidence(
            anomalies,
            series_by_metric,
            onset_step=onset_step,
            event_ids_by_metric=event_ids_by_metric,
            baseline_window=self.config.baseline_window,
            during_window=self.config.during_window,
        )

        hypotheses = rank_hypotheses(
            incident_type, anomalies, buckets, symptom_step=symptom.first_abnormal_step
        )

        missing = self._missing_evidence(buckets, assessments, ordering_candidates, earliest)
        severity = self._severity(incident_type, anomalies)

        incident = Incident(
            run_id=run_id,
            incident_type=incident_type,
            symptom=self._symptom_evidence(symptom, event_ids_by_metric),
            earliest=earliest,
            evidence=buckets,
            multiplicity=detection.multiplicity,
            hypotheses=hypotheses,
            severity=severity,
            detected_step=max(a.first_abnormal_step for a in anomalies),
            detector_ids=sorted({str(a.detector) for a in anomalies}),
            confidence=self._confidence(anomalies, earliest),
            ordering_confidence=str(earliest.ordering_confidence),
            missing_evidence=missing,
            metadata={
                "onset_step": onset_step,
                "anomaly_count": len(anomalies),
                "series_examined": len(detection.series_examined),
                "detector_errors": len(detection.detector_errors),
                "suppressed_findings": len(detection.suppressed),
                "latest_layer_series": len(layer_series),
                "signals_probed": [
                    {
                        "metric_name": a.metric_name,
                        "status": a.status,
                        "relative_change": a.relative_change,
                    }
                    for a in assessments
                ],
            },
        )
        incident.summary = self._summary(incident)
        incident.validate()
        return incident

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_symptom(anomalies: List[Anomaly]) -> Anomaly:
        """The symptom is what a user would have noticed.

        Ordered by (is an observed finding, is a loss/training-level signal,
        earliest step). A per-layer gradient anomaly is *evidence*, not a
        symptom: nobody watches layer 47's gradient norm.
        """
        def rank(anomaly: Anomaly) -> Tuple[int, int, int]:
            observed = 0 if anomaly.level is EvidenceLevel.OBSERVED else 1
            metric = anomaly.metric_name or ""
            symptom_metric = 0 if metric.startswith("train.") else 1
            layer_scope = 0 if anomaly.layer_index is None else 1
            return (observed, symptom_metric + layer_scope, anomaly.first_abnormal_step)

        return sorted(anomalies, key=rank)[0]

    @staticmethod
    def _map_to_events(
        anomalies: List[Anomaly], events: Sequence[TrainingEvent]
    ) -> List[Tuple[TrainingEvent, Anomaly]]:
        """Match each anomaly to the event that produced it.

        Matching is on ``(metric_name, step)`` plus rank and layer when the
        anomaly carries them. Where several events match, the earliest by
        producer sequence wins, because that is the observation that first
        established the anomaly.
        """
        index: Dict[Tuple[str, int], List[TrainingEvent]] = {}
        for event in events:
            if event.metric_name is None or event.step is None:
                continue
            index.setdefault((event.metric_name, event.step), []).append(event)

        mapping: List[Tuple[TrainingEvent, Anomaly]] = []
        for anomaly in anomalies:
            candidates = index.get((anomaly.metric_name, anomaly.first_abnormal_step), [])
            if anomaly.rank is not None:
                filtered = [e for e in candidates if e.rank == anomaly.rank]
                candidates = filtered or candidates
            if anomaly.layer_index is not None:
                filtered = [e for e in candidates if e.layer_index == anomaly.layer_index]
                candidates = filtered or candidates
            if not candidates:
                continue
            candidates.sort(
                key=lambda e: (e.source_seq if e.source_seq is not None else 1 << 62, e.event_id)
            )
            mapping.append((candidates[0], anomaly))
        return mapping

    @staticmethod
    def _symptom_evidence(symptom: Anomaly, event_ids: Dict[str, List[str]]) -> Evidence:
        """The symptom must be OBSERVED -- it is what was actually seen.

        A detector's z-score is an inference about *significance*, but the value
        itself was measured. The symptom therefore carries the observed level
        with the detector recorded as its provenance, which satisfies
        ``Incident.validate`` without overstating anything.
        """
        return Evidence(
            claim=(
                "{metric} = {value:.6g} at step {step} ({detector})".format(
                    metric=symptom.metric_name,
                    value=symptom.observed_value,
                    step=symptom.first_abnormal_step,
                    detector=symptom.detector,
                )
            ),
            level=EvidenceLevel.OBSERVED,
            detector=str(symptom.detector),
            detector_version=symptom.detector_version,
            metric_name=symptom.metric_name,
            value=symptom.observed_value,
            baseline=symptom.baseline,
            threshold=symptom.threshold,
            window=list(symptom.window),
            confidence=symptom.confidence,
            rank=symptom.rank,
            layer_index=symptom.layer_index,
            module_path=symptom.module_path,
            event_ids=list(event_ids.get(symptom.metric_name, [])),
            reason=symptom.reason,
            metadata=dict(symptom.metadata, detector_score=symptom.score),
        )

    @staticmethod
    def _severity(incident_type: IncidentType, anomalies: List[Anomaly]) -> Severity:
        if any(a.level is EvidenceLevel.OBSERVED for a in anomalies):
            return Severity.FATAL
        if incident_type in (IncidentType.LOSS_SPIKE, IncidentType.LOSS_DIVERGENCE,
                             IncidentType.GRADIENT_EXPLOSION):
            return Severity.ERROR
        return Severity.WARN

    @staticmethod
    def _confidence(anomalies: List[Anomaly], earliest: EarliestEvidence) -> float:
        base = max(a.confidence for a in anomalies)
        if earliest.ordering_confidence is OrderingConfidence.PROVEN:
            return min(0.99, base)
        if earliest.ordering_confidence is OrderingConfidence.LOW:
            return min(0.8, base * 0.9)
        return min(0.6, base * 0.75)

    def _missing_evidence(
        self,
        buckets: EvidenceBuckets,
        assessments: Sequence[Any],
        ordering_candidates: Sequence[TrainingEvent],
        earliest: EarliestEvidence,
    ) -> List[str]:
        missing: List[str] = []
        if not ordering_candidates:
            missing.append("stored events for the abnormal findings (ordering not assessable)")
        if earliest.first_abnormal_event_id is None and len(earliest.candidates) > 1:
            missing.append(
                "evidence that would order the {} concurrent abnormal candidates".format(
                    len(earliest.candidates)
                )
            )
        unavailable = [a for a in assessments if a.status == "unavailable"]
        for assessment in unavailable:
            missing.append(
                "{} (not recorded for this run)".format(assessment.label)
            )
        if not buckets.contradicting:
            missing.append(
                "no candidate hypothesis could be actively refuted with the available signals"
            )
        return missing

    @staticmethod
    def _summary(incident: Incident) -> str:
        hypothesis = incident.hypotheses[0].statement if incident.hypotheses else "no hypothesis available"
        return "{type} at step {step}: {hypothesis}".format(
            type=incident.incident_type, step=incident.detected_step, hypothesis=hypothesis
        )
