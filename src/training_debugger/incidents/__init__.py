"""Incident engine: detection results in, defensible incidents out."""

from __future__ import annotations

from .engine import IncidentEngine, IncidentEngineConfig
from .evidence_builder import CANDIDATE_SIGNALS, SignalAssessment, assess_signal, build_evidence
from .ranking import HYPOTHESIS_LIBRARY, HypothesisTemplate, rank_hypotheses
from .relations import (
    DEFAULT_MAX_GAP_STEPS,
    RELATION_RULE_VERSION,
    IncidentRelation,
    RelationType,
    link_consequences,
    relation_registry,
)
from .reconciliation import (
    IncidentGroup,
    ReconciliationResult,
    reconcile_incidents,
)
from .render import confidence_label, render_incident, render_incidents_table, render_verdict, render_verdicts

__all__ = [
    "IncidentEngine",
    "IncidentEngineConfig",
    "build_evidence",
    "assess_signal",
    "SignalAssessment",
    "CANDIDATE_SIGNALS",
    "rank_hypotheses",
    "HYPOTHESIS_LIBRARY",
    "HypothesisTemplate",
    "render_incident",
    "render_incidents_table",
    "confidence_label",
    "render_verdict",
    "render_verdicts",
    "IncidentRelation",
    "RelationType",
    "link_consequences",
    "relation_registry",
    "RELATION_RULE_VERSION",
    "DEFAULT_MAX_GAP_STEPS",
    "IncidentGroup",
    "ReconciliationResult",
    "reconcile_incidents",
]
