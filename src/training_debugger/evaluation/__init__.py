"""Evaluation semantics: the layer that knows ground truth.

Detectors report fires. Only this layer knows which fires were right, and it is
separate precisely because that knowledge must never reach a detector.
"""

from __future__ import annotations

from .consequences import (
    OBSERVATION_CLASSES,
    GroundTruthConsequence,
    classify_observation,
    evaluate_consequences,
)
from .earliest import EarliestEvidenceOutcome, evaluate_earliest_evidence
from .levels import (
    SCORING_INVALID,
    LevelScore,
    ScoringValidity,
    evaluate_levels,
    validate_scoring_inputs,
)
from .relations import (
    RelationOpportunityScore,
    score_relation_opportunities,
)
from .persisted import build_scenarios_from_store, load_persisted_runs, persisted_run_ids
from .layers import LayerScores, empty_scores, evaluate_layers, uncertainty_rates
from .semantics import (
    OUTCOME_KINDS,
    DetectionOutcome,
    DetectionWindow,
    ScenarioEvaluation,
    aggregate,
    classify_detection,
    evaluate_scenario,
)

__all__ = [
    "DetectionWindow",
    "DetectionOutcome",
    "ScenarioEvaluation",
    "classify_detection",
    "evaluate_scenario",
    "aggregate",
    "OUTCOME_KINDS",
    "LayerScores",
    "evaluate_layers",
    "empty_scores",
    "uncertainty_rates",
    "evaluate_consequences",
    "classify_observation",
    "GroundTruthConsequence",
    "OBSERVATION_CLASSES",
    "evaluate_earliest_evidence",
    "EarliestEvidenceOutcome",
    "load_persisted_runs",
    "build_scenarios_from_store",
    "persisted_run_ids",
    "evaluate_levels",
    "validate_scoring_inputs",
    "LevelScore",
    "ScoringValidity",
    "SCORING_INVALID",
    "RelationOpportunityScore",
    "score_relation_opportunities",
]
