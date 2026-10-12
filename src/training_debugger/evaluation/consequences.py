"""K2: consequence-aware scoring.

Phase 1.6 reported `numerics` episode precision 0.314, and that single number mixed
two things that must never be mixed:

* a **true secondary manifestation** -- a `NaN`/`Inf` that genuinely followed a
  confirmed primary fault and is real evidence of it;
* a **duplicate top-level incident** -- the system inventing a second independent root
  cause for a consequence.

The brief is explicit that the first must not be scored as a detector false positive,
while the second must be scored as an error. This module separates them.

Ground truth carries the distinction, and **the detector layer never reads it**:

    primary_fault_type
    allowed_secondary_manifestations
    secondary_window
    scope_constraints

Six observation classes, and the point of six rather than two is that
``PRIMARY_TRUE_POSITIVE`` and ``SECONDARY_TRUE_MANIFESTATION`` are both correct
observations while ``DUPLICATE_TOP_LEVEL_INCIDENT`` is not -- collapsing them would
either reward duplication or punish correct consequence reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Ground truth names a *scenario class*; an episode carries an *anomaly type*.
#: Without this mapping the two never match and every observation is scored as
#: unrelated -- which is exactly what happened the first time these evaluators were
#: run, producing ``primary_precision = 0.0`` from a naming mismatch rather than from
#: a detector failure.
FAULT_CLASS_TO_ANOMALY = {
    "F1_loss_spike": "LOSS_SPIKE",
    "F2_sustained_divergence": "LOSS_DIVERGENCE",
    "F3_gradient_explosion": "GRADIENT_EXPLOSION",
    "F4_nan_loss": "NAN_DETECTED",
    "F5_inf_gradient": "INF_DETECTED",
    "F5b_nan_gradient": "NAN_DETECTED",
    "F6_throughput_drop": "THROUGHPUT_REGRESSION",
    "F10_mixed_anomaly": "NAN_DETECTED",
    "A4_nan_that_recovers": "NAN_DETECTED",
    "A5_inf_one_layer_only": "INF_DETECTED",
    "A3_gradient_spike_no_loss_consequence": "GRADIENT_EXPLOSION",
    "A1_single_huge_harmless_batch": "GRADIENT_EXPLOSION",
    "A2_loss_spike_stable_gradients": "LOSS_SPIKE",
}


def primary_anomaly_type(fault_class: Optional[str]) -> Optional[str]:
    """Map a scenario class onto the anomaly type it is expected to produce."""
    if not fault_class:
        return None
    if fault_class in FAULT_CLASS_TO_ANOMALY:
        return FAULT_CLASS_TO_ANOMALY[fault_class]
    # Already an anomaly type.
    if fault_class.isupper():
        return fault_class
    return None


__all__ = [
    "OBSERVATION_CLASSES",
    "GroundTruthConsequence",
    "classify_observation",
    "evaluate_consequences",
    "FAULT_CLASS_TO_ANOMALY",
    "primary_anomaly_type",
]

#: Every class an observed episode can receive.
OBSERVATION_CLASSES = (
    "PRIMARY_TRUE_POSITIVE",
    "SECONDARY_TRUE_MANIFESTATION",
    "UNRELATED_FALSE_POSITIVE",
    "DUPLICATE_TOP_LEVEL_INCIDENT",
    "CORRECT_ABSTENTION",
    "MISSED_PRIMARY",
)


@dataclass
class GroundTruthConsequence:
    """Evaluation-only description of what a fault was allowed to cause."""

    primary_fault_type: Optional[str] = None
    allowed_secondary_manifestations: Tuple[str, ...] = ()
    secondary_window: Optional[Tuple[int, int]] = None
    scope_constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_fault_type": self.primary_fault_type,
            "allowed_secondary_manifestations": list(self.allowed_secondary_manifestations),
            "secondary_window": list(self.secondary_window or []),
            "scope_constraints": dict(self.scope_constraints),
        }

    @classmethod
    def from_fault(cls, fault: Dict[str, Any]) -> "GroundTruthConsequence":
        injection = fault.get("injection_step")
        window = None
        if injection is not None:
            # A consequence may surface well after the fault; a poisoned optimiser
            # state can take tens of steps to become non-finite.
            window = (injection, injection + 100)
        return cls(
            primary_fault_type=fault.get("primary_fault_type")
            or (None if fault.get("healthy")
                else primary_anomaly_type(fault.get("fault_type"))),
            allowed_secondary_manifestations=tuple(
                fault.get("allowed_secondary_manifestations") or ()
            ),
            secondary_window=window,
            scope_constraints=fault.get("scope_constraints") or {},
        )


def classify_observation(
    episode: Dict[str, Any],
    truth: GroundTruthConsequence,
    *,
    injected: bool,
    acceptable_window: Optional[Sequence[int]],
    opened_top_level_incident: bool,
) -> str:
    """Classify one observed episode against the ground truth.

    ``opened_top_level_incident`` is what separates a correct secondary
    manifestation from a duplicate root incident: both observe the same `NaN`, and
    only the second claims it as an independent cause.
    """
    anomaly_type = episode.get("anomaly_type")
    onset = episode.get("candidate_onset_step")
    confirmed = (
        episode.get("state") == "CONFIRMED"
        or episode.get("outcome") == "CONFIRMED_ANOMALY"
    )

    if not injected:
        # A healthy run has no fault, so nothing observed can be a true positive.
        return (
            "DUPLICATE_TOP_LEVEL_INCIDENT" if opened_top_level_incident
            else "UNRELATED_FALSE_POSITIVE"
        )

    primary_match = anomaly_type == truth.primary_fault_type
    in_window = (
        onset is not None and acceptable_window is not None
        and acceptable_window[0] <= onset <= acceptable_window[1]
    )

    if primary_match and in_window:
        return "PRIMARY_TRUE_POSITIVE" if confirmed else "CORRECT_ABSTENTION"

    # A non-finite manifestation of an allowed primary, inside the consequence window.
    if (
        anomaly_type in truth.allowed_secondary_manifestations
        and onset is not None
        and truth.secondary_window is not None
        and truth.secondary_window[0] <= onset <= truth.secondary_window[1]
    ):
        if opened_top_level_incident:
            return "DUPLICATE_TOP_LEVEL_INCIDENT"
        return "SECONDARY_TRUE_MANIFESTATION" if confirmed else "CORRECT_ABSTENTION"

    if primary_match and not in_window:
        # Right family, wrong place: an early or late detection, not a true positive.
        return "UNRELATED_FALSE_POSITIVE" if confirmed else "CORRECT_ABSTENTION"

    return "UNRELATED_FALSE_POSITIVE" if confirmed else "CORRECT_ABSTENTION"


def evaluate_consequences(scenarios: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Separate primary, secondary, duplicate and relation quality.

    Each ``scenario`` must carry ``episodes``, ``verdicts``, ``injected``,
    ``acceptable_window`` and optionally ``ground_truth`` (a
    :class:`GroundTruthConsequence` dict). Where ground truth is absent the fault
    type itself is used as the primary and ``NAN_DETECTED``/``INF_DETECTED`` are
    treated as allowed consequences of a gradient or loss fault -- which is the
    physical relationship Phase 1.3 identified, not an assumption about scoring.
    """
    counts = {name: 0 for name in OBSERVATION_CLASSES}
    relation_tp = relation_fp = relation_fn = 0
    primary_runs = 0
    primary_found = 0

    for scenario in scenarios:
        if scenario.get("run_error"):
            continue
        injected = bool(scenario.get("injected"))
        raw_truth = scenario.get("ground_truth")
        if raw_truth:
            truth = GroundTruthConsequence(**raw_truth)
        else:
            primary = None if not injected else primary_anomaly_type(scenario.get("fault_type"))
            allowed = ()
            if primary in ("GRADIENT_EXPLOSION", "LOSS_DIVERGENCE", "LOSS_SPIKE"):
                allowed = ("NAN_DETECTED", "INF_DETECTED")
            truth = GroundTruthConsequence(
                primary_fault_type=primary,
                allowed_secondary_manifestations=allowed,
                secondary_window=(
                    (scenario["injection_step"], scenario["injection_step"] + 100)
                    if scenario.get("injection_step") is not None else None
                ),
            )

        verdicts = scenario.get("verdicts", [])
        opened = {
            v.get("anomaly_type")
            for v in verdicts
            if v.get("verdict") == "CONFIRMED_INCIDENT"
        }

        for episode in scenario.get("episodes", []):
            anomaly_type = episode.get("anomaly_type")
            kind = classify_observation(
                episode, truth, injected=injected,
                acceptable_window=scenario.get("acceptable_window"),
                opened_top_level_incident=anomaly_type in opened,
            )
            counts[kind] += 1

        if injected:
            primary_runs += 1
            if truth.primary_fault_type in opened:
                primary_found += 1

        for relation in scenario.get("relations", []):
            target = relation.get("target_anomaly_type")
            allowed = target in truth.allowed_secondary_manifestations
            in_window = (
                relation.get("gap_steps") is not None
                and 0 <= relation["gap_steps"] <= 100
            )
            if allowed and in_window:
                relation_tp += 1
            else:
                relation_fp += 1
        expected_relations = 1 if (
            injected and truth.allowed_secondary_manifestations
            and any(e.get("anomaly_type") in truth.allowed_secondary_manifestations
                    for e in scenario.get("episodes", []))
        ) else 0
        relation_fn += max(0, expected_relations - len(scenario.get("relations", [])))

    def _pr(tp: int, fp: int, fn: int) -> Tuple[Optional[float], Optional[float]]:
        return (
            tp / (tp + fp) if (tp + fp) else None,
            tp / (tp + fn) if (tp + fn) else None,
        )

    primary_precision, primary_recall = _pr(
        counts["PRIMARY_TRUE_POSITIVE"],
        counts["UNRELATED_FALSE_POSITIVE"] + counts["DUPLICATE_TOP_LEVEL_INCIDENT"],
        counts["MISSED_PRIMARY"],
    )
    secondary_precision, secondary_recall = _pr(
        counts["SECONDARY_TRUE_MANIFESTATION"],
        counts["DUPLICATE_TOP_LEVEL_INCIDENT"] + counts["UNRELATED_FALSE_POSITIVE"],
        counts["MISSED_PRIMARY"],
    )
    relation_precision, relation_recall = _pr(relation_tp, relation_fp, relation_fn)

    total_observations = sum(counts.values())
    return {
        "counts": counts,
        "total_observations": total_observations,
        "primary_precision": primary_precision,
        "primary_recall": primary_recall,
        "secondary_precision": secondary_precision,
        "secondary_recall": secondary_recall,
        "duplicate_top_level_incident_rate": (
            counts["DUPLICATE_TOP_LEVEL_INCIDENT"] / total_observations
            if total_observations else None
        ),
        "duplicate_top_level_incidents": counts["DUPLICATE_TOP_LEVEL_INCIDENT"],
        "relation_precision": relation_precision,
        "relation_recall": relation_recall,
        "primary_runs": primary_runs,
        "primary_runs_found": primary_found,
        "misclassification_rate": (
            (counts["UNRELATED_FALSE_POSITIVE"] + counts["DUPLICATE_TOP_LEVEL_INCIDENT"])
            / total_observations if total_observations else None
        ),
        "note": (
            "A SECONDARY_TRUE_MANIFESTATION is a correct observation of a real "
            "consequence. A DUPLICATE_TOP_LEVEL_INCIDENT observes the same thing but "
            "claims it as an independent root cause, and is an error. They are "
            "deliberately different classes."
        ),
    }
