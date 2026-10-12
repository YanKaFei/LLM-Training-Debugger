"""D1: evaluation semantics.

Phase 1.1 reported **negative detection delays** (``grad.robust_zscore`` at
−32 steps). A negative delay is not a fast detector. It is a category error: the
metric was computed as ``first_fire - expected_effect_start``, so a detector that
fired *before* the fault was injected had its pre-injection false positive
reported as an early detection. One number was carrying four distinct concepts.

This module separates them. Four moments are not the same thing:

======================  ==================================================
moment                  meaning
======================  ==================================================
fault injected          ground truth: what the harness did, and when
symptom begins          ground truth: when the effect is first *expected*
acceptable window       ground truth: the grading region
detector fires          detector output: a step, with no notion of truth
======================  ==================================================

The evaluation layer knows the truth; the detector layer does not. That asymmetry
is the entire point, and it is why classification lives here rather than in a
detector.

Separated metrics (D1.2):

* ``pre_injection_false_positives`` -- fired before the effect could exist
* ``detection_delay`` -- from the first **true positive** only
* ``post_fault_duplicates`` -- further fires after the first true positive
* ``missed`` -- no fire in the acceptable or recovery window
* ``late`` -- fired, but only after the acceptable window

A detector that fires at step 43 for a fault injected at step 80 is not early. It
is wrong, and it is counted as a false positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "DetectionWindow",
    "DetectionOutcome",
    "ScenarioEvaluation",
    "classify_detection",
    "evaluate_scenario",
    "aggregate",
    "OUTCOME_KINDS",
]

#: Every classification a single fire can receive.
OUTCOME_KINDS = (
    "true_positive",
    "pre_injection_false_positive",
    "late_detection",
    "post_fault_duplicate",
    "post_fault_false_positive",
)


@dataclass(frozen=True)
class DetectionWindow:
    """Ground truth for when a fault could possibly be observed.

    ``acceptable_detection_window`` is the grading region. It starts at
    ``expected_effect_start`` -- never earlier -- so a fire before the injection
    cannot be graded as a detection however close it lands.
    """

    injection_step: Optional[int]
    expected_effect_start: Optional[int]
    acceptable_detection_window: Optional[Tuple[int, int]]
    recovery_window: Optional[Tuple[int, int]] = None
    pre_fault_window: Optional[Tuple[int, int]] = None
    total_steps: Optional[int] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "injection_step": self.injection_step,
            "expected_effect_start": self.expected_effect_start,
            "acceptable_detection_window": list(self.acceptable_detection_window or []),
            "recovery_window": list(self.recovery_window or []),
            "pre_fault_window": list(self.pre_fault_window or []),
            "total_steps": self.total_steps,
            "note": self.note,
        }

    @classmethod
    def for_injection(
        cls,
        injection_step: int,
        total_steps: int,
        *,
        effect_lag: int = 0,
        tolerance: int = 10,
        recovery: int = 40,
    ) -> "DetectionWindow":
        """Build a window for an injected fault.

        ``effect_lag`` allows a fault whose *symptom* is expected to appear a few
        steps after the injection (a gradient explosion that only moves the loss
        on the following step). ``tolerance`` is the grading region after that.
        """
        effect_start = injection_step + effect_lag
        acceptable_end = min(total_steps, effect_start + tolerance)
        return cls(
            injection_step=injection_step,
            expected_effect_start=effect_start,
            acceptable_detection_window=(effect_start, acceptable_end),
            recovery_window=(acceptable_end + 1, min(total_steps, acceptable_end + recovery)),
            pre_fault_window=(1, max(0, injection_step - 1)),
            total_steps=total_steps,
        )

    @classmethod
    def healthy(cls, total_steps: int) -> "DetectionWindow":
        """A run with no fault. Every fire is a false positive."""
        return cls(
            injection_step=None,
            expected_effect_start=None,
            acceptable_detection_window=None,
            recovery_window=None,
            pre_fault_window=(1, total_steps),
            total_steps=total_steps,
            note="healthy control: any fire is a false positive",
        )


@dataclass
class DetectionOutcome:
    detector: str
    step: int
    kind: str
    delay: Optional[int] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "step": self.step,
            "kind": self.kind,
            "delay": self.delay,
            "reason": self.reason,
        }


def classify_detection(step: int, window: DetectionWindow) -> Tuple[str, Optional[int], str]:
    """Classify one fire. Returns ``(kind, delay, reason)``.

    The ordering of the tests is the whole design:

    1. A healthy control has no fault, so every fire is a false positive.
    2. A fire before ``expected_effect_start`` is a **false positive**, regardless
       of how close it is to the injection. It cannot be a detection, because the
       effect did not exist yet. This is the case Phase 1.1 scored as a negative
       delay.
    3. Inside the acceptable window it is a true positive, and only here is a
       delay computed.
    4. Inside the recovery window it is a late detection (counted as a true
       positive for recall, with the delay reported as ``None`` so it cannot
       flatter the delay statistic).
    5. After that, a false positive.
    """
    if window.expected_effect_start is None:
        return "post_fault_false_positive", None, "healthy control: any fire is a false positive"

    if step < window.expected_effect_start:
        return (
            "pre_injection_false_positive",
            None,
            "fired at step {} but the effect does not begin until step {}".format(
                step, window.expected_effect_start
            ),
        )

    acceptable = window.acceptable_detection_window
    if acceptable and acceptable[0] <= step <= acceptable[1]:
        return (
            "true_positive",
            step - window.expected_effect_start,
            "inside the acceptable window [{}, {}]".format(*acceptable),
        )

    recovery = window.recovery_window
    if recovery and recovery[0] <= step <= recovery[1]:
        return (
            "late_detection",
            None,
            "detected in the recovery window [{}, {}]; too late to carry a delay".format(*recovery),
        )

    return (
        "post_fault_false_positive",
        None,
        "fired at step {}, after the acceptable and recovery windows".format(step),
    )


@dataclass
class ScenarioEvaluation:
    """Separated metrics for one (scenario, detector) pair."""

    scenario: str
    detector: str
    injected: bool = True
    outcomes: List[DetectionOutcome] = field(default_factory=list)

    true_positive: bool = False
    detection_delay: Optional[int] = None
    pre_injection_false_positives: int = 0
    post_fault_duplicates: int = 0
    post_fault_false_positives: int = 0
    late_detections: int = 0
    missed: bool = False
    run_error: Optional[str] = None

    @property
    def total_fires(self) -> int:
        return len(self.outcomes)

    @property
    def false_positives(self) -> int:
        """Fires that assert an anomaly where none was, or where none existed yet."""
        if not self.injected:
            return self.total_fires
        return self.pre_injection_false_positives + self.post_fault_false_positives

    @property
    def fired_before_injection(self) -> bool:
        return self.pre_injection_false_positives > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "detector": self.detector,
            "injected": self.injected,
            "true_positive": self.true_positive,
            "detection_delay": self.detection_delay,
            "pre_injection_false_positives": self.pre_injection_false_positives,
            "post_fault_duplicates": self.post_fault_duplicates,
            "post_fault_false_positives": self.post_fault_false_positives,
            "late_detections": self.late_detections,
            "missed": self.missed,
            "total_fires": self.total_fires,
            "run_error": self.run_error,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def evaluate_scenario(
    scenario: str,
    detector: str,
    fires: Sequence[int],
    window: DetectionWindow,
    *,
    run_error: Optional[str] = None,
) -> ScenarioEvaluation:
    """Evaluate one detector's fires against one scenario's ground truth.

    ``detection_delay`` comes from the **first true positive** only. A later
    duplicate never influences it, and a pre-injection false positive never
    produces a negative one.
    """
    evaluation = ScenarioEvaluation(scenario=scenario, detector=detector, run_error=run_error)
    if run_error:
        evaluation.missed = True
        return evaluation
    evaluation.injected = window.expected_effect_start is not None

    for step in sorted(fires):
        kind, delay, reason = classify_detection(step, window)
        evaluation.outcomes.append(
            DetectionOutcome(detector=detector, step=step, kind=kind, delay=delay, reason=reason)
        )
        if kind == "true_positive":
            if not evaluation.true_positive:
                evaluation.true_positive = True
                evaluation.detection_delay = delay
            else:
                evaluation.post_fault_duplicates += 1
        elif kind == "pre_injection_false_positive":
            evaluation.pre_injection_false_positives += 1
        elif kind == "late_detection":
            evaluation.late_detections += 1
            if not evaluation.true_positive:
                # A late detection still counts for recall, but carries no delay.
                evaluation.true_positive = True
        elif kind == "post_fault_duplicate":
            evaluation.post_fault_duplicates += 1
        else:
            evaluation.post_fault_false_positives += 1

    evaluation.missed = not evaluation.true_positive
    return evaluation


def aggregate(
    evaluations: Sequence[ScenarioEvaluation], *, total_steps: int
) -> Dict[str, Any]:
    """Aggregate per-scenario evaluations into detector-level metrics.

    Reports the separated quantities side by side. Deliberately gives no single
    headline number: collapsing pre-injection false positives, delay, duplicates
    and misses into one figure is what produced the negative-delay defect.
    """
    usable = [e for e in evaluations if not e.run_error]
    injected = [e for e in usable if e.injected]
    healthy = [e for e in usable if not e.injected]

    tp = sum(1 for e in injected if e.true_positive)
    fn = sum(1 for e in injected if e.missed)
    pre_fp = sum(e.pre_injection_false_positives for e in injected)
    post_fp = sum(e.post_fault_false_positives for e in injected)
    healthy_fp = sum(e.total_fires for e in healthy)
    duplicates = sum(e.post_fault_duplicates for e in usable)
    late = sum(e.late_detections for e in usable)
    delays = [e.detection_delay for e in injected if e.detection_delay is not None]

    steps_observed = len(usable) * total_steps
    fires = sum(e.total_fires for e in usable)
    precision = (tp / (tp + pre_fp + post_fp + healthy_fp)) if (tp + pre_fp + post_fp + healthy_fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None
    return {
        "scenarios": len(usable),
        "excluded_run_errors": len(evaluations) - len(usable),
        "injected_scenarios": len(injected),
        "healthy_scenarios": len(healthy),
        "true_positives": tp,
        "false_negatives": fn,
        "pre_injection_false_positives": pre_fp,
        "post_fault_false_positives": post_fp,
        "healthy_false_positives": healthy_fp,
        "post_fault_duplicates": duplicates,
        "late_detections": late,
        "total_fires": fires,
        "recall": recall,
        "precision": precision,
        "f1": (2 * precision * recall / (precision + recall)) if (precision and recall) else None,
        "median_detection_delay": sorted(delays)[len(delays) // 2] if delays else None,
        "delays": delays,
        "false_positives_per_1k_steps": (pre_fp + post_fp + healthy_fp) / steps_observed * 1000.0
        if steps_observed else None,
        "runs_with_a_pre_injection_fire": sum(1 for e in injected if e.fired_before_injection),
        "pre_injection_incident_rate": (
            sum(1 for e in injected if e.fired_before_injection) / len(injected)
            if injected else None
        ),
        "duplicate_rate": duplicates / len(usable) if usable else None,
    }
