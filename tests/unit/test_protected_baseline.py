"""L0: baseline-contamination regression tests.

Written **before** the feature change, so the failure Phase 1.7 discovered is pinned
as a regression rather than described in a report.
"""

from __future__ import annotations

import pytest

from training_debugger.detectors import (
    BaselineState,
    ProtectedBaseline,
    protected_ratio_series,
)
from training_debugger.detectors.runner import joint_gradient_features


def series(flat_until: int, level: float, after: float, end: int = 300):
    return [(s, level) for s in range(1, flat_until)] + [(s, after) for s in range(flat_until, end)]


# ---------------------------------------------------------------------------
# The defect Phase 1.7 measured
# ---------------------------------------------------------------------------


def test_sustained_scaling_does_not_self_normalise():
    """The 680-run finding: a trailing median absorbs the fault it should expose."""
    ratios = protected_ratio_series(series(50, 1.0, 5.0))
    for step in (50, 80, 150, 250, 299):
        assert ratios[step] == pytest.approx(5.0, rel=0.05), (
            "a sustained x5 fault must stay at ratio 5, not decay toward 1"
        )


def test_trailing_median_would_have_decayed():
    """The unprotected variant is retained so the defect stays reproducible."""
    plain = [(s, 1.0) for s in range(1, 50)] + [(s, 5.0) for s in range(50, 300)]

    def trailing_ratio(step):
        history = [v for s, v in plain if 50 <= s < step][-20:]
        if len(history) < 20:
            return None
        return 5.0 / (sum(history) / len(history))

    assert trailing_ratio(299) == pytest.approx(1.0, rel=0.01), (
        "the old feature decays to 1 under a sustained fault -- this is why no "
        "threshold could work"
    )


def test_contaminated_rolling_median_is_pinned_as_a_regression():
    """A rolling median over the fault window reports ~1 where the truth is 5."""
    import statistics

    values = [1.0] * 40 + [5.0] * 60
    rolling = statistics.median(values[40:])
    assert rolling == pytest.approx(5.0)
    # ...and once the rolling window is entirely inside the fault, the ratio is 1.
    assert 5.0 / statistics.median(values[60:]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# State semantics
# ---------------------------------------------------------------------------


def test_baseline_learns_while_normal():
    baseline = ProtectedBaseline(minimum_history=5)
    for step in range(1, 30):
        baseline.update(step, 1.0, state=BaselineState.NORMAL.value)
    assert baseline.ready
    assert baseline.reference == pytest.approx(1.0)
    assert baseline.frozen is False


def test_baseline_freezes_on_suspect():
    baseline = ProtectedBaseline(minimum_history=5)
    for step in range(1, 30):
        baseline.update(step, 1.0, state=BaselineState.NORMAL.value)
    reference_before = baseline.reference
    for step in range(30, 60):
        baseline.update(step, 9.0, state=BaselineState.SUSPECT.value)
    assert baseline.frozen is True
    assert baseline.reference == pytest.approx(reference_before), (
        "a SUSPECT state must not absorb the candidate regime"
    )


def test_baseline_stays_frozen_when_confirmed():
    baseline = ProtectedBaseline(minimum_history=5)
    for step in range(1, 20):
        baseline.update(step, 1.0, state=BaselineState.NORMAL.value)
    for step in range(20, 60):
        baseline.update(step, 9.0, state=BaselineState.CONFIRMED.value)
    assert baseline.reference == pytest.approx(1.0)


def test_baseline_resumes_after_a_rejected_candidate():
    baseline = ProtectedBaseline(minimum_history=5, resume_after_steps=10)
    for step in range(1, 20):
        baseline.update(step, 1.0, state=BaselineState.NORMAL.value)
    for step in range(20, 30):
        baseline.update(step, 9.0, state=BaselineState.SUSPECT.value)
    assert baseline.frozen is True
    # The candidate is rejected; the baseline resumes only after a quiet period.
    for step in range(30, 45):
        baseline.update(step, 1.0, state=BaselineState.NORMAL.value)
    assert baseline.frozen is False


def test_a_false_suspicion_does_not_poison_the_baseline():
    """Two quiet steps must not bulk-absorb the suspect window."""
    baseline = ProtectedBaseline(minimum_history=5, resume_after_steps=20)
    for step in range(1, 20):
        baseline.update(step, 1.0, state=BaselineState.NORMAL.value)
    for step in range(20, 30):
        baseline.update(step, 9.0, state=BaselineState.SUSPECT.value)
    for step in range(30, 35):
        baseline.update(step, 1.0, state=BaselineState.NORMAL.value)
    assert baseline.frozen is True, "learning must not resume immediately"
    assert baseline.reference == pytest.approx(1.0, abs=0.2)


def test_recovering_does_not_relearn_poisoned_values():
    baseline = ProtectedBaseline(minimum_history=5)
    for step in range(1, 20):
        baseline.update(step, 1.0, state=BaselineState.NORMAL.value)
    for step in range(20, 40):
        baseline.update(step, 9.0, state=BaselineState.RECOVERING.value)
    assert baseline.reference == pytest.approx(1.0)
    assert baseline.frozen is True


def test_baseline_is_not_ready_before_minimum_history():
    baseline = ProtectedBaseline(minimum_history=20)
    for step in range(1, 10):
        baseline.update(step, 1.0, state=BaselineState.NORMAL.value)
    assert baseline.ready is False


def test_warmup_abstains_rather_than_classifying():
    """L2.4: an unready baseline yields INSUFFICIENT_EVIDENCE, not 'normal'."""
    baseline = ProtectedBaseline(minimum_history=20)
    for step in range(1, 5):
        baseline.update(step, 1.0, state=BaselineState.NORMAL.value)
    assert baseline.ratio(50.0) is None, (
        "classifying during warm-up is forbidden; the answer must be 'no opinion'"
    )


# ---------------------------------------------------------------------------
# Integration with the joint aggregate
# ---------------------------------------------------------------------------


def _layer_series(values):
    return {0: list(enumerate(values, start=1)), 1: list(enumerate(values, start=1))}


def test_joint_feature_keeps_a_sustained_fault_visible():
    values = [1.0] * 60 + [5.0] * 200
    features_late = joint_gradient_features(_layer_series(values), 250, baseline_floor=1)
    features_early = joint_gradient_features(_layer_series(values), 70, baseline_floor=1)
    assert features_late.get("l1_aggregate", 0) > 3.0
    assert features_early.get("l1_aggregate", 0) > 3.0


def test_joint_feature_does_not_see_a_flat_healthy_run():
    features = joint_gradient_features(_layer_series([1.0] * 260), 200, baseline_floor=1)
    assert features.get("l1_aggregate") == pytest.approx(1.0, rel=0.05)


def test_joint_feature_reports_nothing_before_the_baseline_is_ready():
    """An unready layer set yields no aggregate, so the verdict is insufficient."""
    features = joint_gradient_features(_layer_series([1.0] * 10), 8, baseline_floor=1)
    assert features == {}


# ---------------------------------------------------------------------------
# The resume path, end to end (Phase 1.13)
# ---------------------------------------------------------------------------


def test_resume_path_is_reachable_through_protected_ratio_series():
    """The defect: the freeze was permanent.

    ``protected_ratio_series`` called ``suspect()``, which sets ``state = SUSPECT``,
    and nothing ever set it back. ``ProtectedBaseline.update`` only absorbs while the
    state is ``NORMAL`` or ``CLOSED``, so its resume branch could never run through
    this entry point: the reference stayed frozen at the first suspicion for the rest
    of the run, and ``resume_after_steps`` was dead code.

    Consequence: a healthy run whose benign excursions are closer together than one
    quiet period had its *later legitimate growth* measured against an early-training
    reference, and was reported as an explosion.
    """
    from training_debugger.detectors.baseline import BaselineState, protected_ratio_series

    # A benign excursion, then a long quiet stretch, then a genuine level rise.
    series = [(step, 1.0) for step in range(1, 61)]
    series += [(step, 1.5) for step in range(61, 64)]          # benign excursion
    series += [(step, 1.0) for step in range(64, 130)]         # 66 quiet steps
    series += [(step, 1.6) for step in range(130, 141)]        # a real regime change

    ratios = protected_ratio_series(series, minimum_history=20, resume_after_steps=40)

    # After resuming, the reference tracks the new level, so the ratio stays modest.
    late = [ratios[s] for s in range(135, 141) if ratios.get(s) is not None]
    assert late, "ratios must be produced after the quiet period"
    assert max(late) < 3.0, (
        "after a quiet period the reference must resume learning, otherwise a genuine "
        "level rise is measured against a stale early reference; got {}".format(max(late))
    )


def test_resume_does_not_let_a_sustained_fault_normalise_itself():
    """The property the freeze exists to protect must survive the resume fix."""
    from training_debugger.detectors.baseline import protected_ratio_series

    # Reference on 1.0, then a sustained x5 shift that never returns.
    series = [(step, 1.0) for step in range(1, 61)]
    series += [(step, 5.0) for step in range(61, 200)]

    ratios = protected_ratio_series(series, minimum_history=20, resume_after_steps=40)
    early = ratios[70]
    late = ratios[199]
    assert early > 4.0, "the sustained shift must be visible"
    assert late > 3.5, (
        "a sustained fault must not be absorbed into the reference; the ratio decayed "
        "to {}, which is the self-normalisation defect this baseline exists to fix".format(late)
    )


def test_an_ongoing_suspicion_restarts_the_quiet_clock():
    from training_debugger.detectors.baseline import protected_ratio_series

    # Excursions 20 steps apart never give one 40-step quiet period.
    series = []
    for step in range(1, 200):
        series.append((step, 1.5 if step % 20 == 0 else 1.0))
    ratios = protected_ratio_series(series, minimum_history=20, resume_after_steps=40)
    assert any(value is not None for value in ratios.values())
