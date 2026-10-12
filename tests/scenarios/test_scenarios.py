"""W5 gate: the A0-A4 scenario corpus.

Asserts the structured expectations from ``corpus.py`` against real detector
output. Deliberately not a prose snapshot (brief section 12): every assertion is
over a structured field -- detector id, onset step, incident type, multiplicity.
"""

from __future__ import annotations

import pytest

from scenarios.corpus import SCENARIOS, get_scenario
from scenarios.harness import run_scenario

from training_debugger.schema import EvidenceLevel


@pytest.fixture(scope="module")
def outcomes():
    """Run the corpus once; the scenarios are deterministic so this is safe."""
    return {scenario.name: run_scenario(scenario) for scenario in SCENARIOS}


# ---------------------------------------------------------------------------
# A0 -- the false-positive test
# ---------------------------------------------------------------------------


def test_A0_healthy_run_produces_no_anomalies(outcomes):
    """The single most important assertion in the corpus.

    A detector suite reaches 100% recall by firing always. This is the test that
    forbids it.
    """
    outcome = outcomes["A0_healthy"]
    assert outcome.fired_detectors == [], (
        "healthy run fired %s with onsets %s" % (outcome.fired_detectors, outcome.onsets)
    )
    assert outcome.result.anomalies == []


def test_A0_healthy_run_records_its_denominator(outcomes):
    """Zero findings must still report how many series were examined."""
    outcome = outcomes["A0_healthy"]
    assert outcome.result.series_examined, "the analysis must record what it looked at"
    # 7 since v0.1.8: `activation.health` joined the default set. It reports only
    # threshold-free activation findings and emits nothing unless the opt-in
    # ActivationProbe is attached, so registering it costs nothing at run time.
    assert outcome.result.detectors_run == 7


# ---------------------------------------------------------------------------
# Per-scenario expectations
# ---------------------------------------------------------------------------


REQUIRED = [
    "A1_loss_spike",
    "A1b_loss_spike_without_gradient_precursor",
    "A2_gradient_explosion",
    "A3_nan",
    "A3b_infinite_gradient",
    "A4_throughput_regression",
]


@pytest.mark.parametrize("name", REQUIRED)
def test_required_detectors_fire(outcomes, name):
    scenario = get_scenario(name)
    outcome = outcomes[name]
    for detector in scenario.expected_detectors:
        assert detector in outcome.fired_detectors, (
            "%s: expected %s to fire; fired %s" % (name, detector, outcome.fired_detectors)
        )


@pytest.mark.parametrize("name", REQUIRED)
def test_forbidden_detectors_do_not_fire(outcomes, name):
    scenario = get_scenario(name)
    outcome = outcomes[name]
    for detector in scenario.forbidden_detectors:
        assert detector not in outcome.fired_detectors, (
            "%s: %s must not fire but did (onsets %s)" % (name, detector, outcome.onsets)
        )


@pytest.mark.parametrize("name", REQUIRED)
def test_onset_lands_in_the_expected_window(outcomes, name):
    scenario = get_scenario(name)
    outcome = outcomes[name]
    low, high = scenario.expected_onset_window
    primary = scenario.expected_detectors[0]
    onset = outcome.onsets[primary]
    assert low <= onset <= high, (
        "%s: %s onset at step %d, expected within [%d, %d]"
        % (name, primary, onset, low, high)
    )


# ---------------------------------------------------------------------------
# Scenario-specific semantics
# ---------------------------------------------------------------------------


def test_A1_gradient_anomaly_precedes_the_loss_spike(outcomes):
    """The headline behaviour: the cause is detected before the symptom.

    The engine's whole value proposition is naming an event the user did not
    notice, earlier than the one they did. If the gradient onset is not earlier
    than the loss onset, that claim is false.
    """
    outcome = outcomes["A1_loss_spike"]
    gradient_onset = outcome.onsets["grad.robust_zscore"]
    loss_onset = outcome.onsets["loss.robust_zscore"]
    assert gradient_onset < loss_onset, (
        "gradient onset %d must precede loss onset %d" % (gradient_onset, loss_onset)
    )
    assert loss_onset - gradient_onset >= 5, (
        "the injected gap is 10 steps; a 1-step gap suggests the onset is being "
        "reported at detection time rather than at first-abnormality time"
    )


def test_A1b_does_not_invent_a_gradient_cause(outcomes):
    """No gradient precursor was injected, so none may be reported."""
    outcome = outcomes["A1b_loss_spike_without_gradient_precursor"]
    assert "loss.robust_zscore" in outcome.fired_detectors
    assert "grad.robust_zscore" not in outcome.fired_detectors


def test_A2_attributes_the_explosion_to_a_layer(outcomes):
    outcome = outcomes["A2_gradient_explosion"]
    assert outcome.result.multiplicity.affected_layers, (
        "a per-layer explosion must be attributed to at least one layer"
    )
    summary = outcome.result.multiplicity.summary()
    assert "layers" in summary and "/" in summary, (
        "multiplicity must state an affected/population ratio, got %r" % summary
    )


def test_A3_nan_is_observed_not_inferred(outcomes):
    """NaN is measured, not inferred: confidence 1.0, level OBSERVED."""
    outcome = outcomes["A3_nan"]
    finiteness = [a for a in outcome.result.anomalies if str(a.detector) == "numerics.finiteness"]
    assert finiteness, "the NaN must be reported"
    assert all(a.level is EvidenceLevel.OBSERVED for a in finiteness)
    assert all(a.confidence == 1.0 for a in finiteness)
    assert finiteness[0].first_abnormal_step == 180
    assert finiteness[0].metadata["kind"] == "nan"


def test_A3b_a_non_finite_gradient_element_is_observed_as_nan(outcomes):
    """Corrected in v0.1.5.

    The real `global_gradient_norm` returns **nan** when any element is non-finite and
    leaves `inf` for the case where the float32 reduction overflowed with every element
    finite. The old harness emitted `inf` with `grad.nonfinite_count = 1`, and this test
    asserted that `inf` meant a non-finite gradient was observed. It does not.
    """
    outcome = outcomes["A3b_infinite_gradient"]
    finiteness = [a for a in outcome.result.anomalies if str(a.detector) == "numerics.finiteness"]
    assert finiteness
    kinds = {a.metadata["kind"] for a in finiteness}
    assert "nan" in kinds, "a non-finite gradient element must be reported as nan"
    assert "norm_reduction_overflow" not in kinds, (
        "a genuinely non-finite gradient must not be classified as a reduction artifact"
    )


def test_a_norm_reduction_overflow_is_distinguished_from_a_non_finite_tensor():
    """W0: the two cases must not be conflated.

    `inf` on a `*.global_norm` metric means the reduction overflowed while every
    element was finite. It is a real finding and it is NOT a non-finite tensor.
    """
    from training_debugger.detectors import DetectorConfig, DetectorRunner

    series = {
        "grad.global_norm": [(step, float("inf") if step == 50 else 1.0)
                             for step in range(1, 80)],
    }
    result = DetectorRunner(DetectorConfig()).run(
        "r", series, rank=0, layer_series={}, module_paths={},
        population_layers=1, population_ranks=1,
        coverage={"step_count": 80, "step_min": 1, "step_max": 80},
    )
    finiteness = [a for a in result.anomalies if str(a.detector) == "numerics.finiteness"]
    assert finiteness, "an overflowing reduction must still be reported"
    assert finiteness[0].metadata["kind"] == "norm_reduction_overflow"

    from training_debugger.incidents.relations import is_nonfinite_observation
    assert is_nonfinite_observation(finiteness[0].to_dict()) is False, (
        "a reduction artifact must not count as a non-finite observation"
    )


def test_A4_reports_the_symptom_without_a_root_cause(outcomes):
    """Phase 1 must not attribute a throughput drop to GPU/storage/network."""
    outcome = outcomes["A4_throughput_regression"]
    throughput = [a for a in outcome.result.anomalies if str(a.detector) == "throughput.regression"]
    assert throughput, "the throughput regression must be detected"
    for anomaly in throughput:
        assert anomaly.metadata.get("root_cause") is None, (
            "Phase 1 must not report a root cause for a throughput drop"
        )
        assert anomaly.metadata.get("scope") == "symptom_only"


def test_A4_onset_is_reported_at_the_start_of_the_run_not_at_detection(outcomes):
    """The onset must be the first breaching step, not the step at which the run
    of consecutive breaches became long enough to be conclusive."""
    outcome = outcomes["A4_throughput_regression"]
    throughput = [a for a in outcome.result.anomalies if str(a.detector) == "throughput.regression"]
    earliest = min(a.first_abnormal_step for a in throughput)
    assert earliest <= 165, (
        "onset %d is later than the injection at 160 plus a small tolerance; the "
        "detector is reporting detection time rather than onset" % earliest
    )


# ---------------------------------------------------------------------------
# Cross-cutting properties
# ---------------------------------------------------------------------------


def test_counter_evidence_is_present_on_the_healthy_run(outcomes):
    """A healthy run must be able to say what was tested and found stable."""
    outcome = outcomes["A0_healthy"]
    assert outcome.result.series_examined
    assert outcome.result.multiplicity.event_count == 0


def test_every_anomaly_carries_its_audit_fields(outcomes):
    for name, outcome in outcomes.items():
        for anomaly in outcome.result.anomalies:
            assert anomaly.reason, "%s: anomaly has no reason" % name
            assert anomaly.detector_version
            assert anomaly.window[0] <= anomaly.window[1]
            assert 0.0 <= anomaly.confidence <= 1.0
            assert anomaly.sample_count >= 0
            if anomaly.level is EvidenceLevel.INFERRED:
                assert anomaly.metadata.get("method"), (
                    "%s: inferred anomaly must name its method" % name
                )


def test_detection_is_deterministic():
    """Same scenario twice, byte-identical structured output."""
    scenario = get_scenario("A1_loss_spike")
    first = run_scenario(scenario)
    second = run_scenario(scenario)
    assert first.fired_detectors == second.fired_detectors
    assert first.onsets == second.onsets
    assert [a.to_dict() for a in first.result.anomalies] == [
        a.to_dict() for a in second.result.anomalies
    ]


def test_suppression_never_hides_an_observed_finding(outcomes):
    """If suppression is active, the surviving evidence must include the
    OBSERVED finding, never only the inferred one it superseded."""
    for name, outcome in outcomes.items():
        if not outcome.result.suppressed:
            continue
        assert outcome.result.observed(), (
            "%s suppressed inferred findings without retaining the observed one" % name
        )
