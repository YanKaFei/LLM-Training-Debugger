"""K2/K3: consequence-aware scoring and earliest-evidence evaluation."""

from __future__ import annotations

import pytest

from training_debugger.evaluation import (
    OBSERVATION_CLASSES,
    GroundTruthConsequence,
    classify_observation,
    evaluate_consequences,
    evaluate_earliest_evidence,
)


def episode(anomaly_type, onset, *, confirmed=True, eid="e1"):
    return {
        "episode_id": eid,
        "anomaly_type": anomaly_type,
        "state": "CLOSED",
        "outcome": "CONFIRMED_ANOMALY" if confirmed else "NOT_ANOMALY",
        "candidate_onset_step": onset,
        "confirmed_at_step": onset + 2 if confirmed else None,
    }


def scenario(**overrides):
    payload = {
        "run_id": "r1",
        "injected": True,
        "fault_type": "GRADIENT_EXPLOSION",
        "injection_step": 100,
        "acceptable_window": [100, 110],
        "episodes": [],
        "verdicts": [],
        "relations": [],
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# K2 consequence scoring
# ---------------------------------------------------------------------------


def test_there_are_six_observation_classes():
    assert len(OBSERVATION_CLASSES) == 6
    assert "SECONDARY_TRUE_MANIFESTATION" in OBSERVATION_CLASSES
    assert "DUPLICATE_TOP_LEVEL_INCIDENT" in OBSERVATION_CLASSES


def test_primary_fault_in_window_is_a_true_positive():
    result = evaluate_consequences([
        scenario(
            episodes=[episode("GRADIENT_EXPLOSION", 105)],
            verdicts=[{"anomaly_type": "GRADIENT_EXPLOSION", "verdict": "CONFIRMED_INCIDENT"}],
        )
    ])
    assert result["counts"]["PRIMARY_TRUE_POSITIVE"] == 1
    assert result["primary_precision"] == 1.0


def test_secondary_manifestation_is_not_a_false_positive():
    """The defect Phase 1.6 identified: a real consequence scored as a detector FP."""
    result = evaluate_consequences([
        scenario(
            episodes=[episode("GRADIENT_EXPLOSION", 105), episode("NAN_DETECTED", 140, eid="e2")],
            verdicts=[{"anomaly_type": "GRADIENT_EXPLOSION", "verdict": "CONFIRMED_INCIDENT"}],
            relations=[{"target_anomaly_type": "NAN_DETECTED", "gap_steps": 38}],
        )
    ])
    assert result["counts"]["SECONDARY_TRUE_MANIFESTATION"] == 1
    assert result["counts"]["UNRELATED_FALSE_POSITIVE"] == 0
    assert result["duplicate_top_level_incidents"] == 0
    assert result["relation_precision"] == 1.0


def test_a_second_root_incident_for_a_consequence_is_an_error():
    result = evaluate_consequences([
        scenario(
            episodes=[episode("GRADIENT_EXPLOSION", 105), episode("NAN_DETECTED", 140, eid="e2")],
            verdicts=[
                {"anomaly_type": "GRADIENT_EXPLOSION", "verdict": "CONFIRMED_INCIDENT"},
                {"anomaly_type": "NAN_DETECTED", "verdict": "CONFIRMED_INCIDENT"},
            ],
        )
    ])
    assert result["counts"]["DUPLICATE_TOP_LEVEL_INCIDENT"] == 1
    assert result["duplicate_top_level_incident_rate"] == pytest.approx(0.5)


def test_secondary_outside_the_consequence_window_is_unrelated():
    result = evaluate_consequences([
        scenario(
            episodes=[episode("GRADIENT_EXPLOSION", 105), episode("NAN_DETECTED", 400, eid="e2")],
            verdicts=[{"anomaly_type": "NAN_DETECTED", "verdict": "CONFIRMED_INCIDENT"}],
        )
    ])
    assert result["counts"]["UNRELATED_FALSE_POSITIVE"] == 1


def test_wrong_family_in_the_window_is_unrelated():
    result = evaluate_consequences([
        scenario(
            episodes=[episode("THROUGHPUT_REGRESSION", 105)],
            verdicts=[{"anomaly_type": "THROUGHPUT_REGRESSION", "verdict": "CONFIRMED_INCIDENT"}],
        )
    ])
    assert result["counts"]["UNRELATED_FALSE_POSITIVE"] == 1


def test_healthy_run_observation_is_never_a_true_positive():
    result = evaluate_consequences([
        scenario(injected=False, fault_type="H0_stable_healthy", injection_step=None,
                 acceptable_window=None, episodes=[episode("GRADIENT_EXPLOSION", 50)])
    ])
    assert result["counts"]["PRIMARY_TRUE_POSITIVE"] == 0
    assert result["counts"]["UNRELATED_FALSE_POSITIVE"] == 1


def test_ground_truth_schema_round_trip():
    truth = GroundTruthConsequence.from_fault(
        {"fault_type": "GRADIENT_EXPLOSION", "injection_step": 100,
         "allowed_secondary_manifestations": ["NAN_DETECTED"]}
    )
    assert truth.primary_fault_type == "GRADIENT_EXPLOSION"
    assert truth.secondary_window == (100, 200)
    assert truth.allowed_secondary_manifestations == ("NAN_DETECTED",)
    assert truth.to_dict()["secondary_window"] == [100, 200]


# ---------------------------------------------------------------------------
# K3 earliest evidence
# ---------------------------------------------------------------------------


def test_earliest_evidence_true_when_onset_is_in_the_window():
    result = evaluate_earliest_evidence([
        scenario(run_id="r1", episodes=[episode("GRADIENT_EXPLOSION", 104)])
    ])
    assert result["precision"] == 1.0
    assert result["coverage"] == 1.0
    assert result["absolute_onset_error_median"] == 4
    assert result["signed_onset_error_median"] == 4


def test_earliest_evidence_false_when_onset_precedes_the_fault():
    """A healthy fluctuation at step 40 is a false claim, not an early detection."""
    result = evaluate_earliest_evidence([
        scenario(run_id="r1", episodes=[episode("GRADIENT_EXPLOSION", 40)])
    ])
    assert result["precision"] == 0.0
    assert result["false"] == 1
    assert result["pre_injection_confirmed_incidents"] == 1


def test_earliest_evidence_is_unknown_for_the_wrong_family_not_false():
    """A confirmed episode of another family is not evidence about *this* fault.

    The distinction matters: it is neither a true claim nor a false one, so it is
    UNKNOWN. Counting it as FALSE would penalise a system for correctly declining to
    make a claim it has no evidence for, and counting it as TRUE would be worse.
    """
    result = evaluate_earliest_evidence([
        scenario(run_id="r1", episodes=[episode("LOSS_SPIKE", 105)])
    ])
    assert result["unknown"] == 1
    assert result["false"] == 0
    assert result["precision"] is None
    assert result["coverage"] == 0.0


def test_unknown_is_allowed_and_is_not_a_false_positive():
    result = evaluate_earliest_evidence([
        scenario(
            run_id="r1",
            episodes=[{
                "episode_id": "e1", "anomaly_type": "GRADIENT_EXPLOSION",
                "state": "CLOSED", "outcome": "INSUFFICIENT_EVIDENCE",
                "candidate_onset_step": 104, "confirmed_at_step": None,
            }],
        )
    ])
    assert result["unknown"] == 1
    assert result["false"] == 0
    assert result["precision"] is None
    assert result["coverage"] == 0.0


def test_coverage_guards_against_answering_unknown_to_everything():
    """Precision alone would reward all-UNKNOWN; coverage is the guard."""
    all_unknown = evaluate_earliest_evidence([
        scenario(run_id="r%d" % i, episodes=[]) for i in range(5)
    ])
    assert all_unknown["precision"] is None
    assert all_unknown["coverage"] == 0.0
    assert all_unknown["unknown_rate"] == 1.0


def test_confirmation_delay_is_reported_separately_from_onset_error():
    result = evaluate_earliest_evidence([
        scenario(run_id="r1", episodes=[episode("GRADIENT_EXPLOSION", 104)])
    ])
    assert result["absolute_onset_error_median"] == 4
    assert result["confirmation_delay_median"] == 2


def test_healthy_runs_are_excluded_from_the_earliest_metric():
    result = evaluate_earliest_evidence([
        scenario(injected=False, injection_step=None, acceptable_window=None,
                 episodes=[episode("GRADIENT_EXPLOSION", 50)])
    ])
    assert result["runs"] == 0
