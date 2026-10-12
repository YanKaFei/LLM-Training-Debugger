"""WS1/WS4: the level-separated counting model and fail-closed scoring.

These pin the two defects v0.1.1 shipped:

* **42 runs produced 97 "records".** The unit was the episode, and one run can hold
  many episodes, so an episode count was being read as a run count.
* **`primary_recall` was always 1.0** whenever anything was detected, because its
  false-negative term was never incremented. The metric could not report a miss.
"""

from __future__ import annotations

import pytest

from training_debugger.evaluation import (
    SCORING_INVALID, evaluate_levels, validate_scoring_inputs,
)


def scenario(run_id, *, injected, primary=None, window=None, onset=None,
             episodes=(), verdicts=(), relations=(), injection_step=110):
    return {
        "run_id": run_id,
        "injected": injected,
        "injection_step": injection_step if injected else None,
        "acceptable_window": window,
        "ground_truth": {"primary_fault_type": primary,
                         "allowed_secondary_manifestations": []} if primary else None,
        "episodes": [{"anomaly_type": t, "candidate_onset_step": o} for t, o in episodes],
        "verdicts": [{"verdict": "CONFIRMED_INCIDENT", "anomaly_type": t,
                      "candidate_onset_step": o} for t, o in verdicts],
        "relations": list(relations),
    }


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def test_scoring_refuses_when_an_injected_run_lacks_its_window():
    """Without a window a correct detection is filed as a false positive."""
    scenarios = [scenario("r1", injected=True, primary="GRADIENT_EXPLOSION",
                          window=None, onset=110, episodes=[("GRADIENT_EXPLOSION", 110)])]
    validity = validate_scoring_inputs(scenarios)
    assert validity.valid is False
    assert any(i["issue"] == "MISSING_ACCEPTABLE_WINDOW" for i in validity.issues)

    result = evaluate_levels(scenarios)
    assert result["status"] == SCORING_INVALID
    assert result["levels"] == {}, "no precision may be emitted from invalid inputs"


def test_scoring_refuses_when_the_primary_fault_type_is_missing():
    scenarios = [scenario("r1", injected=True, primary=None, window=[110, 120])]
    assert validate_scoring_inputs(scenarios).valid is False
    assert evaluate_levels(scenarios)["status"] == SCORING_INVALID


def test_scoring_refuses_when_the_injection_step_is_unknown():
    scenarios = [scenario("r1", injected=True, primary="GRADIENT_EXPLOSION",
                          window=[110, 120], injection_step=None)]
    assert validate_scoring_inputs(scenarios).valid is False


def test_scoring_refuses_when_a_run_could_not_be_read():
    scenarios = [{"run_id": "r1", "run_error": "FileNotFoundError: gone"}]
    assert validate_scoring_inputs(scenarios).valid is False
    assert evaluate_levels(scenarios)["status"] == SCORING_INVALID


# ---------------------------------------------------------------------------
# Level separation
# ---------------------------------------------------------------------------


def test_run_level_recall_reports_a_miss():
    """The core v0.1.1 defect: recall read 1.0 because FN was never counted."""
    scenarios = [
        scenario("hit", injected=True, primary="GRADIENT_EXPLOSION",
                 window=[110, 120], episodes=[("GRADIENT_EXPLOSION", 110)],
                 verdicts=[("GRADIENT_EXPLOSION", 110)]),
        scenario("miss", injected=True, primary="GRADIENT_EXPLOSION",
                 window=[110, 120], episodes=[], verdicts=[]),
    ]
    out = evaluate_levels(scenarios)
    assert out["status"] == "VALID"
    run = out["levels"]["RUN"]
    assert run["tp"] == 1 and run["fn"] == 1
    assert run["recall"] == pytest.approx(0.5), (
        "a missed injected fault must lower recall; v0.1.1 reported 1.0 here"
    )
    assert run["recall_denominator"] == "eligible injected runs"


def test_episode_recall_is_not_estimable_rather_than_a_fake_one():
    scenarios = [scenario("r1", injected=True, primary="GRADIENT_EXPLOSION",
                          window=[110, 120], episodes=[("GRADIENT_EXPLOSION", 110)])]
    episode = evaluate_levels(scenarios)["levels"]["EPISODE"]
    assert episode["recall"] is None, (
        "truth names the injected fault, not the number of episodes it should yield"
    )
    assert episode["recall_denominator"].startswith("NOT ESTIMABLE")


def test_episode_counts_are_never_reported_as_run_counts():
    """42 runs produced 97 classification records; the units must stay separate."""
    scenarios = []
    for index in range(4):
        scenarios.append(scenario(
            "r{}".format(index), injected=True, primary="GRADIENT_EXPLOSION",
            window=[110, 120],
            episodes=[("GRADIENT_EXPLOSION", 110), ("GRADIENT_EXPLOSION", 112),
                      ("NAN_DETECTED", 130)],
        ))
    out = evaluate_levels(scenarios)
    accounting = out["unit_accounting"]
    assert accounting["runs"] == 4
    assert accounting["episodes_classified"] == 12
    assert accounting["runs_with_more_than_one_episode"] == 4
    assert "EPISODES, not runs" in accounting["explanation"]


def test_healthy_run_opening_an_incident_is_a_run_level_false_positive():
    scenarios = [scenario("h", injected=False,
                          verdicts=[("GRADIENT_EXPLOSION", 90)],
                          episodes=[("GRADIENT_EXPLOSION", 90)])]
    run = evaluate_levels(scenarios)["levels"]["RUN"]
    assert run["fp"] == 1 and run["tp"] == 0
    assert run["precision"] == 0.0


def test_healthy_run_with_no_incident_is_a_true_negative():
    scenarios = [scenario("h", injected=False)]
    run = evaluate_levels(scenarios)["levels"]["RUN"]
    assert run["tn"] == 1 and run["fp"] == 0


def test_an_onset_outside_the_window_does_not_count_as_a_hit():
    """Guards the pre-injection-onset defect: a match must be inside the window."""
    scenarios = [scenario("r", injected=True, primary="GRADIENT_EXPLOSION",
                          window=[110, 120], onset=75,
                          episodes=[("GRADIENT_EXPLOSION", 75)],
                          verdicts=[("GRADIENT_EXPLOSION", 75)])]
    run = evaluate_levels(scenarios)["levels"]["RUN"]
    assert run["tp"] == 0 and run["fn"] == 1, (
        "an onset before the fault must not be credited as a detection"
    )


# ---------------------------------------------------------------------------
# Relation level
# ---------------------------------------------------------------------------


def test_a_fabricated_relation_between_independent_faults_is_a_false_positive():
    """The measured v0.1.2 defect, pinned.

    Two deliberately independent faults 40 steps apart were linked as
    POSSIBLE_CONSEQUENCE_OF. Temporal adjacency is not evidence of consequence.
    """
    scenarios = [scenario(
        "r", injected=True, primary="GRADIENT_EXPLOSION", window=[80, 92],
        episodes=[("GRADIENT_EXPLOSION", 80), ("NAN_DETECTED", 120)],
        relations=[{"relation_id": "rel1", "relation_type": "POSSIBLE_CONSEQUENCE_OF",
                    "source_incident_id": "e1", "target_incident_id": "e2",
                    "target_anomaly_type": "NAN_DETECTED", "gap_steps": 40}],
    )]
    relation = evaluate_levels(scenarios)["levels"]["RELATION"]
    assert relation["tp"] == 0
    assert relation["fp"] == 1, "no allowed secondary manifestation was declared"


def test_a_relation_inside_a_declared_secondary_window_is_a_true_positive():
    scenarios = [scenario(
        "r", injected=True, primary="GRADIENT_EXPLOSION", window=[80, 92],
        episodes=[("GRADIENT_EXPLOSION", 80), ("NAN_DETECTED", 82)],
        relations=[{"relation_id": "rel1", "relation_type": "POSSIBLE_CONSEQUENCE_OF",
                    "source_incident_id": "e1", "target_incident_id": "e2",
                    "target_anomaly_type": "NAN_DETECTED", "gap_steps": 2}],
    )]
    scenarios[0]["ground_truth"]["allowed_secondary_manifestations"] = ["NAN_DETECTED"]
    scenarios[0]["ground_truth"]["secondary_window"] = [80, 90]
    relation = evaluate_levels(scenarios)["levels"]["RELATION"]
    assert relation["tp"] == 1 and relation["fp"] == 0
    assert relation["precision"] == 1.0


def test_a_missed_consequence_is_a_relation_false_negative():
    scenarios = [scenario(
        "r", injected=True, primary="GRADIENT_EXPLOSION", window=[80, 92],
        episodes=[("GRADIENT_EXPLOSION", 80), ("NAN_DETECTED", 82)],
        relations=[],
    )]
    scenarios[0]["ground_truth"]["allowed_secondary_manifestations"] = ["NAN_DETECTED"]
    scenarios[0]["ground_truth"]["secondary_window"] = [80, 90]
    relation = evaluate_levels(scenarios)["levels"]["RELATION"]
    assert relation["fn"] == 1 and relation["recall"] == 0.0
