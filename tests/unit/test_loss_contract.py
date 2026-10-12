"""A3: the detector/episode confirmation contract.

Phase 1.11 found that ``loss.robust_zscore``'s ``cooldown_steps = 50`` made the
``LOSS_SPIKE`` episode rule — which requires two observations within six steps —
**unsatisfiable by the detector's own configuration**. These tests pin both the defect
and its repair.
"""

from __future__ import annotations

import pytest

from training_debugger.detectors import DetectorConfig, DetectorRunner
from training_debugger.episodes import EpisodeEngine, EpisodeOutcome, EpisodeState
from training_debugger.fusion import Verdict, fuse
from training_debugger.registry import detector_registry


def loss_series(spike_steps, peak=20.0, end=200):
    step_range = list(range(1, end))
    loss = [(s, peak if s in spike_steps else 1.0) for s in step_range]
    return {"train.loss": loss, "train.lr": [(s, 0.05) for s in step_range]}


def detect(by_metric, *, suppress=True, run_id="r"):
    return DetectorRunner(
        DetectorConfig(cooldown_suppresses_evidence=suppress)
    ).run(
        run_id, by_metric, rank=0, layer_series={}, module_paths={},
        population_layers=1, population_ranks=1,
        coverage={"step_count": 200, "step_min": 1, "step_max": 200},
    )


def episodes_for(detection, run_id="r"):
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    engine.consume(detection.anomalies, run_id)
    engine.finish(run_id, final_step=200)
    return engine.all_episodes()


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


def test_cooldown_starves_the_episode_confirmation_rule():
    """Legacy behaviour: a sustained 20-step spike yields ONE observation.

    The `LOSS_SPIKE` rule needs two within six steps, so the rule could never be
    satisfied. This is the defect, pinned so it cannot return silently.
    """
    sustained = set(range(90, 110))
    detection = detect(loss_series(sustained), suppress=True)
    observations = [a for a in detection.anomalies if str(a.detector) == "loss.robust_zscore"]
    assert len(observations) == 1, "cooldown is expected to suppress the rest"

    episodes = episodes_for(detection)
    confirmed = [e for e in episodes if e.confirmed_at_step is not None]
    assert confirmed == [], "with one observation the rule cannot be satisfied"


def test_loss_spike_cooldown_does_not_starve_episode():
    """The fix: the same spike now provides every qualifying observation."""
    sustained = set(range(90, 110))
    detection = detect(loss_series(sustained), suppress=False)
    observations = [a for a in detection.anomalies if str(a.detector) == "loss.robust_zscore"]
    assert len(observations) >= 2, (
        "the episode rule needs at least two observations; the detector must supply them"
    )


# ---------------------------------------------------------------------------
# Evidence vs alert separation
# ---------------------------------------------------------------------------


def test_many_observations_do_not_flood_the_alert_layer():
    """Twenty observations must still yield ONE episode and ONE verdict.

    This is the reason the cooldown is not simply set to zero: deduplication has not
    been removed, it was never the detector's job. `EpisodeEngine` groups contiguous
    observations and `fuse()` returns one verdict per anomaly type.
    """
    sustained = set(range(90, 110))
    detection = detect(loss_series(sustained), suppress=False)
    observations = [a for a in detection.anomalies if str(a.detector) == "loss.robust_zscore"]
    assert len(observations) > 5

    episodes = episodes_for(detection)
    loss_episodes = [e for e in episodes if e.anomaly_type == "LOSS_SPIKE"]
    assert len(loss_episodes) == 1, "contiguous observations must group into one episode"

    verdicts = fuse("r", episodes)
    loss_verdicts = [v for v in verdicts if v.anomaly_type == "LOSS_SPIKE"]
    assert len(loss_verdicts) == 1, "fusion deduplicates per anomaly type"


def test_suppressed_observations_are_flagged_not_hidden():
    sustained = set(range(90, 110))
    detection = detect(loss_series(sustained), suppress=False)
    flags = {
        a.metadata.get("cooldown_suppressed")
        for a in detection.anomalies if str(a.detector) == "loss.robust_zscore"
    }
    assert flags == {False, True}, (
        "an alert-counting consumer must be able to tell which observations a cooldown "
        "would have suppressed"
    )


# ---------------------------------------------------------------------------
# Candidate semantics
# ---------------------------------------------------------------------------


def test_single_observation_creates_a_candidate_not_a_confirmation():
    detection = detect(loss_series({100}), suppress=False)
    episodes = episodes_for(detection)
    assert episodes, "one qualifying observation must open an episode"
    assert all(e.confirmed_at_step is None for e in episodes)
    verdicts = fuse("r", episodes)
    assert all(v.verdict != Verdict.CONFIRMED_INCIDENT.value for v in verdicts)


def test_candidate_recovery_clears_the_verdict():
    """A single fluctuation that resolves leaves no active candidate."""
    detection = detect(loss_series({100}), suppress=False)
    episodes = episodes_for(detection)
    assert episodes
    assert episodes[0].state == EpisodeState.CLOSED.value
    assert episodes[0].outcome == EpisodeOutcome.NOT_ANOMALY.value
    verdicts = fuse("r", episodes)
    assert verdicts[0].verdict == Verdict.NO_ANOMALY.value, (
        "a resolved non-anomaly must not remain a live candidate"
    )


def test_two_distinct_observations_can_confirm():
    detection = detect(loss_series({100, 104}), suppress=False)
    episodes = episodes_for(detection)
    confirmed = [e for e in episodes if e.confirmed_at_step is not None]
    assert confirmed, "two observations within six steps must satisfy the rule"
    assert confirmed[0].candidate_onset_step == 100
    assert confirmed[0].confirmed_at_step == 104


def test_same_observation_cannot_count_twice():
    detection = detect(loss_series({100}), suppress=False)
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    anomalies = [a for a in detection.anomalies if str(a.detector) == "loss.robust_zscore"]
    engine.consume(anomalies, "r")
    # Re-submitting the *same* step must not manufacture a second observation.
    engine.consume(anomalies, "r")
    engine.finish("r", final_step=200)
    episodes = engine.all_episodes()
    assert episodes
    assert episodes[0].confirmed_at_step is None, (
        "one observation resubmitted is still one observation"
    )
    assert episodes[0].signal_count == len(anomalies), "no double counting"


def test_candidate_only_is_reachable_through_production_components():
    """The verdict Phase 1.11 could not produce."""
    detection = detect(loss_series({100, 104}), suppress=False)
    episodes = episodes_for(detection)
    confirmed = [e for e in episodes if e.confirmed_at_step is not None]
    assert confirmed, "the episode must confirm for the candidate verdict to be reachable"
    verdicts = fuse("r", episodes)
    loss_verdicts = [v for v in verdicts if v.anomaly_type == "LOSS_SPIKE"]
    assert loss_verdicts
    assert loss_verdicts[0].verdict == Verdict.CANDIDATE_ONLY.value
    assert loss_verdicts[0].episodes_used, "an abstention must name its episode"


def test_loss_spike_promotion_remains_disabled():
    from training_debugger.fusion import promotion_enabled

    assert promotion_enabled("LOSS_SPIKE") is False


def test_cooldown_does_not_duplicate_incidents():
    """Twenty observations must not become twenty incidents."""
    detection = detect(loss_series(set(range(90, 110))), suppress=False)
    episodes = episodes_for(detection)
    verdicts = fuse("r", episodes)
    confirmed = [v for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value]
    assert confirmed == [], "LOSS_SPIKE promotion is off, so nothing may be confirmed"
    candidates = [v for v in verdicts if v.verdict == Verdict.CANDIDATE_ONLY.value]
    assert len(candidates) <= 1
