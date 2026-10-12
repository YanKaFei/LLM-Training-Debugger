"""C1: pipeline contracts that the smoke run verified, as regression tests.

The eight-scenario smoke (`tests/experiments/full_pipeline_smoke.py`) is an executable
harness, not a gate. These tests pin the contracts it exercised, cheaply enough to run
in the normal suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training_debugger.detectors import DetectorConfig, DetectorRunner
from training_debugger.episodes import AnomalyEpisode, EpisodeEngine, EpisodeState
from training_debugger.fusion import Verdict, fuse
from training_debugger.fusion.quality import DataQuality, DataQualityStatus
from training_debugger.incidents import reconcile_incidents
from training_debugger.incidents.relations import RelationType, link_consequences
from training_debugger.persistence import EpisodeStore
from training_debugger.registry import detector_registry

STEPS = 200
LAYERS = 4


def band_series(ratio: float):
    steps = list(range(1, STEPS + 1))
    layer = {i: [(s, 1.0 if s < 70 else ratio) for s in steps] for i in range(LAYERS)}
    return {"grad.global_norm": [(s, 1.0 if s < 70 else ratio) for s in steps]}, layer


def detect(by_metric, layer_series, run_id="r"):
    return DetectorRunner(DetectorConfig()).run(
        run_id, by_metric, rank=0, layer_series=layer_series, module_paths={},
        population_layers=LAYERS, population_ranks=1,
        coverage={"step_count": STEPS, "step_min": 1, "step_max": STEPS},
    )


def verdicts_for(run_id, ratio):
    by_metric, layer_series = band_series(ratio)
    detection = detect(by_metric, layer_series, run_id)
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    engine.consume(detection.anomalies, run_id)
    engine.finish(run_id, final_step=STEPS)
    episodes = engine.all_episodes()
    return episodes, fuse(run_id, episodes)


# ---------------------------------------------------------------------------
# Band -> verdict contracts
# ---------------------------------------------------------------------------


def test_healthy_band_produces_no_incident():
    episodes, verdicts = verdicts_for("healthy", 1.0)
    assert all(v.verdict != Verdict.CONFIRMED_INCIDENT.value for v in verdicts)


def test_uncertain_band_produces_abstention_not_an_incident():
    episodes, verdicts = verdicts_for("uncertain", 10.0)
    assert all(v.verdict != Verdict.CONFIRMED_INCIDENT.value for v in verdicts)


def test_confirmed_band_produces_an_incident():
    episodes, verdicts = verdicts_for("confirmed", 14.0)
    confirmed = [v for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value]
    assert confirmed, "a strong joint aggregate must be able to open an incident"
    assert confirmed[0].promotion_rule_id
    assert confirmed[0].promotion_rule_version


# ---------------------------------------------------------------------------
# Evidence completeness through the pipeline
# ---------------------------------------------------------------------------


def test_every_verdict_carries_completeness():
    _, verdicts = verdicts_for("quality", 14.0)
    for verdict in verdicts:
        assert verdict.data_quality, "a verdict must state its evidence completeness"
        assert verdict.data_quality["status"] in (
            DataQualityStatus.COMPLETE.value,
            DataQualityStatus.DEGRADED.value,
            DataQualityStatus.UNKNOWN.value,
        )


def test_degraded_completeness_reaches_the_verdict():
    by_metric, layer_series = band_series(14.0)
    detection = detect(by_metric, layer_series, "degraded")
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    engine.consume(detection.anomalies, "degraded")
    engine.finish("degraded", final_step=STEPS)
    quality = DataQuality.from_observer_health({
        "observer": {"events_dropped_total": 9, "queue_overflow_total": 3,
                     "writer_failures_total": 0},
        "queue": {"evicted": 3},
    })
    verdicts = fuse("degraded", engine.all_episodes(), data_quality=quality)
    assert verdicts[0].data_quality["status"] == DataQualityStatus.DEGRADED.value
    assert verdicts[0].completeness_warning


# ---------------------------------------------------------------------------
# Relation and reconciliation contracts
# ---------------------------------------------------------------------------


def anomaly(step, layer, band="CONFIRMED", detector="grad.robust_zscore"):
    from training_debugger.detectors.base import Anomaly, ConfidenceClass
    from training_debugger.schema import EvidenceLevel, Severity

    return Anomaly(
        detector=detector, detector_version="3.0", metric_name="grad.layer_norm",
        first_abnormal_step=step, observed_value=40.0, baseline=1.0, threshold=11.5,
        score=3.0, confidence=0.9, level=EvidenceLevel.INFERRED, severity=Severity.ERROR,
        window=(step - 50, step), rank=0, layer_index=layer,
        confidence_class=ConfidenceClass.STRONG,
        metadata={"gradient_band": band, "population_layers": 4},
    )


def episode(eid, anomaly_type, onset, layers=(0, 1, 2), metric=None):
    return AnomalyEpisode(
        episode_id=eid, run_id="r", detector_id="d", detector_version="1",
        rule_version="v", anomaly_type=anomaly_type, signal_key="k",
        metric_name=metric,
        state=EpisodeState.CLOSED.value, outcome="CONFIRMED_ANOMALY",
        candidate_onset_step=onset, confirmed_at_step=onset + 2,
        affected_layers=list(layers), confidence_class="STRONG", confidence=0.9,
    )


def test_linked_manifestation_is_grouped_without_a_causal_claim():
    """v0.1.3: grouping requires a MECHANISM evidence link, not mere order.

    The fixture used to be a finite GRADIENT_EXPLOSION 40 steps before a NaN. That is
    an order-only relation now, so it correctly does not group. Grouping is tested with
    a genuine propagation: a non-finite gradient, then a non-finite loss one step later.
    """
    episodes = [episode("g", "NAN_DETECTED", 80, metric="grad.global_norm"),
    episode("p", "NAN_DETECTED", 80, metric="param.global_norm"),
                episode("p", "NAN_DETECTED", 80, metric="param.global_norm"),
                episode("n", "NAN_DETECTED", 81, metric="train.loss")]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    # Every stage of the chain is a consequence of the stage before it, so more than
    # one relation is expected and correct: the parameter corruption is itself a
    # downstream manifestation of the gradient fault.
    assert relations
    assert all(r.relation_type == "POSSIBLE_CONSEQUENCE_OF" for r in relations)
    assert all(r.evidence["midstream_nonfinite_witness"] for r in relations)
    result = reconcile_incidents("r", episodes, relations)
    assert result.groups, "a mechanism-supported link must group"
    assert all(g.has_proven_root_cause is False for g in result.groups)


def test_independent_nan_is_never_merged():
    episodes = [episode("g", "GRADIENT_EXPLOSION", 100, layers=(0, 1)),
                episode("far", "NAN_DETECTED", 900, layers=(9,))]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert relations == []
    result = reconcile_incidents("r", episodes, relations)
    assert result.groups == []
    assert set(result.independent_episode_ids) == {"g", "far"}


def test_no_causal_relation_type_exists():
    names = {t.value for t in RelationType}
    assert "CAUSED_BY" not in names and "ROOT_CAUSE_OF" not in names


# ---------------------------------------------------------------------------
# Persistence contracts
# ---------------------------------------------------------------------------


def test_relation_and_episode_survive_a_store_round_trip(tmp_path):
    # v0.1.3: a mechanism-supported pair -- a non-finite gradient, then the non-finite
    # loss on the next step. Mechanism evidence, not merely order.
    episodes = [episode("g", "NAN_DETECTED", 80, metric="grad.global_norm"),
    episode("p", "NAN_DETECTED", 80, metric="param.global_norm"),
                episode("p", "NAN_DETECTED", 80, metric="param.global_norm"),
                episode("n", "NAN_DETECTED", 81, metric="train.loss")]
    relations, linked = link_consequences("r", episodes, mechanism_linker_enabled=True)
    store = EpisodeStore(tmp_path)
    for item in episodes:
        store.append(item)
    store.append_relations("r", relations)

    assert {"g", "n"} <= linked
    restored = {e.episode_id: e for e in
                (store.load("r", eid) for eid in ("g", "n"))}
    assert restored["g"].candidate_onset_step == 80
    assert restored["n"].candidate_onset_step == 81
    assert store.load_relations("r")[0].relation_type == RelationType.POSSIBLE_CONSEQUENCE_OF.value


def test_repeated_persistence_does_not_duplicate_the_episode(tmp_path):
    store = EpisodeStore(tmp_path)
    item = episode("g", "GRADIENT_EXPLOSION", 100)
    assert store.append(item) == 1
    assert store.append(item) == 2
    assert [h["revision"] for h in store.history("r", "g")] == [1, 2]
    assert store.load("r", "g").episode_id == "g"


# ---------------------------------------------------------------------------
# W1 — the onset must be the onset of the abnormality, not of the contiguity run
# ---------------------------------------------------------------------------


def test_benign_precursor_does_not_become_the_onset():
    """The defect: a NORMAL-band twitch inherited the fault next to it.

    The quickstart injects at step 80. A WEAK/NORMAL-band excursion at step 75 fell
    inside `gap_tolerance = 8`, so the episode's first observation was step 75 and the
    incident reported `candidate_onset_step = 75` -- five steps before anything went
    wrong. The band had already dismissed those steps.
    """
    engine = EpisodeEngine()
    precursor = anomaly(75, 0, band="NORMAL")
    precursor.confidence_class = "WEAK"
    precursor.detector_version = "3.0"
    engine.consume([precursor], "r")
    # Steps 78 (same benign regime) then the real fault at 80.
    engine.consume([anomaly(78, 1, band="NORMAL")], "r")
    engine.consume([anomaly(80, layer, band="CONFIRMED") for layer in range(4)], "r")
    engine.finish("r", final_step=200)

    episodes = engine.all_episodes()
    assert len(episodes) == 1, "the run must be one contiguous episode"
    episode = episodes[0]
    assert episode.outcome == "CONFIRMED_ANOMALY"
    assert episode.candidate_onset_step == 80, (
        "onset must be the first genuinely abnormal step, not the benign precursor "
        "the band called NORMAL; got {}".format(episode.candidate_onset_step)
    )
    assert episode.metadata.get("contiguity_onset_step") == 75, (
        "the raw first observation must stay auditable"
    )
    assert episode.metadata.get("onset_reanchored")


def test_uncertain_band_still_anchors_an_early_onset():
    """Genuine early warning must not be discarded along with the false precursor."""
    engine = EpisodeEngine()
    engine.consume([anomaly(70, 0, band="UNCERTAIN")], "r")
    engine.consume([anomaly(74, layer, band="CONFIRMED") for layer in range(4)], "r")
    engine.finish("r", final_step=200)
    episodes = engine.all_episodes()
    assert episodes
    assert episodes[0].candidate_onset_step == 70, (
        "an UNCERTAIN-band observation is genuinely abnormal and must still anchor the "
        "onset; excluding it would discard early warning"
    )


def test_normal_only_episode_is_never_confirmed_or_anchored():
    engine = EpisodeEngine()
    engine.consume([anomaly(60, 0, band="NORMAL")], "r")
    engine.finish("r", final_step=200)
    episodes = engine.all_episodes()
    assert all(e.outcome != "CONFIRMED_ANOMALY" for e in episodes)
