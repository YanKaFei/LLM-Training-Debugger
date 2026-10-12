"""A3 / C2 / C3: invariants that must hold through the *persisted* path.

The unit tests prove these properties in memory. These prove them after a write and a
fresh read, which is where the earlier phases kept finding discrepancies.
"""

from __future__ import annotations

import json

from training_debugger.detectors.base import Anomaly, ConfidenceClass
from training_debugger.detectors.baseline import protected_ratio_series
from training_debugger.episodes import AnomalyEpisode, EpisodeEngine, EpisodeState
from training_debugger.fusion import fuse
from training_debugger.incidents import reconcile_incidents
from training_debugger.incidents.relations import RelationType, link_consequences
from training_debugger.persistence import (
    AnalysisStateStore, EpisodeStore, capture_analysis_state,
)
from training_debugger.schema.enums import EvidenceLevel, Severity


def anomaly(step, layer, band="CONFIRMED", detector="grad.robust_zscore"):
    return Anomaly(
        detector=detector, detector_version="3.0", metric_name="grad.layer_norm",
        first_abnormal_step=step, observed_value=40.0, baseline=1.0, threshold=11.5,
        score=3.0, confidence=0.9, level=EvidenceLevel.INFERRED, severity=Severity.ERROR,
        window=(step - 50, step), rank=0, layer_index=layer,
        confidence_class=ConfidenceClass.STRONG,
        metadata={"gradient_band": band, "population_layers": 4},
    )


def episode(eid, anomaly_type, onset, layers=(0, 1), run_id="r", metric=None):
    return AnomalyEpisode(
        episode_id=eid, run_id=run_id, detector_id="d", detector_version="1",
        rule_version="v", anomaly_type=anomaly_type, signal_key="k",
        metric_name=metric,
        state=EpisodeState.CLOSED.value, outcome="CONFIRMED_ANOMALY",
        candidate_onset_step=onset, confirmed_at_step=onset + 2,
        affected_layers=list(layers), confidence_class="STRONG", confidence=0.9,
    )


# ---------------------------------------------------------------------------
# A3 — no duplicate episode or incident from the same source evidence
# ---------------------------------------------------------------------------


def test_replaying_the_same_anomalies_does_not_create_a_second_episode():
    """A3 at the engine level: identical evidence, fed twice, is still one episode."""
    engine = EpisodeEngine()
    anomalies = [anomaly(80, layer) for layer in (0, 1, 2, 3)]
    engine.consume(anomalies, "r")
    engine.finish("r", final_step=200)
    first = engine.all_episodes()

    replay = EpisodeEngine()
    replay.consume(anomalies, "r")
    replay.consume(anomalies, "r")          # same evidence offered again
    replay.finish("r", final_step=200)
    second = replay.all_episodes()

    assert len(first) == 1
    assert len(second) == 1, "re-feeding the same anomalies must not add an episode"
    assert first[0].episode_id == second[0].episode_id
    assert first[0].signal_count == second[0].signal_count


def test_a_restart_does_not_duplicate_the_persisted_episode(tmp_path):
    store = EpisodeStore(tmp_path)
    engine = EpisodeEngine()
    engine.consume([anomaly(80, layer) for layer in (0, 1)], "r")
    for item in engine.all_episodes():
        store.append(item)

    # A second process resumes, sees the same step for the same layers again, persists.
    resumed = EpisodeEngine()
    opened = store.open_episodes("r")
    resumed.resume(opened)
    resumed.consume([anomaly(80, layer) for layer in (0, 1)], "r")
    resumed.finish("r", final_step=200)
    for item in resumed.all_episodes():
        store.append(item)

    resumed_id = opened[0].episode.episode_id
    history = store.history("r", resumed_id)
    ids = {h["episode_id"] for h in history}
    assert ids == {resumed_id}, "no new episode id may appear"
    assert len(ids) == 1
    assert store.load("r", resumed_id).episode_id == resumed_id


# ---------------------------------------------------------------------------
# C2 — independent NaN protection through the persisted path
# ---------------------------------------------------------------------------


def test_standalone_nan_keeps_its_own_incident_after_a_store_round_trip(tmp_path):
    store = EpisodeStore(tmp_path)
    solo = episode("solo", "NAN_DETECTED", 50, layers=(5,))
    store.append(solo)
    relations, linked = link_consequences("r", [solo])
    store.append_relations("r", relations)

    reloaded = [store.load("r", "solo")]
    restored_relations = store.load_relations("r")
    result = reconcile_incidents("r", reloaded, restored_relations)

    assert result.groups == [], "a standalone NaN must not be grouped with anything"
    assert result.independent_episode_ids == ["solo"]
    assert linked == set()


def test_linked_nan_stays_linked_and_uncaused_after_a_store_round_trip(tmp_path):
    store = EpisodeStore(tmp_path)
    # v0.1.3: a real propagation -- a non-finite gradient, then the loss one step on.
    episodes = [episode("g", "NAN_DETECTED", 80, layers=(0, 1, 2),
                        metric="grad.global_norm"),
                episode("p", "NAN_DETECTED", 80, layers=(0, 1, 2),
                        metric="param.global_norm"),
                episode("n", "NAN_DETECTED", 81, layers=(0, 1, 2),
                        metric="train.loss")]
    for item in episodes:
        store.append(item)
    relations, linked = link_consequences("r", episodes, mechanism_linker_enabled=True)
    store.append_relations("r", relations)

    reloaded = [store.load("r", eid) for eid in ("g", "n")]
    restored = store.load_relations("r")
    result = reconcile_incidents("r", reloaded, restored)

    # v0.1.4: the propagation is observable at every stage -- non-finite gradient,
    # non-finite parameters, non-finite loss -- so this is a mechanism link and it must
    # survive the round trip as one.
    assert result.groups, "a mechanism-supported link must survive the round trip"
    assert all(g.has_proven_root_cause is False for g in result.groups)
    assert {"g", "n"} <= linked
    assert all(r.relation_type == RelationType.POSSIBLE_CONSEQUENCE_OF.value
               for r in restored)


def test_no_relation_type_in_the_store_asserts_causation(tmp_path):
    store = EpisodeStore(tmp_path)
    episodes = [episode("g", "GRADIENT_EXPLOSION", 100), episode("n", "NAN_DETECTED", 140)]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    store.append_relations("r", relations)
    for relation in store.load_relations("r"):
        assert relation.relation_type not in ("CAUSED_BY", "ROOT_CAUSE_OF")


# ---------------------------------------------------------------------------
# C3 — the protected baseline survives a restart
# ---------------------------------------------------------------------------


def test_protected_baseline_is_restored_rather_than_reinitialised(tmp_path):
    """A resumed process must not rebuild the reference from a contaminated stream."""
    series = {i: [(s, 1.0 if s < 80 else 6.0) for s in range(1, 200)] for i in range(4)}
    engine = EpisodeEngine()
    engine.consume([anomaly(80, layer) for layer in (0,)], "r")

    state = capture_analysis_state(
        "r", processed_through_step=80, layer_series=series,
        episodes=engine.all_episodes(),  # type: ignore[arg-type]
        detector_versions={"grad.robust_zscore": "3.0"},
        fusion_rule_version="phase1.3-1.0", episode_rule_version="phase1.3-1.0",
    )
    AnalysisStateStore(tmp_path).append(state)

    restored = AnalysisStateStore(tmp_path).load("r")
    assert restored is not None
    assert restored.processed_through_step == 80
    assert restored.baselines, "the baseline must be persisted, not recomputed"

    # Every restored reference must equal the pre-fault level, NOT the post-fault one.
    for key, snapshot in restored.baselines.items():
        assert snapshot.reference is not None
        assert snapshot.reference < 3.0, (
            "reference {} for {} looks re-initialised from the contaminated stream".format(
                snapshot.reference, key)
        )


def test_analysis_state_round_trips_deterministically(tmp_path):
    series = {i: [(s, 1.0) for s in range(1, 120)] for i in range(2)}
    engine = EpisodeEngine()
    engine.consume([anomaly(60, 0)], "r")
    payload = capture_analysis_state(
        "r", processed_through_step=60, layer_series=series,
        episodes=engine.all_episodes(), detector_versions={"d": "1"},
        fusion_rule_version="v", episode_rule_version="v",
    )
    store = AnalysisStateStore(tmp_path)
    store.append(payload)
    first = store.load("r").to_dict()
    second = store.load("r").to_dict()
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(
        second, sort_keys=True, default=str
    )


# ---------------------------------------------------------------------------
# D3 — fail-open must not mask a completeness problem
# ---------------------------------------------------------------------------


def test_a_forced_overflow_reaches_the_verdict_as_degraded():
    """Dropping events must never be reported as a clean negative.

    The observer is fail-open by design: it drops rather than delaying training. That
    is correct, but it means the analysis layer can be handed a stream with holes in
    it. Before `evidence_completeness` existed, such a run produced an unqualified
    `NO_ANOMALY` -- indistinguishable from a genuinely healthy one.
    """
    from training_debugger.fusion.quality import DataQuality, DataQualityStatus
    from training_debugger.observer import BoundedEventQueue
    from training_debugger.schema import EventPriority, EventType, MetricUnit, TrainingEvent

    def make(step):
        return TrainingEvent(
            run_id="fo", producer="p", source="s", event_type=EventType.LOSS,
            step=step, metric_name="train.loss", metric_value=1.0,
            metric_unit=MetricUnit.DIMENSIONLESS, priority=EventPriority.P3,
        )

    # Deliberately no consumer, so overflow is caused rather than hoped for.
    queue = BoundedEventQueue(maxlen=16)
    accepted = refused = 0
    for step in range(1, 501):
        if queue.put(make(step)):
            accepted += 1
        else:
            refused += 1
    assert refused > 0, "the queue must actually have overflowed"
    assert accepted <= 16, "the queue must stay bounded"

    stats = queue.stats()
    quality = DataQuality.from_observer_health({
        "observer": {"events_dropped_total": refused,
                     "queue_overflow_total": getattr(stats, "evicted", refused),
                     "writer_failures_total": 0, "written_total": accepted,
                     "emitted_total": accepted + refused},
        "queue": {"evicted": getattr(stats, "evicted", 0), "rejected": refused},
    })
    assert quality.status == DataQualityStatus.DEGRADED.value, (
        "dropping {} events must not be reported as complete".format(refused)
    )
    assert quality.dropped_events > 0

    verdict = fuse("fo", [], data_quality=quality)[0]
    assert verdict.verdict == "NO_ANOMALY"
    assert verdict.completeness_warning, (
        "a negative verdict over an incomplete stream must carry a warning"
    )
    assert verdict.data_quality["status"] == "DEGRADED"


def test_dropped_events_are_never_read_as_a_healthy_run():
    from training_debugger.fusion.quality import DataQuality
    from training_debugger.incidents.render import render_verdict

    quality = DataQuality.from_observer_health({
        "observer": {"events_dropped_total": 12, "queue_overflow_total": 0,
                     "writer_failures_total": 0},
        "queue": {"evicted": 0, "rejected": 0},
    })
    text = render_verdict(fuse("r", [], data_quality=quality)[0], [])
    assert "DEGRADED" in text
    assert "positive statement that nothing was found" not in text
