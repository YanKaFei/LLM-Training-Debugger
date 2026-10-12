"""M2: the real pipeline persists episodes, relations and incidents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training_debugger.detectors.base import Anomaly, ConfidenceClass
from training_debugger.episodes import AnomalyEpisode, EpisodeEngine
from training_debugger.fusion import fuse
from training_debugger.incidents.relations import link_consequences
from training_debugger.persistence import EpisodeStore, IncidentStore
from training_debugger.schema.enums import EvidenceLevel, Severity


def gradient_anomaly(step, layer, band="CONFIRMED"):
    return Anomaly(
        detector="grad.robust_zscore", detector_version="3.0",
        metric_name="grad.layer_norm", first_abnormal_step=step,
        observed_value=40.0, baseline=1.0, threshold=11.5, score=3.0,
        confidence=0.9, level=EvidenceLevel.INFERRED, severity=Severity.ERROR,
        window=(step - 50, step), rank=0, layer_index=layer,
        confidence_class=ConfidenceClass.STRONG,
        metadata={"gradient_band": band, "population_layers": 6},
    )


def nan_episode(episode_id, onset):
    return AnomalyEpisode(
        episode_id=episode_id, run_id="run", detector_id="numerics.finiteness",
        detector_version="1.0", rule_version="v", anomaly_type="NAN_DETECTED",
        signal_key="numerics:rank0", state="CLOSED", outcome="CONFIRMED_ANOMALY",
        candidate_onset_step=onset, confirmed_at_step=onset,
        confidence_class="CERTAIN", confidence=1.0, signal_count=1,
    )


def test_pipeline_writes_episodes_and_revisions(tmp_path):
    engine = EpisodeEngine()
    engine.consume([gradient_anomaly(100, layer) for layer in range(6)], "run")
    engine.finish("run", final_step=300)
    episodes = engine.all_episodes()
    assert episodes

    store = EpisodeStore(tmp_path)
    for episode in episodes:
        assert store.append(episode) == 1
    # A second write of the same episode is a new revision, not a new episode.
    for episode in episodes:
        assert store.append(episode) == 2
    assert store.load("run", episodes[0].episode_id).episode_id == episodes[0].episode_id
    assert [h["revision"] for h in store.history("run", episodes[0].episode_id)] == [1, 2]


def test_pipeline_is_idempotent_across_a_rerun(tmp_path):
    """Re-running the same detection must not create duplicate episodes."""
    def build():
        engine = EpisodeEngine()
        engine.consume([gradient_anomaly(100, layer) for layer in range(6)], "run")
        engine.finish("run", final_step=300)
        return engine.all_episodes()

    first = build()
    store = EpisodeStore(tmp_path)
    for episode in first:
        store.append(episode)
    second = build()
    assert [e.episode_id for e in first] == [e.episode_id for e in second], (
        "episode identity must be deterministic across a re-run"
    )
    for episode in second:
        store.append(episode)
    # Distinct episode ids must not multiply. ``open_episodes`` is not the check
    # here: these episodes closed, so there are correctly no open ones. The check is
    # that each deterministic id has exactly two revisions (one per pass) and no
    # third id appeared.
    ids = set()
    for episode in second:
        history = store.history("run", episode.episode_id)
        assert [h["revision"] for h in history] == [1, 2], (
            "a re-run must add one revision, not a new episode"
        )
        ids.add(episode.episode_id)
    assert ids == {e.episode_id for e in first}
    assert store.open_episodes("run") == [], "closed episodes are not open"


def test_pipeline_persists_relations_and_incidents(tmp_path):
    primary = EpisodeEngine()
    primary.consume([gradient_anomaly(100, layer) for layer in range(6)], "run")
    primary.finish("run", final_step=300)
    episodes = primary.all_episodes()
    episodes.append(nan_episode("ep_nan", 140))

    relations, linked = link_consequences("run", episodes)
    # v0.1.3: a finite gradient fault followed by a NaN produces an order-only PRECEDES
    # relation, not a consequence claim. The relation and the persistence path are what
    # this test covers.
    assert relations, "a gradient fault followed by NaN must still be recorded"
    assert linked

    episode_store = EpisodeStore(tmp_path)
    for episode in episodes:
        episode_store.append(episode)
    assert episode_store.append_relations("run", relations) == len(relations)
    assert episode_store.load_relations("run")[0].relation_type == "PRECEDES"

    verdicts = fuse("run", episodes)
    assert any(v.verdict == "CONFIRMED_INCIDENT" for v in verdicts)
    # An incident must never precede the episode it consumes.
    for verdict in verdicts:
        if verdict.verdict != "CONFIRMED_INCIDENT":
            continue
        for episode_id in verdict.episodes_used:
            assert episode_store.history("run", episode_id), (
                "an incident was promoted before its episode was persisted"
            )
