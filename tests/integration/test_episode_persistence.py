"""J3: episode persistence and genuine cross-process continuation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training_debugger.episodes import AnomalyEpisode
from training_debugger.incidents.relations import link_consequences
from training_debugger.persistence import EpisodeStore

ROOT = Path(__file__).resolve().parents[2]


def episode(**overrides):
    payload = dict(
        episode_id="ep_fixed", run_id="run", detector_id="grad.robust_zscore",
        detector_version="3.0", rule_version="phase1.5-1.0",
        anomaly_type="GRADIENT_EXPLOSION", signal_key="grad:rank0",
        state="SUSPECT", candidate_onset_step=80, signal_count=1,
        affected_layers=[2], affected_ranks=[0],
        confidence_class="WEAK", confidence=0.4,
        supporting_event_ids=["ev1"], metadata={"gradient_band": "UNCERTAIN"},
    )
    payload.update(overrides)
    return AnomalyEpisode(**payload)


def test_suspect_then_confirmed_is_two_revisions(tmp_path):
    store = EpisodeStore(tmp_path)
    assert store.append(episode()) == 1
    assert store.append(episode(state="CONFIRMED", confirmed_at_step=82, signal_count=2)) == 2
    assert [h["state"] for h in store.history("run", "ep_fixed")] == ["SUSPECT", "CONFIRMED"]


def test_latest_revision_is_loaded_by_default(tmp_path):
    store = EpisodeStore(tmp_path)
    store.append(episode())
    store.append(episode(state="CONFIRMED", confirmed_at_step=82, signal_count=2))
    loaded = store.load("run", "ep_fixed")
    assert loaded.state == "CONFIRMED"
    assert loaded.candidate_onset_step == 80
    assert loaded.confirmed_at_step == 82
    assert loaded.signal_count == 2


def test_historical_revision_is_loadable(tmp_path):
    store = EpisodeStore(tmp_path)
    store.append(episode())
    store.append(episode(state="CONFIRMED", confirmed_at_step=82, signal_count=2))
    first = store.load("run", "ep_fixed", revision=1)
    assert first.state == "SUSPECT"
    assert first.confirmed_at_step is None


def test_open_episodes_returns_unclosed_only(tmp_path):
    store = EpisodeStore(tmp_path)
    store.append(episode())                                    # SUSPECT, stays open
    store.append(episode(episode_id="ep_done", state="CLOSED",
                         outcome="CONFIRMED_ANOMALY", signal_count=5))
    open_ids = {s.episode.episode_id for s in store.open_episodes("run")}
    assert open_ids == {"ep_fixed"}


def test_gradient_band_survives_persistence(tmp_path):
    """The band is evidence about why an episode was not promoted."""
    store = EpisodeStore(tmp_path)
    store.append(episode())
    assert store.load("run", "ep_fixed").metadata["gradient_band"] == "UNCERTAIN"


def test_relations_persist_and_reload(tmp_path):
    store = EpisodeStore(tmp_path)
    episodes = [
        episode(episode_id="primary", state="CLOSED", outcome="CONFIRMED_ANOMALY",
                anomaly_type="GRADIENT_EXPLOSION", confirmed_at_step=102, signal_count=9),
        episode(episode_id="secondary", state="CLOSED", outcome="CONFIRMED_ANOMALY",
                anomaly_type="NAN_DETECTED", candidate_onset_step=140, confirmed_at_step=140),
    ]
    relations, _ = link_consequences("run", episodes)
    assert relations
    assert store.append_relations("run", relations) == len(relations)
    restored = store.load_relations("run")
    assert len(restored) == len(relations)
    assert restored[0].to_dict() == relations[0].to_dict()
    # v0.1.3: a finite gradient anomaly is not mechanistic evidence, so this is an
    # order-only relation. The point of this test is the round trip, not the type.
    assert restored[0].relation_type == "PRECEDES"


def test_missing_episode_raises(tmp_path):
    store = EpisodeStore(tmp_path)
    with pytest.raises(KeyError):
        store.load("run", "nope")


WRITE = """
import sys
sys.path.insert(0, {src!r})
from training_debugger.episodes import AnomalyEpisode
from training_debugger.persistence import EpisodeStore

store = EpisodeStore({root!r})
episode = AnomalyEpisode(
    episode_id="ep_x", run_id="run", detector_id="grad.robust_zscore",
    detector_version="3.0", rule_version="phase1.5-1.0",
    anomaly_type="GRADIENT_EXPLOSION", signal_key="grad:rank0",
    state="SUSPECT", candidate_onset_step=80, signal_count=1,
    affected_layers=[2], confidence_class="WEAK", confidence=0.4,
)
store.append(episode)
print("wrote revision", 1)
"""

CONTINUE = """
import sys, json
sys.path.insert(0, {src!r})
from training_debugger.persistence import EpisodeStore

store = EpisodeStore({root!r})
snapshots = store.open_episodes("run")
assert snapshots, "the restarted process found no open episode"
snapshot = snapshots[0]
episode = snapshot.episode
assert episode.episode_id == "ep_x"
assert episode.candidate_onset_step == 80
# Continue: two further observations promote the same episode.
episode.signal_count += 2
episode.state = "CONFIRMED"
episode.confirmed_at_step = 82
episode.last_abnormal_step = 82
revision = store.append(episode, revision=snapshot.revision + 1)
json.dump({{
    "episode_id": episode.episode_id,
    "onset": episode.candidate_onset_step,
    "confirmed_at": episode.confirmed_at_step,
    "signal_count": episode.signal_count,
    "revision": revision,
    "states": [h["state"] for h in store.history("run", "ep_x")],
}}, open({out!r}, "w"))
print("continued to revision", revision)
"""


def test_cross_process_episode_continuation(tmp_path):
    """A second process loads a SUSPECT episode and promotes the *same* episode.

    This is the requirement Phase 1.5 stated and did not meet: without it, an
    episode is not durable evidence and a restart silently restarts the analysis.
    """
    src = str(ROOT / "src")
    root = tmp_path / "store"
    result_path = tmp_path / "result.json"

    writer = tmp_path / "write.py"
    writer.write_text(WRITE.format(src=src, root=str(root)), encoding="utf-8")
    result = subprocess.run([sys.executable, str(writer)], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]

    # Separate interpreter, no in-memory state at all.
    cont = tmp_path / "continue.py"
    cont.write_text(
        CONTINUE.format(src=src, root=str(root), out=str(result_path)), encoding="utf-8"
    )
    result = subprocess.run([sys.executable, str(cont)], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]

    payload = json.loads(result_path.read_text())
    assert payload["episode_id"] == "ep_x", "a restart must not create a new episode"
    assert payload["onset"] == 80, "the onset must survive the restart"
    assert payload["confirmed_at"] == 82
    assert payload["signal_count"] == 3
    assert payload["states"] == ["SUSPECT", "CONFIRMED"]

    # And the primary process can read what the second wrote.
    store = EpisodeStore(root)
    restored = store.load("run", "ep_x")
    assert restored.state == "CONFIRMED"
    assert restored.signal_count == 3
