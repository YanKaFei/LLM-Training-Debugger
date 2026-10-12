"""P0: restart readback. A second process must recover the same evidence.

The brief forbids the inconsistency ``Episode = SUSPECT, Baseline = newly
initialized``: a persisted episode whose detector reference was silently rebuilt
would mean the same run judged by two different baselines.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training_debugger.detectors.baseline import ProtectedBaseline
from training_debugger.episodes import AnomalyEpisode, EpisodeEngine
from training_debugger.persistence import (
    AnalysisStateStore,
    EpisodeStore,
    capture_analysis_state,
)

ROOT = Path(__file__).resolve().parents[2]


def safe_series(layer, floor=1, onset=80, before=1.0, after=6.0, end=200):
    """A series whose baseline is fully established before the fault begins."""
    return [(s, before if s < onset else after) for s in range(floor, end)]


def test_snapshot_reference_is_the_pre_fault_level():
    series = {0: safe_series(0), 1: safe_series(1)}
    state = capture_analysis_state("run", processed_through_step=199, layer_series=series)
    for key, snapshot in state.baselines.items():
        assert snapshot.reference == pytest.approx(1.0, rel=0.05), (
            "%s took its reference from the fault regime, not from the healthy one" % key
        )
        assert snapshot.frozen is True
        assert snapshot.freeze_step is not None


def test_snapshot_round_trips_exactly(tmp_path):
    series = {0: safe_series(0)}
    state = capture_analysis_state("run", processed_through_step=199, layer_series=series)
    store = AnalysisStateStore(tmp_path)
    assert store.append(state) == 1
    loaded = store.load("run")
    assert loaded.to_dict()["baselines"] == state.to_dict()["baselines"]
    assert loaded.processed_through_step == 199


def test_episodes_survive_a_reload_unchanged(tmp_path):
    engine = EpisodeEngine()
    from training_debugger.detectors.base import Anomaly, ConfidenceClass
    from training_debugger.schema.enums import EvidenceLevel, Severity

    def anomaly(step, layer):
        return Anomaly(
            detector="grad.robust_zscore", detector_version="3.0",
            metric_name="grad.layer_norm", first_abnormal_step=step,
            observed_value=40.0, baseline=1.0, threshold=11.5, score=3.0,
            confidence=0.9, level=EvidenceLevel.INFERRED, severity=Severity.ERROR,
            window=(step - 50, step), rank=0, layer_index=layer,
            confidence_class=ConfidenceClass.STRONG,
            metadata={"gradient_band": "CONFIRMED", "population_layers": 4},
        )

    engine.consume([anomaly(100, layer) for layer in range(4)], "run")
    episodes = engine.all_episodes()
    store = EpisodeStore(tmp_path)
    for episode in episodes:
        store.append(episode)
    loaded = store.load("run", episodes[0].episode_id)
    assert loaded.to_dict() == episodes[0].to_dict()


WRITE = """
import sys, json
sys.path.insert(0, {src!r})
from training_debugger.detectors.base import Anomaly, ConfidenceClass
from training_debugger.schema.enums import EvidenceLevel, Severity
from training_debugger.episodes import EpisodeEngine
from training_debugger.persistence import EpisodeStore, AnalysisStateStore, capture_analysis_state

run_id = "run-restart"
series = {{i: [(s, 1.0 if s < 80 else 6.0) for s in range(1, 200)] for i in range(4)}}

def anomaly(step, layer):
    return Anomaly(
        detector="grad.robust_zscore", detector_version="3.0",
        metric_name="grad.layer_norm", first_abnormal_step=step,
        observed_value=40.0, baseline=1.0, threshold=11.5, score=3.0,
        confidence=0.9, level=EvidenceLevel.INFERRED, severity=Severity.ERROR,
        window=(step - 50, step), rank=0, layer_index=layer,
        confidence_class=ConfidenceClass.STRONG,
        metadata={{"gradient_band": "CONFIRMED", "population_layers": 4}},
    )

# Process A sees only the first observation of the fault: one layer, one step.
engine = EpisodeEngine()
engine.consume([anomaly(80, 0)], run_id)
open_episodes = engine.all_episodes()
assert open_episodes, "process A produced no episode"
assert open_episodes[0].state == "SUSPECT"
assert open_episodes[0].confirmed_at_step is None

store = EpisodeStore({root!r})
for episode in open_episodes:
    store.append(episode)

state = capture_analysis_state(
    run_id, processed_through_step=80, layer_series=series, episodes=open_episodes,
    detector_versions={{"grad.robust_zscore": "3.0"}},
    fusion_rule_version="phase1.3-1.0", episode_rule_version="phase1.3-1.0",
)
AnalysisStateStore({root!r}).append(state)
json.dump({{
    "episode_id": open_episodes[0].episode_id,
    "onset": open_episodes[0].candidate_onset_step,
    "state": open_episodes[0].state,
    "signal_count": open_episodes[0].signal_count,
}}, open({out!r}, "w"))
print("process A persisted", open_episodes[0].episode_id)
"""

CONTINUE = """
import sys, json
sys.path.insert(0, {src!r})
from training_debugger.detectors.base import Anomaly, ConfidenceClass
from training_debugger.schema.enums import EvidenceLevel, Severity
from training_debugger.episodes import EpisodeEngine
from training_debugger.persistence import EpisodeStore, AnalysisStateStore

run_id = "run-restart"

def anomaly(step, layer):
    return Anomaly(
        detector="grad.robust_zscore", detector_version="3.0",
        metric_name="grad.layer_norm", first_abnormal_step=step,
        observed_value=40.0, baseline=1.0, threshold=11.5, score=3.0,
        confidence=0.9, level=EvidenceLevel.INFERRED, severity=Severity.ERROR,
        window=(step - 50, step), rank=0, layer_index=layer,
        confidence_class=ConfidenceClass.STRONG,
        metadata={{"gradient_band": "CONFIRMED", "population_layers": 4}},
    )

store = EpisodeStore({root!r})
open_episodes = store.open_episodes(run_id)
assert open_episodes, "process B found no open episode to resume"

state = AnalysisStateStore({root!r}).load(run_id)
assert state is not None, "process B found no analysis state"

engine = EpisodeEngine()
restored = engine.resume(open_episodes)
assert restored == len(open_episodes)

# Continue with the remaining observations of the same fault.
engine.consume([anomaly(80, layer) for layer in (1, 2, 3)], run_id)
engine.finish(run_id, final_step=400)
episodes = engine.all_episodes()

json.dump({{
    "restored": restored,
    "episode_count": len(episodes),
    "episode_id": episodes[0].episode_id,
    "onset": episodes[0].candidate_onset_step,
    "confirmed_at": episodes[0].confirmed_at_step,
    "signal_count": episodes[0].signal_count,
    "state": episodes[0].state,
    "outcome": episodes[0].outcome,
    "baseline_refs": {{k: v.reference for k, v in state.baselines.items()}},
    "state_revision": state.revision,
}}, open({out!r}, "w"))
print("process B resumed", restored, "episode(s)")
"""


def test_cross_process_restart_readback(tmp_path):
    """Process A persists a SUSPECT episode; process B resumes and confirms *it*."""
    src = str(ROOT / "src")
    root = tmp_path / "store"
    written = tmp_path / "a.json"
    continued = tmp_path / "b.json"

    a = tmp_path / "a.py"
    a.write_text(WRITE.format(src=src, root=str(root), out=str(written)), encoding="utf-8")
    result = subprocess.run([sys.executable, str(a)], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]

    b = tmp_path / "b.py"
    b.write_text(CONTINUE.format(src=src, root=str(root), out=str(continued)), encoding="utf-8")
    result = subprocess.run([sys.executable, str(b)], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]

    first = json.loads(written.read_text())
    second = json.loads(continued.read_text())

    assert second["episode_id"] == first["episode_id"], (
        "a restart created a new episode instead of resuming the persisted one"
    )
    assert second["onset"] == first["onset"] == 80
    # ``finish()`` closes a confirmed episode, so the terminal state is CLOSED
    # carrying the CONFIRMED_ANOMALY outcome -- not "CONFIRMED" left open.
    assert second["outcome"] == "CONFIRMED_ANOMALY"
    assert second["confirmed_at"] is not None
    assert second["signal_count"] == 4, "all four observations must be counted once"
    assert second["episode_count"] == 1, "a restart must not create a duplicate episode"
    assert second["baseline_refs"], "the protected baseline must survive the restart"
    for reference in second["baseline_refs"].values():
        assert reference == pytest.approx(1.0, rel=0.05), (
            "the restarted baseline must be the pre-fault reference, not a fresh one"
        )


def test_resume_ignores_closed_episodes():
    from training_debugger.episodes import EpisodeOutcome, EpisodeState

    closed = AnomalyEpisode(
        episode_id="ep_closed", run_id="run", detector_id="d", detector_version="1",
        rule_version="v", anomaly_type="GRADIENT_EXPLOSION", signal_key="k",
        state=EpisodeState.CLOSED.value, outcome=EpisodeOutcome.CONFIRMED_ANOMALY.value,
        candidate_onset_step=10, confirmed_at_step=12,
    )
    engine = EpisodeEngine()
    assert engine.resume([closed]) == 0
    assert engine.all_episodes() == []
