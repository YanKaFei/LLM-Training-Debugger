"""S8: cross-process restart smoke, driven by the real stores."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_suffix(".json")
WORKER = ROOT / "tests" / "experiments" / "restart_worker.py"
WORKER.write_text('''
import sys, json
sys.path.insert(0, {src!r})
from training_debugger.detectors.base import Anomaly, ConfidenceClass
from training_debugger.schema.enums import EvidenceLevel, Severity
from training_debugger.episodes import EpisodeEngine
from training_debugger.persistence import EpisodeStore, AnalysisStateStore, capture_analysis_state

run_id = "s8-restart"
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

engine = EpisodeEngine()
engine.consume([anomaly(80, 0)], run_id)
open_episodes = engine.all_episodes()
store = EpisodeStore({root!r})
for episode in open_episodes:
    store.append(episode)
state = capture_analysis_state(
    run_id, processed_through_step=80, layer_series=series, episodes=open_episodes,
    detector_versions={{"grad.robust_zscore": "3.0"}},
    fusion_rule_version="phase1.3-1.0", episode_rule_version="phase1.3-1.0",
)
AnalysisStateStore({root!r}).append(state)
assert open_episodes and open_episodes[0].state == "SUSPECT"
''')

CONTINUE = ROOT / "tests" / "experiments" / "restart_continue.py"
CONTINUE.write_text('''
import sys, json
sys.path.insert(0, {src!r})
from training_debugger.detectors.base import Anomaly, ConfidenceClass
from training_debugger.schema.enums import EvidenceLevel, Severity
from training_debugger.episodes import EpisodeEngine
from training_debugger.persistence import EpisodeStore, AnalysisStateStore

run_id = "s8-restart"

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
state = AnalysisStateStore({root!r}).load(run_id)
engine = EpisodeEngine()
restored = engine.resume(open_episodes)
engine.consume([anomaly(80, layer) for layer in (1, 2, 3)], run_id)
engine.finish(run_id, final_step=400)
episodes = engine.all_episodes()
out = {{
    "restored": restored, "episode_count": len(episodes),
    "episode_id": episodes[0].episode_id, "onset": episodes[0].candidate_onset_step,
    "signal_count": episodes[0].signal_count, "outcome": episodes[0].outcome,
    "baseline_count": len(state.baselines),
    "baseline_refs": {{k: v.reference for k, v in state.baselines.items()}},
}}
assert episodes[0].candidate_onset_step == 80, "onset must survive the restart"
assert restored == 1, "exactly one episode must be resumed"
assert len(episodes) == 1, "a restart must not duplicate the episode"
assert state is not None and state.baselines, "the protected baseline must survive"
json.dump(out, open({out!r}, "w"))
''')

src = str(ROOT / "src")
root = str(ROOT / "outputs" / "phase1-12c" / "s8_store")
Path(root).mkdir(parents=True, exist_ok=True)
out = str(SCRIPT)
for script, args in ((WORKER, []), (CONTINUE, [])):
    code = script.read_text().format(src=src, root=root, out=out)
    path = script.with_suffix(".py")
    path.write_text(code)
    completed = subprocess.run([sys.executable, str(path)], capture_output=True,
                               text=True, timeout=600, cwd=str(ROOT))
    if completed.returncode != 0:
        print(completed.stderr[-1200:])
        raise SystemExit(completed.returncode)
print(json.loads(Path(out).read_text()))
