"""W2: adversarial development challenge corpus.

The v0.1.3 rule scored precision 1.0 / recall 1.0 on the 50-run mixed corpus. W1 showed
that result rests on a **generation signature**: the rule discriminates on whether the
*antecedent* observation was non-finite, and the M1/M3 scenarios were generated so that
exactly that differed.

This corpus attacks that. The centrepiece is **A8, observational equivalence**: two cases
whose production-observable fields are identical while their true causal structure
differs. No diagnostic can separate them from those fields, so the correct output is
abstention. Anything else is invented certainty.

```
A1  gradient norm inf, update suppressed   -> no propagation possible
A2  gradient norm inf, then an INDEPENDENT non-finite value 1 step later
A3  finite gradient norm, downstream failure                   (M3 shape)
A4  genuine propagation DELAYED past the 2-step bound
A5  large gradient anomaly, no downstream failure
A6  standalone numerical fault, no gradient antecedent
A7  optimizer update skipped, then an unrelated numerical anomaly
A8  X: identical observations, propagated truth
    Y: identical observations, independent truth
```

DEVELOPMENT challenge data. NOT a blind holdout.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
from artifacts import write_artifact  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tiny_model import make_tiny_lm, synthetic_batch  # noqa: E402

from training_debugger.adapters.pytorch import (  # noqa: E402
    LayerMetricCollector, LayerSelection, PyTorchTrainingAdapter,
)
from training_debugger.detectors import DetectorConfig, DetectorRunner  # noqa: E402
from training_debugger.episodes import RULE_VERSION, EpisodeEngine  # noqa: E402
from training_debugger.fusion import FUSION_RULE_VERSION, fuse  # noqa: E402
from training_debugger.fusion.quality import DataQuality  # noqa: E402
from training_debugger.incidents.relations import (  # noqa: E402
    RELATION_RULE_VERSION, link_consequences,
)
from training_debugger.observer import (  # noqa: E402
    MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver,
)
from training_debugger.persistence import (  # noqa: E402
    AnalysisStateStore, EpisodeStore, IncidentStore, capture_analysis_state,
)
from training_debugger.registry import detector_registry  # noqa: E402

OUT = ROOT / "outputs" / "alpha" / "adversarial" / "v015"
STORE = OUT / "store"
TRUTH = OUT / "ground_truth"

STEPS = 160
INJECTION = 80
DELAYED_AT = 86                      # A4: past the 2-step propagation bound
CONFIGS: Tuple[Tuple[int, int], ...] = ((4, 32),)
SEEDS = tuple(range(95000, 95005))   # 5 seeds per case

CASES: Tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8X", "A8Y")

#: Cases whose training environment this project cannot actually realise.
UNSUPPORTED: Dict[str, str] = {}


def run_id_for(case: str, layers: int, d_model: int, seed: int) -> str:
    return "adv-{}-L{}d{}-s{}".format(case, layers, d_model, seed)


def train(case: str, layers: int, d_model: int, seed: int) -> Dict[str, Any]:
    """One real training run implementing the case's injection semantics."""
    torch.manual_seed(seed)
    model, config = make_tiny_lm(seed=seed, n_layers=layers, d_model=d_model, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "adv", writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384), rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="adv")
    collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=32), cadence=1)
    adapter.run_start(steps=STEPS)
    nonfinite_steps: List[int] = []
    suppressed = False
    collapsed = False

    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            effective = loss

            # A2 / A8Y: an INDEPENDENT non-finite value, injected one step after the
            # gradient anomaly, with nothing propagating from the gradient.
            if case in ("A2", "A8Y") and step == INJECTION + 1:
                effective = effective * float("nan")
            # A4: a genuine downstream failure, but delayed past the bound.
            if case == "A4" and step == DELAYED_AT:
                effective = effective * float("nan")
            # A3 / A7: an unrelated later numerical fault.
            if case in ("A3", "A7") and step == INJECTION + 40:
                effective = effective * float("nan")
            # A6: standalone, nothing upstream.
            if case == "A6" and step == INJECTION:
                effective = effective * float("nan")

            if not math.isfinite(float(effective.detach())):
                nonfinite_steps.append(step)
            tracker.record_loss(effective)

            if math.isfinite(float(effective.detach())) and not collapsed:
                effective.backward()
                if step == INJECTION:
                    if case in ("A1", "A2", "A8Y"):
                        # Gradient becomes non-finite in the RECORDED norm, but the
                        # update is suppressed, so nothing can propagate.
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.fill_(1e28)
                        suppressed = True
                    elif case in ("A3", "A7", "A5"):
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.mul_(20.0)
                    elif case in ("A4", "A8X"):
                        # A genuine propagation chain.
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.fill_(1e28)
            tracker.record_gradients()
            collector.collect(step)

            with tracker.phase("optimizer"):
                if math.isfinite(float(effective.detach())) and not collapsed:
                    try:
                        if suppressed:
                            # A1/A2/A8Y/A7: the update is a NO-OP. The gradient was
                            # observed as non-finite, but the parameters never see it.
                            optimizer.zero_grad(set_to_none=True)
                            suppressed = False
                        else:
                            optimizer.step()
                    except Exception:
                        collapsed = True
            optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()

    adapter.run_end()
    observer.flush(timeout=60.0)
    health = observer.health()
    observer.close()
    by_metric: Dict[str, Dict[int, float]] = {}
    layer_series: Dict[int, Dict[int, float]] = {}
    for event in writer.events:
        if event.metric_name is None or event.step is None or event.metric_value is None:
            continue
        if event.metric_name == "grad.layer_norm" and event.layer_index is not None:
            layer_series.setdefault(int(event.layer_index), {})[event.step] = event.metric_value
        else:
            by_metric.setdefault(str(event.metric_name), {})[event.step] = event.metric_value
    del model, optimizer, observer, adapter, collector, writer
    gc.collect()
    return {
        "by_metric": {k: sorted(v.items()) for k, v in by_metric.items()},
        "layer_series": {k: sorted(v.items()) for k, v in layer_series.items()},
        "health": health,
        "nonfinite_steps": nonfinite_steps,
    }


def analyse(run_id: str, layers: int, evidence: Dict[str, Any]) -> Dict[str, Any]:
    detection = DetectorRunner(DetectorConfig()).run(
        run_id, evidence["by_metric"], rank=0, layer_series=evidence["layer_series"],
        module_paths={}, population_layers=layers, population_ranks=1,
        coverage={"step_count": STEPS, "step_min": 1, "step_max": STEPS},
    )
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    engine.consume(detection.anomalies, run_id)
    engine.finish(run_id, final_step=STEPS)
    episodes = engine.all_episodes()
    quality = DataQuality.from_observer_health(evidence["health"])
    verdicts = fuse(run_id, episodes, data_quality=quality)
    relations, linked = link_consequences(run_id, episodes)
    return {
        "run_id": run_id,
        "episodes": [
            {"episode_id": e.episode_id, "anomaly_type": e.anomaly_type,
             "metric_name": e.metric_name,
             "candidate_onset_step": e.candidate_onset_step,
             "confirmed_at_step": e.confirmed_at_step,
             "outcome": e.outcome, "affected_layers": sorted(e.affected_layers or [])}
            for e in episodes
        ],
        "relations": [
            {"relation_id": r.relation_id, "relation_type": r.relation_type,
             "source": r.source_incident_id, "target": r.target_incident_id,
             "gap_steps": r.gap_steps, "tier": (r.evidence or {}).get("tier"),
             "antecedent_metric": (r.evidence or {}).get("antecedent_metric"),
             "antecedent_is_nonfinite": (r.evidence or {}).get("antecedent_is_nonfinite")}
            for r in relations
        ],
        "verdicts": [v.verdict for v in verdicts],
        "nonfinite_steps": evidence["nonfinite_steps"],
    }


#: Ground truth per case. `expectation` is the *relation-opportunity* class.
GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "A1": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "the gradient norm overflows but the update is suppressed, so no "
                     "value propagated; any consequence claim is fabricated"},
    "A2": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "the downstream non-finite value was injected independently one "
                     "step later; the gradient could not have caused it"},
    "A3": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "a finite scale anomaly is not a mechanism"},
    "A4": {"expectation": "CONSEQUENCE", "opportunity": "positive",
           "reason": "a genuine propagation, delayed past the 2-step bound"},
    "A5": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "no downstream failure occurred at all"},
    "A6": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "standalone non-finite value with no antecedent"},
    "A7": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "the update was skipped and a later unrelated fault followed"},
    "A8X": {"expectation": "CONSEQUENCE", "opportunity": "positive",
            "reason": "genuine propagation; observations identical to A8Y"},
    "A8Y": {"expectation": "INDEPENDENT", "opportunity": "negative",
            "reason": "independent fault; observations identical to A8X"},
}


def main() -> int:
    STORE.mkdir(parents=True, exist_ok=True)
    TRUTH.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    truth: Dict[str, Any] = {}
    signatures: Dict[str, Dict[str, Any]] = {}

    for case in CASES:
        if case in UNSUPPORTED:
            continue
        for layers, d_model in CONFIGS:
            for seed in SEEDS:
                run_id = run_id_for(case, layers, d_model, seed)
                print("[ADV] {} ...".format(run_id))
                evidence = train(case, layers, d_model, seed)
                row = analyse(run_id, layers, evidence)
                row["case"] = case
                results.append(row)

                # The production-observable signature: only fields a diagnostic can
                # read. Ground truth and case id are deliberately NOT included.
                signatures[run_id] = {
                    "episodes": [
                        {k: e[k] for k in ("anomaly_type", "metric_name",
                                           "candidate_onset_step", "affected_layers")}
                        for e in row["episodes"]
                    ],
                    "nonfinite_steps": row["nonfinite_steps"],
                }

                entry = dict(GROUND_TRUTH[case])
                entry.update({"case": case, "seed": seed, "layers": layers,
                              "d_model": d_model, "injection_step": INJECTION})
                truth[run_id] = entry

                # -- persist through the production stores -------------------
                episode_store = EpisodeStore(STORE)
                from training_debugger.episodes import AnomalyEpisode
                engine = EpisodeEngine(
                    enabled_detectors=detector_registry()["enabled_detector_ids"])
                detection = DetectorRunner(DetectorConfig()).run(
                    run_id, evidence["by_metric"], rank=0,
                    layer_series=evidence["layer_series"], module_paths={},
                    population_layers=layers, population_ranks=1,
                    coverage={"step_count": STEPS, "step_min": 1, "step_max": STEPS},
                )
                engine.consume(detection.anomalies, run_id)
                engine.finish(run_id, final_step=STEPS)
                episodes = engine.all_episodes()
                for episode in episodes:
                    episode_store.append(episode)
                relations, _ = link_consequences(run_id, episodes)
                episode_store.append_relations(run_id, relations)
                AnalysisStateStore(STORE).append(capture_analysis_state(
                    run_id, processed_through_step=STEPS,
                    layer_series=evidence["layer_series"], config=DetectorConfig(),
                    episodes=episodes,
                    detector_versions={str(a.detector): str(a.detector_version)
                                       for a in detection.anomalies},
                    fusion_rule_version=FUSION_RULE_VERSION,
                    episode_rule_version=RULE_VERSION, rule_version="v0.1.5-alpha",
                ))

    write_artifact(TRUTH / "adversarial_ground_truth.json", truth, provenance="sibling")
    (OUT / "observable_signatures.json").write_text(
        json.dumps(signatures, indent=2), encoding="utf-8")
    payload = {
        "cases": list(CASES),
        "unsupported": UNSUPPORTED,
        "runs": len(results),
        "store": str(STORE),
        "ground_truth": str(TRUTH / "adversarial_ground_truth.json"),
        "results": results,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    write_artifact(OUT / "adversarial_results.json", payload)
    print("")
    print("persisted {} runs".format(len(results)))
    for case in CASES:
        rows = [r for r in results if r["case"] == case]
        if not rows:
            continue
        types = set()
        for r in rows:
            for rel in r["relations"]:
                types.add(rel["relation_type"])
        print("  %-4s runs=%-3d relations=%-3d types=%s" % (
            case, len(rows), sum(len(r["relations"]) for r in rows),
            sorted(types) or "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
