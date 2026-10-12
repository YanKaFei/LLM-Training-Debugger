"""Workstream B: C3 / C4 / C11 / C12 with injections AFTER detector readiness.

Workstream A established the root cause of the "11-step latency":

```
baseline_floor = warmup_steps = 50   +   minimum_history = 20
->  the protected baseline has no usable ratio until step 70
->  an injection at step 60 cannot be scored until step 70
->  with an injection at step 85 it fires at step 85: latency ZERO
```

The latency was **baseline readiness**, not smoothing. v0.1.6's C3 injected at step 60,
inside that window, so no gradient antecedent could ever be observed in time and the case
was refused for the wrong reason. Every case here injects at ``INJECTION = 90``.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tiny_model import make_tiny_lm, synthetic_batch  # noqa: E402

from training_debugger.adapters.pytorch import (  # noqa: E402
    LayerMetricCollector, LayerSelection, OptimizerWitness, PyTorchTrainingAdapter,
)
from training_debugger.detectors import DetectorConfig, DetectorRunner  # noqa: E402
from training_debugger.episodes import RULE_VERSION, EpisodeEngine  # noqa: E402
from training_debugger.fusion import FUSION_RULE_VERSION, fuse  # noqa: E402
from training_debugger.fusion.quality import DataQuality  # noqa: E402
from training_debugger.incidents.relations import link_consequences  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver,
)
from training_debugger.persistence import (  # noqa: E402
    AnalysisStateStore, EpisodeStore, capture_analysis_state,
)
from training_debugger.registry import detector_registry  # noqa: E402

OUT = ROOT / "outputs" / "v0.1.7-alpha"
STORE = OUT / "store"
TRUTH = OUT / "ground_truth"

STEPS = 200
#: Past the detector's readiness boundary (step 70). Injecting before it makes a case
#: untestable rather than hard, which is what invalidated v0.1.6's C3.
INJECTION = 90
MAX_F32 = 3.0e38
CONFIGS: Tuple[Tuple[int, int], ...] = ((4, 32), (6, 48))
SEEDS = tuple(range(98000, 98004))          # 4 seeds x 2 configs = 8 runs per case

CASES: Tuple[str, ...] = ("C3", "C4", "C11", "C12")

GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "C3": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "the optimizer IS invoked and the gradient evidence PRECEDES the "
                     "downstream NaN, but the NaN is independently injected and the "
                     "parameters never became non-finite"},
    "C4": {"expectation": "CONSEQUENCE", "opportunity": "positive",
           "reason": "the update leaves the parameters finite but enormous; the NEXT "
                     "forward pass genuinely overflows"},
    "C11": {"expectation": "INDEPENDENT", "opportunity": "negative",
            "reason": "an early suspicious gradient that recovers with no failure at all"},
    "C12": {"expectation": "INDEPENDENT", "opportunity": "negative",
            "reason": "gradient evidence precedes the downstream NaN, the optimizer is "
                      "invoked, and the two are still unrelated"},
}


def run_id_for(case: str, layers: int, d_model: int, seed: int) -> str:
    return "tl-{}-L{}d{}-s{}".format(case, layers, d_model, seed)


def train(case: str, layers: int, d_model: int, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    lr = 1.0
    model, config = make_tiny_lm(seed=seed, n_layers=layers, d_model=d_model, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "tl", writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384), rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="tl")
    collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=32), cadence=1)
    adapter.run_start(steps=STEPS)
    witness = OptimizerWitness(enabled=True, optimizer_id="optimizer:main")
    attached = witness.attach(optimizer)

    nonfinite_steps: List[int] = []
    param_nonfinite_steps: List[int] = []
    collapse_source: List[str] = []

    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            effective = loss
            pre_finite = math.isfinite(float(effective.detach()))

            # C3 / C12: an INDEPENDENT NaN, injected two steps AFTER the gradient
            # evidence at the injection step, so the observed order is
            # gradient-anomaly -> invoked update -> independent NaN.
            if case in ("C3", "C12") and step == INJECTION + 2:
                effective = effective * float("nan")
                collapse_source.append("injected_nan")

            if not math.isfinite(float(effective.detach())):
                nonfinite_steps.append(step)
            tracker.record_loss(effective)

            if pre_finite:
                effective.backward()
                if step == INJECTION:
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        if case in ("C3", "C12"):
                            # x30 is the magnitude C11 measured as sufficient to fire the
                            # aggregate detector at this very step while leaving every
                            # parameter FINITE. x100 was used first and it genuinely
                            # corrupted the parameters (param non-finite in 8/8 runs),
                            # which turned the case into a real propagation and destroyed
                            # the thing it was meant to test.
                            p.grad.mul_(30.0)
                        elif case == "C4":
                            # Finite but enormous: the update leaves the parameters
                            # finite, and the NEXT forward pass overflows naturally.
                            p.grad.fill_(1e20)
                        elif case == "C11":
                            p.grad.mul_(30.0)
                # C11: the suspicious gradient recovers -- the update still runs normally.
            tracker.record_gradients()
            collector.collect(step)

            invoked = False
            if pre_finite:
                witness.note_attempted(training_step_id=step)
                with tracker.phase("optimizer"):
                    try:
                        optimizer.step()
                        invoked = True
                    except Exception:
                        pass
                witness.note_invoked()
            optimizer.zero_grad(set_to_none=True)

            params_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
            if not params_finite:
                param_nonfinite_steps.append(step)
            tracker.record_parameters()
            tracker.record_learning_rate()

    adapter.run_end()
    observer.flush(timeout=60.0)
    health = observer.health()
    observer.close()
    witness.detach()

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
        "param_nonfinite_steps": param_nonfinite_steps,
        "witness_records": witness.records,
        "witness_summary": witness.summary(),
        "witness_attached": attached,
    }


def main() -> int:
    STORE.mkdir(parents=True, exist_ok=True)
    TRUTH.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    truth: Dict[str, Any] = {}

    for case in CASES:
        for layers, d_model in CONFIGS:
            for seed in SEEDS:
                run_id = run_id_for(case, layers, d_model, seed)
                print("[B] {} ...".format(run_id))
                evidence = train(case, layers, d_model, seed)
                detection = DetectorRunner(DetectorConfig()).run(
                    run_id, evidence["by_metric"], rank=0,
                    layer_series=evidence["layer_series"], module_paths={},
                    population_layers=layers, population_ranks=1,
                    coverage={"step_count": STEPS, "step_min": 1, "step_max": STEPS},
                )
                engine = EpisodeEngine(
                    enabled_detectors=detector_registry()["enabled_detector_ids"])
                engine.consume(detection.anomalies, run_id)
                engine.finish(run_id, final_step=STEPS)
                episodes = engine.all_episodes()
                quality = DataQuality.from_observer_health(evidence["health"])
                verdicts = fuse(run_id, episodes, data_quality=quality)
                relations, _ = link_consequences(run_id, episodes)

                episode_store = EpisodeStore(STORE)
                for episode in episodes:
                    episode_store.append(episode)
                episode_store.append_relations(run_id, relations)
                AnalysisStateStore(STORE).append(capture_analysis_state(
                    run_id, processed_through_step=STEPS,
                    layer_series=evidence["layer_series"], config=DetectorConfig(),
                    episodes=episodes,
                    detector_versions={str(a.detector): str(a.detector_version)
                                       for a in detection.anomalies},
                    fusion_rule_version=FUSION_RULE_VERSION,
                    episode_rule_version=RULE_VERSION, rule_version="v0.1.7-alpha",
                ))
                (STORE / "runs" / run_id).mkdir(parents=True, exist_ok=True)
                (STORE / "runs" / run_id / "witness.json").write_text(
                    json.dumps({"records": evidence["witness_records"],
                                "summary": evidence["witness_summary"],
                                "attached": evidence["witness_attached"]}, indent=2,
                               default=str), encoding="utf-8")

                # The observed timeline: when did gradient evidence first appear,
                # and when did the downstream non-finite value appear?
                grad_fires = sorted(a.first_abnormal_step for a in detection.anomalies
                                    if str(a.detector) == "grad.robust_zscore")
                finite_fires = sorted(a.first_abnormal_step for a in detection.anomalies
                                      if str(a.detector) == "numerics.finiteness")
                rows.append({
                    "run_id": run_id, "case": case,
                    "episodes": [e.to_dict() for e in episodes],
                    "relations": [r.to_dict() for r in relations],
                    "gradient_first_fire": grad_fires[0] if grad_fires else None,
                    "nonfinite_first_fire": finite_fires[0] if finite_fires else None,
                    "nonfinite_steps": evidence["nonfinite_steps"],
                    "param_nonfinite_steps": evidence["param_nonfinite_steps"],
                })
                entry = dict(GROUND_TRUTH[case])
                entry.update({"case": case, "seed": seed, "run_id": run_id,
                              "injection_step": INJECTION,
                              "gradient_first_fire": rows[-1]["gradient_first_fire"],
                              "nonfinite_first_fire": rows[-1]["nonfinite_first_fire"],
                              "param_nonfinite_steps": evidence["param_nonfinite_steps"],
                              "witness_summary": evidence["witness_summary"]})
                truth[run_id] = entry

    (TRUTH / "timeline_ground_truth.json").write_text(
        json.dumps(truth, indent=2, default=str), encoding="utf-8")
    payload = {"cases": list(CASES), "injection_step": INJECTION, "runs": len(rows),
               "store": str(STORE), "results": rows,
               "elapsed_s": round(time.perf_counter() - started, 1)}
    (OUT / "timeline_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("")
    print("%-5s %-8s %-10s %-10s %-10s %s" % ("case", "grad1st", "nonfin1st", "inj", "paramNonfin", "relations"))
    for case in CASES:
        subset = [r for r in rows if r["case"] == case]
        g = [r["gradient_first_fire"] for r in subset if r["gradient_first_fire"]]
        n = [r["nonfinite_first_fire"] for r in subset if r["nonfinite_first_fire"]]
        pn = sum(1 for r in subset if r["param_nonfinite_steps"])
        types = sorted({rel["relation_type"] for r in subset for rel in r["relations"]})
        print("%-5s %-8s %-10s %-10s %-10d %s" % (
            case, min(g) if g else "-", min(n) if n else "-", INJECTION, pn, types or "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
