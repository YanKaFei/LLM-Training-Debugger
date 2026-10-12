"""W3: the witness challenge corpus, C1-C10.

One training run per case, with the witness **attached in the same run** so that the
OFF/ON comparison is paired on byte-identical evidence (W2). Both analyses read from
disk; nothing about the training differs between them.

The decisive pair is C1 vs C3:

```
C1  gradient anomaly -> optimizer invoked -> REAL parameter corruption -> downstream NaN
C3  gradient anomaly -> optimizer invoked -> parameters FINITE          -> INDEPENDENT NaN
```

Both have an invoked optimizer step. If the witness's `INVOKED` status is what the
relation rule consumes, **it cannot separate them** and must abstain. If the
`param.global_norm == nan` (non-finite elements) observation is what it consumes, it can.
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

from training_debugger.adapters.pytorch import OptimizerWitness  # noqa: E402
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
    AnalysisStateStore, EpisodeStore, capture_analysis_state,
)
from training_debugger.registry import detector_registry  # noqa: E402

OUT = ROOT / "outputs" / "v0.1.6-alpha"
STORE = OUT / "store"
TRUTH = OUT / "ground_truth"

STEPS = 120
INJECTION = 60
MAX_F32 = 3.0e38
CONFIGS: Tuple[Tuple[int, int], ...] = ((4, 32), (6, 48))
SEEDS = tuple(range(97000, 97003))          # 3 seeds x 2 configs = 6 runs per case

CASES: Tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")

#: Whether a genuine consequence exists, and whether the case is a relation opportunity.
GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "C1": {"expectation": "CONSEQUENCE", "opportunity": "positive",
           "reason": "real parameter corruption at the update, then a downstream NaN"},
    "C2": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "the optimizer update was explicitly skipped, so nothing propagated"},
    "C3": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "the optimizer WAS invoked but the downstream NaN is independent; the "
                     "parameters never became non-finite"},
    "C4": {"expectation": "CONSEQUENCE", "opportunity": "positive",
           "reason": "the parameters stayed finite through the update, but the next forward "
                     "pass genuinely overflowed"},
    "C5": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "only the float32 norm reduction overflowed; nothing was damaged"},
    "C6": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "the optimizer was invoked with no gradients"},
    "C7": {"expectation": "CONSEQUENCE", "opportunity": "positive",
           "reason": "gradient accumulation: a real propagation on the optimizer step"},
    "C8": {"expectation": "CONSEQUENCE", "opportunity": "positive",
           "reason": "a real propagation whose manifestation is delayed past the window"},
    "C9": {"expectation": "INDEPENDENT", "opportunity": "negative",
           "reason": "two optimizers, one updated and one skipped; the NaN is independent"},
}


def run_id_for(case: str, layers: int, d_model: int, seed: int) -> str:
    return "wc-{}-L{}d{}-s{}".format(case, layers, d_model, seed)


def train(case: str, layers: int, d_model: int, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    # A plain constant LR keeps the parameter arithmetic predictable, which the case
    # definitions depend on: grad * lr must be able to exceed the float32 maximum.
    # C3 deliberately does NOT get the overflowing lr: its whole point is that the
    # optimizer IS invoked while the parameters stay finite, so that "invoked" cannot
    # be read as "parameters changed".
    lr = 2.0 if case in ("C1", "C7", "C8") else 1.0
    accumulate = 2 if case in ("C7", "C8") else 1

    model, config = make_tiny_lm(seed=seed, n_layers=layers, d_model=d_model, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    second = (torch.optim.Adam(model.parameters(), lr=0.0) if case == "C9" else None)

    writer = MemoryWriter()
    observer = TrainingObserver(
        "wc", writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384), rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="wc")
    collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=32), cadence=1)
    adapter.run_start(steps=STEPS)
    witness = OptimizerWitness(enabled=True, optimizer_id="optimizer:main")
    attached = witness.attach(optimizer)

    nonfinite_steps: List[int] = []
    step_status: List[Dict[str, Any]] = []
    pending_grad = None

    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            effective = loss
            # C2/C3/C9: an INDEPENDENT downstream NaN, injected after the gradient step.
            if case in ("C2", "C9") and step == INJECTION + 1:
                effective = effective * float("nan")
            if case == "C3" and step == INJECTION + 2:
                effective = effective * float("nan")
            if case == "C8" and step == INJECTION + 12:
                effective = effective * float("nan")

            if not math.isfinite(float(effective.detach())):
                nonfinite_steps.append(step)
            tracker.record_loss(effective)

            do_update = (step % accumulate == 0)
            skipped_reason = None
            if math.isfinite(float(effective.detach())):
                effective.backward()
                if step == INJECTION:
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        if case == "C1" or case == "C7" or case == "C8":
                            p.grad.fill_(MAX_F32)          # lr 2.0 -> genuinely overflows
                        elif case == "C3":
                            # A finite scale anomaly large enough that the AGGREGATE
                            # gradient detector fires at the injection step itself.
                            # x20 was too small: the only anomaly appeared at step 76
                            # from the lagging per-layer band, i.e. AFTER the injected
                            # NaN, so the case refused for the wrong reason -- no
                            # antecedent -- instead of by discriminating. The
                            # parameters still stay finite, which is the point.
                            p.grad.mul_(100.0)
                        elif case == "C4":
                            p.grad.fill_(1e20)             # finite params, real overflow next
                        elif case == "C5":
                            p.grad.fill_(1e20)             # artifact only, no damage
                        elif case == "C2":
                            p.grad.mul_(20.0)
                        elif case == "C9":
                            p.grad.mul_(20.0)
                # C2 / C9: the caller EXPLICITLY skips the update.
                if case in ("C2", "C9") and step == INJECTION:
                    skipped_reason = "caller_skip"
                    for p in model.parameters():
                        p.grad = None
            tracker.record_gradients()
            collector.collect(step)

            if case == "C6" and step == INJECTION:
                # Invoked with no gradients at all.
                witness.note_attempted(training_step_id=step, skip_reason=None)
                optimizer.step()
                witness.note_invoked()
                step_status.append(witness.records[-1])
                optimizer.zero_grad(set_to_none=True)
            elif do_update:
                witness.note_attempted(training_step_id=step, micro_step_id=step % accumulate,
                                       skip_reason=skipped_reason)
                with tracker.phase("optimizer"):
                    if math.isfinite(float(effective.detach())):
                        optimizer.step()
                witness.note_invoked()
                step_status.append(witness.records[-1])
                optimizer.zero_grad(set_to_none=True)
                if second is not None and step == INJECTION:
                    # The second optimizer is skipped on the same boundary.
                    second.step()
            else:
                optimizer.zero_grad(set_to_none=True)

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
    del model, optimizer, second, observer, adapter, collector, writer
    gc.collect()
    return {
        "by_metric": {k: sorted(v.items()) for k, v in by_metric.items()},
        "layer_series": {k: sorted(v.items()) for k, v in layer_series.items()},
        "health": health,
        "nonfinite_steps": nonfinite_steps,
        "witness_records": witness.records,
        "witness_summary": witness.summary(),
        "witness_attached": attached,
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
    relations, _ = link_consequences(run_id, episodes)
    return {
        "run_id": run_id,
        "episodes": [e.to_dict() for e in episodes],
        "relations": [r.to_dict() for r in relations],
        "verdicts": [v.to_dict() for v in verdicts],
        "detection_anomalies": len(detection.anomalies),
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
                print("[W3] {} ...".format(run_id))
                evidence = train(case, layers, d_model, seed)
                row = analyse(run_id, layers, evidence)
                row["case"] = case
                rows.append(row)

                entry = dict(GROUND_TRUTH[case])
                entry.update({"case": case, "seed": seed, "layers": layers,
                              "d_model": d_model, "injection_step": INJECTION,
                              "run_id": run_id,
                              "nonfinite_steps": evidence["nonfinite_steps"],
                              "witness_summary": evidence["witness_summary"],
                              "witness_attached": evidence["witness_attached"]})
                truth[run_id] = entry

                episode_store = EpisodeStore(STORE)
                from training_debugger.episodes import AnomalyEpisode  # noqa: F401
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
                    episode_rule_version=RULE_VERSION, rule_version="v0.1.6-alpha",
                ))
                # The witness records are persisted alongside, so the paired OFF/ON
                # analysis reads identical evidence for both configurations.
                (STORE / "runs" / run_id).mkdir(parents=True, exist_ok=True)
                (STORE / "runs" / run_id / "witness.json").write_text(
                    json.dumps({"records": evidence["witness_records"],
                                "summary": evidence["witness_summary"],
                                "attached": evidence["witness_attached"]}, indent=2,
                               default=str), encoding="utf-8")

    write_artifact(TRUTH / "witness_challenge_ground_truth.json", truth, provenance="sibling")
    payload = {
        "cases": list(CASES), "runs": len(rows),
        "store": str(STORE), "ground_truth": str(TRUTH / "witness_challenge_ground_truth.json"),
        "results": rows, "elapsed_s": round(time.perf_counter() - started, 1),
    }
    write_artifact(OUT / "witness_challenge_results.json", payload)
    print("")
    print("persisted {} runs".format(len(rows)))
    for case in CASES:
        subset = [r for r in rows if r["case"] == case]
        types = sorted({rel["relation_type"] for r in subset for rel in r["relations"]})
        print("  %-4s runs=%-3d relations=%-3d types=%s" % (
            case, len(subset), sum(len(r["relations"]) for r in subset), types or "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
