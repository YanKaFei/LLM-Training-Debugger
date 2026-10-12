"""Cost and detection quality of the layer-metric observability cadence.

`param.layer.norm` was 8 of 32 events per step -- 25 % of all event volume -- and is
consumed by **nothing**: it appears downstream only in the presentation sink's
`FORBIDDEN_PREFIXES`. `grad.layer.norm` by contrast feeds `grad.robust_zscore`.

The change emits at the rate the consumer needs: unconsumed observability metrics at
`observability_cadence`, detector inputs unchanged.

Two things must be shown, and the second matters more:

1. the cost falls
2. **detection does not change** -- measured against the paired corpus, where every
   injection step is known exactly
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tiny_model import make_tiny_lm, synthetic_batch  # noqa: E402

from training_debugger.adapters.pytorch import (  # noqa: E402
    LayerMetricCollector, LayerSelection, PyTorchTrainingAdapter,
)
from training_debugger.detectors import DetectorConfig, DetectorRunner  # noqa: E402
from training_debugger.episodes import EpisodeEngine  # noqa: E402
from training_debugger.fusion import fuse  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver,
)
from training_debugger.registry import detector_registry  # noqa: E402
from training_debugger.storage import DuckDBReader  # noqa: E402

OUT = ROOT / "outputs" / "v0.1.8-alpha"
PAIRED = OUT / "paired-store"
STEPS = 120
WARMUP = 20
REPEATS = 7
CADENCES = (1, 5)


def timed(cadence: int) -> Dict[str, Any]:
    torch.manual_seed(909)
    model, cfg = make_tiny_lm(seed=909, n_layers=4, d_model=32, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "cad", writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384),
        rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="cad")
    collector = LayerMetricCollector(
        model, observer, selection=LayerSelection(top_k=32), cadence=1,
        observability_cadence=cadence,
    )
    adapter.run_start(steps=STEPS)
    durations: List[float] = []
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8,
                               vocab_size=cfg.vocab_size, seed=step)
        started = time.perf_counter()
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            tracker.record_loss(loss)
            loss.backward()
            tracker.record_gradients()
            collector.collect(step)
            with tracker.phase("optimizer"):
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()
        durations.append(time.perf_counter() - started)
    adapter.run_end()
    observer.flush(timeout=60.0)
    health = observer.health()["observer"]
    observer.close()
    return {
        "median_step_s": statistics.median(durations[WARMUP:]),
        "events": int(health.get("emitted_total", 0) or 0),
        "emit_us": health.get("emit_time_total_ns", 0) / max(1, health.get("emitted_total", 1)) / 1000.0,
    }


def detection_quality(cadence: int) -> Dict[str, Any]:
    """Re-detect the stored paired corpus with grad.layer_norm resampled.

    Only `grad.layer.norm` is resampled here: it is the detector input, so this is the
    conservative test. If quality holds for the detector's own input it certainly holds
    for the unconsumed observability metrics.
    """
    if not PAIRED.exists():
        return {}
    reader = DuckDBReader(PAIRED)
    truth = json.loads((OUT / "paired_ground_truth.json").read_text())
    outcomes: List[Dict[str, Any]] = []
    try:
        for entry in truth.values():
            if entry["fault"] == "none":
                continue
            run_id = entry["run_b"]
            table = reader.events(run_id)
            by_metric: Dict[str, Dict[int, float]] = {}
            layers: Dict[int, Dict[int, float]] = {}
            for name, step, value, layer in zip(
                table.column("metric_name").to_pylist(),
                table.column("step").to_pylist(),
                table.column("metric_value").to_pylist(),
                table.column("layer_index").to_pylist(),
            ):
                if name is None or step is None or value is None:
                    continue
                if name == "grad.layer_norm" and layer is not None:
                    layers.setdefault(int(layer), {})[int(step)] = float(value)
                else:
                    by_metric.setdefault(str(name), {})[int(step)] = float(value)
            if cadence > 1:
                layers = {
                    index: {s: v for s, v in series.items() if s % cadence == 0}
                    for index, series in layers.items()
                }
            detection = DetectorRunner(DetectorConfig()).run(
                run_id, {k: sorted(v.items()) for k, v in by_metric.items()},
                rank=0,
                layer_series={k: sorted(v.items()) for k, v in layers.items()},
                module_paths={}, population_layers=4, population_ranks=1,
                coverage={"step_count": 150, "step_min": 1, "step_max": 150},
            )
            engine = EpisodeEngine(
                enabled_detectors=detector_registry()["enabled_detector_ids"])
            engine.consume(detection.anomalies, run_id)
            engine.finish(run_id, final_step=150)
            verdicts = fuse(run_id, engine.all_episodes())
            fires = sorted(a.first_abnormal_step for a in detection.anomalies
                           if str(a.detector) == "grad.robust_zscore")
            outcomes.append({
                "run": run_id, "fault": entry["fault"],
                "grad_first_fire": fires[0] if fires else None,
                "incidents": sum(1 for v in verdicts if v.verdict == "CONFIRMED_INCIDENT"),
            })
    finally:
        reader.close()
    latencies = [o["grad_first_fire"] - 90 for o in outcomes
                 if o["grad_first_fire"] is not None]
    return {
        "runs": len(outcomes),
        "fired": sum(1 for o in outcomes if o["grad_first_fire"] is not None),
        "incidents": sum(o["incidents"] for o in outcomes),
        "median_latency": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "signature": sorted((o["run"], o["grad_first_fire"], o["incidents"])
                            for o in outcomes),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"steps": STEPS, "repeats": REPEATS, "timing": {},
                              "detection": {}}
    for _ in range(2):
        for cadence in CADENCES:
            timed(cadence)

    samples: Dict[int, List[Dict[str, Any]]] = {c: [] for c in CADENCES}
    for repeat in range(REPEATS):
        order = list(CADENCES)
        if repeat % 2:
            order.reverse()
        for cadence in order:
            samples[cadence].append(timed(cadence))

    for cadence in CADENCES:
        rows = samples[cadence]
        report["timing"][str(cadence)] = {
            "median_step_s": statistics.median(r["median_step_s"] for r in rows),
            "events": rows[0]["events"],
            "events_per_step": rows[0]["events"] / (STEPS - WARMUP),
            "emit_us": statistics.median(r["emit_us"] for r in rows),
        }
        report["detection"][str(cadence)] = detection_quality(cadence)

    (OUT / "LAYER_CADENCE_AUDIT.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    base = report["timing"]["1"]["median_step_s"]
    print("%-12s %-12s %-10s %-12s %s" % (
        "cadence", "median ms", "events/step", "emit us/ev", "added vs cadence=1"))
    for cadence in CADENCES:
        row = report["timing"][str(cadence)]
        delta = ("baseline" if cadence == 1 else
                 "{:+.3f} ms".format((row["median_step_s"] - base) * 1e3))
        print("%-12s %-12.4f %-10.1f %-12.2f %s" % (
            cadence, row["median_step_s"] * 1e3, row["events_per_step"],
            row["emit_us"], delta))
    print()
    print("%-12s %-8s %-10s %-10s %s" % (
        "cadence", "runs", "fired", "incidents", "median latency"))
    for cadence in CADENCES:
        row = report["detection"][str(cadence)]
        if not row:
            continue
        print("%-12s %-8s %-10s %-10s %s" % (
            cadence, row["runs"], row["fired"], row["incidents"], row["median_latency"]))
    a = report["detection"]["1"].get("signature")
    b = report["detection"]["5"].get("signature")
    if a is not None and b is not None:
        print()
        print("detection signature identical:" , a == b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
