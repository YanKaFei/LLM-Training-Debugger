"""R2: measure whether healthy heavy-tail and mild gradient faults are separable.

Phase 1.3 ended on an unresolved question: `H1_heavy_tail_gradients` alarmed in 11 of
14 runs, and the detector's boundary sits between x2 and x3 while that healthy class
reaches x2.6 by construction. The brief is explicit about what to do with this:

    If the evidence cannot separate them, the correct answer is
    INSUFFICIENT_EVIDENCE -- not a cleverer threshold.

So this script measures separability **before** any rule is built, using only
signals already collected (T1 per-layer gradient norms plus the global norm; no new
tensor capture). It reports the overlap honestly and computes the minimum reliably
detectable magnitude, which is a product capability boundary rather than a failure.

Run::

    .venv/bin/python tests/experiments/gradient_boundary.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tiny_model import make_tiny_lm, synthetic_batch  # noqa: E402

from training_debugger.adapters.pytorch import (  # noqa: E402
    LayerMetricCollector,
    LayerSelection,
    PyTorchTrainingAdapter,
)
from training_debugger.observer import (  # noqa: E402
    MemoryWriter,
    ObserverConfig,
    SamplingPolicy,
    TrainingObserver,
)

OUTPUT_DIR = ROOT / "outputs" / "phase1-4"
STEPS = 200
INJECTION = 120
MAGNITUDES = (1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
LIVE_WINDOW = 25


def collect(seed: int, *, layers: int = 4, d_model: int = 32,
            heavy_tail: bool = False, magnitude: Optional[float] = None
            ) -> Tuple[List[int], Dict[int, float], Dict[int, Dict[int, float]]]:
    torch.manual_seed(seed)
    model, config = make_tiny_lm(seed=seed, n_layers=layers, d_model=d_model, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "b-%d" % seed, writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384), rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="b")
    collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=32), cadence=1)
    adapter.run_start(steps=STEPS)
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            tracker.record_loss(loss)
            loss.backward()
            if magnitude is not None and step >= INJECTION:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(magnitude)
            if heavy_tail and step % 17 == 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(2.6)
            tracker.record_gradients()
            collector.collect(step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()
    adapter.run_end()
    observer.flush(timeout=30.0)
    observer.close()

    global_norm: Dict[int, float] = {}
    per_layer: Dict[int, Dict[int, float]] = {}
    for event in writer.events:
        if event.step is None or event.metric_value is None:
            continue
        if event.metric_name == "grad.global_norm":
            global_norm[event.step] = float(event.metric_value)
        elif event.metric_name == "grad.layer_norm" and event.layer_index is not None:
            per_layer.setdefault(int(event.layer_index), {})[event.step] = float(event.metric_value)
    del model, optimizer, observer, adapter, collector
    gc.collect()
    return sorted(global_norm), global_norm, per_layer


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return float("nan")
    return float(ordered[n // 2]) if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] * (1 - (pos - lo)) + ordered[hi] * (pos - lo))


def features(global_norm: Dict[int, float], per_layer: Dict[int, Dict[int, float]],
             step: int, *, baseline_floor: int = 50, window: int = 100) -> Dict[str, float]:
    """Joint features available at ``step`` from signals already collected."""
    layers = sorted(per_layer)
    if not layers:
        return {}
    # Normalise each layer by its own trailing baseline, then aggregate jointly.
    normalized: List[float] = []
    ratio_over_median: List[float] = []
    for index in layers:
        series = per_layer[index]
        history = [series[s] for s in sorted(series) if baseline_floor <= s < step][-window:]
        if len(history) < 20 or step not in series:
            continue
        base = _median(history)
        if base <= 0:
            continue
        normalized.append(series[step] / base)
        ratio_over_median.append(series[step] / base)
    glob = global_norm.get(step)
    glob_history = [global_norm[s] for s in sorted(global_norm) if baseline_floor <= s < step][-window:]
    glob_norm = (glob / _median(glob_history)) if glob and glob_history and _median(glob_history) > 0 else float("nan")

    ordered = sorted(normalized)
    out: Dict[str, float] = {
        # -- univariate (what Phase 1.3 used) --------------------------
        "global_norm_ratio": glob_norm,
        # -- joint deterministic statistics ----------------------------
        "n_layers": float(len(normalized)),
        "mean_layer_ratio": statistics.fmean(normalized) if normalized else float("nan"),
        "max_layer_ratio": max(normalized) if normalized else float("nan"),
        "l2_aggregate": math.sqrt(sum(v * v for v in normalized)) if normalized else float("nan"),
        "l1_aggregate": sum(normalized) if normalized else float("nan"),
        "fraction_above_1p5": (
            sum(1 for v in normalized if v > 1.5) / len(normalized) if normalized else float("nan")
        ),
        "fraction_above_2": (
            sum(1 for v in normalized if v > 2.0) / len(normalized) if normalized else float("nan")
        ),
        "top_k_mass": (
            sum(ordered[-max(1, len(ordered) // 3):]) / sum(ordered) if ordered and sum(ordered) > 0
            else float("nan")
        ),
        "max_over_median": (
            max(normalized) / _median(normalized) if normalized and _median(normalized) > 0
            else float("nan")
        ),
        # Trimmed aggregate: ignores the single largest layer, so a lone outlier
        # cannot carry the statistic on its own.
        "trimmed_l2": (
            math.sqrt(sum(v * v for v in sorted(normalized)[:-1])) if len(normalized) > 1
            else float("nan")
        ),
    }
    return out


def windowed(feature_name: str, steps: Sequence[int], global_norm, per_layer,
             start: int, end: int) -> float:
    """Peak feature value over a step window."""
    values = []
    for step in steps:
        if start <= step <= end:
            payload = features(global_norm, per_layer, step)
            value = payload.get(feature_name)
            if value is not None and value == value:
                values.append(value)
    return max(values) if values else float("nan")


def auc(healthy: Sequence[float], fault: Sequence[float]) -> Optional[float]:
    """Probability a random fault value exceeds a random healthy value.

    0.5 means the feature carries no information at all; 1.0 means perfect
    separation. Reported instead of a threshold sweep because it is
    threshold-independent, so it answers "is there signal here?" rather than
    "did we pick a good cut?".
    """
    clean_h = [v for v in healthy if v == v]
    clean_f = [v for v in fault if v == v]
    if len(clean_h) < 3 or len(clean_f) < 3:
        return None
    wins = 0.0
    for f in clean_f:
        for h in clean_h:
            if f > h:
                wins += 1.0
            elif f == h:
                wins += 0.5
    return wins / (len(clean_f) * len(clean_h))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Gradient boundary separability")
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    seeds = list(range(7000, 7000 + args.seeds))
    print("[R2] healthy heavy-tail runs ...")
    healthy_runs = [collect(s, heavy_tail=True) for s in seeds]
    print("[R2] healthy stable runs ...")
    stable_runs = [collect(s, heavy_tail=False) for s in seeds]

    feature_names = list(features(*_unpack(healthy_runs[0])[1:], step=INJECTION).keys()) if False else None
    sample = features(healthy_runs[0][1], healthy_runs[0][2], INJECTION)
    feature_names = sorted(sample.keys())

    results: Dict[str, Any] = {"healthy_heavy_tail": {}, "healthy_stable": {}, "faults": {}}

    for name in feature_names:
        results["healthy_heavy_tail"][name] = [
            windowed(name, run[0], run[1], run[2], INJECTION, INJECTION + LIVE_WINDOW)
            for run in healthy_runs
        ]
        results["healthy_stable"][name] = [
            windowed(name, run[0], run[1], run[2], INJECTION, INJECTION + LIVE_WINDOW)
            for run in stable_runs
        ]

    for magnitude in MAGNITUDES:
        print("[R2] fault magnitude x{} ...".format(magnitude))
        runs = [collect(s, magnitude=magnitude) for s in seeds[: max(4, args.seeds // 2)]]
        results["faults"][str(magnitude)] = {
            name: [
                windowed(name, run[0], run[1], run[2], INJECTION, INJECTION + LIVE_WINDOW)
                for run in runs
            ]
            for name in feature_names
        }

    # -- separability -----------------------------------------------------
    separability: Dict[str, Any] = {}
    for magnitude, block in sorted(results["faults"].items(), key=lambda kv: float(kv[0])):
        per_feature: Dict[str, Any] = {}
        for name in feature_names:
            healthy_values = results["healthy_heavy_tail"][name]
            fault_values = block[name]
            clean_h = [v for v in healthy_values if v == v]
            clean_f = [v for v in fault_values if v == v]
            if not clean_h or not clean_f:
                continue
            h_max = max(clean_h)
            per_feature[name] = {
                "auc_vs_heavy_tail": auc(healthy_values, fault_values),
                "healthy_median": _median(clean_h),
                "healthy_p95": _quantile(sorted(clean_h), 0.95),
                "healthy_max": h_max,
                "fault_median": _median(clean_f),
                "fault_min": min(clean_f),
                "overlap": min(clean_f) <= h_max,
            }
        separability[magnitude] = per_feature

    payload = {
        "config": {
            "seeds": seeds, "steps": STEPS, "injection_step": INJECTION,
            "live_window": LIVE_WINDOW, "magnitudes": list(MAGNITUDES),
        },
        "feature_values": results,
        "separability": separability,
    }
    (OUTPUT_DIR / "gradient_boundary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print("")
    print("elapsed {:.0f}s".format(time.perf_counter() - started))
    print("report: {}".format(OUTPUT_DIR / "gradient_boundary.json"))
    return 0


def _unpack(run):
    return run


if __name__ == "__main__":
    raise SystemExit(main())
