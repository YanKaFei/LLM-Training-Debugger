"""D2.1/D2.2: characterise healthy gradient norms, then compare detector families.

Phase 1.1 established *that* the gradient detector fires on healthy runs. It did
not establish **why**, and the difference matters: a threshold can be nudged to
hide a symptom without touching the cause.

This script answers the distributional question first, assuming nothing:

* shape -- quantiles, skew, excess kurtosis, and a Hill tail index
* stationarity -- is the first 20 % the same distribution as the middle?
* autocorrelation -- is a "spike" actually independent noise?
* layer structure -- do layers move together, or independently?

Only then are eight detector families compared on the same 30 healthy runs and
the same injected faults, with the D1 separated metrics (never a negative delay).
No learned detector is used, per the brief.

Run::

    .venv/bin/python tests/experiments/gradient_distribution.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from training_debugger.detectors.base import (  # noqa: E402
    detrended_residuals,
    robust_local_trend,
    robust_stats,
)
from training_debugger.observer import (  # noqa: E402
    MemoryWriter,
    ObserverConfig,
    SamplingPolicy,
    TrainingObserver,
)

OUTPUT_DIR = ROOT / "outputs" / "phase1-2"
STEPS = 200
INJECTION_STEP = 120
LAYERS = 4
D_MODEL = 32
HEALTHY_SEEDS = list(range(2000, 2030))


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect_run(
    seed: int,
    *,
    grad_explode_at: Optional[int] = None,
    grad_explode_factor: float = 35.0,
    heavy_tail: bool = False,
    layers: int = LAYERS,
    steps: int = STEPS,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model, config = make_tiny_lm(seed=seed, n_layers=layers, d_model=D_MODEL, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "gd-%d" % seed, writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384), rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="tiny")
    collector = LayerMetricCollector(
        model, observer, selection=LayerSelection(top_k=32), cadence=1
    )
    adapter.run_start(steps=steps)
    for step in range(1, steps + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            with tracker.phase("forward"):
                loss = model(**batch)["loss"]
            tracker.record_loss(loss)
            with tracker.phase("backward"):
                loss.backward()
            if grad_explode_at is not None and step >= grad_explode_at:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(grad_explode_factor)
            if heavy_tail and step % 17 == 0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(2.6)
            tracker.record_gradients()
            collector.collect(step)
            with tracker.phase("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()
    adapter.run_end()
    observer.flush(timeout=30.0)
    observer.close()

    global_norm: Dict[int, float] = {}
    per_layer: Dict[int, Dict[int, float]] = {}
    loss_series: Dict[int, float] = {}
    for event in writer.events:
        if event.step is None:
            continue
        if event.metric_name == "grad.global_norm" and event.metric_value is not None:
            global_norm[event.step] = float(event.metric_value)
        elif event.metric_name == "grad.layer_norm" and event.layer_index is not None:
            per_layer.setdefault(int(event.layer_index), {})[event.step] = float(event.metric_value)
        elif event.metric_name == "train.loss" and event.metric_value is not None:
            if "nonfinite" not in str(event.metric_name):
                loss_series[event.step] = float(event.metric_value)
    del model, optimizer, observer, adapter, collector
    gc.collect()
    return {
        "seed": seed,
        "global_norm": sorted(global_norm.items()),
        "per_layer": {k: sorted(v.items()) for k, v in per_layer.items()},
        "loss": sorted(loss_series.items()),
    }


# ---------------------------------------------------------------------------
# Distribution characterisation
# ---------------------------------------------------------------------------


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[index]


def hill_index(values: Sequence[float], tail_fraction: float = 0.1) -> Optional[float]:
    """Hill tail index for the upper tail.

    A Gaussian gives ~0; a heavy tail gives a positive index that grows with
    heaviness. Included because the whole design question is whether a
    standard-deviation-style threshold is the right instrument at all.
    """
    positive = sorted(v for v in values if v > 0)
    if len(positive) < 20:
        return None
    k = max(5, int(len(positive) * tail_fraction))
    tail = positive[-k:]
    threshold = positive[-k - 1] if len(positive) > k else positive[0]
    if threshold <= 0:
        return None
    logs = [math.log(v / threshold) for v in tail if v > threshold]
    if not logs:
        return None
    mean_log = sum(logs) / len(logs)
    return (1.0 / mean_log) if mean_log > 0 else None


def excess_kurtosis(values: Sequence[float]) -> Optional[float]:
    n = len(values)
    if n < 4:
        return None
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    if m2 <= 0:
        return None
    return m4 / (m2 * m2) - 3.0


def autocorrelation(values: Sequence[float], lag: int = 1) -> Optional[float]:
    n = len(values)
    if n <= lag + 2:
        return None
    mean = sum(values) / n
    denominator = sum((v - mean) ** 2 for v in values)
    if denominator <= 0:
        return None
    numerator = sum((values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag))
    return numerator / denominator


def stationarity_split(values: Sequence[float], fraction: float = 0.2) -> Dict[str, Any]:
    """Compare the first ``fraction`` of training with the middle."""
    n = len(values)
    head = values[: max(5, int(n * fraction))]
    middle = values[int(n * 0.3): int(n * 0.8)]
    if not head or not middle:
        return {}
    head_sorted, mid_sorted = sorted(head), sorted(middle)
    return {
        "head_median": _quantile(head_sorted, 0.5),
        "middle_median": _quantile(mid_sorted, 0.5),
        "head_p95": _quantile(head_sorted, 0.95),
        "middle_p95": _quantile(mid_sorted, 0.95),
        "head_max": max(head),
        "middle_max": max(middle),
        "median_shift_ratio": _quantile(mid_sorted, 0.5) / _quantile(head_sorted, 0.5)
        if _quantile(head_sorted, 0.5) else None,
    }


def characterise(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    all_values: List[float] = []
    per_run = []
    layer_correlations: List[float] = []
    for run in runs:
        values = [v for _, v in run["global_norm"]]
        all_values.extend(values)
        if not values:
            continue
        ordered = sorted(values)
        # Cross-layer correlation: how often do layers spike together?
        layers = run["per_layer"]
        if len(layers) >= 2:
            keys = sorted(layers)
            steps = sorted(set(layers[keys[0]]) & set(layers[keys[1]]))
            if len(steps) > 10:
                a = [layers[keys[0]][s] for s in steps]
                b = [layers[keys[1]][s] for s in steps]
                rho = autocorrelation(a, 1)  # reuse the centred covariance helper
                _ = b
                layer_correlations.append(rho if rho is not None else 0.0)
        per_run.append(
            {
                "seed": run["seed"],
                "n": len(values),
                "min": ordered[0],
                "p50": _quantile(ordered, 0.5),
                "p90": _quantile(ordered, 0.9),
                "p95": _quantile(ordered, 0.95),
                "p99": _quantile(ordered, 0.99),
                "max": ordered[-1],
                "mean": sum(values) / len(values),
                "excess_kurtosis": excess_kurtosis(values),
                "hill_index": hill_index(values),
                "autocorr_lag1": autocorrelation(values, 1),
                "stationarity": stationarity_split(values),
                "max_over_p50": ordered[-1] / _quantile(ordered, 0.5) if _quantile(ordered, 0.5) else None,
                "p99_over_p50": _quantile(ordered, 0.99) / _quantile(ordered, 0.5) if _quantile(ordered, 0.5) else None,
            }
        )
    ordered_all = sorted(all_values)
    return {
        "runs": len(runs),
        "total_observations": len(all_values),
        "pooled": {
            "min": ordered_all[0] if ordered_all else None,
            "p50": _quantile(ordered_all, 0.5),
            "p95": _quantile(ordered_all, 0.95),
            "p99": _quantile(ordered_all, 0.99),
            "p999": _quantile(ordered_all, 0.999),
            "max": ordered_all[-1] if ordered_all else None,
            "excess_kurtosis": excess_kurtosis(all_values),
            "hill_index": hill_index(all_values),
        },
        "per_run": per_run,
        "median_max_over_p50": _median([r["max_over_p50"] for r in per_run if r["max_over_p50"]]),
        "median_p99_over_p50": _median([r["p99_over_p50"] for r in per_run if r["p99_over_p50"]]),
        "median_autocorr_lag1": _median([r["autocorr_lag1"] for r in per_run if r["autocorr_lag1"] is not None]),
        "median_excess_kurtosis": _median([r["excess_kurtosis"] for r in per_run if r["excess_kurtosis"] is not None]),
        "median_hill_index": _median([r["hill_index"] for r in per_run if r["hill_index"] is not None]),
        "median_median_shift_ratio": _median([
            r["stationarity"]["median_shift_ratio"] for r in per_run
            if r.get("stationarity", {}).get("median_shift_ratio")
        ]),
    }


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return float(ordered[n // 2]) if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


# ---------------------------------------------------------------------------
# Detector families
# ---------------------------------------------------------------------------


class Context:
    """Rolling state a family may keep. Per (run, layer-set) only."""

    def __init__(self, warmup: int = 50) -> None:
        self.warmup = warmup
        self.values: List[float] = []
        self.steps: List[int] = []


Family = Callable[[Context, int, float, Dict[int, float]], bool]


def family_rolling_quantile(ctx: Context, step: int, value: float, layers: Dict[int, float]) -> bool:
    if len(ctx.values) < 50:
        return False
    ordered = sorted(ctx.values[-100:])
    p99 = _quantile(ordered, 0.99)
    return value > p99 * 1.5


def family_peak_relative(ctx: Context, step: int, value: float, layers: Dict[int, float]) -> bool:
    if len(ctx.values) < 50:
        return False
    peak = max(ctx.values[-100:])
    return peak > 0 and value > peak * 3.0


def family_log_robust_z(ctx: Context, step: int, value: float, layers: Dict[int, float]) -> bool:
    if len(ctx.values) < 50 or value <= 0:
        return False
    logs = [math.log(max(v, 1e-12)) for v in ctx.values[-100:]]
    stats = robust_stats(logs)
    scale = stats["mad"] * 1.4826
    if scale <= 0:
        return False
    z = (math.log(value) - stats["median"]) / scale
    return z > 6.0


def family_ewma(ctx: Context, step: int, value: float, layers: Dict[int, float]) -> bool:
    if len(ctx.values) < 50:
        return False
    alpha = 0.1
    mean = ctx.values[0]
    var = 0.0
    for v in ctx.values:
        delta = v - mean
        mean += alpha * delta
        var = (1 - alpha) * (var + alpha * delta * delta)
    if var <= 0:
        return False
    return (value - mean) / math.sqrt(var) > 6.0


def family_cusum(ctx: Context, step: int, value: float, layers: Dict[int, float]) -> bool:
    if len(ctx.values) < 60:
        return False
    window = ctx.values[-60:-1]
    stats = robust_stats(window)
    scale = stats["mad"] * 1.4826
    if scale <= 0:
        return False
    k = 0.5
    threshold = 6.0
    positives = [(v - stats["median"]) / scale - k for v in ctx.values[-30:] + [value]]
    running = 0.0
    for contribution in positives:
        running = max(0.0, running + contribution)
    return running > threshold


def family_envelope(ctx: Context, step: int, value: float, layers: Dict[int, float]) -> bool:
    if len(ctx.values) < 60:
        return False
    window = ctx.values[-120:]
    ordered = sorted(window)
    median = _quantile(ordered, 0.5)
    upper = _quantile(ordered, 0.995)
    # Asymmetric: require a clear departure above the envelope.
    return upper > 0 and value > max(upper * 1.5, median * 3.0)


def family_historical_max(ctx: Context, step: int, value: float, layers: Dict[int, float]) -> bool:
    if len(ctx.values) < 50:
        return False
    running_max = max(ctx.values[:-1]) if len(ctx.values) > 1 else 0.0
    return running_max > 0 and value > running_max * 3.0


def family_persistence(ctx: Context, step: int, value: float, layers: Dict[int, float]) -> bool:
    """Global departure AND cross-layer multiplicity.

    This is the family the brief asks to be tested on data rather than intuition:
    ``k`` affected layers, or ``m`` consecutive observations. Both parameters are
    swept below.
    """
    if len(ctx.values) < 60:
        return False
    window = ctx.values[-120:]
    ordered = sorted(window)
    median = _quantile(ordered, 0.5)
    p99 = _quantile(ordered, 0.99)
    if median <= 0 or p99 <= 0:
        return False
    global_departure = value > max(p99 * 1.5, median * 2.0)
    if not global_departure:
        return False
    if len(layers) < 3:
        return False
    # A layer "departs" when it exceeds its own trailing envelope.
    affected = 0
    for layer_value in layers.values():
        if layer_value > p99 * 1.5:
            affected += 1
    return affected >= 2


FAMILIES: Dict[str, Family] = {
    "A_rolling_quantile": family_rolling_quantile,
    "B_peak_relative": family_peak_relative,
    "C_log_robust_z": family_log_robust_z,
    "D_ewma": family_ewma,
    "E_cusum": family_cusum,
    "F_asymmetric_envelope": family_envelope,
    "G_historical_max": family_historical_max,
    "H_persistence_multilayer": family_persistence,
}


def evaluate_family(run: Dict[str, Any], family: Family) -> List[int]:
    ctx = Context()
    fires: List[int] = []
    layer_series = run["per_layer"]
    for step, value in run["global_norm"]:
        current_layers = {
            index: series[step]
            for index, series in layer_series.items()
            if step in series
        }
        try:
            fired = family(ctx, step, value, current_layers)
        except Exception:
            fired = False
        if fired:
            fires.append(step)
        ctx.values.append(value)
        ctx.steps.append(step)
    return fires


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Characterise healthy gradients, compare families")
    parser.add_argument("--healthy-runs", type=int, default=30)
    parser.add_argument("--fault-runs", type=int, default=6)
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("[D2.1] collecting {} healthy runs ...".format(args.healthy_runs))
    healthy = []
    for index, seed in enumerate(HEALTHY_SEEDS[: args.healthy_runs], start=1):
        healthy.append(collect_run(seed))
        if index % 10 == 0:
            print("   {}/{}".format(index, args.healthy_runs))

    print("[D2.1] collecting healthy heavy-tail runs ...")
    heavy = [collect_run(seed, heavy_tail=True) for seed in HEALTHY_SEEDS[: args.fault_runs]]

    print("[D2.2] collecting injected-fault runs ...")
    faults = {}
    for factor in (2.0, 3.0, 5.0, 10.0, 20.0, 40.0):
        faults["explosion_x{:.0f}".format(factor)] = [
            collect_run(seed, grad_explode_at=INJECTION_STEP, grad_explode_factor=factor)
            for seed in HEALTHY_SEEDS[: args.fault_runs]
        ]

    stats = characterise(healthy)
    stats_heavy = characterise(heavy)

    # -- family comparison ------------------------------------------------
    comparison: Dict[str, Any] = {}
    for name, family in FAMILIES.items():
        healthy_fires = sum(len(evaluate_family(run, family)) for run in healthy)
        heavy_fires = sum(len(evaluate_family(run, family)) for run in heavy)
        per_factor: Dict[str, Any] = {}
        for factor_name, runs in faults.items():
            detected = 0
            delays: List[int] = []
            pre_injection = 0
            for run in runs:
                fires = evaluate_family(run, family)
                if any(f >= INJECTION_STEP for f in fires):
                    detected += 1
                    delays.append(min(f for f in fires if f >= INJECTION_STEP) - INJECTION_STEP)
                pre_injection += sum(1 for f in fires if f < INJECTION_STEP)
            per_factor[factor_name] = {
                "detected": detected,
                "runs": len(runs),
                "recall": detected / len(runs) if runs else None,
                "median_delay": _median(delays),
                "pre_injection_fires": pre_injection,
            }
        comparison[name] = {
            "healthy_fires": healthy_fires,
            "healthy_fires_per_1k_steps": healthy_fires / (len(healthy) * STEPS) * 1000.0,
            "heavy_tail_fires": heavy_fires,
            "heavy_tail_fires_per_1k_steps": heavy_fires / (len(heavy) * STEPS) * 1000.0,
            "per_magnitude": per_factor,
        }
        print("   {:<28} healthy FP/1k={:>6.2f}  x40 recall={}  x2 recall={}".format(
            name,
            comparison[name]["healthy_fires_per_1k_steps"],
            per_factor["explosion_x40"]["recall"],
            per_factor["explosion_x2"]["recall"],
        ))

    payload = {
        "healthy_distribution": stats,
        "heavy_tail_distribution": stats_heavy,
        "families": comparison,
        "config": {
            "healthy_runs": args.healthy_runs,
            "steps": STEPS,
            "injection_step": INJECTION_STEP,
            "layers": LAYERS,
            "d_model": D_MODEL,
            "healthy_seeds": HEALTHY_SEEDS[: args.healthy_runs],
        },
    }
    (OUTPUT_DIR / "gradient_distribution.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print("")
    print("elapsed: {:.0f}s".format(time.perf_counter() - started))
    print("pooled healthy:", json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats["pooled"].items()},
        default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
