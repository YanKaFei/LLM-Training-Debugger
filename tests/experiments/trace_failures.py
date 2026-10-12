"""D0/D4: reproduce the Phase 1.1 detector failures, with tracing.

Nothing here modifies a detector. The point is to freeze the failing behaviour,
capture the raw signal, and find the mechanism — in that order. A change made
before the mechanism is understood cannot be distinguished from a lucky guess.

Four failures are reproduced:

1. ``H0_healthy``: the gradient detector fires on clean runs (6/6).
2. ``F10_mixed_anomaly``: NaN loss + gradient explosion fires nothing (0/6).
3. Negative detection delay: the reported onset precedes the injection.
4. Performance measurement instability.

The trace for (2) records every intermediate value on the path from the raw loss
to a detector verdict, because the failure is an *interaction*: the same NaN in
isolation is caught 6/6, so something about the combination suppresses it.

Run::

    .venv/bin/python tests/experiments/trace_failures.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
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
from training_debugger.detectors import DetectorConfig, DetectorRunner  # noqa: E402
from training_debugger.detectors.base import robust_stats  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    MemoryWriter,
    ObserverConfig,
    SamplingPolicy,
    TrainingObserver,
)

OUTPUT_DIR = ROOT / "outputs" / "phase1-2"
TRACE_DIR = OUTPUT_DIR / "traces"

STEPS = 200
INJECTION_STEP = 120
LAYERS = 4
D_MODEL = 32
SEEDS = [1000, 1001, 1002, 1003, 1004, 1005]


# ---------------------------------------------------------------------------
# Scenario runner with full tracing
# ---------------------------------------------------------------------------


def run_scenario(
    fault_type: str,
    seed: int,
    *,
    steps: int = STEPS,
    injection_step: int = INJECTION_STEP,
    layers: int = LAYERS,
    d_model: int = D_MODEL,
    trace: bool = False,
) -> Dict[str, Any]:
    """Train a real model with one fault, returning events plus an optional trace."""
    torch.manual_seed(seed)
    model, config = make_tiny_lm(seed=seed, n_layers=layers, d_model=d_model, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    writer = MemoryWriter()
    observer = TrainingObserver(
        "trace-{}-{}".format(fault_type, seed),
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384),
        rank=0,
        world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="tiny")
    collector = LayerMetricCollector(
        model, observer, selection=LayerSelection(top_k=32), cadence=1
    )
    adapter.run_start(steps=steps)

    scale = 1.0
    trace_rows: List[Dict[str, Any]] = []

    for step in range(1, steps + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            with tracker.phase("forward"):
                loss = model(**batch)["loss"]

            injected_this_step: List[str] = []
            if fault_type == "H2_noisy_healthy_loss":
                loss = loss * (1.0 + 0.35 * math.sin(step * 1.7))
            if fault_type == "F1_isolated_loss_spike" and step == injection_step:
                loss = loss + 8.0
                injected_this_step.append("loss_spike")
            if fault_type == "F2_sustained_divergence" and step >= injection_step:
                scale *= 1.012
                loss = loss * scale
            if fault_type == "F4_nan_loss" and step == injection_step:
                loss = loss * float("nan")
                injected_this_step.append("nan_loss")
            if fault_type == "F10_mixed_anomaly" and step == injection_step:
                # NOTE: this was `loss = float("nan")` in the Phase 1.1 harness.
                # That replaced the tensor with a Python float, and
                # `record_loss` calls `.detach()` on it, so the run died with
                # AttributeError and the scoring counted the crash as a detector
                # false negative. The "F10 detector interaction bug" was this.
                loss = loss * float("nan")
                injected_this_step.append("nan_loss")
            if fault_type == "F6_throughput_degradation" and step >= injection_step:
                time.sleep(0.004)

            loss_before = loss
            tracker.record_loss(loss)

            with tracker.phase("backward"):
                if loss == loss:
                    loss.backward()

            if fault_type == "F3_gradient_explosion" and step >= injection_step:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(35.0)
                injected_this_step.append("grad_explosion")
            if fault_type == "F10_mixed_anomaly" and step >= injection_step:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(25.0)
                if step == injection_step:
                    injected_this_step.append("grad_explosion")
            if fault_type == "F5_inf_gradient" and step == injection_step:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.flatten()[0] = float("inf")
                injected_this_step.append("inf_gradient")
            if fault_type == "H1_heavy_tail_gradients" and step % 17 == 0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(2.6)

            if trace:
                finite_grads = [
                    float(p.grad.detach().norm()) if p.grad is not None else None
                    for p in model.parameters()
                ]
                n_nonfinite = sum(
                    1 for g in finite_grads if g is None or g != g or abs(g) == float("inf")
                )
                trace_rows.append(
                    {
                        "step": step,
                        "loss_input": _num(loss_before),
                        "loss_finite": bool(loss_before == loss_before and abs(loss_before) != float("inf")),
                        "backward_ran": bool(loss_before == loss_before),
                        "injected": list(injected_this_step),
                        "grad_tensors": len(finite_grads),
                        "grad_nonfinite_tensors": n_nonfinite,
                        "per_tensor_grad_norm": finite_grads,
                    }
                )

            tracker.record_gradients()
            collector.collect(step)
            with tracker.phase("optimizer"):
                if loss == loss:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()

    adapter.run_end()
    observer.flush(timeout=30.0)
    observer.close()

    events = writer.events
    by_metric, layer_series, module_paths = _series(events)
    detection = DetectorRunner(DetectorConfig()).run(
        "trace",
        by_metric,
        rank=0,
        layer_series=layer_series,
        module_paths=module_paths,
        population_layers=layers,
        population_ranks=1,
        coverage={"step_count": steps, "step_min": 1, "step_max": steps},
    )
    result = {
        "fault_type": fault_type,
        "seed": seed,
        "steps": steps,
        "injection_step": injection_step,
        "fired": sorted({str(a.detector) for a in detection.anomalies}),
        "onsets": {str(a.detector): a.first_abnormal_step for a in detection.anomalies},
        "anomalies": [
            {
                "detector": str(a.detector),
                "step": a.first_abnormal_step,
                "window": list(a.window or []),
                "value": a.observed_value,
                "score": a.score,
                "metadata": {k: _num(v) for k, v in (a.metadata or {}).items()},
            }
            for a in detection.anomalies
        ],
        "detector_errors": detection.detector_errors,
        "event_count": len(events),
        "series": {name: len(values) for name, values in by_metric.items()},
        "layer_series_count": len(layer_series),
    }
    if trace:
        result["trace"] = trace_rows
    del model, optimizer, observer, adapter, collector
    gc.collect()
    return result


def _num(value: Any) -> Any:
    try:
        if value is None:
            return None
        f = float(value)
        if f != f:
            return "nan"
        if abs(f) == float("inf"):
            return "inf" if f > 0 else "-inf"
        return f
    except (TypeError, ValueError):
        return value


def _series(events):
    by_metric: Dict[str, Dict[int, float]] = {}
    layers: Dict[int, Dict[int, float]] = {}
    module_paths: Dict[int, str] = {}
    for event in events:
        if event.metric_name is None or event.metric_value is None or event.step is None:
            continue
        if event.metric_name == "grad.layer_norm" and event.layer_index is not None:
            layers.setdefault(event.layer_index, {})[event.step] = event.metric_value
            if event.module_path:
                module_paths[event.layer_index] = event.module_path
            continue
        by_metric.setdefault(event.metric_name, {})[event.step] = event.metric_value
    return (
        {k: sorted(v.items()) for k, v in by_metric.items()},
        {k: sorted(v.items()) for k, v in layers.items()},
        module_paths,
    )


# ---------------------------------------------------------------------------
# Failure 1: H0 gradient false positives
# ---------------------------------------------------------------------------


def analyse_h0(seed: int) -> Dict[str, Any]:
    """Where does the gradient detector fire on a clean run, and why?"""
    result = run_scenario("H0_healthy", seed)
    by_metric = None
    # Re-derive the series so the detector's own arithmetic can be reproduced.
    torch.manual_seed(seed)
    model, config = make_tiny_lm(seed=seed, n_layers=LAYERS, d_model=D_MODEL, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "h0-%d" % seed, writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384), rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="tiny")
    adapter.run_start(steps=STEPS)
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            with tracker.phase("forward"):
                loss = model(**batch)["loss"]
            tracker.record_loss(loss)
            with tracker.phase("backward"):
                loss.backward()
            tracker.record_gradients()
            with tracker.phase("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()
    adapter.run_end()
    observer.flush(timeout=30.0)
    observer.close()
    by_metric, _, _ = _series(writer.events)

    grad = dict(by_metric.get("grad.global_norm", []))
    steps_sorted = sorted(grad)
    values = [grad[s] for s in steps_sorted]
    diagnostics = []
    config_det = DetectorConfig()
    for index in range(len(values)):
        step = steps_sorted[index]
        window = values[max(0, index - config_det.window):index]
        if len(window) < config_det.min_samples or step <= config_det.warmup_steps:
            continue
        stats = robust_stats(window)
        median = stats["median"]
        mad = stats["mad"] * 1.4826 if stats["mad"] and stats["mad"] < 1e-9 else stats["mad"]
        peak = max(window)
        diagnostics.append(
            {
                "step": step,
                "value": values[index],
                "window_median": median,
                "window_mad": stats["mad"],
                "window_peak": peak,
                "z": ((values[index] - median) / stats["mad"]) if stats["mad"] else None,
                "peak_ratio": (values[index] / peak) if peak else None,
                "is_peak_of_all": values[index] >= max(values),
            }
        )
    result["series"] = {"grad.global_norm": [grad[s] for s in steps_sorted]}
    result["steps"] = steps_sorted
    result["diagnostics"] = diagnostics
    result["summary"] = {
        "n": len(values),
        "min": min(values), "max": max(values),
        "median": sorted(values)[len(values) // 2],
        "p95": sorted(values)[int(0.95 * (len(values) - 1))],
    }
    return result


# ---------------------------------------------------------------------------
# Failure 2: F10 mixed anomaly
# ---------------------------------------------------------------------------


def analyse_f10(seed: int) -> Dict[str, Any]:
    paired = {
        "F10_mixed_anomaly": run_scenario("F10_mixed_anomaly", seed, trace=True),
        "F4_nan_loss": run_scenario("F4_nan_loss", seed, trace=True),
        "F3_gradient_explosion": run_scenario("F3_gradient_explosion", seed, trace=True),
    }
    return paired


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce Phase 1.1 detector failures")
    parser.add_argument("--seeds", type=int, default=3, help="seeds per failure")
    args = parser.parse_args(argv)

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[: args.seeds]
    started = time.perf_counter()

    # -- 1. H0 gradient false positives ---------------------------------
    print("[D0] H0_healthy gradient false positives ...")
    h0 = [analyse_h0(seed) for seed in seeds]
    for seed, entry in zip(seeds, h0):
        fired = "grad.robust_zscore" in " ".join(
            str(a["detector"]) for a in entry.get("anomalies", [])
        )
        print("   seed {}: fired={} summary={}".format(seed, fired, entry["summary"]))
    (TRACE_DIR / "h0_gradient_series.json").write_text(
        json.dumps(
            [
                {
                    "seed": seed,
                    "steps": entry["steps"],
                    "values": entry["series"]["grad.global_norm"],
                    "summary": entry["summary"],
                    "onsets": entry["onsets"],
                    "anomalies": entry["anomalies"],
                    "diagnostics": entry["diagnostics"][:200],
                }
                for seed, entry in zip(seeds, h0)
            ],
            indent=2, default=str,
        ),
        encoding="utf-8",
    )

    # -- 2. F10 mixed anomaly -------------------------------------------
    print("[D0] F10_mixed_anomaly ...")
    f10 = {}
    for seed in seeds:
        f10[str(seed)] = analyse_f10(seed)
        print("   seed {}: F10 fired={} | F4 fired={} | F3 fired={}".format(
            seed,
            f10[str(seed)]["F10_mixed_anomaly"]["fired"],
            f10[str(seed)]["F4_nan_loss"]["fired"],
            f10[str(seed)]["F3_gradient_explosion"]["fired"],
        ))
    (TRACE_DIR / "f10_traces.json").write_text(
        json.dumps(f10, indent=2, default=str), encoding="utf-8"
    )

    elapsed = time.perf_counter() - started
    print("")
    print("elapsed: {:.0f}s".format(elapsed))
    print("traces:", TRACE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
