"""Performance audit: the observer's real cost, and where it goes.

Two existing benchmarks disagree by 5x on the same workload:

    benchmark_overhead.py    tiny: core = +2.15 ms/step   (median of per-repeat medians)
    benchmark_cost_ledger.py tiny: observed = +0.41 ms/step (MIN of per-repeat medians)

They are not comparable: one reports a median, the other a minimum, and they use
different workload parameters. The ledger also reports a **negative residual** --
accounted 3.86 ms against observed 0.41 ms -- which cannot be true of a valid
attribution: the parts cannot cost more than the whole.

This script establishes one defensible number, and the per-event cost that should explain
it. Design choices that matter:

* **Paired and interleaved.** Off and on are measured inside the same repeat, alternating,
  so thermal drift and background load hit both arms equally.
* **Enough repeats for a confidence interval**, not just a point estimate.
* **The per-event cost is measured directly**, because at ~14 events per step it is the
  only thing that could plausibly account for a millisecond of overhead -- and ~100 us for
  a queue append is not credible, so it needs checking rather than quoting.
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tiny_model import make_tiny_lm, synthetic_batch  # noqa: E402

from training_debugger.adapters.pytorch import (  # noqa: E402
    LayerMetricCollector, LayerSelection, PyTorchTrainingAdapter,
)
from training_debugger.observer import (  # noqa: E402
    MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver,
)

OUT = ROOT / "outputs" / "v0.1.8-alpha"
STEPS = 120
WARMUP = 20
REPEATS = 9

MODES: Tuple[str, ...] = ("off", "observer_only", "core_layers")


def build(mode: str, seed: int = 909):
    torch.manual_seed(seed)
    model, cfg = make_tiny_lm(seed=seed, n_layers=4, d_model=32, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    writer = MemoryWriter()
    observer = None
    adapter = None
    collector = None
    if mode != "off":
        observer = TrainingObserver(
            "audit", writer=writer, sampling=SamplingPolicy.uniform(1),
            config=ObserverConfig(flush_interval_s=0.01, batch_size=8192),
            rank=0, world_size=1,
        )
        adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="audit")
        if mode == "core_layers":
            collector = LayerMetricCollector(
                model, observer, selection=LayerSelection(top_k=32), cadence=1)
    return model, cfg, optimizer, observer, adapter, collector, writer


def measure(mode: str) -> Dict[str, Any]:
    model, cfg, optimizer, observer, adapter, collector, writer = build(mode)
    if adapter is not None:
        adapter.run_start(steps=STEPS)
    durations: List[float] = []
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8,
                               vocab_size=cfg.vocab_size, seed=step)
        started = time.perf_counter()
        if adapter is None:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()
        else:
            with adapter.step(step) as tracker:
                tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
                loss = model(**batch)["loss"]
                tracker.record_loss(loss)
                loss.backward()
                tracker.record_gradients()
                if collector is not None:
                    collector.collect(step)
                with tracker.phase("optimizer"):
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                tracker.record_parameters()
                tracker.record_learning_rate()
        durations.append(time.perf_counter() - started)
    events = 0
    emit_us = None
    if adapter is not None:
        adapter.run_end()
        observer.flush(timeout=60.0)
        health = observer.health()["observer"]
        events = int(health.get("emitted_total", 0) or 0)
        n = max(1, events)
        emit_us = health.get("emit_time_total_ns", 0) / n / 1000.0
        observer.close()
    return {
        "median_step_s": statistics.median(durations[WARMUP:]),
        "events": events,
        "us_per_event": emit_us,
    }


def paired() -> Dict[str, Any]:
    """Alternate arms inside each repeat so drift cancels."""
    samples: Dict[str, List[float]] = {mode: [] for mode in MODES}
    events: Dict[str, int] = {mode: 0 for mode in MODES}
    per_event: Dict[str, List[Optional[float]]] = {}
    gc.collect()
    for _ in range(2):                      # discarded warm-up rounds
        for mode in MODES:
            measure(mode)
    for repeat in range(REPEATS):
        order = list(MODES)
        if repeat % 2:
            order.reverse()
        for mode in order:
            outcome = measure(mode)
            samples[mode].append(outcome["median_step_s"])
            events[mode] = outcome["events"]
            per_event.setdefault(mode, []).append(outcome["us_per_event"])
    return {"samples": samples, "events": events, "per_event": per_event}


def per_event_cost() -> Dict[str, float]:
    """The cost of emitting one event, measured on the observer alone."""
    def emit_n(n: int) -> float:
        from training_debugger.observer import MemoryWriter as _W
        from training_debugger.schema import EventPriority, EventType
        from training_debugger.schema.event import MetricUnit
        writer = _W()
        observer = TrainingObserver(
            "emit", writer=writer, sampling=SamplingPolicy.uniform(1),
            config=ObserverConfig(flush_interval_s=0.001, batch_size=65536),
            rank=0, world_size=1,
        )
        for step in range(1, n + 1):
            observer.metric("grad.global_norm", float(step), MetricUnit.DIMENSIONLESS,
                            step=step, event_type=EventType.GRADIENT_METRIC,
                            priority=EventPriority.P2)
        observer.flush(timeout=60.0)
        observer.close()
        return statistics.median([]) if False else 0.0

    def timed(n: int) -> float:
        from training_debugger.observer import MemoryWriter as _W
        from training_debugger.schema import EventPriority, EventType
        from training_debugger.schema.event import MetricUnit
        writer = _W()
        observer = TrainingObserver(
            "emit", writer=writer, sampling=SamplingPolicy.uniform(1),
            config=ObserverConfig(flush_interval_s=0.001, batch_size=65536),
            rank=0, world_size=1,
        )
        started = time.perf_counter()
        for step in range(1, n + 1):
            observer.metric("grad.global_norm", float(step), MetricUnit.DIMENSIONLESS,
                            step=step, event_type=EventType.GRADIENT_METRIC,
                            priority=EventPriority.P2)
        elapsed = time.perf_counter() - started
        observer.flush(timeout=60.0)
        observer.close()
        return elapsed

    emit_n(200)                             # warm-up
    small = statistics.median([timed(2000) for _ in range(7)])
    large = statistics.median([timed(20000) for _ in range(7)])
    return {
        "us_per_event": 1e6 * (large - small) / (20000 - 2000),
        "small_s": small, "large_s": large,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"steps": STEPS, "warmup": WARMUP, "repeats": REPEATS}
    report["emit_cost"] = per_event_cost()
    paired_result = paired()
    report.update(paired_result)

    samples = report["samples"]
    base = statistics.median(samples["off"])
    report["summary"] = {}
    for mode in MODES:
        values = samples[mode]
        median = statistics.median(values)
        stderr = (statistics.stdev(values) / (len(values) ** 0.5)) if len(values) > 1 else 0.0
        report["summary"][mode] = {
            "median_step_s": median, "stderr_s": stderr,
            "added_ms": (median - base) * 1e3,
            "added_ms_stderr": ((stderr ** 2 + (statistics.stdev(samples["off"]) /
                                                (len(samples["off"]) ** 0.5)) ** 2) ** 0.5) * 1e3
            if mode != "off" else 0.0,
            "relative_pct": 100.0 * (median - base) / base if mode != "off" else 0.0,
            "events": report["events"][mode],
            "us_per_event": (statistics.median([v for v in paired_result["per_event"].get(mode, []) if v])
                             if any(paired_result["per_event"].get(mode)) else None),
            "n": len(values),
        }
    (OUT / "PERFORMANCE_AUDIT.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("per-event emit cost: {:.3f} us  (2000 -> 20000 events, 7 reps)".format(
        report["emit_cost"]["us_per_event"]))
    print()
    print("%-16s %-11s %-21s %-11s %-9s %-7s %s" % (
        "mode", "median ms", "added vs off", "relative", "events", "ev/step", "emit/ev"))
    for mode in MODES:
        row = report["summary"][mode]
        added = ("baseline" if mode == "off" else
                 "{:+.3f} ± {:.3f} ms".format(row["added_ms"], row["added_ms_stderr"]))
        rel = ("-" if mode == "off" else "{:+.1f}%".format(row["relative_pct"]))
        inst = STEPS - WARMUP
        ev_step = row["events"] / inst if inst else 0
        cost = ("-" if row.get("us_per_event") is None
                else "{:.2f} us".format(row["us_per_event"]))
        print("%-16s %-11.4f %-21s %-11s %-9d %-7.1f %s" % (
            mode, row["median_step_s"] * 1e3, added, rel, row["events"], ev_step, cost))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
