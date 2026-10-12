"""W4 performance gate: observer and T1 layer-hook overhead.

The brief sets two budgets: core observer overhead **< 3%** median, T1 layer
hooks **< 8%**. This script measures four configurations against the same real
training workload and reports:

* **absolute** cost per step (ms) -- the transferable number
* **relative** cost against the same workload with the debugger off
* tokens/second delta
* Python heap delta and device memory delta

One caveat is stated up front in the report rather than buried: this benchmark runs
a deliberately tiny model on a CPU/MPS device, so a *fixed* per-step cost appears
as a *large percentage*. On a production model whose step takes seconds, the same
absolute cost is negligible. Both numbers are reported so neither can be quoted
alone.

Run::

    .venv/bin/python tests/performance/benchmark_overhead.py
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from training_debugger.env_info import capture_environment  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    NullWriter,
    ObserverConfig,
    SamplingPolicy,
    TrainingObserver,
)
from training_debugger.schema import EventType, MetricUnit  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "phase1" / "w4"
REPORT = OUTPUT_DIR / "OVERHEAD_BENCHMARK.md"
RESULTS = OUTPUT_DIR / "overhead_results.json"

WARMUP_STEPS = 20
MEASURED_STEPS = 120
REPEATS = 3
BATCH_SIZE = 4
SEQ_LEN = 8
N_LAYERS = 4
SEED = 4321

#: ``shipping`` uses DEFAULT_POLICY_PHASE1 -- the configuration that actually
#: ships -- so the budget verdict is not based on a uniform(1) worst case alone.
MODES = ("off", "observer_only", "core", "core_layers", "shipping")

#: Cadence for layer metrics in the benchmark. Matches DEFAULT_POLICY_PHASE1 so
#: the benchmark measures the configuration that actually ships.
LAYER_CADENCE = 10


def _build(mode: str):
    """Fresh, deterministic training setup for one measurement."""
    torch.manual_seed(SEED)
    model, config = make_tiny_lm(seed=SEED, n_layers=N_LAYERS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    observer, adapter, collector = None, None, None
    if mode != "off":
        if mode == "shipping":
            from training_debugger.observer import DEFAULT_POLICY_PHASE1
            sampling = DEFAULT_POLICY_PHASE1.clone()
        else:
            sampling = SamplingPolicy.uniform(1)
        observer = TrainingObserver(
            "bench-overhead",
            writer=NullWriter(),
            sampling=sampling,
            config=ObserverConfig(max_queue=20_000, batch_size=1024, flush_interval_s=0.05),
        )
        adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="tiny")
        if mode in ("core_layers", "shipping"):
            collector = LayerMetricCollector(
                model, observer, selection=LayerSelection(top_k=8), cadence=LAYER_CADENCE
            )
    return model, config, optimizer, observer, adapter, collector


def _train_step(step: int, mode: str, model, config, optimizer, adapter, collector) -> int:
    """One identical training step for every mode. Returns tokens processed."""
    batch = synthetic_batch(
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, vocab_size=config.vocab_size, seed=step
    )
    if mode == "off":
        loss = model(**batch)["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return int(batch["input_ids"].numel())

    with adapter.step(step) as tracker:
        tracker.set_batch(samples=BATCH_SIZE, tokens=int(batch["input_ids"].numel()))
        with tracker.phase("forward"):
            loss = model(**batch)["loss"]
        tracker.record_loss(loss)
        with tracker.phase("backward"):
            loss.backward()
        tracker.record_gradients()
        if collector is not None:
            collector.collect(step)
        with tracker.phase("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if mode in ("core", "core_layers", "shipping"):
            tracker.record_parameters()
            tracker.record_learning_rate()
    return int(batch["input_ids"].numel())


def measure(mode: str) -> Dict[str, Any]:
    """Run one configuration and return timing and memory statistics."""
    per_repeat: List[Dict[str, Any]] = []
    for _ in range(REPEATS):
        model, config, optimizer, observer, adapter, collector = _build(mode)
        # Warm-up: the first steps include allocator and lazy-init costs that are
        # not part of steady-state overhead.
        for step in range(1, WARMUP_STEPS + 1):
            _train_step(step, mode, model, config, optimizer, adapter, collector)

        gc.collect()
        tracemalloc.start()
        baseline_heap = tracemalloc.get_traced_memory()[0]
        device_before = _device_bytes()

        durations: List[float] = []
        tokens = 0
        for step in range(WARMUP_STEPS + 1, WARMUP_STEPS + MEASURED_STEPS + 1):
            started = time.perf_counter()
            tokens += _train_step(step, mode, model, config, optimizer, adapter, collector)
            durations.append(time.perf_counter() - started)

        heap_peak = tracemalloc.get_traced_memory()[1]
        heap_delta = heap_peak - baseline_heap
        device_after = _device_bytes()
        tracemalloc.stop()

        stats = None
        if observer is not None:
            observer.flush(timeout=30.0)
            identity = observer.accounting_identity()
            stats = {
                "observer": observer.stats().to_dict(),
                "identity": identity,
                "collector": collector.stats() if collector is not None else None,
            }
            observer.close()

        per_repeat.append(
            {
                "durations": durations,
                "median_step_s": statistics.median(durations),
                "p95_step_s": _percentile(durations, 95),
                "mean_step_s": statistics.fmean(durations),
                "total_s": sum(durations),
                "tokens": tokens,
                "tokens_per_sec": tokens / sum(durations) if durations else 0.0,
                "heap_peak_bytes": heap_peak,
                "heap_delta_bytes": heap_delta,
                "device_delta_bytes": (
                    None if device_before is None or device_after is None
                    else device_after - device_before
                ),
                "stats": stats,
            }
        )
        del model, optimizer, observer, adapter, collector
        gc.collect()

    medians = [r["median_step_s"] for r in per_repeat]
    return {
        "mode": mode,
        "repeats": REPEATS,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "median_step_s": statistics.median(medians),
        "median_step_s_per_repeat": medians,
        "p95_step_s": statistics.median([r["p95_step_s"] for r in per_repeat]),
        "tokens_per_sec": statistics.median([r["tokens_per_sec"] for r in per_repeat]),
        "heap_peak_bytes": max(r["heap_peak_bytes"] for r in per_repeat),
        "device_delta_bytes": next(
            (r["device_delta_bytes"] for r in per_repeat if r["device_delta_bytes"] is not None),
            None,
        ),
        "stats": per_repeat[-1]["stats"],
    }


def _device_bytes() -> Optional[int]:
    try:
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "current_allocated_memory") and mps.is_available():
            return int(mps.current_allocated_memory())
    except Exception:
        return None
    return None


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def render(results: Dict[str, Dict[str, Any]], env: Dict[str, Any]) -> str:
    base = results["off"]
    lines: List[str] = []
    add = lines.append

    add("# W4 OVERHEAD BENCHMARK")
    add("")
    add("**Generated by:** `tests/performance/benchmark_overhead.py`")
    add("**Workload:** tiny transformer, {} layers, batch {} x seq {}, {} warm-up + {} measured steps, {} repeats.".format(
        N_LAYERS, BATCH_SIZE, SEQ_LEN, WARMUP_STEPS, MEASURED_STEPS, REPEATS))
    add("**Budgets:** core observer median < 3 %, T1 layer hooks median < 8 %.")
    add("")

    add("## Headline")
    add("")
    add("| Mode | Median step | Added vs off | **Relative overhead** | Tokens/s |")
    add("|---|---|---|---|---|")
    for mode in MODES:
        entry = results[mode]
        added_ms = (entry["median_step_s"] - base["median_step_s"]) * 1000
        relative = (
            (entry["median_step_s"] - base["median_step_s"]) / base["median_step_s"] * 100.0
        )
        label = "baseline" if mode == "off" else "{:+.2f} ms".format(added_ms)
        pct = "—" if mode == "off" else "**{:+.1f} %**".format(relative)
        add("| `{}` | {:.2f} ms | {} | {} | {:,.0f} |".format(
            mode, entry["median_step_s"] * 1000, label, pct, entry["tokens_per_sec"]))
    add("")

    core_added = results["core"]["median_step_s"] - base["median_step_s"]
    layers_added = results["core_layers"]["median_step_s"] - base["median_step_s"]
    core_pct = core_added / base["median_step_s"] * 100.0
    layers_pct = layers_added / base["median_step_s"] * 100.0
    core_only = results["core_layers"]["median_step_s"] - results["core"]["median_step_s"]
    core_only_pct = core_only / base["median_step_s"] * 100.0

    add("### Budget verdict")
    add("")
    add("| Budget | Target | Measured | Verdict |")
    add("|---|---|---|---|")
    add("| observer + core metrics | < 3 % | {:+.1f} % ({:+.2f} ms/step) | **{}** |".format(
        core_pct, core_added * 1000, "PASS" if core_pct < 3.0 else "FAIL"))
    add("| T1 layer hooks (total) | < 8 % | {:+.1f} % ({:+.2f} ms/step) | **{}** |".format(
        layers_pct, layers_added * 1000, "PASS" if layers_pct < 8.0 else "FAIL"))
    add("| T1 layer hooks (incremental over core) | — | {:+.1f} % ({:+.2f} ms/step) | reported |".format(
        core_only_pct, core_only * 1000))
    add("")

    add("## The scaling caveat, stated plainly")
    add("")
    add("This benchmark trains a **tiny** model, so one step takes about {:.2f} ms and a".format(
        base["median_step_s"] * 1000))
    add("fixed per-step cost is a large fraction of it. The absolute cost is what")
    add("transfers to a real workload:")
    add("")
    add("* core observer: **{:+.2f} ms per step**".format(core_added * 1000))
    add("* with T1 layer hooks: **{:+.2f} ms per step**".format(layers_added * 1000))
    add("")
    add("On a production step of, say, 2000 ms, {:.2f} ms is {:.3f} % — but that".format(
        layers_added * 1000, layers_added / 2.0 * 100 / 1000 * 100))
    add("projection is an *extrapolation from this machine* and is **NOT VERIFIED** here.")
    add("The only verified claims are the absolute per-step costs above, measured on the")
    add("workload described, on the hardware in the environment block.")
    add("")

    add("## Detail per mode")
    add("")
    for mode in MODES:
        entry = results[mode]
        add("### `{}`".format(mode))
        add("")
        add("- median step: **{:.2f} ms** (per repeat: {})".format(
            entry["median_step_s"] * 1000,
            ", ".join("{:.2f}".format(v * 1000) for v in entry["median_step_s_per_repeat"])))
        add("- p95 step: {:.2f} ms".format(entry["p95_step_s"] * 1000))
        add("- tokens/s: {:,.0f}".format(entry["tokens_per_sec"]))
        add("- peak Python heap (tracemalloc): {:.2f} MB".format(entry["heap_peak_bytes"] / 1048576))
        device = entry["device_delta_bytes"]
        add("- device memory delta: {}".format(
            "not reported on this backend" if device is None else "{:.2f} MB".format(device / 1048576)))
        stats = entry.get("stats")
        if stats:
            observer_stats = stats["observer"]
            add("- events written: {:,}; dropped: {:,}; queue overflow: {:,}".format(
                observer_stats["written_total"], observer_stats["events_dropped_total"],
                observer_stats["queue_overflow_total"]))
            add("- unaccounted events: {}".format(stats["identity"]["unaccounted"]))
            add("- emit cost per event: {:.2f} us".format(
                (observer_stats["emit_time_total_ns"] / max(1, observer_stats["emitted_total"])) / 1000))
            collector = stats.get("collector")
            if collector:
                add("- layer collector: {} of {} steps collected (cadence {}), {} events, {} layers".format(
                    collector["steps_collected"], collector["steps_considered"], collector["cadence"],
                    collector["events_emitted"], collector["layers_selected"]))
        add("")

    add("## Environment")
    add("")
    add("```json")
    add(json.dumps(env, indent=2, sort_keys=True, default=str))
    add("```")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        print("measuring mode: {} ...".format(mode))
        results[mode] = measure(mode)
        print("  median step {:.2f} ms".format(results[mode]["median_step_s"] * 1000))

    env = capture_environment(
        extra={
            "benchmark": "observer_overhead",
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "repeats": REPEATS,
            "layer_cadence": LAYER_CADENCE,
            "model_layers": N_LAYERS,
            "batch": [BATCH_SIZE, SEQ_LEN],
        }
    )
    RESULTS.write_text(
        json.dumps({"results": results, "environment": env}, indent=2, default=str),
        encoding="utf-8",
    )
    REPORT.write_text(render(results, env), encoding="utf-8")
    print("report:", REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
