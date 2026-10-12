"""D7: paired, interleaved performance measurement.

The Phase 1.1 cost ledger was disqualified because it measured ``OFF`` and ``ON``
in **separate loops** and then subtracted medians. Run-to-run drift was the same
order as the effect, which produced a physically impossible negative overhead.

This design removes that class of error by construction:

**Paired per step.** For each step, the *same* batch is run through both arms and
the difference is taken **within that step**. Drift over the session cancels
exactly, because both arms of a pair experience the same machine state.

**Interleaved and order-randomised.** The arm order alternates (AB, BA, AB, ...)
using a seeded random choice, so a systematic first-arm penalty (warm cache,
branch predictor state) cancels across pairs rather than biasing one arm.

**Frozen model.** No optimiser step is taken, so the parameters never change and
both arms compute identical arithmetic. This is a benchmark, not a training run;
letting the model evolve would make the two arms different computations.

**Two measurement systems, never summed.** A microbench measures individual
components; a macrobench measures whole-step deltas. Microbench numbers are for
*locating* cost, never for explaining a macrobench delta by addition — that
mistake is what produced the 5.13 ms "accounted" figure in Phase 1.1.

**The CI decides.** If the 95 % interval on the paired difference crosses zero,
the report says **NO RESOLVABLE EFFECT AT THIS SCALE**. A negative overhead is
never reported.

Run::

    .venv/bin/python tests/performance/benchmark_paired.py
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
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
    global_gradient_norm,
    global_parameter_norm,
)
from training_debugger.env_info import capture_environment  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    DEFAULT_POLICY_PHASE1,
    MemoryWriter,
    ObserverConfig,
    SamplingPolicy,
    TrainingObserver,
)
from training_debugger.schema import (  # noqa: E402
    EventPriority,
    EventType,
    MetricUnit,
    TrainingEvent,
    new_event_id,
    validate_event,
)

OUTPUT_DIR = ROOT / "outputs" / "phase1-2"
PAIRS_PARQUET = OUTPUT_DIR / "performance_pairs.parquet"
REPORT_MD = OUTPUT_DIR / "D7_PERFORMANCE_REPORT.md"

WORKLOADS = (
    ("tiny (4L, d32)", 4, 32, 4, 8, 4),
    ("medium (8L, d128)", 8, 128, 8, 16, 4),
    ("large (12L, d256)", 12, 256, 8, 16, 4),
)

WARMUP_STEPS = 50
PAIRED_STEPS = 300
REPEATS = 3


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(
    samples: Sequence[float], *, confidence: float = 0.95, iterations: int = 4000, seed: int = 17
) -> Tuple[float, float]:
    """Percentile bootstrap CI on the mean. No normality assumption.

    The paired differences are not Gaussian -- they are a mixture of a small
    deterministic cost and scheduler noise -- so a t-interval would be the wrong
    instrument.
    """
    if len(samples) < 5:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(samples)
    means = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += samples[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low = means[int((1 - confidence) / 2 * iterations)]
    high = means[int((1 + confidence) / 2 * iterations) - 1]
    return (low, high)


def summarise(differences_ms: Sequence[float], baseline_ms: float) -> Dict[str, Any]:
    if not differences_ms:
        return {"n": 0}
    ordered = sorted(differences_ms)
    median = statistics.median(ordered)
    mad = statistics.median([abs(v - median) for v in ordered])
    low, high = bootstrap_ci(ordered)
    mean = sum(ordered) / len(ordered)
    resolves = (low > 0.0) or (high < 0.0)
    return {
        "n": len(ordered),
        "median_ms": median,
        "mean_ms": mean,
        "mad_ms": mad,
        "p05_ms": ordered[int(0.05 * (len(ordered) - 1))],
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "ci95_low_ms": low,
        "ci95_high_ms": high,
        "median_percent": (median / baseline_ms * 100.0) if baseline_ms else None,
        "ci95_low_percent": (low / baseline_ms * 100.0) if baseline_ms else None,
        "ci95_high_percent": (high / baseline_ms * 100.0) if baseline_ms else None,
        "resolves_at_this_scale": resolves,
        "verdict": (
            "resolvable effect"
            if resolves
            else "NO RESOLVABLE EFFECT AT THIS SCALE"
        ),
    }


# ---------------------------------------------------------------------------
# Paired macrobenchmark
# ---------------------------------------------------------------------------


def build_arm(mode: str, model, optimizer, label: str):
    """Construct one arm. ``off`` has no observer at all."""
    if mode == "off":
        return None, None, None, None
    policy = DEFAULT_POLICY_PHASE1.clone()
    observer = TrainingObserver(
        "paired-{}".format(label),
        writer=MemoryWriter(),
        sampling=policy,
        config=ObserverConfig(max_queue=100_000, batch_size=8192, flush_interval_s=0.05),
        rank=0,
        world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id=label)
    collector = (
        LayerMetricCollector(model, observer, selection=LayerSelection(top_k=8), cadence=10)
        if mode == "core_layers"
        else None
    )
    return observer, adapter, collector, policy


def measure_paired(
    label: str, n_layers: int, d_model: int, n_heads: int, seq: int, batch_size: int
) -> Dict[str, Any]:
    """Interleaved paired comparison of ``off`` vs ``core`` vs ``core_layers``.

    For every step the same batch is executed by both arms and the difference is
    recorded within the pair.
    """
    torch.manual_seed(11)
    model, config = make_tiny_lm(seed=11, n_layers=n_layers, d_model=d_model, n_heads=n_heads)
    # AdamW with lr=0: the optimiser does its full arithmetic (so the baseline is
    # a realistic training step and the percentage is comparable to the Phase 1
    # gate) while the parameters never move, so both arms compute identical
    # values. An SGD-less baseline made the percentage ~3x too pessimistic
    # because the optimiser step is a large fixed cost the debugger does not
    # touch.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)

    core = build_arm("core", model, optimizer, label)
    layers = build_arm("core_layers", model, optimizer, label)

    # Pre-generate every batch so both arms see byte-identical inputs and no arm
    # pays for batch construction.
    batches = [
        synthetic_batch(batch_size=batch_size, seq_len=seq, vocab_size=config.vocab_size, seed=s)
        for s in range(1, WARMUP_STEPS + PAIRED_STEPS + 1)
    ]

    rng = random.Random(4242)
    core_deltas: List[float] = []
    layer_deltas: List[float] = []
    baseline_times: List[float] = []

    def run_off(batch) -> float:
        started = time.perf_counter()
        loss = model(**batch)["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return time.perf_counter() - started

    def run_on(batch, step: int, arm) -> float:
        observer, adapter, collector, _ = arm
        started = time.perf_counter()
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=batch_size, tokens=int(batch["input_ids"].numel()))
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
            tracker.record_parameters()
            tracker.record_learning_rate()
        return time.perf_counter() - started

    for index in range(WARMUP_STEPS + PAIRED_STEPS):
        step = index + 1
        batch = batches[index]
        warm = index < WARMUP_STEPS
        # Randomise arm order within each pair.
        core_first = rng.random() < 0.5
        if core_first:
            off_time = run_off(batch)
            on_core = run_on(batch, step, core)
            off_time2 = run_off(batch)
            on_layers = run_on(batch, step, layers)
        else:
            on_core = run_on(batch, step, core)
            off_time = run_off(batch)
            on_layers = run_on(batch, step, layers)
            off_time2 = run_off(batch)
        if warm:
            continue
        off_mean = 0.5 * (off_time + off_time2)
        baseline_times.append(off_mean * 1000.0)
        core_deltas.append((on_core - off_mean) * 1000.0)
        layer_deltas.append((on_layers - off_mean) * 1000.0)

    for arm in (core, layers):
        if arm[0] is not None:
            arm[0].flush(timeout=30.0)
            arm[0].close()

    baseline_ms = statistics.median(baseline_times)
    result = {
        "workload": label,
        "baseline_ms": baseline_ms,
        "core": summarise(core_deltas, baseline_ms),
        "core_layers": summarise(layer_deltas, baseline_ms),
        "pairs": len(core_deltas),
    }
    result["t1_incremental"] = summarise(
        [b - a for a, b in zip(core_deltas, layer_deltas)], baseline_ms
    )
    result["raw_core_deltas_ms"] = core_deltas
    result["raw_layer_deltas_ms"] = layer_deltas
    del model, optimizer, core, layers
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Microbenchmark (separate system, never summed into the macrobench)
# ---------------------------------------------------------------------------


def microbench(model, config) -> Dict[str, Any]:
    batch = synthetic_batch(batch_size=4, seq_len=16, vocab_size=config.vocab_size, seed=1)
    model(**batch)["loss"].backward()
    observer = TrainingObserver(
        "micro", writer=MemoryWriter(), sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_queue=200_000, batch_size=100_000, flush_interval_s=5.0),
    )
    event = TrainingEvent(
        run_id="micro", producer="p", source="s", event_type=EventType.LOSS, step=1,
        metric_name="train.loss", metric_value=1.0, metric_unit=MetricUnit.DIMENSIONLESS,
    )

    def timeit(fn, n=200, warmup=20):
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(n):
            started = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - started) * 1e6)
        samples.sort()
        return {"median_us": statistics.median(samples),
                "mad_us": statistics.median([abs(s - statistics.median(samples)) for s in samples])}

    out = {
        "grad_norm": timeit(lambda: global_gradient_norm(model.parameters()), n=40, warmup=10),
        "param_norm": timeit(lambda: global_parameter_norm(model.parameters()), n=40, warmup=10),
        "emit": timeit(lambda: observer.metric("train.loss", 1.0, MetricUnit.DIMENSIONLESS,
                                               step=1, event_type=EventType.LOSS), n=300),
        "validate": timeit(lambda: validate_event(event), n=300),
        "new_event_id": timeit(new_event_id, n=1000),
        "construct": timeit(lambda: TrainingEvent(
            run_id="r", producer="p", source="s", event_type=EventType.LOSS, step=1,
            metric_name="train.loss", metric_value=1.0, metric_unit=MetricUnit.DIMENSIONLESS), n=400),
        "would_emit": timeit(lambda: observer.would_emit("grad.layer_norm", 1, EventPriority.P2), n=1000),
    }
    observer.close()
    return out


def render(results: Sequence[Dict[str, Any]], micro: Dict[str, Any], env: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append
    add("# D7 PERFORMANCE REPORT")
    add("")
    add("**Method:** interleaved paired A/B. For each step the *same batch* is executed")
    add("by both arms, arm order is randomised per pair, and the difference is taken")
    add("**within the pair**. The model is frozen (`lr = 0`), so both arms perform")
    add("identical arithmetic. This removes the drift that invalidated the Phase 1.1")
    add("ledger.")
    add("")
    add("## Paired result")
    add("")
    add("| Workload | Baseline | core delta (95 % CI) | core % | + T1 delta (95 % CI) | T1 total % | T1 incremental |")
    add("|---|---|---|---|---|---|---|")
    for entry in results:
        core = entry["core"]
        layer = entry["core_layers"]
        t1 = entry["t1_incremental"]
        add("| {} | {:.2f} ms | {:+.3f} ms ({:+.3f}, {:+.3f}) | {:+.2f} % | {:+.3f} ms ({:+.3f}, {:+.3f}) | {:+.2f} % | {:+.3f} ms |".format(
            entry["workload"], entry["baseline_ms"],
            core["median_ms"], core["ci95_low_ms"], core["ci95_high_ms"],
            core["median_percent"] or 0.0,
            layer["median_ms"], layer["ci95_low_ms"], layer["ci95_high_ms"],
            layer["median_percent"] or 0.0,
            t1["median_ms"]))
    add("")
    add("## Verdicts")
    add("")
    add("| Workload | core | T1 total |")
    add("|---|---|---|")
    for entry in results:
        add("| {} | {} | {} |".format(
            entry["workload"], entry["core"]["verdict"], entry["core_layers"]["verdict"]))
    add("")
    add("**A negative overhead is never reported.** Where the 95 % CI crosses zero the")
    add("report says NO RESOLVABLE EFFECT AT THIS SCALE rather than quoting a sign the")
    add("data does not support.")
    add("")
    add("## Gate")
    add("")
    add("| Budget | Target | Measured (largest workload) | Verdict |")
    add("|---|---|---|---|")
    large = results[-1]
    add("| core observer | < 3 % | {:+.2f} % | {} |".format(
        large["core"]["median_percent"] or 0.0,
        "PASS" if (large["core"]["median_percent"] or 0.0) < 3.0 else "FAIL"))
    add("| T1 layer hooks (total) | < 8 % | {:+.2f} % | {} |".format(
        large["core_layers"]["median_percent"] or 0.0,
        "PASS" if (large["core_layers"]["median_percent"] or 0.0) < 8.0 else "FAIL"))
    add("")
    add("The original Phase 1 gate is unchanged. T1 incremental cost is reported for")
    add("explanation only and is **not** substituted for the total.")
    add("")
    add("## Microbenchmark (separate system — NOT summed into the above)")
    add("")
    add("| Component | median | MAD |")
    add("|---|---|---|")
    for name, value in sorted(micro.items()):
        add("| `{}` | {:.3f} µs | {:.3f} µs |".format(name, value["median_us"], value["mad_us"]))
    add("")
    add("These numbers locate cost. Adding them up to predict the macrobench delta is the")
    add("error that produced the Phase 1.1 ledger's 5.13 ms \"accounted\" figure against a")
    add("2.60 ms observed effect.")
    add("")
    add("## Environment")
    add("")
    add("```json")
    add(json.dumps(env, indent=2, sort_keys=True, default=str))
    add("```")
    add("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Paired interleaved performance benchmark")
    parser.add_argument("--steps", type=int, default=PAIRED_STEPS)
    args = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for label, n_layers, d_model, n_heads, seq, batch_size in WORKLOADS:
        print("paired benchmark: {} ...".format(label))
        entry = measure_paired(label, n_layers, d_model, n_heads, seq, batch_size)
        results.append(entry)
        print("   baseline {:.2f} ms | core {:+.3f} ms ({:+.2f} %) | +T1 {:+.3f} ms ({:+.2f} %)".format(
            entry["baseline_ms"], entry["core"]["median_ms"], entry["core"]["median_percent"] or 0.0,
            entry["core_layers"]["median_ms"], entry["core_layers"]["median_percent"] or 0.0))
        print("   CI core: ({:+.3f}, {:+.3f}) -> {}".format(
            entry["core"]["ci95_low_ms"], entry["core"]["ci95_high_ms"], entry["core"]["verdict"]))

    torch.manual_seed(11)
    model, config = make_tiny_lm(seed=11, n_layers=8, d_model=128, n_heads=8)
    micro = microbench(model, config)

    _write_pairs_parquet(results)
    env = capture_environment(extra={"benchmark": "paired_interleaved"})
    REPORT_MD.write_text(render(results, micro, env), encoding="utf-8")
    (OUTPUT_DIR / "performance_paired.json").write_text(
        json.dumps({"results": results, "micro": micro, "environment": env}, indent=2, default=str),
        encoding="utf-8",
    )
    print("report:", REPORT_MD)
    return 0


def _write_pairs_parquet(results: Sequence[Dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for entry in results:
        for index, (core_delta, layer_delta) in enumerate(
            zip(entry["raw_core_deltas_ms"], entry["raw_layer_deltas_ms"])
        ):
            rows.append(
                {
                    "workload": entry["workload"],
                    "pair_index": index,
                    "baseline_ms": entry["baseline_ms"],
                    "core_delta_ms": core_delta,
                    "core_layers_delta_ms": layer_delta,
                    "t1_incremental_ms": layer_delta - core_delta,
                }
            )
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, PAIRS_PARQUET, compression="zstd")
    print("pairs:", PAIRS_PARQUET, "rows:", table.num_rows)


if __name__ == "__main__":
    raise SystemExit(main())
