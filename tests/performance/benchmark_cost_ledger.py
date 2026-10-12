"""NOTE: the residual this script reports has been NEGATIVE on tiny and medium
workloads (accounted 3.86 ms against observed 0.41 ms), which is not a valid
attribution -- the parts cannot cost more than the whole. The micro-costs are measured
in tight loops where they are cache-hot and uncontended, and they do not sum in situ.

It also reports the MINIMUM of per-repeat medians, while ``benchmark_overhead.py``
reports the MEDIAN. The two are not comparable and differed by 5x on the same workload.

For a defensible number use ``tests/performance/audit_observer_cost.py``, which pairs
the arms inside each repeat and reports a confidence interval. See
``outputs/v0.1.8-alpha/PERFORMANCE_AUDIT.md``.


C0: the observer cost ledger.

Phase 1 left ~0.8 ms/step of the largest-workload overhead **unattributed**. A
number that large cannot be left as "unknown": it is 60 % of the total cost, so
any optimisation decision taken on top of it is guesswork.

This module measures each component of the debugger's per-step cost in isolation,
on the same three workloads the overhead benchmark uses, and reconciles the parts
against the whole:

    sum(measured components) + residual  ~=  observed overhead

The residual is reported even when it is uncomfortable. If it cannot be brought
under the run-to-run dispersion, that fact is the finding.

Components measured, each with the reason it is a candidate:

===========================  ==========================================
component                    why it might cost
===========================  ==========================================
metric computation           reductions over parameter tensors
tensor traversal             ``named_parameters()``/``parameters()`` walks
grad norm                    fused L2 over gradients
param norm                   fused L2 over parameters
loss extraction              ``.item()`` on the loss -> a device sync
memory snapshot              allocator query; may sync
Python object construction   a 39-field dataclass per event
enum conversion              ``str(Enum)`` per field per event
UUID / id generation         ULID shaping per event
timestamp calls              ``time.time_ns`` + ``time.monotonic_ns`` per event
dict creation                ``to_dict``/metadata copies
validation                   ~34 checks per event
sampling decision            ``would_emit`` + policy matching
queue insertion              lock, eviction scan, counters
serialization preparation    JSON sizing for the metadata bound
writer interaction           writer-thread handoff and batch write
thread synchronization       locks, event waits, GIL contention
allocation / GC              object churn
Trackio disabled path        the C2 presenter when no sink is configured
===========================  ==========================================

Run::

    .venv/bin/python tests/performance/benchmark_cost_ledger.py
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    memory_snapshot,
)
from training_debugger.env_info import capture_environment  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    DEFAULT_POLICY_PHASE1,
    BoundedEventQueue,
    NullWriter,
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
from training_debugger.schema.validation import ProducerSequenceTracker  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "phase1-1" / "performance"
LEDGER_MD = OUTPUT_DIR / "OBSERVER_COST_LEDGER.md"
LEDGER_JSON = OUTPUT_DIR / "cost_ledger_results.json"

WARMUP = 50
MEASURED = 200
REPEATS = 5

WORKLOADS = (
    ("tiny (4L, d32)", 4, 32, 4, 8, 4),
    ("medium (8L, d128)", 8, 128, 8, 16, 4),
    ("large (12L, d256)", 12, 256, 8, 16, 4),
)


def bench_us(fn: Callable[[], Any], *, warmup: int = 50, n: int = 400) -> Dict[str, float]:
    """Median and p95 wall time in microseconds, plus MAD for dispersion."""
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(n):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e6)
    samples.sort()
    median = statistics.median(samples)
    return {
        "median_us": median,
        "p95_us": samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))],
        "mad_us": statistics.median([abs(s - median) for s in samples]),
        "n": n,
    }


def bench_ms(fn: Callable[[], Any], *, warmup: int = 20, n: int = 60) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(n):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    median = statistics.median(samples)
    return {
        "median_ms": median,
        "p95_ms": samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))],
        "mad_ms": statistics.median([abs(s - median) for s in samples]),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Component micro-benchmarks
# ---------------------------------------------------------------------------


def scaling_component_costs(model, device) -> Dict[str, Dict[str, float]]:
    """Components whose cost depends on the model, measured on THIS model.

    ``global_gradient_norm`` cost 0.14 ms on a 4-layer d32 model and 2.44 ms on a
    12-layer d256 model. Applying one model's figure to every workload made the
    reconciliation meaningless -- the first version of this ledger reported a
    negative residual because a single 8L d128 measurement was subtracted from
    three different workloads' observed overhead.
    """
    batch = synthetic_batch(batch_size=4, seq_len=16, vocab_size=32, seed=1)
    try:
        model(**batch)["loss"].backward()
    except Exception:
        pass
    costs: Dict[str, Dict[str, float]] = {}
    costs["tensor_traversal_parameters"] = bench_us(lambda: list(model.parameters()), n=200)
    costs["grad_norm"] = bench_ms(lambda: global_gradient_norm(model.parameters()), warmup=10, n=40)
    costs["param_norm"] = bench_ms(lambda: global_parameter_norm(model.parameters()), warmup=10, n=40)
    return costs


def component_costs(model, config, device) -> Dict[str, Dict[str, float]]:
    """Measure each candidate component on one model instance."""
    batch = synthetic_batch(batch_size=4, seq_len=16, vocab_size=config.vocab_size, seed=1)
    model(**batch)["loss"].backward()

    observer = TrainingObserver(
        "ledger",
        writer=NullWriter(),
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_queue=200_000, batch_size=100_000, flush_interval_s=5.0),
        autostart=True,
    )
    collector = LayerMetricCollector(
        model, observer, selection=LayerSelection(top_k=8), cadence=10
    )
    counts = {
        "parameters": sum(1 for _ in model.parameters()),
        "named_parameters": sum(1 for _ in model.named_parameters()),
        "grad_tensors": sum(1 for p in model.parameters() if p.grad is not None),
    }

    event = TrainingEvent(
        run_id="ledger", producer="pytorch-adapter:r0:pid1", source="pytorch-adapter",
        event_type=EventType.LOSS, step=10, source_seq=10,
        metric_name="train.loss", metric_value=1.5, metric_unit=MetricUnit.DIMENSIONLESS,
    )

    costs: Dict[str, Dict[str, float]] = {}
    costs["tensor_traversal_parameters"] = bench_us(lambda: list(model.parameters()), n=300)
    costs["tensor_traversal_named_parameters"] = bench_us(
        lambda: list(model.named_parameters()), n=300
    )
    costs["grad_norm"] = bench_ms(lambda: global_gradient_norm(model.parameters()))
    costs["param_norm"] = bench_ms(lambda: global_parameter_norm(model.parameters()))
    costs["loss_item_sync"] = bench_us(
        lambda: float(model(**batch)["loss"].detach().to(torch.float32).item()), n=200
    )
    costs["memory_snapshot"] = bench_us(lambda: memory_snapshot(device), n=500)
    costs["event_construction"] = bench_us(lambda: TrainingEvent(
        run_id="r", producer="p", source="s", event_type=EventType.LOSS, step=1,
        metric_name="train.loss", metric_value=1.0, metric_unit=MetricUnit.DIMENSIONLESS,
    ), n=800)
    costs["enum_conversion"] = bench_us(lambda: (str(EventType.LOSS), str(EventPriority.P1)), n=2000)
    costs["event_id_generation"] = bench_us(lambda: new_event_id(), n=2000)
    costs["timestamp_calls"] = bench_us(lambda: (time.time_ns(), time.monotonic_ns()), n=2000)
    costs["to_dict"] = bench_us(lambda: event.to_dict(), n=800)
    costs["copy_with"] = bench_us(lambda: event.copy_with(rank=0), n=800)
    costs["metadata_size_bytes"] = bench_us(lambda: event.metadata_size_bytes(), n=800)
    costs["validate_event"] = bench_us(lambda: validate_event(event), n=800)
    costs["validate_with_tracker"] = bench_us(
        lambda: validate_event(event, seq_tracker=ProducerSequenceTracker()), n=400
    )
    costs["sampling_decision"] = bench_us(
        lambda: observer.would_emit("grad.layer_norm", 10, EventPriority.P2), n=2000
    )
    queue = BoundedEventQueue(maxlen=1000)
    costs["queue_insert"] = bench_us(lambda: queue.put(event), n=2000)
    costs["observer_emit_full"] = bench_us(
        lambda: observer.metric("train.loss", 1.0, MetricUnit.DIMENSIONLESS, step=1,
                                event_type=EventType.LOSS), n=800
    )
    costs["collector_compute"] = bench_ms(lambda: collector.compute(1), warmup=5, n=30)
    costs["json_dumps_metadata"] = bench_us(
        lambda: json.dumps(event.to_dict(), separators=(",", ":"), sort_keys=True), n=400
    )
    costs["gc_collect"] = bench_ms(lambda: gc.collect(), warmup=2, n=10)

    observer.close()
    return {"costs": costs, "model_counts": counts}


# ---------------------------------------------------------------------------
# Whole-step measurement
# ---------------------------------------------------------------------------


def _train_step(step, mode, model, config, optimizer, adapter, collector, device):
    batch = synthetic_batch(batch_size=4, seq_len=16, vocab_size=config.vocab_size, seed=step)
    if mode == "off":
        loss = model(**batch)["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return
    with adapter.step(step) as tracker:
        tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
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


def measure_mode(label, n_layers, d_model, n_heads, seq, batch_size, mode) -> Dict[str, float]:
    """Warm-up + measured steps, min of per-repeat medians across repeats."""
    per_repeat: List[float] = []
    for _ in range(REPEATS):
        torch.manual_seed(7)
        model, config = make_tiny_lm(seed=7, n_layers=n_layers, d_model=d_model, n_heads=n_heads)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        observer = adapter = collector = None
        if mode != "off":
            sampling = DEFAULT_POLICY_PHASE1.clone()
            observer = TrainingObserver(
                "ledger-run", writer=NullWriter(), sampling=sampling,
                config=ObserverConfig(max_queue=50_000, batch_size=2048, flush_interval_s=0.05),
            )
            adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id=label)
            if mode == "core_layers":
                collector = LayerMetricCollector(
                    model, observer, selection=LayerSelection(top_k=8), cadence=10
                )
        for step in range(1, WARMUP + 1):
            _train_step(step, mode, model, config, optimizer, adapter, collector, None)
        durations: List[float] = []
        for step in range(WARMUP + 1, WARMUP + MEASURED + 1):
            started = time.perf_counter()
            _train_step(step, mode, model, config, optimizer, adapter, collector, None)
            durations.append(time.perf_counter() - started)
        if observer is not None:
            observer.flush(timeout=20.0)
            observer.close()
        durations.sort()
        per_repeat.append(statistics.median(durations))
        del model, optimizer, observer, adapter, collector
        gc.collect()
    per_repeat.sort()
    return {
        "median_s": per_repeat[0],  # min of medians: contention only adds time
        "per_repeat_s": per_repeat,
    }


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def attribute_workload(label, n_layers, d_model, n_heads, seq, batch_size, costs, scaled) -> Dict[str, Any]:
    """Reconcile measured components against the observed whole-step overhead."""
    baseline = measure_mode(label, n_layers, d_model, n_heads, seq, batch_size, "off")
    core = measure_mode(label, n_layers, d_model, n_heads, seq, batch_size, "core")
    layers = measure_mode(label, n_layers, d_model, n_heads, seq, batch_size, "core_layers")

    observed_core_ms = (core["median_s"] - baseline["median_s"]) * 1000.0
    observed_t1_ms = (layers["median_s"] - baseline["median_s"]) * 1000.0
    dispersion_ms = max(
        max(core["per_repeat_s"]) - min(core["per_repeat_s"]),
        max(layers["per_repeat_s"]) - min(layers["per_repeat_s"]),
        max(baseline["per_repeat_s"]) - min(baseline["per_repeat_s"]),
    ) * 1000.0

    # Per-step contributions. Cadences come from DEFAULT_POLICY_PHASE1.
    # Workload-scaled components use THIS workload's measurement; the Python
    # components below are workload-independent and come from the shared table.
    per_step = {
        "grad_norm (every 2nd step)": scaled["grad_norm"]["median_ms"] * 0.5,
        "param_norm (every 10th step)": scaled["param_norm"]["median_ms"] * 0.1,
        "loss .item() sync (every step)": costs["loss_item_sync"]["median_us"] / 1000.0,
        "memory snapshot (every 10th step)": costs["memory_snapshot"]["median_us"] / 1000.0 * 0.1,
    }
    # Event volume: the adapter emits ~10 events/step at the core tier.
    events_per_step = 10.0
    emit_us = costs["observer_emit_full"]["median_us"]
    per_step["event emission (~10/step)"] = emit_us * events_per_step / 1000.0

    accounted = sum(per_step.values())
    residual = observed_core_ms - accounted

    t1_incremental = observed_t1_ms - observed_core_ms
    collector_ms = scaled.get("collector_compute", costs["collector_compute"])["median_ms"] * 0.1
    t1_accounted = collector_ms
    t1_residual = t1_incremental - t1_accounted

    return {
        "workload": label,
        "baseline_ms": baseline["median_s"] * 1000.0,
        "core_ms": core["median_s"] * 1000.0,
        "core_layers_ms": layers["median_s"] * 1000.0,
        "observed_core_overhead_ms": observed_core_ms,
        "observed_core_overhead_pct": observed_core_ms / (baseline["median_s"] * 1000.0) * 100.0,
        "observed_t1_total_ms": observed_t1_ms,
        "observed_t1_total_pct": observed_t1_ms / (baseline["median_s"] * 1000.0) * 100.0,
        "t1_incremental_ms": t1_incremental,
        "t1_incremental_pct_of_baseline": t1_incremental / (baseline["median_s"] * 1000.0) * 100.0,
        "components_ms": per_step,
        "scaled_costs": {
            k: (v.get("median_ms") if "median_ms" in v else v.get("median_us", 0.0) / 1000.0)
            for k, v in scaled.items()
        },
        "accounted_ms": accounted,
        "residual_ms": residual,
        "residual_pct_of_observed": (residual / observed_core_ms * 100.0) if observed_core_ms else 0.0,
        "dispersion_ms": dispersion_ms,
        "residual_within_dispersion": residual <= dispersion_ms,
        "t1_accounted_ms": t1_accounted,
        "t1_residual_ms": t1_residual,
        "t1_dispersion_ms": max(layers["per_repeat_s"]) - min(layers["per_repeat_s"]) * 1000.0
        if False else (max(layers["per_repeat_s"]) - min(layers["per_repeat_s"])) * 1000.0,
    }


def render(costs, counts, attributions, env) -> str:
    lines: List[str] = []
    add = lines.append
    add("# OBSERVER COST LEDGER")
    add("")
    add("**Generated by:** `tests/performance/benchmark_cost_ledger.py`")
    add("")
    add("Phase 1 left ~0.8 ms/step unattributed at the largest workload — about 60 %")
    add("of the total cost. This ledger measures each candidate component in isolation")
    add("and reconciles the parts against the whole.")
    add("")
    add("## Reconciliation")
    add("")
    add("| Workload | Baseline | Observed core overhead | Accounted | **Residual** | Dispersion |")
    add("|---|---|---|---|---|---|")
    for a in attributions:
        add("| {} | {:.2f} ms | {:+.2f} ms ({:+.1f} %) | {:.2f} ms | **{:+.2f} ms ({:.0f} % of observed)** | ±{:.2f} ms |".format(
            a["workload"], a["baseline_ms"], a["observed_core_overhead_ms"],
            a["observed_core_overhead_pct"], a["accounted_ms"], a["residual_ms"],
            a["residual_pct_of_observed"], a["dispersion_ms"]))
    add("")
    add("## Per-step attribution")
    add("")
    keys = list(attributions[0]["components_ms"].keys())
    add("| Component | " + " | ".join(a["workload"] for a in attributions) + " |")
    add("|---" * (len(attributions) + 1) + "|")
    for key in keys:
        add("| {} | ".format(key) + " | ".join(
            "{:.3f} ms".format(a["components_ms"][key]) for a in attributions) + " |")
    add("| **accounted** | " + " | ".join(
        "**{:.3f} ms**".format(a["accounted_ms"]) for a in attributions) + " |")
    add("| **residual** | " + " | ".join(
        "**{:+.3f} ms**".format(a["residual_ms"]) for a in attributions) + " |")
    add("")
    add("## Component micro-costs")
    add("")
    add("| Component | median | p95 | MAD | n |")
    add("|---|---|---|---|---|")
    for name, value in sorted(costs.items()):
        if "us" in next(iter(value)):
            add("| {} | {:.3f} µs | {:.3f} µs | {:.3f} µs | {} |".format(
                name, value["median_us"], value["p95_us"], value["mad_us"], value["n"]))
        else:
            add("| {} | {:.3f} ms | {:.3f} ms | {:.3f} ms | {} |".format(
                name, value["median_ms"], value["p95_ms"], value["mad_ms"], value["n"]))
    add("")
    add("Model shape for the micro-costs: {} parameter tensors, {} parameters, {} with gradients.".format(
        counts["parameters"], counts["named_parameters"], counts["grad_tensors"]))
    add("")
    add("## T1 layer hooks: incremental cost")
    add("")
    add("| Workload | T1 total | **T1 incremental over core** | accounted | residual |")
    add("|---|---|---|---|---|")
    for a in attributions:
        add("| {} | {:+.2f} ms ({:+.1f} %) | **{:+.2f} ms ({:+.1f} % of baseline)** | {:.2f} ms | {:+.2f} ms |".format(
            a["workload"], a["observed_t1_total_ms"], a["observed_t1_total_pct"],
            a["t1_incremental_ms"], a["t1_incremental_pct_of_baseline"],
            a["t1_accounted_ms"], a["t1_residual_ms"]))
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
    torch.manual_seed(7)
    model, config = make_tiny_lm(seed=7, n_layers=8, d_model=128, n_heads=8)
    costs_payload = component_costs(model, config, None)
    costs = costs_payload["costs"]
    counts = costs_payload["model_counts"]
    print("component micro-costs measured")

    attributions = []
    for label, n_layers, d_model, n_heads, seq, batch_size in WORKLOADS:
        print("attributing workload: {} ...".format(label))
        torch.manual_seed(7)
        wl_model, _ = make_tiny_lm(seed=7, n_layers=n_layers, d_model=d_model, n_heads=n_heads)
        scaled = scaling_component_costs(wl_model, None)
        scaled["collector_compute"] = bench_ms(
            lambda m=wl_model: LayerMetricCollector(
                m, TrainingObserver("x", writer=NullWriter(), sampling=SamplingPolicy.uniform(1),
                                    config=ObserverConfig(batch_size=200_000, flush_interval_s=5.0)),
                selection=LayerSelection(top_k=8), cadence=10).compute(1),
            warmup=3, n=15,
        )
        attributions.append(
            attribute_workload(label, n_layers, d_model, n_heads, seq, batch_size, costs, scaled)
        )
        a = attributions[-1]
        print("   observed {:+.2f} ms, accounted {:.2f} ms, residual {:+.2f} ms".format(
            a["observed_core_overhead_ms"], a["accounted_ms"], a["residual_ms"]))

    env = capture_environment(extra={
        "benchmark": "cost_ledger", "warmup": WARMUP, "measured": MEASURED, "repeats": REPEATS,
    })
    LEDGER_JSON.write_text(json.dumps(
        {"costs": costs, "model_counts": counts, "attributions": attributions, "environment": env},
        indent=2, default=str), encoding="utf-8")
    LEDGER_MD.write_text(render(costs, counts, attributions, env), encoding="utf-8")
    print("ledger:", LEDGER_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
