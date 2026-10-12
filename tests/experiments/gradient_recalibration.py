"""K0/K1: expanded gradient calibration corpus and authoritative boundary re-derivation.

Phase 1.6 established two facts that together invalidate the previous operating point:

1. The three-way band *can* suppress healthy false alarms (28.6 % -> 1.19 %).
2. The boundaries 8.5 / 11.5 were fitted while ``aggregate_confirms=True`` meant the
   band **never participated in confirmation** at all. Two consecutive re-runs
   produced byte-identical numbers, which is only possible if the band was not in the
   path it appeared to be in.

So the old boundaries were fitted against a code path that did not use them. This
script asks the question the brief poses:

    "Now that the band is actually authoritative, where should the operating
     boundaries really be?"

and answers it with enough runs to be statistically meaningful: >= 20 runs per
retained magnitude (the brief's floor) and >= 100 healthy runs, across two model
configurations.

**What is swept.** `A` (the NORMAL/UNCERTAIN split) cannot change precision or
recall, because both bands are non-promotable; it only moves observations between
NORMAL and UNCERTAIN. `B` (the UNCERTAIN/CONFIRMED split) drives every quality
metric. Both are swept and both are reported, and the frontier records which is
which rather than presenting a single pair as if it were the only candidate.

Run::

    .venv/bin/python tests/experiments/gradient_recalibration.py --healthy 100 --per-magnitude 20
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
from training_debugger.detectors import DetectorConfig, DetectorRunner  # noqa: E402
from training_debugger.detectors.runner import joint_gradient_features  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    MemoryWriter,
    ObserverConfig,
    SamplingPolicy,
    TrainingObserver,
)

OUTPUT_DIR = ROOT / "outputs" / "phase1-7"

STEPS = 200
INJECTION = 110
LIVE_WINDOW = 30

#: The brief's magnitude list. Runs are collected per magnitude; a magnitude may be
#: dropped after analysis only if it is redundant, and never below 20 runs.
MAGNITUDES = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 20.0, 25.0, 40.0)

#: Two development model configurations, so a boundary cannot simply fit one shape.
CONFIGS = (
    {"name": "A", "layers": 4, "d_model": 32},
    {"name": "B", "layers": 6, "d_model": 48},
)

HEALTHY_KINDS = ("stable", "heavy_tail", "scheduler", "clipping", "noisy")


def collect(seed: int, config: Dict[str, Any], kind: str,
            magnitude: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """One real training run. Returns its peak joint aggregate, or None on failure."""
    torch.manual_seed(seed)
    model, model_config = make_tiny_lm(
        seed=seed, n_layers=config["layers"], d_model=config["d_model"], n_heads=4
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "k-%d" % seed, writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384), rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="k")
    collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=32), cadence=1)
    adapter.run_start(steps=STEPS)
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=model_config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            if kind == "noisy":
                loss = loss * (1.0 + 0.35 * math.sin(step * 1.7))
            tracker.record_loss(loss)
            loss.backward()
            if magnitude is not None and step >= INJECTION:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(magnitude)
            if kind == "heavy_tail" and step % 17 == 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(2.6)
            if kind == "clipping":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            tracker.record_gradients()
            collector.collect(step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            if kind == "scheduler" and step == INJECTION:
                for group in optimizer.param_groups:
                    group["lr"] = 0.18
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

    layer_series = {k: sorted(v.items()) for k, v in per_layer.items()}
    by_metric = {"grad.global_norm": sorted(global_norm.items())}

    # Did the *marginal* detector fire at all? Independent of A/B, so it is measured
    # once here rather than re-derived per candidate boundary.
    detection = DetectorRunner(DetectorConfig()).run(
        "k", by_metric, rank=0, layer_series=layer_series, module_paths={},
        population_layers=config["layers"], population_ranks=1,
        coverage={"step_count": STEPS, "step_min": 1, "step_max": STEPS},
    )
    gradient_fires = sorted(
        a.first_abnormal_step for a in detection.anomalies
        if str(a.detector) == "grad.robust_zscore" and a.first_abnormal_step is not None
    )
    if not gradient_fires:
        return {
            "seed": seed, "config": config["name"], "kind": kind, "magnitude": magnitude,
            "detector_fired": False, "peak_aggregate": None, "peak_step": None,
            "fault_window_peak": None, "pre_fault_peak": None,
        }

    def aggregate_at(step: int) -> Optional[float]:
        payload = joint_gradient_features(layer_series, step, baseline_floor=50, window=100)
        return payload.get("l1_aggregate")

    fire_aggregates = {s: aggregate_at(s) for s in gradient_fires}
    valid = {s: v for s, v in fire_aggregates.items() if v is not None and v == v}
    if not valid:
        return {
            "seed": seed, "config": config["name"], "kind": kind, "magnitude": magnitude,
            "detector_fired": True, "peak_aggregate": None, "peak_step": None,
            "fault_window_peak": None, "pre_fault_peak": None,
        }
    peak_step = max(valid, key=lambda s: valid[s])
    window = [s for s in valid if INJECTION <= s <= INJECTION + LIVE_WINDOW]
    pre = [s for s in valid if s < INJECTION]
    return {
        "seed": seed,
        "config": config["name"],
        "kind": kind,
        "magnitude": magnitude,
        "detector_fired": True,
        "peak_aggregate": valid[peak_step],
        "peak_step": peak_step,
        "fault_window_peak": max((valid[s] for s in window), default=None),
        "pre_fault_peak": max((valid[s] for s in pre), default=None),
    }


# ---------------------------------------------------------------------------
# K1: boundary derivation
# ---------------------------------------------------------------------------


def _quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return float("nan")
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (1 - (pos - lo)) + ordered[hi] * (pos - lo)


def evaluate_boundary(rows: Sequence[Dict[str, Any]], a: float, b: float) -> Dict[str, Any]:
    """Grade a candidate (A, B) pair.

    A run "confirms" when the marginal detector fired **and** its peak joint
    aggregate reaches B. A sits strictly below B and only partitions the
    non-promotable observations.
    """
    healthy = [r for r in rows if r["magnitude"] is None and r["detector_fired"]]
    healthy_all = [r for r in rows if r["magnitude"] is None]
    faults = [r for r in rows if r["magnitude"] is not None]

    def peak(row: Dict[str, Any]) -> Optional[float]:
        value = row.get("peak_aggregate")
        return value if value is not None and value == value else None

    healthy_confirmed = sum(1 for r in healthy if (peak(r) or -1) >= b)
    healthy_uncertain = sum(
        1 for r in healthy if a <= (peak(r) or -1) < b
    )
    healthy_normal = sum(1 for r in healthy if (peak(r) or -1) < a)

    tp = fn = fault_confirmed = fault_uncertain = fault_normal = 0
    for row in faults:
        value = peak(row)
        if value is None:
            fn += 1
            continue
        if value >= b:
            tp += 1
            fault_confirmed += 1
        elif value >= a:
            fn += 1
            fault_uncertain += 1
        else:
            fn += 1
            fault_normal += 1

    precision = tp / (tp + healthy_confirmed) if (tp + healthy_confirmed) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    n_healthy = max(1, len(healthy_all))
    n_fault = max(1, len(faults))
    return {
        "A": a,
        "B": b,
        "confirmed_precision": precision,
        "confirmed_recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if (precision and recall) else None,
        "healthy_confirmed_rate": healthy_confirmed / n_healthy,
        "healthy_uncertainty_rate": healthy_uncertain / n_healthy,
        "healthy_normal_rate": healthy_normal / n_healthy,
        "fault_uncertainty_rate": fault_uncertain / n_fault,
        "fault_abstention_rate": (fault_uncertain + fault_normal) / n_fault,
        "true_positives": tp,
        "false_positives": healthy_confirmed,
        "false_negatives": fn,
    }


def magnitude_curve(rows: Sequence[Dict[str, Any]], b: float) -> List[Dict[str, Any]]:
    curve = []
    magnitudes = sorted({r["magnitude"] for r in rows if r["magnitude"] is not None})
    for magnitude in magnitudes:
        group = [r for r in rows if r["magnitude"] == magnitude]
        confirmed = sum(
            1 for r in group
            if (r.get("peak_aggregate") or -1) >= b
        )
        curve.append(
            {
                "magnitude": magnitude,
                "runs": len(group),
                "confirmed": confirmed,
                "p_confirmed": confirmed / len(group) if group else None,
            }
        )
    return curve


def monotonicity_report(curve: Sequence[Dict[str, Any]], tolerance: float = 0.0) -> Dict[str, Any]:
    """Count every decrease in P(CONFIRMED | magnitude).

    Phase 1.8 reported ``monotonicity violations = 0`` while printing a curve that
    visibly decreased (0.23 at x1.5 -> 0.12 at x2.0). The cause was not the data and
    not the curve: the reporter only flagged a drop larger than an arbitrary **0.25**,
    so every real decrease in that curve fell below its own threshold and the metric
    silently returned zero.

    A monotonicity claim is only meaningful if the detector for it cannot be tuned to
    hide the thing it measures. The tolerance defaults to **0.0**: any decrease is an
    inversion, and the magnitude of the largest inversion is reported alongside the
    count so a small inversion cannot be confused with a large one.
    """
    inversions = []
    for previous, current in zip(curve, curve[1:]):
        before = previous["p_confirmed"] or 0.0
        after = current["p_confirmed"] or 0.0
        drop = before - after
        if drop > tolerance:
            inversions.append(
                {
                    "from_magnitude": previous["magnitude"],
                    "to_magnitude": current["magnitude"],
                    "from_p": before,
                    "to_p": after,
                    "drop": drop,
                }
            )
    rates = [c["p_confirmed"] or 0.0 for c in curve]
    # Spearman-style rank correlation between magnitude rank and confirmation rate,
    # reported so "is it monotone?" has a number rather than a boolean.
    n = len(rates)
    rank_correlation = None
    if n > 2:
        magnitude_ranks = list(range(n))
        order = sorted(range(n), key=lambda i: rates[i])
        rate_ranks = [0] * n
        for position, index in enumerate(order):
            rate_ranks[index] = position
        mean = (n - 1) / 2.0
        numerator = sum((magnitude_ranks[i] - mean) * (rate_ranks[i] - mean) for i in range(n))
        denominator = sum((magnitude_ranks[i] - mean) ** 2 for i in range(n))
        if denominator:
            rank_correlation = numerator / denominator
    return {
        "points": n,
        "tolerance": tolerance,
        "inversion_count": len(inversions),
        "inversions": inversions,
        "largest_inversion": max((i["drop"] for i in inversions), default=0.0),
        "rank_correlation": rank_correlation,
        "monotone": len(inversions) == 0,
        "note": (
            "Any decrease counts as an inversion (tolerance 0.0). A tolerance above "
            "zero lets the reporter hide real decreases, which is exactly what "
            "Phase 1.8 did with 0.25."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="K0/K1 gradient recalibration")
    parser.add_argument("--healthy", type=int, default=100, help="healthy runs per config")
    parser.add_argument("--per-magnitude", type=int, default=20)
    parser.add_argument("--magnitudes", type=float, nargs="*", default=None)
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    magnitudes = tuple(args.magnitudes) if args.magnitudes else MAGNITUDES
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    # -- K0 healthy ------------------------------------------------------
    per_kind = max(1, args.healthy // len(HEALTHY_KINDS))
    for config in CONFIGS:
        for kind_index, kind in enumerate(HEALTHY_KINDS):
            for index in range(per_kind):
                seed = 9000 + kind_index * 1000 + index + (0 if config["name"] == "A" else 500)
                try:
                    rows.append(collect(seed, config, kind))
                except Exception as exc:
                    print("   healthy {} {} {} failed: {}".format(config["name"], kind, seed, exc))
        print("[K0] healthy {} done ({} runs)".format(config["name"], len(rows)))

    # -- K0 faults -------------------------------------------------------
    for config in CONFIGS:
        for magnitude in magnitudes:
            for index in range(args.per_magnitude):
                seed = 20000 + int(magnitude * 10) * 100 + index + (
                    0 if config["name"] == "A" else 5000
                )
                try:
                    rows.append(collect(seed, config, "fault", magnitude=magnitude))
                except Exception as exc:
                    print("   fault {} x{} {} failed: {}".format(config["name"], magnitude, seed, exc))
            print("[K0] fault {} x{} done ({}/{} configs, {} rows)".format(
                config["name"], magnitude, CONFIGS.index(config) + 1, len(CONFIGS), len(rows)))

    rows = [r for r in rows if r is not None]
    fired = [r for r in rows if r["detector_fired"]]
    print("[K0] {} runs collected, {} with a marginal detector fire".format(len(rows), len(fired)))

    # -- K1 boundary sweep ------------------------------------------------
    healthy_peaks = sorted(
        r["peak_aggregate"] for r in fired
        if r["magnitude"] is None and r["peak_aggregate"] is not None
    )
    fault_peaks = sorted(
        r["peak_aggregate"] for r in fired if r["magnitude"] is not None
    )
    if not healthy_peaks or not fault_peaks:
        print("ERROR: not enough data to sweep boundaries")
        return 2

    candidates: List[Tuple[float, float]] = []
    # B candidates: spanning the healthy upper tail to above the fault median.
    for q in (1.0, 0.999, 0.995, 0.99, 0.98, 0.95, 0.90):
        candidates.append((_quantile(healthy_peaks, q) * 1.02, None))  # type: ignore
    b_values = sorted({
        round(_quantile(healthy_peaks, q) * factor, 3)
        for q in (1.0, 0.999, 0.995, 0.99, 0.98, 0.95)
        for factor in (1.0, 1.02, 1.1)
    })
    b_values = [b for b in b_values if b > 0]
    frontier = []
    for b in b_values:
        for a in (b * 0.5, b * 0.75, b * 0.9):
            frontier.append(evaluate_boundary(fired, round(a, 3), b))
    frontier.sort(key=lambda e: (-(e["f1"] or 0), -e["healthy_confirmed_rate"]))

    # Prefer points that satisfy the brief's priority order.
    def passes(entry: Dict[str, Any]) -> bool:
        return (
            entry["healthy_confirmed_rate"] <= 0.03
            and (entry["confirmed_precision"] or 0) >= 0.90
            and (entry["confirmed_recall"] or 0) >= 0.80
        )
    qualifying = [e for e in frontier if passes(e)]
    chosen = qualifying[0] if qualifying else (frontier[0] if frontier else None)

    curve = magnitude_curve(fired, chosen["B"]) if chosen else []
    monotonicity = monotonicity_report(curve)

    payload = {
        "corpus": {
            "runs_total": len(rows),
            "runs_with_detector_fire": len(fired),
            "healthy_runs": sum(1 for r in rows if r["magnitude"] is None),
            "healthy_with_fire": len(healthy_peaks),
            "per_magnitude": args.per_magnitude,
            "magnitudes": list(magnitudes),
            "configs": [c["name"] for c in CONFIGS],
            "healthy_kinds": list(HEALTHY_KINDS),
        },
        "distributions": {
            "healthy_peak_aggregate": {
                "n": len(healthy_peaks), "min": min(healthy_peaks),
                "p50": _quantile(healthy_peaks, 0.5),
                "p95": _quantile(healthy_peaks, 0.95),
                "p99": _quantile(healthy_peaks, 0.99),
                "max": max(healthy_peaks),
            },
            "fault_peak_aggregate": {
                "n": len(fault_peaks), "min": min(fault_peaks),
                "p50": _quantile(fault_peaks, 0.5),
                "p95": _quantile(fault_peaks, 0.95),
                "max": max(fault_peaks),
            },
        },
        "frontier": frontier[:60],
        "qualifying_points": qualifying[:20],
        "chosen": chosen,
        "magnitude_curve": curve,
        "monotonicity": monotonicity,
        "rows": rows,
    }
    (OUTPUT_DIR / "gradient_boundary_frontier.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2, default=str),
        encoding="utf-8",
    )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(
            pa.Table.from_pylist([
                {k: (str(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in rows
            ]),
            OUTPUT_DIR / "gradient_magnitude_sweep.parquet",
            compression="zstd",
        )
    except Exception as exc:
        print("parquet write failed:", exc)

    print("")
    print("healthy peak aggregate: n={} p50={:.2f} p99={:.2f} max={:.2f}".format(
        len(healthy_peaks), _quantile(healthy_peaks, 0.5),
        _quantile(healthy_peaks, 0.99), max(healthy_peaks)))
    if chosen:
        print("chosen A={} B={}  precision={} recall={} healthy_confirmed={:.3f}".format(
            chosen["A"], chosen["B"],
            "n/a" if chosen["confirmed_precision"] is None else "{:.3f}".format(chosen["confirmed_precision"]),
            "n/a" if chosen["confirmed_recall"] is None else "{:.3f}".format(chosen["confirmed_recall"]),
            chosen["healthy_confirmed_rate"]))
        print("qualifying operating points: {}".format(len(qualifying)))
    print("monotonicity: monotone={} inversions={} largest_drop={:.3f} rank_corr={}".format(
        monotonicity["monotone"], monotonicity["inversion_count"],
        monotonicity["largest_inversion"],
        "n/a" if monotonicity["rank_correlation"] is None else round(monotonicity["rank_correlation"], 3)))
    print("elapsed {:.0f}s".format(time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
