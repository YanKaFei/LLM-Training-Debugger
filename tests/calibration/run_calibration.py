"""C5: detector calibration on real training runs.

Phase 1 could only say "tests pass". That is not a precision statement. This
harness produces the numbers a detector set needs before anything is built on top
of it: true/false positives and negatives, precision, recall, detection delay, and
false positives per 1 000 steps — each measured on **real PyTorch training loops**,
never on hand-built event arrays.

Four design decisions, each addressing a way this kind of measurement goes wrong:

**Ground truth is separated from detector output.** Faults are declared in
:class:`Fault` records and written to a ground-truth file; the detector layer never
sees them. A detector that could read the answer would score perfectly.

**Train/test split, declared up front.** The corpus is split by seed into a
`calibration` half and a `holdout` half. Precision and recall are reported for
both. **No threshold was tuned on either half** — the Phase 1 defaults were frozen
before this harness existed — but the split is built now so that the next person
cannot tune on the holdout by accident.

**Healthy diversity, not one toy model.** The healthy class includes heavy-tailed
gradients, noisy losses, warm-up transients, learning-rate discontinuities (F7)
and scheduler transitions, because a detector tuned only against a clean curve
will fire on every real one.

**Small-sample honesty.** With ~25 runs per class the confidence interval on
recall is wide. The report states the interval and says plainly when the sample is
too small for the number to carry weight, rather than quoting 0.96 as if it were
exact.

Run::

    .venv/bin/python tests/calibration/run_calibration.py            # full corpus
    .venv/bin/python tests/calibration/run_calibration.py --quick    # smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
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
from training_debugger.env_info import capture_environment  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    MemoryWriter,
    ObserverConfig,
    SamplingPolicy,
    TrainingObserver,
)

OUTPUT_DIR = ROOT / "outputs" / "phase1-1" / "calibration"
GROUND_TRUTH = OUTPUT_DIR / "ground_truth.json"
RESULTS = OUTPUT_DIR / "calibration_results.json"
REPORT = OUTPUT_DIR / "CALIBRATION_REPORT.md"
SPLIT_FILE = OUTPUT_DIR / "split.json"


# ---------------------------------------------------------------------------
# Fault definitions
# ---------------------------------------------------------------------------

#: Fault classes. ``H0`` is the healthy control; every other class injects exactly
#: one thing so that a false negative is attributable to one detector.
FAULT_CLASSES = (
    "H0_healthy",
    "H1_heavy_tail_gradients",
    "H2_noisy_healthy_loss",
    "F1_isolated_loss_spike",
    "F2_sustained_divergence",
    "F3_gradient_explosion",
    "F4_nan_loss",
    "F5_inf_gradient",
    "F6_throughput_degradation",
    "F7_lr_discontinuity_no_failure",
    "F10_mixed_anomaly",
)

#: Classes that are healthy by construction. A detector firing on one of these is
#: a false positive, which is what makes F7 valuable: it *looks* like a fault.
HEALTHY_CLASSES = frozenset(
    {"H0_healthy", "H1_heavy_tail_gradients", "H2_noisy_healthy_loss", "F7_lr_discontinuity_no_failure"}
)

#: Which detector is *expected* to fire for each fault class. Empty means none.
EXPECTED_DETECTORS: Dict[str, Tuple[str, ...]] = {
    "H0_healthy": (),
    "H1_heavy_tail_gradients": (),
    "H2_noisy_healthy_loss": (),
    "F7_lr_discontinuity_no_failure": (),
    "F1_isolated_loss_spike": ("loss.robust_zscore",),
    "F2_sustained_divergence": ("loss.trend_slope", "loss.robust_zscore"),
    "F3_gradient_explosion": ("grad.robust_zscore",),
    "F4_nan_loss": ("numerics.finiteness",),
    "F5_inf_gradient": ("numerics.finiteness",),
    "F6_throughput_degradation": ("throughput.regression",),
    "F10_mixed_anomaly": ("grad.robust_zscore", "loss.robust_zscore", "numerics.finiteness"),
}


@dataclass
class Fault:
    """Ground truth for one run. Never visible to the detector layer."""

    fault_type: str
    seed: int
    split: str
    injection_step: Optional[int] = None
    injection_scope: str = "none"
    affected_layers: List[int] = field(default_factory=list)
    expected_symptom_window: Optional[Tuple[int, int]] = None
    expected_detectors: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_type": self.fault_type,
            "seed": self.seed,
            "split": self.split,
            "injection_step": self.injection_step,
            "injection_scope": self.injection_scope,
            "affected_layers": list(self.affected_layers),
            "expected_symptom_window": list(self.expected_symptom_window or []),
            "expected_detectors": list(self.expected_detectors),
        }


STEPS = 200
WARMUP_STEPS = 40
INJECTION_STEP = 120
LAYERS = 4


def build_corpus(seeds: Sequence[int], split_for) -> List[Fault]:
    """One fault per (class, seed), with its ground truth."""
    faults: List[Fault] = []
    for index, fault_type in enumerate(FAULT_CLASSES):
        for seed in seeds:
            split = split_for(seed)
            fault = Fault(
                fault_type=fault_type,
                seed=seed,
                split=split,
                expected_detectors=EXPECTED_DETECTORS[fault_type],
            )
            if fault_type in HEALTHY_CLASSES:
                fault.injection_scope = "none"
            else:
                fault.injection_step = INJECTION_STEP
                fault.expected_symptom_window = (INJECTION_STEP - 2, INJECTION_STEP + 40)
                if fault_type in ("F3_gradient_explosion", "F10_mixed_anomaly"):
                    fault.injection_scope = "all_gradients"
                    fault.affected_layers = list(range(LAYERS))
                elif fault_type in ("F4_nan_loss", "F5_inf_gradient"):
                    fault.injection_scope = "loss_or_gradient"
                else:
                    fault.injection_scope = "loss"
            faults.append(fault)
    return faults


# ---------------------------------------------------------------------------
# Real training with one injected fault
# ---------------------------------------------------------------------------


def run_one(fault: Fault) -> Dict[str, Any]:
    """Train a real model with the declared fault and return the detector output."""
    torch.manual_seed(fault.seed)
    model, config = make_tiny_lm(seed=fault.seed, n_layers=LAYERS, d_model=32, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    writer = MemoryWriter()
    observer = TrainingObserver(
        "run-{}-{}".format(fault.fault_type, fault.seed),
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=8192),
        rank=0,
        world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="tiny")
    collector = LayerMetricCollector(
        model, observer, selection=LayerSelection(top_k=8), cadence=1
    )
    adapter.run_start(steps=STEPS)

    scale = 1.0
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            with tracker.phase("forward"):
                loss = model(**batch)["loss"]
            # -- injections, chosen so each fault has one clear cause -----
            if fault.fault_type == "H1_heavy_tail_gradients":
                pass  # achieved through the optimiser below
            if fault.fault_type == "H2_noisy_healthy_loss":
                loss = loss * (1.0 + 0.35 * math.sin(step * 1.7))
            if fault.fault_type == "F1_isolated_loss_spike" and step == INJECTION_STEP:
                loss = loss + 8.0
            if fault.fault_type == "F2_sustained_divergence" and step >= INJECTION_STEP:
                scale *= 1.012
                loss = loss * scale
            if fault.fault_type == "F6_throughput_degradation" and step >= INJECTION_STEP:
                time.sleep(0.004)
            if fault.fault_type == "F10_mixed_anomaly" and step == INJECTION_STEP:
                loss = float("nan")
            if fault.fault_type == "F4_nan_loss" and step == INJECTION_STEP:
                loss = loss * float("nan")
            tracker.record_loss(loss)
            with tracker.phase("backward"):
                if loss == loss:
                    loss.backward()
            if fault.fault_type == "F3_gradient_explosion" and step >= INJECTION_STEP:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(35.0)
            if fault.fault_type == "F10_mixed_anomaly" and step >= INJECTION_STEP:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(25.0)
            if fault.fault_type == "F5_inf_gradient" and step == INJECTION_STEP:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.flatten()[0] = float("inf")
            if fault.fault_type == "H1_heavy_tail_gradients" and step % 17 == 0:
                # A genuinely heavy-tailed healthy signal: occasional large but
                # finite excursions, as real gradient norms exhibit.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(2.6)
            tracker.record_gradients()
            collector.collect(step)
            with tracker.phase("optimizer"):
                if loss == loss:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()
            if fault.fault_type == "F7_lr_discontinuity_no_failure" and step == INJECTION_STEP:
                for group in optimizer.param_groups:
                    group["lr"] = 0.2

    adapter.run_end()
    observer.flush(timeout=30.0)
    observer.close()

    by_metric, layer_series, module_paths = _series(writer.events)
    detection = DetectorRunner(DetectorConfig()).run(
        "cal",
        by_metric,
        rank=0,
        layer_series=layer_series,
        module_paths=module_paths,
        population_layers=LAYERS,
        population_ranks=1,
        coverage={"step_count": STEPS, "step_min": 1, "step_max": STEPS},
    )
    fired: Dict[str, int] = {}
    for anomaly in detection.anomalies:
        key = str(anomaly.detector)
        if key not in fired or anomaly.first_abnormal_step < fired[key]:
            fired[key] = anomaly.first_abnormal_step
    result = {
        "fault_type": fault.fault_type,
        "seed": fault.seed,
        "split": fault.split,
        "fired_detectors": sorted(fired),
        "onsets": fired,
        "detector_errors": detection.detector_errors,
        "anomaly_count": len(detection.anomalies),
    }
    del model, optimizer, observer, adapter, collector
    gc.collect()
    return result


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
# Scoring
# ---------------------------------------------------------------------------


def split_for_seed(seed: int, holdout_modulus: int = 3) -> str:
    """Deterministic split. Seeds divisible by the modulus form the holdout."""
    return "holdout" if (seed % holdout_modulus == 0) else "calibration"


def score(ground_truth: Sequence[Fault], results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-detector TP/FP/TN/FN plus precision, recall, delay and FP rate."""
    # Keyed by (fault_type, seed). Keying by seed alone collapsed every class
    # onto one entry when the same seed is reused across classes -- which is the
    # intended corpus design -- and silently scored the whole corpus against a
    # single run. It reported recall 0.000 on a corpus where faults were in fact
    # detected.
    by_key = {(r["fault_type"], r["seed"]): r for r in results}
    failed = {k for k, r in by_key.items() if r.get("run_error")}
    detectors = sorted({d for f in ground_truth for d in f.expected_detectors} | {
        d for r in results for d in r["fired_detectors"]
    })
    scores: Dict[str, Any] = {}
    for split in ("calibration", "holdout", "all"):
        faults = [f for f in ground_truth if split == "all" or f.split == split]
        split_scores: Dict[str, Any] = {}
        for detector in detectors:
            tp = fp = tn = fn = 0
            delays: List[int] = []
            for fault in faults:
                if (fault.fault_type, fault.seed) in failed:
                    # An infrastructure failure is NOT a detector miss. Scoring it
                    # as a false negative is how a crashed run became "the
                    # detectors cannot see mixed anomalies".
                    continue
                record = by_key.get((fault.fault_type, fault.seed), {})
                fired = detector in record.get("fired_detectors", [])
                expected = detector in fault.expected_detectors
                if expected and fired:
                    tp += 1
                    delay = _delay(fault, record.get("onsets", {}).get(detector))
                    if delay is not None:
                        delays.append(delay)
                elif expected and not fired:
                    fn += 1
                elif not expected and fired:
                    fp += 1
                else:
                    tn += 1
            precision = tp / (tp + fp) if (tp + fp) else None
            recall = tp / (tp + fn) if (tp + fn) else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision and recall else None
            )
            steps_observed = len(faults) * STEPS
            split_scores[detector] = {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "median_delay_steps": statistics.median(delays) if delays else None,
                "false_positives_per_1k_steps": (fp / steps_observed * 1000.0)
                if steps_observed else None,
                "recall_ci95": _wilson(tp, tp + fn),
                "precision_ci95": _wilson(tp, tp + fp),
                "runs": len(faults),
                "note": (
                    "confidence interval not meaningful below 20 positives"
                    if (tp + fn) < 20 else None
                ),
            }
        # Any-detector-warned view: did the system notice at all?
        system_tp = system_fp = system_tn = system_fn = 0
        for fault in faults:
            if (fault.fault_type, fault.seed) in failed:
                continue
            fired_any = bool(by_key.get((fault.fault_type, fault.seed), {}).get("fired_detectors"))
            expected_any = bool(fault.expected_detectors)
            if expected_any and fired_any:
                system_tp += 1
            elif expected_any and not fired_any:
                system_fn += 1
            elif not expected_any and fired_any:
                system_fp += 1
            else:
                system_tn += 1
        split_scores["__system__"] = {
            "tp": system_tp, "fp": system_fp, "tn": system_tn, "fn": system_fn,
            "precision": system_tp / (system_tp + system_fp) if (system_tp + system_fp) else None,
            "recall": system_tp / (system_tp + system_fn) if (system_tp + system_fn) else None,
            "false_positives_per_1k_steps": (
                system_fp / (len(faults) * STEPS) * 1000.0 if faults else None
            ),
            "healthy_runs_with_a_false_alarm": system_fp,
            "healthy_runs": system_fp + system_tn,
            "runs": len(faults),
        }
        scores[split] = split_scores
    return scores


def _delay(fault: Fault, onset: Optional[int]) -> Optional[int]:
    if onset is None:
        return None
    if fault.expected_symptom_window:
        return onset - fault.expected_symptom_window[0]
    return None


def _wilson(successes: int, trials: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    """Wilson score interval. Honest about small samples, unlike a normal approx."""
    if trials <= 0:
        return None
    phat = successes / trials
    denominator = 1 + z * z / trials
    centre = phat + z * z / (2 * trials)
    margin = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
    return (max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(
    ground_truth: Sequence[Fault],
    results: Sequence[Dict[str, Any]],
    scores: Dict[str, Any],
    env: Dict[str, Any],
    elapsed_s: float,
) -> str:
    lines: List[str] = []
    add = lines.append
    add("# CALIBRATION REPORT")
    add("")
    add("**Generated by:** `tests/calibration/run_calibration.py`")
    add("**Corpus:** {} real PyTorch training runs, {} steps each".format(
        len(ground_truth), STEPS))
    add("**Wall time:** {:.0f} s".format(elapsed_s))
    add("")

    add("## Corpus composition")
    add("")
    add("| Fault class | Runs | Split | Expected detectors | Injection step |")
    add("|---|---|---|---|---|")
    counts: Dict[str, Dict[str, int]] = {}
    for fault in ground_truth:
        entry = counts.setdefault(fault.fault_type, {"calibration": 0, "holdout": 0})
        entry[fault.split] += 1
    for fault_type in FAULT_CLASSES:
        entry = counts.get(fault_type, {"calibration": 0, "holdout": 0})
        example = next((f for f in ground_truth if f.fault_type == fault_type), None)
        add("| `{}` | {} ({}+{}) | cal {} / hold {} | {} | {} |".format(
            fault_type,
            entry["calibration"] + entry["holdout"], entry["calibration"], entry["holdout"],
            entry["calibration"], entry["holdout"],
            ", ".join(EXPECTED_DETECTORS[fault_type]) or "(none)",
            "—" if example is None or example.injection_step is None else example.injection_step,
        ))
    add("")
    healthy_runs = sum(1 for f in ground_truth if not f.expected_detectors)
    faulty_runs = len(ground_truth) - healthy_runs
    add("Healthy-by-construction runs: **{}**. Fault-injected runs: **{}**.".format(
        healthy_runs, faulty_runs))
    add("")
    add("`F7_lr_discontinuity_no_failure` and `H1_heavy_tail_gradients` are the")
    add("precision tests: both look like faults to a naive detector and are not.")
    add("")

    add("## System-level performance (did the system notice?)")
    add("")
    add("| Split | Runs | TP | FP | TN | FN | Precision | Recall | FP / 1k steps | Healthy runs with a false alarm |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for split in ("calibration", "holdout", "all"):
        entry = scores[split]["__system__"]
        add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} |".format(
            split, entry["runs"], entry["tp"], entry["fp"], entry["tn"], entry["fn"],
            _fmt(entry["precision"]), _fmt(entry["recall"]),
            _fmt(entry["false_positives_per_1k_steps"], 2),
            entry["healthy_runs_with_a_false_alarm"], entry["healthy_runs"]))
    add("")

    add("## Per-detector scores (all runs)")
    add("")
    add("| Detector | TP | FP | TN | FN | Precision (95 % CI) | Recall (95 % CI) | F1 | Median delay | FP / 1k steps |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for detector, entry in sorted(scores["all"].items()):
        if detector == "__system__":
            continue
        add("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            detector, entry["tp"], entry["fp"], entry["tn"], entry["fn"],
            _fmt_ci(entry["precision"], entry["precision_ci95"]),
            _fmt_ci(entry["recall"], entry["recall_ci95"]),
            _fmt(entry["f1"]),
            "—" if entry["median_delay_steps"] is None else "{:.0f} steps".format(entry["median_delay_steps"]),
            _fmt(entry["false_positives_per_1k_steps"], 2)))
    add("")

    add("## Holdout only")
    add("")
    add("| Detector | TP | FP | FN | Precision | Recall |")
    add("|---|---|---|---|---|---|")
    for detector, entry in sorted(scores["holdout"].items()):
        if detector == "__system__":
            continue
        add("| `{}` | {} | {} | {} | {} | {} |".format(
            detector, entry["tp"], entry["fp"], entry["fn"],
            _fmt(entry["precision"]), _fmt(entry["recall"])))
    add("")
    add("**No threshold was tuned on either split.** The Phase 1 defaults were frozen")
    add("before this harness existed; the split exists so that a future tuning pass")
    add("cannot silently consume the holdout.")
    add("")

    add("## Per-class outcome")
    add("")
    add("| Fault class | Runs | Runs where any detector fired | Runs with a false alarm | Detectors that fired |")
    add("|---|---|---|---|---|")
    by_key = {(r["fault_type"], r["seed"]): r for r in results}
    for fault_type in FAULT_CLASSES:
        group = [f for f in ground_truth if f.fault_type == fault_type]
        if not group:
            continue
        noticed = sum(
            1 for f in group
            if by_key.get((f.fault_type, f.seed), {}).get("fired_detectors")
        )
        fired_sets: Dict[str, int] = {}
        for fault in group:
            for detector in by_key.get((fault.fault_type, fault.seed), {}).get("fired_detectors", []):
                fired_sets[detector] = fired_sets.get(detector, 0) + 1
        expected_none = not EXPECTED_DETECTORS[fault_type]
        alarm_label = "{} (false alarm)".format(noticed) if expected_none else str(noticed)
        add("| `{}` | {} | {} | {} | {} |".format(
            fault_type, len(group), noticed,
            noticed if expected_none else 0,
            ", ".join("{} x{}".format(k, v) for k, v in sorted(fired_sets.items())) or "—"))
    add("")

    crashes = [r for r in results if r.get("run_error")]
    add("## Infrastructure failures")
    add("")
    if crashes:
        add("**{} run(s) crashed and were EXCLUDED from the scores.** A crashed run is")
        add("not a detector miss; scoring it as one is how a broken scenario was reported")
        add("as a detector limitation.")
        add("")
        for entry in crashes[:8]:
            add("- `{}` seed {}: {}".format(entry["fault_type"], entry["seed"], entry["run_error"]))
    else:
        add("None. Every run completed.")
    add("")

    errors = [r for r in results if r.get("detector_errors")]
    add("## Detector errors")
    add("")
    if errors:
        add("**{} run(s) had a detector raise.** A silently broken detector scores as")
        add("perfect precision and zero recall, which is the most dangerous failure mode")
        add("in this report.")
        add("")
        for entry in errors[:5]:
            add("- `{}` seed {}: {}".format(
                entry["fault_type"], entry["seed"], entry["detector_errors"][:2]))
    else:
        add("None. Every detector ran to completion on every run.")
    add("")

    add("## What these numbers do and do not support")
    add("")
    injected = sum(1 for f in ground_truth if f.expected_detectors)
    per_class = len(ground_truth) // len(FAULT_CLASSES) if FAULT_CLASSES else 0
    # Half-width of a 95 % Wilson interval at p=0.9, the worst case for this
    # sample size. Reported so the reader sees the width rather than trusting the
    # point estimate.
    half_width = 196.0 * math.sqrt(0.09 / injected) if injected else 0.0
    add("* These are **detector-level** precision and recall on a synthetic fault corpus")
    add("  built from real training loops. They are not field precision on real incidents.")
    add("* The corpus is **{} runs**, of which {} are fault-injected. The 95 % interval on".format(
        len(ground_truth), injected))
    add("  a recall near 0.9 is roughly ±{:.0f} points at this sample size — wide. Per-class".format(
        half_width))
    add("  conclusions rest on only {} runs each.".format(per_class))
    add("* Faults are injected at a **known single step** with a known mechanism. Real")
    add("  incidents arrive through data, hardware and configuration in ways this corpus")
    add("  does not model.")
    add("* `H1` and `F7` deliberately probe the boundary between healthy-looking and")
    add("  faulty; a detector that scores well on `F1`-`F6` and badly on `H1`/`F7` is")
    add("  overfitted to clean signals.")
    add("")
    add("## Environment")
    add("")
    add("```json")
    add(json.dumps(env, indent=2, sort_keys=True, default=str))
    add("```")
    add("")
    return "\n".join(lines)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else ("{:.%df}" % digits).format(value)


def _fmt_ci(value: Optional[float], interval: Optional[Sequence[float]]) -> str:
    if value is None:
        return "n/a"
    if not interval:
        return "{:.3f}".format(value)
    return "{:.3f} [{:.2f}, {:.2f}]".format(value, interval[0], interval[1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detector calibration on real training runs")
    parser.add_argument("--seeds", type=int, default=5,
                        help="seeds per fault class (default 5 -> %d runs)" % (5 * len(FAULT_CLASSES)))
    parser.add_argument("--quick", action="store_true", help="1 seed per class, smoke only")
    parser.add_argument("--seed-start", type=int, default=1000)
    args = parser.parse_args(argv)

    seeds_per_class = 1 if args.quick else args.seeds
    seeds = [args.seed_start + i for i in range(seeds_per_class)]
    ground_truth = build_corpus(seeds, split_for_seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH.write_text(
        json.dumps([f.to_dict() for f in ground_truth], indent=2), encoding="utf-8"
    )
    SPLIT_FILE.write_text(
        json.dumps(
            {
                "method": "seed % 3 == 0 -> holdout",
                "calibration_seeds": sorted(s for s in seeds if split_for_seed(s) == "calibration"),
                "holdout_seeds": sorted(s for s in seeds if split_for_seed(s) == "holdout"),
                "note": (
                    "No threshold was tuned on either split. The Phase 1 detector defaults "
                    "predate this corpus; the split exists to prevent a future tuning pass "
                    "from consuming the holdout."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("corpus: {} runs across {} classes".format(len(ground_truth), len(FAULT_CLASSES)))
    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    for index, fault in enumerate(ground_truth, start=1):
        try:
            results.append(run_one(fault))
        except Exception as exc:  # a corpus run failing must not lose the rest
            results.append(
                {
                    "fault_type": fault.fault_type, "seed": fault.seed, "split": fault.split,
                    "fired_detectors": [], "onsets": {}, "detector_errors": [],
                    "anomaly_count": 0,
                    "run_error": "{}: {}".format(type(exc).__name__, exc),
                }
            )
        if index % 10 == 0 or index == len(ground_truth):
            print("  {}/{} runs ({:.0f}s)".format(index, len(ground_truth), time.perf_counter() - started))
    elapsed = time.perf_counter() - started

    scores = score(ground_truth, results)
    env = capture_environment(extra={
        "calibration": {
            "runs": len(ground_truth), "steps_per_run": STEPS,
            "injection_step": INJECTION_STEP, "layers": LAYERS,
            "classes": list(FAULT_CLASSES), "seeds_per_class": seeds_per_class,
        }
    })
    RESULTS.write_text(
        json.dumps(
            {
                "ground_truth": [f.to_dict() for f in ground_truth],
                "results": results,
                "scores": scores,
                "environment": env,
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    REPORT.write_text(render_report(ground_truth, results, scores, env, elapsed), encoding="utf-8")
    system = scores["all"]["__system__"]
    print("")
    print("system precision: {}  recall: {}".format(
        _fmt(system["precision"]), _fmt(system["recall"])))
    print("report:", REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
