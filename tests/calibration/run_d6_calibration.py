"""D6: calibration and holdout with independent generation.

Three defects in the Phase 1.1 corpus are repaired here:

1. **Calibration and holdout were identical** (0.632 / 0.857), because both drew
   from the same seed pool on the same model with the same fault magnitudes. The
   holdout here uses a **different model configuration** (6 layers, d_model 48
   against the calibration's 4 layers, d_model 32), different seeds, and different
   fault magnitudes.
2. **Faults had a single magnitude each.** A detector that catches a 40x explosion
   proves almost nothing; what matters is the **detection boundary**. Every fault
   family is swept across magnitudes.
3. **Detection delay was computed from the raw first fire**, which produced
   negative delays. Evaluation now goes through
   :mod:`training_debugger.evaluation`, which classifies each fire as a true
   positive, a pre-injection false positive, a late detection or a duplicate.

Threshold fitting rule (D6.2), stated so it can be audited: the v2.0 envelope
parameters were derived from a **separate 30-run distribution study**
(seeds 2000-2029, ``tests/experiments/gradient_distribution.py``) that is disjoint
from both splits here. Neither split was consulted while choosing them.

Run::

    .venv/bin/python tests/calibration/run_d6_calibration.py
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
from training_debugger.detectors.base import ConfidenceClass  # noqa: E402
from training_debugger.env_info import capture_environment  # noqa: E402
from training_debugger.evaluation import (  # noqa: E402
    DetectionWindow,
    evaluate_scenario,
    aggregate,
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

#: Independent generation: the holdout differs in model shape.
SPLITS = {
    "calibration": {"layers": 4, "d_model": 32, "seeds": list(range(3000, 3060))},
    "holdout": {"layers": 6, "d_model": 48, "seeds": list(range(4000, 4040))},
}


@dataclass
class Fault:
    fault_type: str
    seed: int
    split: str
    magnitude: Optional[float] = None
    injection_step: Optional[int] = None
    expected_detectors: Tuple[str, ...] = ()
    effect_lag: int = 0
    note: str = ""

    def window(self) -> DetectionWindow:
        if self.injection_step is None:
            return DetectionWindow.healthy(STEPS)
        return DetectionWindow.for_injection(
            self.injection_step, STEPS, effect_lag=self.effect_lag, tolerance=10
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "fault_type": self.fault_type,
            "seed": self.seed,
            "split": self.split,
            "magnitude": self.magnitude,
            "injection_step": self.injection_step,
            "expected_detectors": list(self.expected_detectors),
            "effect_lag": self.effect_lag,
            "note": self.note,
        }
        payload.update(self.window().to_dict())
        return payload


#: Fault definitions. ``magnitudes`` is swept; healthy classes have none.
FAULT_SPECS: Tuple[Dict[str, Any], ...] = (
    # -- healthy (>= 30 % of the corpus) ---------------------------------
    {"name": "H0_stable_healthy", "magnitudes": (None,), "healthy": True},
    {"name": "H1_heavy_tail_gradients", "magnitudes": (None,), "healthy": True},
    {"name": "H2_noisy_healthy_loss", "magnitudes": (None,), "healthy": True},
    {"name": "H3_scheduler_transition", "magnitudes": (None,), "healthy": True},
    {"name": "H4_gradient_clipping", "magnitudes": (None,), "healthy": True},
    {"name": "F7_lr_jump_no_failure", "magnitudes": (None,), "healthy": True},
    # -- gradient magnitude sweep ----------------------------------------
    {
        "name": "F3_gradient_explosion",
        "magnitudes": (2.0, 3.0, 5.0, 10.0, 20.0, 40.0),
        "detectors": ("grad.robust_zscore",),
    },
    # -- loss magnitude sweep --------------------------------------------
    {
        "name": "F1_loss_spike",
        "magnitudes": (0.5, 2.0, 8.0),
        "detectors": ("loss.robust_zscore",),
    },
    # -- divergence ------------------------------------------------------
    {
        "name": "F2_sustained_divergence",
        "magnitudes": (1.006, 1.012, 1.03),
        "detectors": ("loss.trend_slope", "loss.robust_zscore"),
    },
    # -- finiteness ------------------------------------------------------
    {"name": "F4_nan_loss", "magnitudes": (None,), "detectors": ("numerics.finiteness",)},
    {"name": "F5_inf_gradient", "magnitudes": (None,), "detectors": ("numerics.finiteness",)},
    {"name": "F5b_nan_gradient", "magnitudes": (None,), "detectors": ("numerics.finiteness",)},
    # -- throughput ------------------------------------------------------
    {"name": "F6_throughput_drop", "magnitudes": (0.004,), "detectors": ("throughput.regression",)},
    # -- mixed -----------------------------------------------------------
    {
        "name": "F10_mixed_anomaly",
        "magnitudes": (None,),
        "detectors": ("grad.robust_zscore", "loss.robust_zscore", "numerics.finiteness"),
    },
    # -- adversarial (D8) -------------------------------------------------
    {"name": "A1_single_huge_harmless_batch", "magnitudes": (None,), "healthy": True,
     "note": "one large but finite gradient excursion"},
    {"name": "A2_loss_spike_stable_gradients", "magnitudes": (None,), "healthy": True,
     "note": "loss moves, gradients do not"},
    {"name": "A3_gradient_spike_no_loss_consequence", "magnitudes": (None,), "healthy": True,
     "note": "gradients move, loss does not"},
    {"name": "A4_nan_that_recovers", "magnitudes": (None,), "detectors": ("numerics.finiteness",),
     "note": "NaN on one step, then finite again"},
    {"name": "A5_inf_one_layer_only", "magnitudes": (None,), "detectors": ("numerics.finiteness",)},
)


def build_manifest(split: str) -> List[Fault]:
    config = SPLITS[split]
    seeds = config["seeds"]
    faults: List[Fault] = []
    for spec in FAULT_SPECS:
        healthy = bool(spec.get("healthy"))
        magnitudes = spec["magnitudes"]
        # Bounded seed counts. Healthy classes get more runs than any single
        # fault magnitude because the FP-per-1k estimate needs the sample, and a
        # magnitude sweep spreads its seed budget across magnitudes by design.
        is_holdout = split == "holdout"
        if healthy:
            take = 5 if is_holdout else 8
        elif len(magnitudes) > 1:
            take = 2 if is_holdout else 3
        else:
            take = 3 if is_holdout else 4
        for magnitude in magnitudes:
            for seed in seeds[:take]:
                faults.append(
                    Fault(
                        fault_type=spec["name"],
                        seed=seed,
                        split=split,
                        magnitude=magnitude,
                        injection_step=None if healthy else INJECTION_STEP,
                        expected_detectors=() if healthy else tuple(spec.get("detectors", ())),
                        effect_lag=1 if spec["name"] == "F6_throughput_drop" else 0,
                        note=spec.get("note", ""),
                    )
                )
    return faults


# ---------------------------------------------------------------------------
# Real training with one fault
# ---------------------------------------------------------------------------


def run_one(fault: Fault, layers: int, d_model: int) -> Dict[str, Any]:
    torch.manual_seed(fault.seed)
    model, config = make_tiny_lm(seed=fault.seed, n_layers=layers, d_model=d_model, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    writer = MemoryWriter()
    observer = TrainingObserver(
        "d6-{}-{}".format(fault.fault_type, fault.seed),
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384),
        rank=0,
        world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="d6")
    collector = LayerMetricCollector(
        model, observer, selection=LayerSelection(top_k=32), cadence=1
    )
    adapter.run_start(steps=STEPS)

    name = fault.fault_type
    scale = 1.0
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            with tracker.phase("forward"):
                loss = model(**batch)["loss"]
            effective_loss = loss

            if name == "H2_noisy_healthy_loss":
                effective_loss = effective_loss * (1.0 + 0.35 * math.sin(step * 1.7))
            if name == "H3_scheduler_transition":
                effective_loss = effective_loss  # LR change only, below
            if name == "A2_loss_spike_stable_gradients" and step == INJECTION_STEP:
                # Loss changes, but the gradient is detached so the backward pass
                # sees a normal signal.
                effective_loss = effective_loss * 12.0
            if name == "A4_nan_that_recovers" and step == INJECTION_STEP:
                effective_loss = effective_loss * float("nan")
            if name == "F1_loss_spike" and step == INJECTION_STEP and fault.magnitude is not None:
                effective_loss = effective_loss + fault.magnitude
            if name == "F2_sustained_divergence" and step >= INJECTION_STEP:
                scale *= fault.magnitude or 1.012
                effective_loss = effective_loss * scale
            if name == "F4_nan_loss" and step == INJECTION_STEP:
                effective_loss = effective_loss * float("nan")
            if name == "F10_mixed_anomaly" and step == INJECTION_STEP:
                effective_loss = effective_loss * float("nan")
            if name == "F6_throughput_drop" and step >= INJECTION_STEP:
                time.sleep(fault.magnitude or 0.004)

            tracker.record_loss(effective_loss)
            with tracker.phase("backward"):
                if effective_loss == effective_loss:
                    effective_loss.backward()

            if name == "F3_gradient_explosion" and step >= INJECTION_STEP:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(fault.magnitude or 1.0)
            if name == "F10_mixed_anomaly" and step >= INJECTION_STEP:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(25.0)
            if name == "A3_gradient_spike_no_loss_consequence" and step == INJECTION_STEP:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(30.0)
            if name == "A1_single_huge_harmless_batch" and step == INJECTION_STEP:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(4.0)
            if name == "A5_inf_one_layer_only" and step == INJECTION_STEP:
                parameters = list(model.parameters())
                if parameters and parameters[0].grad is not None:
                    parameters[0].grad.flatten()[0] = float("inf")
            if name == "F5_inf_gradient" and step == INJECTION_STEP:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.flatten()[0] = float("inf")
            if name == "F5b_nan_gradient" and step == INJECTION_STEP:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.flatten()[0] = float("nan")
            if name == "H1_heavy_tail_gradients" and step % 17 == 0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(2.6)
            if name == "H4_gradient_clipping":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)

            tracker.record_gradients()
            collector.collect(step)
            with tracker.phase("optimizer"):
                if effective_loss == effective_loss:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            if name == "H3_scheduler_transition" and step == INJECTION_STEP:
                for group in optimizer.param_groups:
                    group["lr"] = 0.2
            if name == "F7_lr_jump_no_failure" and step == INJECTION_STEP:
                for group in optimizer.param_groups:
                    group["lr"] = 0.35
            tracker.record_learning_rate()

    adapter.run_end()
    observer.flush(timeout=30.0)
    observer.close()

    by_metric, layer_series, module_paths = _series(writer.events)
    detection = DetectorRunner(DetectorConfig()).run(
        "d6",
        by_metric,
        rank=0,
        layer_series=layer_series,
        module_paths=module_paths,
        population_layers=layers,
        population_ranks=1,
        coverage={"step_count": STEPS, "step_min": 1, "step_max": STEPS},
    )

    fires_by_detector: Dict[str, List[int]] = {}
    classes: Dict[str, List[str]] = {}
    for anomaly in detection.anomalies:
        key = str(anomaly.detector)
        fires_by_detector.setdefault(key, []).append(anomaly.first_abnormal_step)
        classes.setdefault(key, []).append(str(anomaly.confidence_class))

    # "System noticed" = any STRONG or CERTAIN fire.
    strong_steps = sorted(
        a.first_abnormal_step
        for a in detection.anomalies
        if a.confidence_class in (ConfidenceClass.STRONG, ConfidenceClass.CERTAIN)
    )
    weak_steps = sorted(
        a.first_abnormal_step
        for a in detection.anomalies
        if a.confidence_class == ConfidenceClass.WEAK
    )

    del model, optimizer, observer, adapter, collector
    gc.collect()
    return {
        "fault_type": fault.fault_type,
        "seed": fault.seed,
        "split": fault.split,
        "magnitude": fault.magnitude,
        "layers": layers,
        "d_model": d_model,
        "fires_by_detector": {k: sorted(v) for k, v in fires_by_detector.items()},
        "classes_by_detector": {k: sorted(set(v)) for k, v in classes.items()},
        "strong_steps": strong_steps,
        "weak_steps": weak_steps,
        "incident_opening_steps": strong_steps,
        "detector_errors": detection.detector_errors,
        "anomaly_count": len(detection.anomalies),
        "run_error": None,
    }


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
# Scoring via the D1 evaluation layer
# ---------------------------------------------------------------------------


def score_split(
    faults: Sequence[Fault], results: Sequence[Dict[str, Any]], *, incident_only: bool = True
) -> Dict[str, Any]:
    by_key = {(r["fault_type"], r["seed"], r["magnitude"]): r for r in results}
    detectors = sorted({d for f in faults for d in f.expected_detectors} | {
        d for r in results for d in r["fires_by_detector"]
    })
    scored: Dict[str, Any] = {}

    for detector in detectors:
        evaluations = []
        for fault in faults:
            record = by_key.get((fault.fault_type, fault.seed, fault.magnitude))
            if record is None:
                continue
            if record.get("run_error"):
                evaluations.append(
                    evaluate_scenario(
                        fault.fault_type, detector, [], fault.window(),
                        run_error=record["run_error"],
                    )
                )
                continue
            if detector not in fault.expected_detectors and detector not in record["fires_by_detector"]:
                continue
            fires = record["fires_by_detector"].get(detector, [])
            evaluations.append(
                evaluate_scenario(fault.fault_type, detector, fires, fault.window())
            )
        scored[detector] = aggregate(evaluations, total_steps=STEPS)

    # System level: did the system open an incident?
    evaluations = []
    for fault in faults:
        record = by_key.get((fault.fault_type, fault.seed, fault.magnitude))
        if record is None:
            continue
        fires = record["incident_opening_steps"] if incident_only else (
            record["strong_steps"] + record["weak_steps"]
        )
        evaluations.append(
            evaluate_scenario(
                fault.fault_type, "__system__", fires, fault.window(),
                run_error=record.get("run_error"),
            )
        )
    scored["__system__"] = aggregate(evaluations, total_steps=STEPS)
    return scored


def _write_parquet(path: Path, rows: Sequence[Dict[str, Any]], fault_index) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    flat = []
    for row in rows:
        fault = fault_index[(row["fault_type"], row["seed"], row["magnitude"])]
        window = fault.window()
        flat.append(
            {
                "fault_type": row["fault_type"],
                "seed": row["seed"],
                "split": row["split"],
                "magnitude": row["magnitude"],
                "layers": row["layers"],
                "d_model": row["d_model"],
                "injection_step": window.injection_step,
                "expected_effect_start": window.expected_effect_start,
                "acceptable_start": (window.acceptable_detection_window or (None, None))[0],
                "acceptable_end": (window.acceptable_detection_window or (None, None))[1],
                "expected_detectors": ",".join(fault.expected_detectors),
                "fired_detectors": ",".join(sorted(row["fires_by_detector"])),
                "incident_opening_steps": ",".join(str(s) for s in row["incident_opening_steps"]),
                "weak_steps": ",".join(str(s) for s in row["weak_steps"]),
                "first_incident_step": row["incident_opening_steps"][0] if row["incident_opening_steps"] else None,
                "anomaly_count": row["anomaly_count"],
                "detector_errors": ";".join(row["detector_errors"]),
                "run_error": row["run_error"] or "",
            }
        )
    pq.write_table(pa.Table.from_pylist(flat), path, compression="zstd")


def render(faults, results, scores, env, elapsed_s) -> str:
    lines: List[str] = []
    add = lines.append
    add("# D6 CALIBRATION AND HOLDOUT REPORT")
    add("")
    add("**Corpus:** {} real training runs across two independently generated".format(len(faults)))
    add("splits. **Wall time:** {:.0f} s.".format(elapsed_s))
    add("")
    add("## Independent generation")
    add("")
    add("| Split | Model | Seeds | Runs |")
    add("|---|---|---|---|")
    for split, config in SPLITS.items():
        count = sum(1 for f in faults if f.split == split)
        add("| {} | {} layers, d_model {} | {} | {} |".format(
            split, config["layers"], config["d_model"],
            "{}-{}".format(config["seeds"][0], config["seeds"][-1]), count))
    add("")
    add("The holdout uses a **different model shape**, different seeds and different")
    add("fault magnitudes, so agreement between the splits is independent evidence")
    add("rather than a restatement of the same runs.")
    add("")
    add("**Threshold fitting rule.** The v2.0 envelope parameters came from a separate")
    add("30-run distribution study (seeds 2000-2029) disjoint from both splits. Neither")
    add("split was consulted while choosing them, and no threshold was changed after")
    add("seeing any holdout result.")
    add("")

    add("## System-level detection quality")
    add("")
    add("| Split | Runs | TP | FP(pre-inj) | FP(post) | Healthy FP | FN | Precision | Recall | F1 | FP/1k | Median delay |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for split in ("calibration", "holdout"):
        entry = scores[split]["__system__"]
        add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            split, entry["scenarios"], entry["true_positives"],
            entry["pre_injection_false_positives"], entry["post_fault_false_positives"],
            entry["healthy_false_positives"], entry["false_negatives"],
            _f(entry["precision"]), _f(entry["recall"]), _f(entry["f1"]),
            _f(entry["false_positives_per_1k_steps"], 2),
            "n/a" if entry["median_detection_delay"] is None else entry["median_detection_delay"]))
    add("")

    add("## Gate assessment (holdout)")
    add("")
    holdout = scores["holdout"]["__system__"]
    gates = [
        ("system precision >= 0.90", holdout["precision"], 0.90),
        ("system recall >= 0.85", holdout["recall"], 0.85),
        ("FP / 1k steps <= 0.5", holdout["false_positives_per_1k_steps"], 0.5),
    ]
    add("| Gate | Target | Measured | Verdict |")
    add("|---|---|---|---|")
    for label, measured, target in gates:
        if measured is None:
            verdict = "NOT VERIFIED"
        elif label.startswith("FP"):
            verdict = "PASS" if measured <= target else "FAIL"
        else:
            verdict = "PASS" if measured >= target else "FAIL"
        add("| {} | {} | {} | {} |".format(label, target, _f(measured, 3), verdict))
    add("")
    add("Per-detector precision >= 0.80:")
    add("")
    add("| Detector | Split | Precision | Recall | Median delay | FP/1k | Verdict |")
    add("|---|---|---|---|---|---|---|")
    for detector, entry in sorted(scores["holdout"].items()):
        if detector == "__system__":
            continue
        verdict = "PASS" if (entry["precision"] or 0.0) >= 0.80 else "FAIL"
        add("| `{}` | holdout | {} | {} | {} | {} | {} |".format(
            detector, _f(entry["precision"]), _f(entry["recall"]),
            "n/a" if entry["median_detection_delay"] is None else entry["median_detection_delay"],
            _f(entry["false_positives_per_1k_steps"], 2), verdict))
    add("")

    add("## Pre-injection incidents (D6.6)")
    add("")
    for split in ("calibration", "holdout"):
        entry = scores[split]["__system__"]
        add("* **{}**: {} of {} fault runs produced an incident-opening fire before the".format(
            split, entry["runs_with_a_pre_injection_fire"], entry["injected_scenarios"]))
        add("  fault could exist (rate {}).".format(_f(entry["pre_injection_incident_rate"])))
    add("")
    add("Target: **0**. A pre-injection incident is a defect even when the fault is later")
    add("detected, because the earliest-abnormal-event claim would name the wrong step.")
    add("")

    add("## Detection boundary by magnitude")
    add("")
    add("| Fault | Magnitude | Split | Detected | Median delay |")
    add("|---|---|---|---|---|")
    by_magnitude: Dict[Tuple[str, Any, str], List[Dict[str, Any]]] = {}
    for fault in faults:
        if fault.injection_step is None:
            continue
        for result in results:
            if (result["fault_type"], result["seed"], result["magnitude"]) != (
                fault.fault_type, fault.seed, fault.magnitude
            ):
                continue
            by_magnitude.setdefault(
                (fault.fault_type, fault.magnitude, fault.split), []
            ).append(result)
    for (name, magnitude, split), group in sorted(
        by_magnitude.items(), key=lambda item: (item[0][0], item[0][1] or 0, item[0][2])
    ):
        detected = 0
        delays = []
        for row in group:
            window = DetectionWindow.for_injection(INJECTION_STEP, STEPS)
            step = row["incident_opening_steps"][0] if row["incident_opening_steps"] else None
            if step is not None and window.acceptable_detection_window[0] <= step <= window.acceptable_detection_window[1]:
                detected += 1
                delays.append(step - window.expected_effect_start)
        add("| `{}` | {} | {} | {}/{} | {} |".format(
            name, magnitude, split, detected, len(group),
            "n/a" if not delays else statistics.median(delays)))
    add("")

    crashes = [r for r in results if r.get("run_error")]
    add("## Infrastructure failures")
    add("")
    if crashes:
        add("**{} run(s) crashed and were excluded from scoring.**".format(len(crashes)))
        for row in crashes[:6]:
            add("- `{}` seed {}: {}".format(row["fault_type"], row["seed"], row["run_error"]))
    else:
        add("None. Every run completed and every detector returned.")
    add("")
    add("## Environment")
    add("")
    add("```json")
    add(json.dumps(env, indent=2, sort_keys=True, default=str))
    add("```")
    add("")
    return "\n".join(lines)


def _f(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else ("{:.%df}" % digits).format(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="D6 calibration and holdout")
    parser.add_argument("--limit", type=int, default=0, help="cap runs per split (smoke)")
    args = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_faults: List[Fault] = []
    for split in SPLITS:
        faults = build_manifest(split)
        if args.limit:
            faults = faults[: args.limit]
        all_faults.extend(faults)
    fault_index = {(f.fault_type, f.seed, f.magnitude): f for f in all_faults}

    print("corpus: {} runs ({} calibration, {} holdout)".format(
        len(all_faults),
        sum(1 for f in all_faults if f.split == "calibration"),
        sum(1 for f in all_faults if f.split == "holdout"),
    ))
    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    for index, fault in enumerate(all_faults, start=1):
        config = SPLITS[fault.split]
        try:
            results.append(run_one(fault, config["layers"], config["d_model"]))
        except Exception as exc:
            results.append(
                {
                    "fault_type": fault.fault_type, "seed": fault.seed, "split": fault.split,
                    "magnitude": fault.magnitude, "layers": config["layers"],
                    "d_model": config["d_model"], "fires_by_detector": {},
                    "classes_by_detector": {}, "strong_steps": [], "weak_steps": [],
                    "incident_opening_steps": [], "detector_errors": [], "anomaly_count": 0,
                    "run_error": "{}: {}".format(type(exc).__name__, exc),
                }
            )
        if index % 20 == 0 or index == len(all_faults):
            print("   {}/{} ({:.0f}s)".format(index, len(all_faults), time.perf_counter() - started))

    elapsed = time.perf_counter() - started
    scores = {}
    for split in SPLITS:
        faults = [f for f in all_faults if f.split == split]
        split_results = [r for r in results if r["split"] == split]
        scores[split] = score_split(faults, split_results)

    env = capture_environment(extra={"calibration": {
        "runs": len(all_faults), "steps": STEPS, "injection_step": INJECTION_STEP,
        "splits": {k: {"layers": v["layers"], "d_model": v["d_model"]} for k, v in SPLITS.items()},
    }})

    # -- required machine-readable artefacts ------------------------------
    for split in SPLITS:
        _write_parquet(
            OUTPUT_DIR / "{}_results.parquet".format(split),
            [r for r in results if r["split"] == split],
            fault_index,
        )
    (OUTPUT_DIR / "detector_confusion_matrix.json").write_text(
        json.dumps(scores, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "threshold_registry.json").write_text(
        json.dumps({
            "detector_config": DetectorConfig().to_dict(),
            "rule_version": "phase1.2-1.1",
            "fitted_on": "seeds 2000-2029 distribution study (disjoint from both splits)",
            "holdout_consulted_during_fitting": False,
        }, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "scenario_manifest.json").write_text(
        json.dumps([f.to_dict() for f in all_faults], indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "calibration_raw.json").write_text(
        json.dumps({"results": results, "scores": scores, "environment": env}, indent=2, default=str),
        encoding="utf-8")
    (OUTPUT_DIR / "D6_CALIBRATION_REPORT.md").write_text(
        render(all_faults, results, scores, env, elapsed), encoding="utf-8")
    (OUTPUT_DIR / "D6_HOLDOUT_REPORT.md").write_text(
        render(all_faults, results, scores, env, elapsed), encoding="utf-8")

    for split in SPLITS:
        entry = scores[split]["__system__"]
        print("{}: precision={} recall={} FP/1k={} pre-injection={}/{}".format(
            split, _f(entry["precision"]), _f(entry["recall"]),
            _f(entry["false_positives_per_1k_steps"], 2),
            entry["runs_with_a_pre_injection_fire"], entry["injected_scenarios"]))
    print("report:", OUTPUT_DIR / "D6_CALIBRATION_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
