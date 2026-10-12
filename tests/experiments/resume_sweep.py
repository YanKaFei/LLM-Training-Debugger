"""W2: pre-registered sweep of `resume_after_steps`.

Design frozen in ``outputs/alpha/W2_PRE_REGISTRATION.md`` before this ran.

Trains every (config, seed, magnitude) **once**, caches the per-layer gradient series,
then evaluates every candidate on that identical evidence. The comparison is therefore
paired: a difference between candidates cannot be seed noise.
"""

from __future__ import annotations

import gc
import json
import math
import pickle
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
    LayerMetricCollector, LayerSelection, PyTorchTrainingAdapter,
)
from training_debugger.detectors import DetectorConfig, DetectorRunner  # noqa: E402
from training_debugger.episodes import EpisodeEngine  # noqa: E402
from training_debugger.fusion import fuse  # noqa: E402
from training_debugger.registry import detector_registry  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver,
)

OUTPUT_DIR = ROOT / "outputs" / "alpha"
CACHE = OUTPUT_DIR / "w2_evidence_cache.pkl"

# ---- frozen in the pre-registration -------------------------------------
CANDIDATES: Sequence[Optional[int]] = (20, 30, 40, 60, None)   # None == "never resume"
CONFIGS = ((4, 32), (6, 48))
HEALTHY_SEEDS = tuple(range(70000, 70060))
MAGNITUDES = (1.5, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 40.0)
FAULT_SEEDS = tuple(range(71000, 71012))
STEPS = 200
INJECTION = 110
GATE_PRECISION, GATE_RECALL, GATE_HEALTHY_FP = 0.90, 0.85, 0.05


def run_once(n_layers: int, d_model: int, seed: int, magnitude: Optional[float]) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model, config = make_tiny_lm(seed=seed, n_layers=n_layers, d_model=d_model, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "w2", writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384), rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="w2")
    collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=32), cadence=1)
    adapter.run_start(steps=STEPS)
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            tracker.record_loss(loss)
            loss.backward()
            if magnitude is not None and step == INJECTION:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(magnitude)
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
    by_metric: Dict[str, Dict[int, float]] = {}
    layer_series: Dict[int, Dict[int, float]] = {}
    for event in writer.events:
        if event.metric_name is None or event.step is None or event.metric_value is None:
            continue
        if event.metric_name == "grad.layer_norm" and event.layer_index is not None:
            layer_series.setdefault(int(event.layer_index), {})[event.step] = event.metric_value
        else:
            by_metric.setdefault(str(event.metric_name), {})[event.step] = event.metric_value
    del model, optimizer, observer, adapter, collector, writer
    gc.collect()
    return {
        "n_layers": n_layers, "d_model": d_model, "seed": seed, "magnitude": magnitude,
        "by_metric": {k: sorted(v.items()) for k, v in by_metric.items()},
        "layer_series": {k: sorted(v.items()) for k, v in layer_series.items()},
    }


def evaluate(evidence: Dict[str, Any], resume: Optional[int]) -> Dict[str, Any]:
    """Run the full analysis path with one candidate value."""
    config = DetectorConfig()
    config.baseline_resume_after_steps = resume if resume is not None else 10 ** 6
    detection = DetectorRunner(config).run(
        "w2", evidence["by_metric"], rank=0, layer_series=evidence["layer_series"],
        module_paths={}, population_layers=evidence["n_layers"], population_ranks=1,
        coverage={"step_count": STEPS, "step_min": 1, "step_max": STEPS},
    )
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    engine.consume(detection.anomalies, "w2")
    engine.finish("w2", final_step=STEPS)
    episodes = engine.all_episodes()
    verdicts = fuse("w2", episodes)
    confirmed = [v for v in verdicts if v.verdict == "CONFIRMED_INCIDENT"]
    onset = min((v.candidate_onset_step for v in confirmed
                 if v.candidate_onset_step is not None), default=None)
    confirmed_step = min((v.confirmed_at_step for v in confirmed
                          if v.confirmed_at_step is not None), default=None)
    return {
        "incident": bool(confirmed),
        "episodes": len(episodes),
        "onset": onset,
        "confirmed_at": confirmed_step,
        "abstained": bool(verdicts) and all(
            v.verdict == "INSUFFICIENT_EVIDENCE" for v in verdicts
        ),
    }


def build_cache() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total = len(CONFIGS) * len(HEALTHY_SEEDS) + len(MAGNITUDES) * len(FAULT_SEEDS)
    done = 0
    started = time.perf_counter()
    for n_layers, d_model in CONFIGS:
        for seed in HEALTHY_SEEDS:
            rows.append({"kind": "healthy", **run_once(n_layers, d_model, seed, None)})
            done += 1
            if done % 20 == 0:
                print("  cached {}/{} ({:.0f}s)".format(
                    done, total, time.perf_counter() - started))
    for magnitude in MAGNITUDES:
        for seed in FAULT_SEEDS:
            rows.append({"kind": "fault", **run_once(4, 32, seed, magnitude)})
            done += 1
            if done % 20 == 0:
                print("  cached {}/{} ({:.0f}s)".format(
                    done, total, time.perf_counter() - started))
    return rows


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        print("[W2] loading cached evidence ...")
        with CACHE.open("rb") as handle:
            rows = pickle.load(handle)
    else:
        print("[W2] training {} runs once ...".format(
            len(CONFIGS) * len(HEALTHY_SEEDS) + len(MAGNITUDES) * len(FAULT_SEEDS)))
        rows = build_cache()
        with CACHE.open("wb") as handle:
            pickle.dump(rows, handle, protocol=4)
    print("[W2] evidence cached: {} runs".format(len(rows)))

    results: Dict[str, Any] = {}
    for candidate in CANDIDATES:
        label = "never" if candidate is None else str(candidate)
        print("[W2] evaluating resume_after_steps={} ...".format(label))
        tp = fp = fn = 0
        healthy_runs = healthy_incidents = 0
        fault_runs = fault_abstained = 0
        delays: List[int] = []
        per_magnitude: Dict[str, List[int]] = {str(m): [] for m in MAGNITUDES}
        for row in rows:
            outcome = evaluate(row, candidate)
            if row["kind"] == "healthy":
                healthy_runs += 1
                healthy_incidents += 1 if outcome["incident"] else 0
            else:
                fault_runs += 1
                fault_abstained += 1 if outcome["abstained"] else 0
                per_magnitude[str(row["magnitude"])].append(1 if outcome["incident"] else 0)
                if outcome["incident"]:
                    tp += 1
                    if outcome["onset"] is not None:
                        delays.append(outcome["onset"] - INJECTION)
                else:
                    fn += 1
        fp = healthy_incidents
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        healthy_rate = healthy_incidents / healthy_runs if healthy_runs else None
        results[label] = {
            "resume_after_steps": candidate,
            "tp": tp, "fp": fp, "fn": fn,
            "healthy_runs": healthy_runs, "healthy_incidents": healthy_incidents,
            "precision": precision, "recall": recall, "healthy_rate": healthy_rate,
            "fault_abstention_rate": fault_abstained / fault_runs if fault_runs else None,
            "median_confirmation_delay": (
                sorted(delays)[len(delays) // 2] if delays else None
            ),
            "magnitude_response": {
                m: (sum(v) / len(v) if v else None) for m, v in per_magnitude.items()
            },
            "per_run": None,
        }

    # -- paired comparison against the default (40) ----------------------
    baseline_label = "40"
    paired: Dict[str, Any] = {}
    for row in rows:
        row["_outcomes"] = {("never" if c is None else str(c)): evaluate(row, c)
                            for c in CANDIDATES}
    for label in results:
        if label == baseline_label:
            continue
        discordant_a = discordant_b = 0
        for row in rows:
            if row["kind"] != "fault":
                continue
            a = row["_outcomes"][label]["incident"]
            b = row["_outcomes"][baseline_label]["incident"]
            if a and not b:
                discordant_a += 1
            elif b and not a:
                discordant_b += 1
        n = discordant_a + discordant_b
        paired[label] = {
            "gained_confirmed": discordant_a, "lost_confirmed": discordant_b,
            "discordant_pairs": n,
            "sign_test_p": (
                None if n == 0 else
                sum(math.comb(n, k) for k in range(0, min(discordant_a, discordant_b) + 1))
                / (2 ** n) * 2
            ),
        }

    qualifying = [
        label for label, r in results.items()
        if (r["precision"] or 0) >= GATE_PRECISION
        and (r["recall"] or 0) >= GATE_RECALL
        and (r["healthy_rate"] if r["healthy_rate"] is not None else 1) <= GATE_HEALTHY_FP
    ]
    payload = {
        "pre_registration": "outputs/alpha/W2_PRE_REGISTRATION.md",
        "candidates": ["never" if c is None else str(c) for c in CANDIDATES],
        "runs": len(rows),
        "gate": {"precision": GATE_PRECISION, "recall": GATE_RECALL,
                 "healthy_fp": GATE_HEALTHY_FP},
        "results": results,
        "paired_vs_default_40": paired,
        "qualifying_candidates": qualifying,
        "selection": (
            "none: no candidate satisfies the pre-registered gates" if not qualifying
            else "tie-break: highest recall, then lowest healthy rate"
        ),
    }
    (OUTPUT_DIR / "w2_resume_sweep.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("")
    print("resume  tp   fp   fn   precision  recall  healthy_fp  abstain  delay")
    for label in results:
        r = results[label]
        print("%-7s %-4d %-4d %-4d %-10s %-7s %-11s %-8s %s" % (
            label, r["tp"], r["fp"], r["fn"],
            "n/a" if r["precision"] is None else "{:.3f}".format(r["precision"]),
            "n/a" if r["recall"] is None else "{:.3f}".format(r["recall"]),
            "n/a" if r["healthy_rate"] is None else "{:.3f}".format(r["healthy_rate"]),
            "n/a" if r["fault_abstention_rate"] is None else "{:.3f}".format(r["fault_abstention_rate"]),
            r["median_confirmation_delay"]))
    print("")
    print("qualifying candidates:", qualifying or "NONE")
    print("paired vs default(40):", json.dumps(paired, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
