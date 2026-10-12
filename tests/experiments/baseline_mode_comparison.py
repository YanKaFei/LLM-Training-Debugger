"""A: frozen vs EMA baseline, measured on the real paired corpus.

Three questions, each with a number:

1. **Readiness.** The v0.1.7 root cause: the frozen baseline produces no usable ratio
   until ``baseline_floor + minimum_history`` = 70 steps. The EMA needs only its warm-up.
2. **Detection latency.** On the paired corpus the injection step is known exactly, so
   the step at which the detector first fires can be compared against it.
3. **Healthy-run false incidents.** The risk of a shorter warm-up: a 25-step warm-up is a
   *narrower evidence base*. This project's healthy FP rate is already 7.7 % against a
   5 % gate, so this is the number that can falsify the change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from training_debugger.detectors import DetectorConfig, DetectorRunner  # noqa: E402
from training_debugger.detectors.baseline import (  # noqa: E402
    adaptive_ratio_series, protected_ratio_series,
)
from training_debugger.episodes import EpisodeEngine  # noqa: E402
from training_debugger.fusion import fuse  # noqa: E402
from training_debugger.registry import detector_registry  # noqa: E402
from training_debugger.storage import DuckDBReader  # noqa: E402

OUT = ROOT / "outputs" / "v0.1.8-alpha"
PAIRED = OUT / "paired-store"
INJECTION = 90
MODES = ("frozen", "ema")


def load_series(reader: DuckDBReader, run_id: str):
    table = reader.events(run_id)
    by_metric: Dict[str, Dict[int, float]] = {}
    layers: Dict[int, Dict[int, float]] = {}
    for name, step, value, layer in zip(
        table.column("metric_name").to_pylist(),
        table.column("step").to_pylist(),
        table.column("metric_value").to_pylist(),
        table.column("layer_index").to_pylist(),
    ):
        if name is None or step is None or value is None:
            continue
        if name == "grad.layer_norm" and layer is not None:
            layers.setdefault(int(layer), {})[int(step)] = float(value)
        else:
            by_metric.setdefault(str(name), {})[int(step)] = float(value)
    return ({k: sorted(v.items()) for k, v in by_metric.items()},
            {k: sorted(v.items()) for k, v in layers.items()})


def readiness() -> Dict[str, Optional[int]]:
    """First step at which each mode can produce a ratio, in production configuration."""
    config = DetectorConfig()
    series = [(s, 1.0) for s in range(1, 120)]
    frozen = protected_ratio_series(
        [(s, v) for s, v in series if s >= config.warmup_steps], minimum_history=20)
    ema = adaptive_ratio_series(series, warmup=config.baseline_warmup_steps)
    return {
        "frozen_first_ratio_step": min((s for s, v in frozen.items() if v is not None),
                                       default=None),
        "ema_first_ratio_step": min((s for s, v in ema.items() if v is not None),
                                    default=None),
        "baseline_floor": config.warmup_steps,
        "frozen_minimum_history": 20,
        "ema_warmup": config.baseline_warmup_steps,
    }


def analyse(by_metric, layers, mode: str):
    config = DetectorConfig(baseline_mode=mode)
    detection = DetectorRunner(config).run(
        "probe", by_metric, rank=0, layer_series=layers, module_paths={},
        population_layers=4, population_ranks=1,
        coverage={"step_count": 150, "step_min": 1, "step_max": 150},
    )
    gradient_fires = sorted(a.first_abnormal_step for a in detection.anomalies
                            if str(a.detector) == "grad.robust_zscore")
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    engine.consume(detection.anomalies, "probe")
    engine.finish("probe", final_step=150)
    verdicts = fuse("probe", engine.all_episodes())
    return {
        "gradient_first_fire": gradient_fires[0] if gradient_fires else None,
        "incidents": sum(1 for v in verdicts if v.verdict == "CONFIRMED_INCIDENT"),
        "episodes": len(engine.all_episodes()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"readiness": readiness(), "paired": {}, "healthy": {}}

    truth_path = OUT / "paired_ground_truth.json"
    if truth_path.exists() and PAIRED.exists():
        truth = json.loads(truth_path.read_text())
        reader = DuckDBReader(PAIRED)
        try:
            for mode in MODES:
                fault_fires: List[Optional[int]] = []
                clean_incidents = 0
                clean_runs = 0
                for entry in truth.values():
                    by_metric, layers = load_series(reader, entry["run_b"])
                    outcome = analyse(by_metric, layers, mode)
                    if entry["fault"] == "none":
                        clean_runs += 1
                        clean_incidents += outcome["incidents"]
                    else:
                        fault_fires.append(outcome["gradient_first_fire"])
                latencies = [f - INJECTION for f in fault_fires if f is not None]
                report["paired"][mode] = {
                    "fault_runs": len(fault_fires),
                    "runs_with_a_gradient_fire": sum(1 for f in fault_fires if f is not None),
                    "median_latency_vs_injection": (
                        sorted(latencies)[len(latencies) // 2] if latencies else None),
                    "latencies": latencies,
                    "clean_runs": clean_runs,
                    "clean_runs_with_incident": clean_incidents,
                }
        finally:
            reader.close()

    (OUT / "baseline_mode_comparison.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    r = report["readiness"]
    print("READINESS (production configuration)")
    print("  frozen : first usable ratio at step {}   (floor {} + minimum_history {})".format(
        r["frozen_first_ratio_step"], r["baseline_floor"], r["frozen_minimum_history"]))
    print("  ema    : first usable ratio at step {}   (warmup {})".format(
        r["ema_first_ratio_step"], r["ema_warmup"]))
    if r["frozen_first_ratio_step"] and r["ema_first_ratio_step"]:
        print("  -> {} fewer blind steps".format(
            r["frozen_first_ratio_step"] - r["ema_first_ratio_step"]))
    print()
    for mode in MODES:
        row = report["paired"].get(mode)
        if not row:
            continue
        print("PAIRED CORPUS — {} mode".format(mode))
        print("  gradient fired on {}/{} fault runs".format(
            row["runs_with_a_gradient_fire"], row["fault_runs"]))
        print("  median latency vs injection : {}".format(row["median_latency_vs_injection"]))
        print("  clean runs with an incident : {}/{}".format(
            row["clean_runs_with_incident"], row["clean_runs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
