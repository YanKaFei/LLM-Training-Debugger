"""P0 from docs/research/EXTERNAL_LANDSCAPE.md: EMA + clipped-statistics vs freeze.

ZClip (arxiv 2504.02507 §2.3.4) solves the same problem the protected baseline solves --
"a spike must not be absorbed into the reference" -- without freezing:

    when a spike is detected, update the running statistics with the CLIPPED value,
    not the raw one. Including the spike skews the statistics and desensitises future
    detection; excluding it biases them low and makes the detector permanently
    over-aggressive.

The freeze design does only the second half and pays for it: this project measured that
freezing **never resumes**, costing precision 0.973->0.964 and recall 0.468->0.457
(D-010). This script measures whether the EMA variant beats it on three shapes that
separate the two failure modes.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from training_debugger.detectors.baseline import protected_ratio_series  # noqa: E402

OUT = ROOT / "outputs" / "v0.1.7-alpha"

ALPHA = 0.97            # ZClip default EMA smoothing
Z_THRES = 2.5           # ZClip default z threshold
WARMUP = 25             # ZClip default N_w


def ema_ratio_series(series, *, alpha=ALPHA, z_thres=Z_THRES, warmup=WARMUP,
                     **_) -> Dict[int, float]:
    """ZClip-style: EMA of mean and variance, spike values clipped for the update."""
    ratios: Dict[int, float] = {}
    window: List[float] = []
    mean = variance = None
    for step, value in series:
        if value is None or value != value or not value > 0:
            ratios[step] = None
            continue
        if mean is None:
            window.append(value)
            if len(window) < warmup:
                ratios[step] = None
                continue
            mean = statistics.fmean(window)
            variance = statistics.pvariance(window) if len(window) > 1 else 0.0
            ratios[step] = 1.0
            continue
        sigma = math.sqrt(variance) if variance > 0 else 1e-12
        z = (value - mean) / sigma
        ratios[step] = value / mean if mean > 0 else None
        # The clipped update: a spike contributes the clipped value, never the raw one.
        update = value
        if z > z_thres:
            update = mean + (z_thres ** 2 / z) * sigma
        mean = alpha * mean + (1 - alpha) * update
        variance = alpha * variance + (1 - alpha) * (update - mean) ** 2
    return ratios


def shapes() -> Dict[str, List[Tuple[int, float]]]:
    """Three shapes that separate the two failure modes."""
    out: Dict[str, List[Tuple[int, float]]] = {}

    # 1. a sustained fault: must NOT be absorbed into the reference
    out["sustained_x5"] = [(s, 1.0) for s in range(1, 61)] + \
                          [(s, 5.0) for s in range(61, 200)]

    # 2. benign recurring excursions, then real growth: the reference MUST keep up.
    #    This is the shape that made a healthy run look like an explosion (A2/S1).
    series = []
    for s in range(1, 200):
        value = 1.0
        if s % 20 == 0:
            value = 1.5              # benign excursion
        if s > 120:
            value = 2.2              # genuine secular growth
        series.append((s, value))
    out["benign_recurring_then_growth"] = series

    # 3. a genuine level rise that starts late: the ratio should reveal it
    out["late_level_rise"] = [(s, 1.0) for s in range(1, 140)] + \
                             [(s, 4.0) for s in range(140, 200)]
    return out


def summarise(name, ratios) -> Dict[str, float]:
    values = {s: v for s, v in ratios.items() if v is not None}
    if not values:
        return {}
    last = values[max(values)]
    return {
        "first_ready_step": min(values),
        "max_ratio": max(values.values()),
        "final_ratio": last,
        "n_scored": len(values),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"config": {"alpha": ALPHA, "z_thres": Z_THRES, "warmup": WARMUP},
              "shapes": {}}
    for name, series in shapes().items():
        freeze = protected_ratio_series(series, minimum_history=20)
        ema = ema_ratio_series(series)
        report["shapes"][name] = {
            "freeze": summarise(name, freeze),
            "ema_clipped": summarise(name, ema),
        }

    (OUT / "baseline_ema_vs_freeze.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("%-32s %-28s %s" % ("shape", "freeze (current)", "ema_clipped (ZClip-style)"))
    for name, row in report["shapes"].items():
        f, e = row["freeze"], row["ema_clipped"]
        print("%-32s ready=%-4s final=%-6.2f max=%-7.2f  ready=%-4s final=%-6.2f max=%.2f" % (
            name, f.get("first_ready_step"), f.get("final_ratio", 0),
            f.get("max_ratio", 0), e.get("first_ready_step"),
            e.get("final_ratio", 0), e.get("max_ratio", 0)))
    print()
    print("readiness: freeze needs minimum_history=20 AFTER a floor of 50;")
    print("           ema_clipped is ready after warmup=%d. That is the latency gap." % WARMUP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
