"""W5: paired A/B measurement of the Optimizer Boundary Witness.

A and B are interleaved on the same machine with identical seeds, model and inputs, so
drift in the environment affects both equally. Reporting a single run's difference would
be the mistake Phase 1.1 made when it produced a *negative* overhead.

The two standing product targets are **not** redefined by this measurement:
``core observer overhead < 3%`` and ``T1 total overhead < 8%``.
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tiny_model import make_tiny_lm, synthetic_batch  # noqa: E402

from training_debugger.adapters.pytorch import OptimizerWitness  # noqa: E402

OUT = ROOT / "outputs" / "alpha" / "v015"
STEPS = 120
REPEATS = 7


def one_run(enable: bool, seed: int = 4242) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model, config = make_tiny_lm(seed=seed, n_layers=4, d_model=32, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    witness = OptimizerWitness(enabled=enable)
    if enable:
        witness.attach(optimizer)

    latencies: List[float] = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        t0 = time.perf_counter()
        loss = model(**batch)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if enable:
            witness.note_attempted(training_step_id=step)
        optimizer.step()
        if enable:
            witness.note_invoked()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    total = time.perf_counter() - started

    if enable:
        witness.detach()
    records = len(witness.records)
    del model, optimizer, witness
    gc.collect()
    ordered = sorted(latencies)
    return {
        "total_s": total,
        "median_ms": statistics.median(latencies),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "witness_records": records,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[W5] warming up ...")
    one_run(False)
    one_run(True)

    off: List[Dict[str, Any]] = []
    on: List[Dict[str, Any]] = []
    for repeat in range(REPEATS):
        print("[W5] repeat {}/{} ...".format(repeat + 1, REPEATS))
        # Interleaved, and the order alternates, so a monotonic thermal drift cannot
        # be mistaken for a treatment effect.
        if repeat % 2 == 0:
            off.append(one_run(False))
            on.append(one_run(True))
        else:
            on.append(one_run(True))
            off.append(one_run(False))

    def summarise(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
        values = [r[key] for r in rows]
        median = statistics.median(values)
        if len(values) > 1:
            spread = statistics.stdev(values)
            stderr = spread / (len(values) ** 0.5)
        else:
            stderr = 0.0
        return {"median": median, "min": min(values), "max": max(values),
                "stderr": stderr}

    report: Dict[str, Any] = {"steps": STEPS, "repeats": REPEATS,
                              "raw": {"off": off, "on": on}}
    for key in ("total_s", "median_ms", "p95_ms"):
        a = summarise(off, key)
        b = summarise(on, key)
        delta = b["median"] - a["median"]
        pct = (delta / a["median"] * 100.0) if a["median"] else None
        report[key] = {
            "off": a, "on": b, "delta": delta, "delta_pct": pct,
            "paired_stderr": (a["stderr"] ** 2 + b["stderr"] ** 2) ** 0.5,
        }
    report["witness_event_volume"] = {
        "records_off": off[0]["witness_records"],
        "records_on": on[0]["witness_records"],
    }
    report["targets"] = {"core_observer_overhead_pct": 3.0, "t1_total_overhead_pct": 8.0}
    report["note"] = (
        "This measures the witness increment only. It does NOT re-measure, and must not "
        "be read as resolving, the separate standing finding that observer overhead "
        "exceeds the <3% core target at the largest workload."
    )
    (OUT / "witness_overhead.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("")
    print("%-12s %12s %12s %10s %10s" % ("metric", "off", "on", "delta", "delta%"))
    for key in ("total_s", "median_ms", "p95_ms"):
        row = report[key]
        print("%-12s %12.4f %12.4f %10.4f %9s%%" % (
            key, row["off"]["median"], row["on"]["median"], row["delta"],
            "n/a" if row["delta_pct"] is None else "{:.2f}".format(row["delta_pct"])))
    print("")
    print("witness records: off=%d on=%d" % (
        report["witness_event_volume"]["records_off"],
        report["witness_event_volume"]["records_on"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
