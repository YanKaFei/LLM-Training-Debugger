"""What per-module activation observation costs.

The probe reads activations, which the rest of this project deliberately does not. That
is a new cost on the training path, so it is measured before anything is enabled by
default. Three configurations, paired and interleaved:

    off                 no hooks at all
    sampled  8192       bounded per-module read
    full               exact per-module read
"""

from __future__ import annotations

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

from training_debugger.adapters.pytorch import PyTorchTrainingAdapter  # noqa: E402
from training_debugger.adapters.pytorch.activations import ActivationProbe  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver,
)

OUT = ROOT / "outputs" / "v0.1.8-alpha"
STEPS = 80
REPEATS = 5
MODULES = ("blocks.0.mlp.2", "blocks.1.mlp.2", "blocks.2.mlp.2", "blocks.3.mlp.2")

CONFIGS: Dict[str, Any] = {
    "probe_off": {"enabled": False},
    "probe_sampled_8192": {"enabled": True, "sample_limit": 8192},
    "probe_full": {"enabled": True, "sample_limit": None},
}


def one_run(config: Dict[str, Any], seed: int = 4242) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model, cfg = make_tiny_lm(seed=seed, n_layers=4, d_model=32, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "actcost", writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=8192),
        rank=0, world_size=1,
    )
    probe = ActivationProbe(selection=list(MODULES), **config)
    attached = probe.attach(model)
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="actcost")
    adapter.run_start(steps=STEPS)

    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=cfg.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            tracker.record_loss(loss)
            loss.backward()
            tracker.record_gradients()
            if config["enabled"]:
                probe.collect(step, observer)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
    total = time.perf_counter() - started
    adapter.run_end()
    observer.flush(timeout=30.0)
    observer.close()
    probe.detach()
    return {"total_s": total, "hooks": attached,
            "activation_observations": len(probe.records)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS.values():
        one_run(config)

    results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in CONFIGS}
    for repeat in range(REPEATS):
        order = list(CONFIGS)
        if repeat % 2:
            order.reverse()
        for name in order:
            results[name].append(one_run(CONFIGS[name]))

    baseline = statistics.median(r["total_s"] for r in results["probe_off"])
    report: Dict[str, Any] = {"steps": STEPS, "repeats": REPEATS, "configs": {}}
    for name, rows in results.items():
        totals = [r["total_s"] for r in rows]
        median = statistics.median(totals)
        stderr = statistics.stdev(totals) / (len(totals) ** 0.5) if len(totals) > 1 else 0.0
        report["configs"][name] = {
            "median_total_s": median,
            "stderr": stderr,
            "delta_vs_off_pct": 100.0 * (median - baseline) / baseline if baseline else None,
            "hooks": rows[0]["hooks"],
            "activation_observations": rows[0]["activation_observations"],
            "raw": totals,
        }
    (OUT / "activation_probe_cost.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("%-22s %-12s %-12s %-16s %s" % (
        "config", "median s", "stderr", "delta vs off", "observations"))
    for name, row in report["configs"].items():
        print("%-22s %-12.4f %-12.5f %-16s %d" % (
            name, row["median_total_s"], row["stderr"],
            "baseline" if name == "probe_off" else "{:+.2f}%".format(row["delta_vs_off_pct"]),
            row["activation_observations"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
