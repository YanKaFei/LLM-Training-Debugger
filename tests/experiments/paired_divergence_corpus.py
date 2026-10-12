"""A paired corpus with EVENTS persisted, so `compare` can be validated on real data.

Two gaps this closes:

1. Every existing corpus wrote only episodes and incidents. Nothing wrote the raw
   `TrainingEvent`s to Parquet, so no event-level query — including the new `compare`
   command — could be reproduced from disk. That is a reproducibility defect.
2. `compare` localizes a divergence step. Its accuracy had never been checked against a
   run pair whose true divergence step is known.

Each pair is the SAME scenario and seed, with the fault injected into run B only at a
known step. `compare A B` should report that step.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tiny_model import make_tiny_lm, synthetic_batch  # noqa: E402

from training_debugger.adapters.pytorch import (  # noqa: E402
    LayerMetricCollector, LayerSelection, PyTorchTrainingAdapter,
)
from training_debugger.observer import (  # noqa: E402
    ObserverConfig, SamplingPolicy, TrainingObserver,
)
from training_debugger.storage import ParquetEventWriter  # noqa: E402

OUT = ROOT / "outputs" / "v0.1.8-alpha"
STORE = OUT / "paired-store"

STEPS = 150
#: Past the detector warm-up, so any downstream analysis is not blocked by readiness.
INJECTION = 90

#: (pair name, fault kind) — run A is always clean, run B carries the fault.
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("grad_explosion", "grad"),
    ("loss_spike", "loss"),
    ("nonfinite_grad", "nonfinite"),
    ("clean_control", "none"),
)
SEEDS = (11000, 11001, 11002)
CONFIGS = ((4, 32),)


def train(run_id: str, fault: str, layers: int, d_model: int, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model, config = make_tiny_lm(seed=seed, n_layers=layers, d_model=d_model, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    writer = ParquetEventWriter(STORE, run_id, rows_per_file=50_000)
    observer = TrainingObserver(
        run_id, writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=8192),
        rank=0, world_size=1, model_id="paired",
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="paired")
    collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=32), cadence=1)
    adapter.run_start(steps=STEPS)

    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            effective = loss
            if fault == "loss" and step == INJECTION:
                effective = effective * 60.0
            tracker.record_loss(effective)
            if math.isfinite(float(effective.detach())):
                effective.backward()
                if step == INJECTION:
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        if fault == "grad":
                            p.grad.mul_(150.0)
                        elif fault == "nonfinite":
                            p.grad.mul_(float("inf"))
            tracker.record_gradients()
            collector.collect(step)
            with tracker.phase("optimizer"):
                if math.isfinite(float(effective.detach())):
                    try:
                        optimizer.step()
                    except Exception:
                        pass
            optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()

    adapter.run_end()
    observer.flush(timeout=60.0)
    health = observer.health()
    observer.close()
    del model, optimizer, observer, adapter, collector
    gc.collect()
    return {"health": health}


def main() -> int:
    STORE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    truth: Dict[str, Any] = {}
    for pair, fault in PAIRS:
        for layers, d_model in CONFIGS:
            for seed in SEEDS:
                run_a = "A_{}_{}_{}".format(pair, layers, seed)
                run_b = "B_{}_{}_{}".format(pair, layers, seed)
                print("[paired] {} ...".format(pair))
                train(run_a, "none", layers, d_model, seed)
                train(run_b, fault, layers, d_model, seed)
                truth["{}|{}".format(run_a, run_b)] = {
                    "pair": pair, "fault": fault, "seed": seed,
                    "run_a": run_a, "run_b": run_b,
                    "injection_step": INJECTION if fault != "none" else None,
                    "true_divergence_step": INJECTION if fault != "none" else None,
                }
    (OUT / "paired_ground_truth.json").write_text(
        json.dumps(truth, indent=2), encoding="utf-8")
    print("")
    print("wrote {} run pairs to {} ({:.0f}s)".format(
        len(truth), STORE, time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
