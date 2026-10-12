"""D2: how far does the evidence grow, and does fail-open stay visible?

Measures, on the real observer and stores:

    raw events emitted / persisted / dropped  at increasing step counts
    wall time, peak RSS, on-disk bytes
    detector observations, episodes, incidents as the fault widens
    whether a forced overflow reaches the *verdict* as DEGRADED (D3)

The point is to answer "is growth bounded and explicable?" with numbers rather than a
reassurance.
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

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
from training_debugger.fusion.quality import DataQuality, DataQualityStatus  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    BoundedEventQueue, ObserverConfig, SamplingPolicy, TrainingObserver,
)
from training_debugger.storage import ParquetEventWriter  # noqa: E402
from training_debugger.registry import detector_registry  # noqa: E402
from training_debugger.schema import (  # noqa: E402
    EventPriority, EventType, MetricUnit, TrainingEvent,
)

OUTPUT_DIR = ROOT / "outputs" / "alpha"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def disk_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def observe(steps: int, tmp: Path) -> Dict[str, Any]:
    torch.manual_seed(77)
    model, config = make_tiny_lm(seed=77, n_layers=4, d_model=32, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    writer = ParquetEventWriter(tmp, "scale", rows_per_file=50_000)
    observer = TrainingObserver(
        "scale", writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=8192), rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="scale")
    collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=8), cadence=1)
    adapter.run_start(steps=steps)
    before = rss_mb()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            tracker.record_loss(loss)
            loss.backward()
            tracker.record_gradients()
            collector.collect(step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()
    adapter.run_end()
    observer.flush(timeout=60.0)
    health = observer.health()
    observer.close()
    elapsed = time.perf_counter() - started
    writer.close() if hasattr(writer, "close") else None
    payload = {
        "steps": steps,
        "events_persisted": (health.get("observer") or {}).get("written_total"),
        "emitted": (health.get("observer") or {}).get("emitted_total"),
        "written": (health.get("observer") or {}).get("written_total"),
        "dropped": (health.get("observer") or {}).get("events_dropped_total"),
        "overflow": (health.get("observer") or {}).get("queue_overflow_total"),
        "wall_s": round(elapsed, 2),
        "rss_delta_mb": round(rss_mb() - before, 1),
        "disk_bytes": disk_bytes(tmp),
        "bytes_per_event": round(disk_bytes(tmp) / max(1, int((health.get("observer") or {}).get("written_total") or 1)), 2),
    }
    del model, optimizer, observer, adapter, collector, writer
    gc.collect()
    return payload


def fault_volume(widths=(1, 20, 100, 1000)) -> List[Dict[str, Any]]:
    rows = []
    for width in widths:
        spike = set(range(90, 90 + width))
        steps = list(range(1, 400))
        by_metric = {
            "train.loss": [(s, 20.0 if s in spike else 1.0) for s in steps],
            "train.lr": [(s, 0.05) for s in steps],
        }
        started = time.perf_counter()
        detection = DetectorRunner(DetectorConfig(cooldown_suppresses_evidence=False)).run(
            "v", by_metric, rank=0, layer_series={}, module_paths={},
            population_layers=1, population_ranks=1,
            coverage={"step_count": 400, "step_min": 1, "step_max": 400},
        )
        observations = [a for a in detection.anomalies if str(a.detector) == "loss.robust_zscore"]
        engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
        engine.consume(detection.anomalies, "v")
        engine.finish("v", final_step=400)
        episodes = engine.all_episodes()
        verdicts = fuse("v", episodes)
        rows.append({
            "spike_width": width,
            "observations": len(observations),
            "episodes": len(episodes),
            "verdicts": len(verdicts),
            "incidents": sum(1 for v in verdicts if v.verdict == "CONFIRMED_INCIDENT"),
            "ms": round((time.perf_counter() - started) * 1000, 1),
        })
    return rows


def fail_open_visibility() -> Dict[str, Any]:
    """D3: a forced overflow must reach the verdict, not be swallowed."""
    def make(step: int) -> TrainingEvent:
        return TrainingEvent(
            run_id="fo", producer="p", source="s", event_type=EventType.LOSS,
            step=step, metric_name="train.loss", metric_value=1.0,
            metric_unit=MetricUnit.DIMENSIONLESS, priority=EventPriority.P3,
        )

    queue = BoundedEventQueue(maxlen=16)
    accepted = refused = 0
    for step in range(1, 501):
        if queue.put(make(step)):
            accepted += 1
        else:
            refused += 1
    stats = queue.stats()
    health = {
        "observer": {
            "events_dropped_total": refused,
            "queue_overflow_total": getattr(stats, "evicted", refused),
            "writer_failures_total": 0,
            "written_total": accepted,
            "emitted_total": accepted + refused,
        },
        "queue": {"evicted": getattr(stats, "evicted", 0), "rejected": refused},
    }
    quality = DataQuality.from_observer_health(health)
    verdict = fuse("fo", [], data_quality=quality)[0]
    return {
        "offered": 500, "accepted": accepted, "refused": refused,
        "queue_capacity": 16,
        "quality_status": quality.status,
        "quality_dropped": quality.dropped_events,
        "verdict": verdict.verdict,
        "completeness_warning_present": bool(verdict.completeness_warning),
        "warning": verdict.completeness_warning,
        "masks_loss": quality.status == DataQualityStatus.COMPLETE.value and refused > 0,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix="td-scale-"))

    observer_rows = []
    for steps in (100, 500, 2000):
        print("[D2] observer at {} steps ...".format(steps))
        observer_rows.append(observe(steps, tmp))

    print("[D2] fault volume ...")
    volume = fault_volume()
    print("[D3] fail-open visibility ...")
    visibility = fail_open_visibility()

    payload = {
        "observer_scale": observer_rows,
        "fault_volume": volume,
        "fail_open_visibility": visibility,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    (OUTPUT_DIR / "scale_boundary_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("")
    print("observer scale:")
    for row in observer_rows:
        print("  steps=%-5d events=%-6d emitted=%-6s dropped=%-4s wall=%-6ss rss_delta=%-6sMB bytes/event=%s" % (
            row["steps"], row["events_persisted"], row["emitted"], row["dropped"],
            row["wall_s"], row["rss_delta_mb"], row["bytes_per_event"]))
    print("fault volume:")
    for row in volume:
        print("  spike=%-5d observations=%-5d episodes=%-3d incidents=%-3d ms=%s" % (
            row["spike_width"], row["observations"], row["episodes"], row["incidents"], row["ms"]))
    print("fail-open: %s" % json.dumps(visibility, default=str)[:260])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
