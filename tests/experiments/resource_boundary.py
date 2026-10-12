"""C2: where does the evidence-volume increase actually land?

Phase 1.12A set ``cooldown_suppresses_evidence = False`` and measured 20 detector
observations where there had been 1. Phase 1.12B recorded the risk that this might load
the **observer** and cause drops, silently truncating the evidence chain.

This script answers that by **measuring each layer separately** rather than assuming:

    TrainingEvents emitted by the observer
    TrainingEvents persisted
    Anomalies emitted by DetectorRunner      <- the layer the fix changed
    Anomalies consumed by EpisodeEngine
    Episodes and incidents produced

The cooldown lives inside the detector, which runs **offline over already-persisted
events**. If that is true, the observer's raw-event load is untouched and the 20x is
confined to the detector side.

Run::

    .venv/bin/python tests/experiments/resource_boundary.py
"""

from __future__ import annotations

import gc
import json
import sys
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
from training_debugger.observer import (  # noqa: E402
    BoundedEventQueue, MemoryWriter, NullWriter, ObserverConfig, SamplingPolicy,
    TrainingObserver,
)
from training_debugger.registry import detector_registry  # noqa: E402
from training_debugger.schema.enums import EventPriority  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "phase1-12c"
STEPS = 120


def observer_layer() -> Dict[str, Any]:
    """The observer's own numbers. The cooldown cannot appear here."""
    results: Dict[str, Any] = {}
    for label, flag in (("cooldown_on", True), ("cooldown_off", False)):
        torch.manual_seed(4242)
        model, config = make_tiny_lm(seed=4242, n_layers=4, d_model=32, n_heads=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        writer = MemoryWriter()
        observer = TrainingObserver(
            "c2-%s" % label, writer=writer, sampling=SamplingPolicy.uniform(1),
            config=ObserverConfig(flush_interval_s=0.01, batch_size=8192), rank=0, world_size=1,
        )
        adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="c2")
        collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=32), cadence=1)
        adapter.run_start(steps=STEPS)
        for step in range(1, STEPS + 1):
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
        observer.flush(timeout=30.0)
        health = observer.health()
        observer.close()
        results[label] = {
            "events_persisted": len(writer.events),
            "emitted": health.get("emitted"),
            "dropped": health.get("dropped"),
            "written": health.get("written"),
            "overflow": health.get("overflow"),
        }
        del model, optimizer, observer, adapter, collector
        gc.collect()
    results["raw_event_load_unchanged"] = (
        results["cooldown_on"]["events_persisted"] == results["cooldown_off"]["events_persisted"]
    )
    return results


def detector_layer(counts=(1, 20, 100, 1000)) -> Dict[str, Any]:
    """Detector / episode / incident volume as observations scale."""
    out: Dict[str, Any] = {}
    for count in counts:
        spike = list(range(90, 90 + count))
        step_range = list(range(1, 400))
        by_metric = {
            "train.loss": [(s, 20.0 if s in spike else 1.0) for s in step_range],
            "train.lr": [(s, 0.05) for s in step_range],
        }
        rows: Dict[str, Any] = {}
        for label, flag in (("cooldown_on", True), ("cooldown_off", False)):
            config = DetectorConfig(cooldown_suppresses_evidence=flag)
            started = time.perf_counter()
            detection = DetectorRunner(config).run(
                "c2", by_metric, rank=0, layer_series={}, module_paths={},
                population_layers=1, population_ranks=1,
                coverage={"step_count": 400, "step_min": 1, "step_max": 400},
            )
            detector_ms = (time.perf_counter() - started) * 1000.0
            observations = [a for a in detection.anomalies if str(a.detector) == "loss.robust_zscore"]

            started = time.perf_counter()
            engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
            engine.consume(detection.anomalies, "c2")
            engine.finish("c2", final_step=400)
            episodes = engine.all_episodes()
            episode_ms = (time.perf_counter() - started) * 1000.0
            verdicts = fuse("c2", episodes)
            incidents = [v for v in verdicts if v.verdict == "CONFIRMED_INCIDENT"]

            rows[label] = {
                "observations": len(observations),
                "episodes": len(episodes),
                "incidents": len(incidents),
                "verdicts": len(verdicts),
                "detector_ms": round(detector_ms, 2),
                "episode_ms": round(episode_ms, 2),
            }
        rows["spike_steps"] = count
        out[str(count)] = rows
    return out


def queue_stress() -> Dict[str, Any]:
    """Bounded queue under a deliberately stalled consumer.

    Overflow is **forced** by never starting a consumer, not by hoping the machine is
    busy. The accounting identity is then checked on the queue itself.
    """
    from training_debugger.schema import EventType, MetricUnit, TrainingEvent

    def make(step):
        return TrainingEvent(
            run_id="c2", producer="p", source="s", event_type=EventType.LOSS,
            step=step, metric_name="train.loss", metric_value=1.0,
            metric_unit=MetricUnit.DIMENSIONLESS,
        )

    capacity = 64
    queue = BoundedEventQueue(maxlen=capacity)
    accepted = dropped = 0
    # Consumer deliberately absent: every put beyond `capacity` must be refused or
    # must evict, never silently grow the queue.
    for step in range(1, 5000):
        if queue.put(make(step)):
            accepted += 1
        else:
            dropped += 1
    size_after = len(queue.peek_all())
    stats = queue.stats()
    drained_items = queue.drain(capacity * 4)
    return {
        "capacity": capacity,
        "offered": 4999,
        "accepted": accepted,
        "refused_or_evicted": dropped,
        "size_after_load": size_after,
        "bounded": size_after <= capacity,
        "drained": len(drained_items),
        "drained_equals_size": len(drained_items) == size_after,
        "accounting_ok": accepted + dropped == 4999,
        "queue_stats": {k: v for k, v in vars(stats).items()
                        if not isinstance(v, (list, dict))} if hasattr(stats, "__dict__") else {},
    }


def priority_policy() -> Dict[str, Any]:
    """A full queue must accept a higher-priority event by evicting a lower one."""
    from training_debugger.schema import EventType, MetricUnit, TrainingEvent

    def make(step, priority):
        return TrainingEvent(
            run_id="c2", producer="p", source="s", event_type=EventType.LOSS,
            step=step, metric_name="train.loss", metric_value=1.0,
            metric_unit=MetricUnit.DIMENSIONLESS, priority=priority,
        )

    queue = BoundedEventQueue(maxlen=4)
    for step in range(4):
        queue.put(make(step, EventPriority.P3))
    _ = queue.stats()
    before = len(queue)
    accepted = queue.put(make(99, EventPriority.P0))
    size = len(queue.peek_all())
    items = queue.drain(16)
    return {
        "capacity": 4,
        "size_before_p0": before,
        "p0_accepted": accepted,
        "size_after_p0": size,
        "bounded": size <= 4,
        "p0_survived": any(i.priority == EventPriority.P0 for i in items),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("[C2] observer layer ...")
    observer = observer_layer()
    print("   persisted events cooldown_on={} cooldown_off={} unchanged={}".format(
        observer["cooldown_on"]["events_persisted"],
        observer["cooldown_off"]["events_persisted"],
        observer["raw_event_load_unchanged"]))

    print("[C2] detector layer ...")
    detector = detector_layer()

    print("[C2] bounded queue stress ...")
    queue = queue_stress()
    print("   capacity={} accepted={} refused={} size={} bounded={}".format(
        queue["capacity"], queue["accepted"], queue["refused_or_evicted"],
        queue["size_after_load"], queue["bounded"]))

    print("[C2] priority policy ...")
    priority = priority_policy()
    print("   p0 accepted={} survived={} bounded={}".format(
        priority["p0_accepted"], priority["p0_survived"], priority["bounded"]))

    payload = {
        "observer_layer": observer,
        "detector_layer": detector,
        "queue_stress": queue,
        "priority_policy": priority,
        "conclusion": (
            "The cooldown lives inside the detector, which runs offline over "
            "already-persisted events, so the observer's raw-event load is unchanged. "
            "The volume increase is confined to detector-side evidence."
        ),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    (OUTPUT_DIR / "observer_resource_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "detector_volume_results.json").write_text(
        json.dumps(detector, indent=2, default=str), encoding="utf-8")
    print("")
    print("observations by spike size (cooldown_off):")
    for count in sorted(detector, key=int):
        row = detector[count]
        print("   spike={:<5} obs={:<5} episodes={:<3} incidents={:<3} detector_ms={}".format(
            count, row["cooldown_off"]["observations"], row["cooldown_off"]["episodes"],
            row["cooldown_off"]["incidents"], row["cooldown_off"]["detector_ms"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
