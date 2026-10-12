"""Runnable end-to-end demo: real PyTorch training -> incidents -> CLI.

Run::

    python examples/quickstart.py

It trains a tiny transformer, injects one controlled gradient explosion, and shows
the whole chain: observer -> events -> Parquet -> detector -> episode -> fusion ->
persisted incident -> rendered verdict.

Nothing here is mocked. The training is real autograd, the observer is the shipped
observer, and the incident comes from the shipped detector and fusion rules.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tiny_model import make_tiny_lm, synthetic_batch  # noqa: E402

from training_debugger.adapters.pytorch import (  # noqa: E402
    LayerMetricCollector, LayerSelection, PyTorchTrainingAdapter,
)
from training_debugger.detectors import DetectorConfig, DetectorRunner  # noqa: E402
from training_debugger.episodes import EpisodeEngine  # noqa: E402
from training_debugger.fusion import fuse  # noqa: E402
from training_debugger.fusion.quality import DataQuality  # noqa: E402
from training_debugger.incidents.render import render_verdicts  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver,
)
from training_debugger.persistence import (  # noqa: E402
    AnalysisStateStore, EpisodeStore, capture_analysis_state,
)
from training_debugger.registry import detector_registry  # noqa: E402
from training_debugger.storage import ParquetEventWriter  # noqa: E402

STEPS = 130
INJECT_AT = 80
FACTOR = 40.0


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="td-quickstart-"))
    store = workdir / "data"
    print("dataset root: {}".format(store))

    torch.manual_seed(20240)
    model, config = make_tiny_lm(seed=20240, n_layers=4, d_model=32, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    writer = ParquetEventWriter(store, "quickstart", rows_per_file=50_000)
    observer = TrainingObserver(
        "quickstart", writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=4096),
        rank=0, world_size=1, model_id="tiny-transformer-lm",
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="tiny-transformer-lm")
    collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=8), cadence=1)
    adapter.run_start(steps=STEPS)

    losses = []
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            tracker.record_loss(loss)
            loss.backward()
            if step == INJECT_AT:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(FACTOR)
            tracker.record_gradients()
            collector.collect(step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()
        losses.append(float(loss.detach().item()))

    adapter.run_end()
    observer.flush(timeout=30.0)
    health = observer.health()
    observer.close()
    print("training done: {} steps, final loss {:.4f}".format(STEPS, losses[-1]))
    observed = health.get("observer") or {}
    print("observer health: emitted={} written={} dropped={} overflow={}".format(
        observed.get("emitted_total"), observed.get("written_total"),
        observed.get("events_dropped_total"), observed.get("queue_overflow_total")))

    # -- analysis --------------------------------------------------------
    from training_debugger.storage import DuckDBReader

    reader = DuckDBReader(store)
    table = reader.events("quickstart")
    by_metric, layer_series = {}, {}
    for name, step, value, layer in zip(
        table.column("metric_name").to_pylist(), table.column("step").to_pylist(),
        table.column("metric_value").to_pylist(), table.column("layer_index").to_pylist(),
    ):
        if name is None or step is None or value is None:
            continue
        if name == "grad.layer_norm" and layer is not None:
            layer_series.setdefault(int(layer), {})[int(step)] = float(value)
            continue
        by_metric.setdefault(str(name), {})[int(step)] = float(value)
    reader.close()

    detection = DetectorRunner(DetectorConfig()).run(
        "quickstart",
        {k: sorted(v.items()) for k, v in by_metric.items()},
        rank=0, layer_series={k: sorted(v.items()) for k, v in layer_series.items()},
        module_paths={}, population_layers=4, population_ranks=1,
        coverage={"step_count": STEPS, "step_min": 1, "step_max": STEPS},
    )
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    engine.consume(detection.anomalies, "quickstart")
    engine.finish("quickstart", final_step=STEPS)
    episodes = engine.all_episodes()

    quality = DataQuality.from_observer_health(health)
    verdicts = fuse("quickstart", episodes, data_quality=quality)

    episode_store = EpisodeStore(store)
    for episode in episodes:
        episode_store.append(episode)
    AnalysisStateStore(store).append(capture_analysis_state(
        "quickstart", processed_through_step=STEPS,
        layer_series={k: sorted(v.items()) for k, v in layer_series.items()},
        config=DetectorConfig(), episodes=episodes,
        detector_versions={str(a.detector): str(a.detector_version) for a in detection.anomalies},
        fusion_rule_version="phase1.3-1.0", episode_rule_version="phase1.3-1.0",
        rule_version="v0.1.0-alpha",
    ))
    print("persisted {} episode(s), {} observation(s), quality={}".format(
        len(episodes), sum(e.signal_count for e in episodes), quality.status))

    print("")
    print("=" * 72)
    print(render_verdicts(verdicts, episodes))
    print("=" * 72)
    print("")
    print("Query it with:")
    print("  PYTHONPATH=src python -m training_debugger.cli.main --root {} runs".format(store))
    print("  PYTHONPATH=src python -m training_debugger.cli.main --root {} verdicts quickstart".format(store))
    print("")
    print("artifacts kept at {}".format(store))
    _ = shutil
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
