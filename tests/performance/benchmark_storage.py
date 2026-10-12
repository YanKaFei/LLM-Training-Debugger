"""W3 performance gate: 1,000,000 events through the storage layer.

The brief is explicit that testing 1,000 rows proves nothing, so this generates a
million and measures what actually matters:

* write throughput (events/second, MB/second)
* **file count** -- the small-file problem the brief calls out
* disk size, and the compression ratio against the in-memory representation
* query latency for each of the required query shapes
* peak resident memory

Events are generated and written in bounded chunks rather than materialising a
million Python objects, because peak RSS is one of the things being measured and
a generator that itself costs 2 GB would make the number meaningless.

Run::

    .venv/bin/python tests/performance/benchmark_storage.py
"""

from __future__ import annotations

import gc
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from training_debugger.env_info import capture_environment  # noqa: E402
from training_debugger.schema import (  # noqa: E402
    EventPriority,
    EventType,
    MetricUnit,
    Severity,
    TrainingEvent,
    make_producer_id,
    new_event_id,
)
from training_debugger.storage import DuckDBReader, ParquetEventWriter  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "phase1" / "w3"
REPORT = OUTPUT_DIR / "STORAGE_BENCHMARK.md"
RESULTS = OUTPUT_DIR / "storage_benchmark_results.json"

TOTAL_EVENTS = 1_000_000
RUN_ID = "bench-1m"
STEPS = 100_000
LAYERS = 8
RANKS = 1
CHUNK = 50_000


def peak_rss_mb() -> float:
    """Peak RSS in MB. ``ru_maxrss`` is bytes on macOS and KB on Linux."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024.0


def build_chunk(start_index: int, count: int, producer: str) -> List[TrainingEvent]:
    """Generate ``count`` events starting at ``start_index``.

    The mix mirrors a real run: mostly per-layer gradient metrics (the highest
    cardinality signal), plus core scalars, step boundaries and the occasional
    lifecycle event. Measuring only low-cardinality scalars would flatter the
    result.
    """
    events: List[TrainingEvent] = []
    for offset in range(count):
        index = start_index + offset
        seq = index + 1
        step = (index // 10) % STEPS + 1
        bucket = index % 10
        if bucket == 0:
            events.append(
                TrainingEvent(
                    event_id=new_event_id(), run_id=RUN_ID, producer=producer,
                    source="pytorch-adapter", event_type=EventType.LOSS, step=step,
                    rank=0, local_rank=0, world_size=RANKS, source_seq=seq,
                    metric_name="train.loss", metric_value=2.0 + (index % 100) / 1000.0,
                    metric_unit=MetricUnit.DIMENSIONLESS, priority=EventPriority.P1,
                )
            )
        elif bucket == 1:
            events.append(
                TrainingEvent(
                    event_id=new_event_id(), run_id=RUN_ID, producer=producer,
                    source="pytorch-adapter", event_type=EventType.LR_CHANGE, step=step,
                    rank=0, local_rank=0, world_size=RANKS, source_seq=seq,
                    metric_name="train.lr", metric_value=1e-3,
                    metric_unit=MetricUnit.DIMENSIONLESS, priority=EventPriority.P1,
                )
            )
        elif bucket == 2:
            events.append(
                TrainingEvent(
                    event_id=new_event_id(), run_id=RUN_ID, producer=producer,
                    source="pytorch-adapter", event_type=EventType.STEP_END, step=step,
                    rank=0, local_rank=0, world_size=RANKS, source_seq=seq,
                    metric_name="train.step_time", metric_value=120.0 + (index % 40),
                    metric_unit=MetricUnit.MILLISECONDS, priority=EventPriority.P1,
                    metadata={"step_time_ms": 120.0},
                )
            )
        elif bucket == 3:
            events.append(
                TrainingEvent(
                    event_id=new_event_id(), run_id=RUN_ID, producer=producer,
                    source="pytorch-adapter", event_type=EventType.GRADIENT_METRIC, step=step,
                    rank=0, local_rank=0, world_size=RANKS, source_seq=seq,
                    metric_name="grad.global_norm", metric_value=1.0 + (index % 50) / 100.0,
                    metric_unit=MetricUnit.DIMENSIONLESS, priority=EventPriority.P1,
                    metadata={"norm_type": "L2", "tensor_count": 32, "element_count": 1_000_000},
                )
            )
        else:
            layer = index % LAYERS
            events.append(
                TrainingEvent(
                    event_id=new_event_id(), run_id=RUN_ID, producer=producer,
                    source="pytorch-adapter", event_type=EventType.GRADIENT_METRIC, step=step,
                    rank=0, local_rank=0, world_size=RANKS, source_seq=seq,
                    layer_index=layer, module_path="layers.{}.mlp".format(layer),
                    metric_name="grad.layer_norm", metric_value=0.5 + (index % 70) / 100.0,
                    metric_unit=MetricUnit.DIMENSIONLESS, priority=EventPriority.P2,
                    severity=Severity.INFO,
                )
            )
    return events


def timed(fn, repeats: int = 3) -> Tuple[float, Any]:
    """Run ``fn`` ``repeats`` times; return (median seconds, last result)."""
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), result


def write_phase(root: Path) -> Dict[str, Any]:
    writer = ParquetEventWriter(root, RUN_ID, rows_per_file=100_000, compression="zstd")
    producer = make_producer_id("pytorch-adapter", 0, "bench")
    written = 0
    started = time.perf_counter()
    per_chunk = []
    while written < TOTAL_EVENTS:
        count = min(CHUNK, TOTAL_EVENTS - written)
        chunk_started = time.perf_counter()
        events = build_chunk(written, count, producer)
        producer_seconds = time.perf_counter() - chunk_started
        writer.write_batch(events)
        per_chunk.append((count, time.perf_counter() - chunk_started, producer_seconds))
        written += count
        del events
        gc.collect()
    writer.flush()
    writer.close()
    elapsed = time.perf_counter() - started
    files = writer.part_files()
    disk_bytes = sum(p.stat().st_size for p in files)
    return {
        "events": writer.stats.rows_written,
        "seconds": elapsed,
        "events_per_second": writer.stats.rows_written / elapsed if elapsed else 0.0,
        "files": len(files),
        "disk_bytes": disk_bytes,
        "disk_mb": disk_bytes / (1024 * 1024),
        "bytes_per_event": disk_bytes / max(1, writer.stats.rows_written),
        "peak_rss_mb": peak_rss_mb(),
        "batches": writer.stats.batches,
        "chunk_generation_share": (
            sum(c[2] for c in per_chunk) / elapsed if elapsed else 0.0
        ),
        "rows_per_file_config": writer.rows_per_file,
        "compression": writer.compression,
    }


def query_phase(reader: DuckDBReader) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    coverage = reader.coverage(RUN_ID)
    results["coverage"] = coverage.to_dict()

    median, table = timed(lambda: reader.series(RUN_ID, "train.loss"))
    results["query_series_full_run"] = {"median_s": median, "rows": len(table)}

    median, table = timed(
        lambda: reader.metrics(RUN_ID, ["train.loss"], step_range=(50_000, 50_000))
    )
    results["query_metrics_step_range"] = {"median_s": median, "rows": table.num_rows}

    median, table = timed(
        lambda: reader.incident_window(RUN_ID, step=50_000, back=50, forward=0)
    )
    results["query_incident_window"] = {"median_s": median, "rows": table.num_rows}

    median, table = timed(lambda: reader.events(RUN_ID, rank=0, limit=10_000))
    results["query_events_by_rank"] = {"median_s": median, "rows": table.num_rows}

    median, table = timed(
        lambda: reader.events(RUN_ID, layer_index=3, metric_name="grad.layer_norm")
    )
    results["query_events_by_layer"] = {"median_s": median, "rows": table.num_rows}

    median, table = timed(lambda: reader.event_type_counts(RUN_ID))
    results["query_event_type_counts"] = {"median_s": median, "rows": table.num_rows}

    median, table = timed(lambda: reader.series_by_layer(RUN_ID, "grad.layer_norm"))
    results["query_series_by_layer"] = {
        "median_s": median,
        "layers": len(table),
        "rows": sum(len(v) for v in table.values()),
    }

    median, table = timed(lambda: reader.metrics(RUN_ID, ["grad.layer_norm"], limit=1000))
    results["query_metrics_limited"] = {"median_s": median, "rows": table.num_rows}
    return results


def observer_phase(root: Path, events: int = 200_000) -> Dict[str, Any]:
    """End-to-end emit throughput: observer -> queue -> writer thread -> Parquet.

    A smaller N than the storage benchmark, because this measures the *hot path*
    including validation, sampling and the writer thread, and 1M events would
    take long enough to make the number noisy.
    """
    sys.path.insert(0, str(ROOT / "tests"))
    from training_debugger.observer import (  # noqa: PLC0415
        ObserverConfig,
        SamplingPolicy,
        TrainingObserver,
    )

    config = ObserverConfig(max_queue=50_000, batch_size=2048, flush_interval_s=0.05)
    writer = ParquetEventWriter(root, "bench-observer", rows_per_file=100_000)
    observer = TrainingObserver(
        "bench-observer", writer=writer, sampling=SamplingPolicy.uniform(1), config=config
    )
    producer = make_producer_id("pytorch-adapter", 0, "bench")
    started = time.perf_counter()
    emitted = 0
    while emitted < events:
        count = min(CHUNK, events - emitted)
        for event in build_chunk(emitted, count, producer):
            observer.emit(event.copy_with(run_id="bench-observer"))
        emitted += count
    emit_seconds = time.perf_counter() - started
    observer.flush(timeout=60.0)
    total_seconds = time.perf_counter() - started
    stats = observer.stats()
    health = observer.health()
    observer.close()

    return {
        "events": events,
        "emit_seconds": emit_seconds,
        "emit_events_per_second": events / emit_seconds if emit_seconds else 0.0,
        "total_seconds": total_seconds,
        "written": stats.written_total,
        "dropped": stats.events_dropped_total,
        "queue_overflow": stats.queue_overflow_total,
        "writer_failures": stats.writer_failures_total,
        "internal_errors": stats.observer_internal_errors_total,
        "validation_failures": stats.validation_failures_total,
        "max_queue_depth": health["queue"]["max_depth"],
        "emit_time_total_ns": stats.emit_time_total_ns,
        "us_per_event": (stats.emit_time_total_ns / max(1, stats.emitted_total)) / 1000.0,
        "peak_rss_mb": peak_rss_mb(),
    }


def render(write: Dict[str, Any], queries: Dict[str, Any], observer: Dict[str, Any], env: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append
    add("# W3 STORAGE BENCHMARK")
    add("")
    add("**Generated by:** `tests/performance/benchmark_storage.py`")
    add("**Events:** {:,}".format(TOTAL_EVENTS))
    add("")
    add("## Write throughput")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add("| Events written | {:,} |".format(write["events"]))
    add("| Wall time | {:.2f} s |".format(write["seconds"]))
    add("| **Throughput** | **{:,.0f} events/s** |".format(write["events_per_second"]))
    add("| Batches | {:,} |".format(write["batches"]))
    add("| Rows per file (configured) | {:,} |".format(write["rows_per_file_config"]))
    add("| Compression | {} |".format(write["compression"]))
    add("| Event construction share of wall time | {:.1%} |".format(write["chunk_generation_share"]))
    add("")
    add("## File count and disk size")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add("| **Files written** | **{:,}** |".format(write["files"]))
    add("| Rows per file (actual) | {:,.0f} |".format(write["events"] / max(1, write["files"])))
    add("| Disk size | {:.1f} MB |".format(write["disk_mb"]))
    add("| Bytes per event (compressed) | {:.1f} |".format(write["bytes_per_event"]))
    add("| Peak RSS during write | {:.0f} MB |".format(write["peak_rss_mb"]))
    add("")
    add("**Small-file check:** {:,} events produced {:,} files at a target of {:,} rows/file.".format(
        write["events"], write["files"], write["rows_per_file_config"]))
    add("A file-per-event layout would have produced {:,} files; batching reduces that by a factor of about {:,.0f}.".format(
        write["events"], write["events"] / max(1, write["files"])))
    add("")
    add("## Query latency (median of 3)")
    add("")
    add("| Query | Median | Rows returned |")
    add("|---|---|---|")
    labels = {
        "query_series_full_run": "full-run series for one metric",
        "query_metrics_step_range": "metrics for one step",
        "query_incident_window": "incident window (+/- 50 steps, all event types)",
        "query_events_by_rank": "events by rank (capped at 10k)",
        "query_events_by_layer": "events by layer, one metric",
        "query_event_type_counts": "event-type counts (full scan, aggregate)",
        "query_series_by_layer": "per-layer series for all layers",
        "query_metrics_limited": "metrics with LIMIT 1000",
    }
    for key, label in labels.items():
        if key not in queries:
            continue
        entry = queries[key]
        add("| {} | {:.1f} ms | {:,} |".format(label, entry["median_s"] * 1000, entry["rows"]))
    add("")
    coverage = queries["coverage"]
    add("Coverage: {:,} rows across {:,} files, steps {}..{}, {} event types, {} skipped files.".format(
        coverage["row_count"], coverage["files_total"], coverage["step_min"],
        coverage["step_max"], len(coverage["event_types"]), coverage["files_skipped"]))
    add("")
    add("## End-to-end observer throughput")
    add("")
    add("This includes schema validation, sampling, the bounded queue and the writer thread -- the real hot path.")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add("| Events emitted | {:,} |".format(observer["events"]))
    add("| Emit-path throughput | {:,.0f} events/s |".format(observer["emit_events_per_second"]))
    add("| Emit-path cost per event | {:.2f} us |".format(observer["us_per_event"]))
    add("| Events written | {:,} |".format(observer["written"]))
    add("| **Events dropped** | **{:,}** |".format(observer["dropped"]))
    add("| Queue overflows | {:,} |".format(observer["queue_overflow"]))
    add("| Writer failures | {:,} |".format(observer["writer_failures"]))
    add("| Internal errors | {:,} |".format(observer["internal_errors"]))
    add("| Peak queue depth | {:,} |".format(observer["max_queue_depth"]))
    add("| Peak RSS | {:.0f} MB |".format(observer["peak_rss_mb"]))
    add("")
    add("## Environment")
    add("")
    add("```json")
    add(json.dumps(env, indent=2, sort_keys=True, default=str))
    add("```")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    import tempfile

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="training-debugger-bench-"))
    print("working directory:", workdir)

    print("write phase: {:,} events...".format(TOTAL_EVENTS))
    write = write_phase(workdir)
    print("  {:,} events/s, {:,} files, {:.1f} MB".format(
        write["events_per_second"], write["files"], write["disk_mb"]))

    print("query phase...")
    reader = DuckDBReader(workdir)
    queries = query_phase(reader)
    reader.close()
    for key, value in queries.items():
        if key != "coverage":
            print("  {:32s} {:.1f} ms".format(key, value["median_s"] * 1000))

    print("observer phase...")
    observer = observer_phase(workdir)
    print("  emit {:,.0f} events/s, dropped {:,}".format(
        observer["emit_events_per_second"], observer["dropped"]))

    env = capture_environment(extra={"benchmark": "storage", "events": TOTAL_EVENTS})
    RESULTS.write_text(json.dumps(
        {"write": write, "queries": queries, "observer": observer, "environment": env},
        indent=2, default=str), encoding="utf-8")
    REPORT.write_text(render(write, queries, observer, env), encoding="utf-8")
    print("report:", REPORT)

    # Leave the data on disk only if it is small enough to be useful; a 1M-event
    # dataset is not committed, so clean up.
    import shutil

    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
