"""W3 gate: Parquet storage and the DuckDB query layer."""

from __future__ import annotations

import json
import os

import pyarrow.parquet as pq
import pytest

import helpers
from training_debugger.observer import MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver
from training_debugger.schema import EventType
from training_debugger.storage import (
    ARROW_SCHEMA,
    DuckDBReader,
    ParquetEventWriter,
    list_part_files,
    run_dir,
)


def _write_run(root, run_id="run-w3", steps=30, layer_count=3, rows_per_file=100):
    writer = ParquetEventWriter(root, run_id, rows_per_file=rows_per_file)
    observer = TrainingObserver(
        run_id, writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01),
        rank=0, local_rank=0, world_size=1,
    )
    helpers.synthetic_training(observer, steps=steps, layer_count=layer_count)
    observer.flush(timeout=10.0)
    observer.close()
    return writer


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_writer_round_trips_every_event(tmp_path):
    writer = _write_run(tmp_path, steps=20, layer_count=2)
    reader = DuckDBReader(tmp_path)
    coverage = reader.coverage("run-w3")
    assert coverage.row_count == writer.stats.rows_written
    assert coverage.files_skipped == 0
    reader.close()


def test_schema_is_explicit_and_stable(tmp_path):
    _write_run(tmp_path, steps=2, layer_count=1)
    files = list_part_files(tmp_path, "run-w3")
    assert files
    metadata = pq.read_metadata(str(files[0]))
    written = metadata.schema.to_arrow_schema()
    # The explicit schema must survive the round trip, or a later reader can be
    # surprised by an inferred type.
    for field in ARROW_SCHEMA:
        assert field.name in written.names, field.name


def test_row_count_is_split_across_files_not_exploded(tmp_path):
    """The small-file problem: rows_per_file must actually bound file size."""
    writer = _write_run(tmp_path, steps=200, layer_count=3, rows_per_file=500)
    files = writer.part_files()
    assert len(files) >= 2, "a large run must be split into several files"
    assert writer.stats.rows_written > 1000
    # No file should be wildly larger than the configured threshold.
    for path in files:
        assert pq.read_metadata(str(path)).num_rows <= 1000


def test_manifest_is_written_and_readable(tmp_path):
    _write_run(tmp_path, steps=5, layer_count=1)
    manifest = json.loads((run_dir(tmp_path, "run-w3") / "manifest.json").read_text())
    assert manifest["run_id"] == "run-w3"
    assert manifest["schema_version"] == "0.1"
    assert manifest["rows_written"] > 0
    assert manifest["closed"] is True


def test_ingest_seq_is_assigned_by_the_writer_and_is_dense(tmp_path):
    writer = _write_run(tmp_path, steps=10, layer_count=1)
    reader = DuckDBReader(tmp_path)
    table = reader.events("run-w3")
    sequences = [s for s in table.column("ingest_seq").to_pylist() if s is not None]
    # The query returns rows in *step* order, so ingest_seq values are permuted
    # within the result. The invariant is that they are dense and unique -- one
    # storage slot per event, no gaps -- not that they arrive sorted.
    assert sorted(sequences) == list(range(len(sequences)))
    reader.close()


# ---------------------------------------------------------------------------
# Crash safety
# ---------------------------------------------------------------------------


def test_a_corrupt_part_is_skipped_and_the_rest_stays_readable(tmp_path):
    """The most likely moment to read a run is right after it crashed."""
    _write_run(tmp_path, steps=40, layer_count=2, rows_per_file=100)
    files = list_part_files(tmp_path, "run-w3")
    assert len(files) >= 2
    # Truncate one file in place, simulating a partially flushed write.
    victim = files[-1]
    original_size = victim.stat().st_size
    with open(victim, "r+b") as handle:
        handle.truncate(original_size // 2)

    reader = DuckDBReader(tmp_path)
    coverage = reader.coverage("run-w3")
    assert coverage.files_skipped == 1, "the truncated file must be detected"
    assert coverage.skipped_paths
    assert coverage.row_count > 0, "the remaining files must still be readable"
    assert len(reader.series("run-w3", "train.loss")) > 0
    reader.close()


def test_a_temp_file_left_by_a_crash_is_invisible(tmp_path):
    """Write-then-rename is the crash-safety story: readers glob *.parquet only."""
    _write_run(tmp_path, steps=5, layer_count=1)
    run_events = run_dir(tmp_path, "run-w3") / "events"
    stray = run_events / ".part-9999-deadbeef.parquet.tmp"
    stray.write_bytes(b"this is not parquet")
    reader = DuckDBReader(tmp_path)
    coverage = reader.coverage("run-w3")
    assert coverage.files_skipped == 0
    assert coverage.row_count > 0
    reader.close()
    assert stray.exists(), "temp files are ignored, not deleted"


def test_reader_refresh_picks_up_newly_written_data(tmp_path):
    writer = ParquetEventWriter(tmp_path, "run-w3", rows_per_file=100)
    observer = TrainingObserver("run-w3", writer=writer, sampling=SamplingPolicy.uniform(1),
                                config=ObserverConfig(flush_interval_s=0.01))
    helpers.synthetic_training(observer, steps=5, layer_count=1)
    observer.flush(timeout=5.0)
    reader = DuckDBReader(tmp_path)
    before = reader.coverage("run-w3").row_count
    helpers.synthetic_training(observer, steps=5, layer_count=1)
    observer.flush(timeout=5.0)
    observer.close()
    reader.refresh("run-w3")
    after = reader.coverage("run-w3").row_count
    assert after > before
    reader.close()


# ---------------------------------------------------------------------------
# The queries the brief requires
# ---------------------------------------------------------------------------


def test_metrics_for_run_and_around_step(tmp_path):
    _write_run(tmp_path, steps=50, layer_count=2)
    reader = DuckDBReader(tmp_path)
    all_losses = reader.metrics("run-w3", ["train.loss"])
    assert all_losses.num_rows == 50

    around = reader.metrics_around("run-w3", step=25, window=4, metric_names=["train.loss"])
    steps = sorted(around.column("step").to_pylist())
    assert steps == list(range(21, 30))
    reader.close()


def test_events_by_rank_and_by_layer(tmp_path):
    _write_run(tmp_path, steps=20, layer_count=3)
    reader = DuckDBReader(tmp_path)
    by_rank = reader.events_by_rank("run-w3", 0)
    assert by_rank.num_rows > 0
    by_layer = reader.events_by_layer("run-w3", 2, metric_name="grad.layer_norm")
    assert by_layer.num_rows > 0
    assert set(by_layer.column("layer_index").to_pylist()) == {2}
    reader.close()


def test_event_type_counts(tmp_path):
    _write_run(tmp_path, steps=10, layer_count=1)
    reader = DuckDBReader(tmp_path)
    counts = {row["event_type"]: row["n"] for row in reader.event_type_counts("run-w3").to_pylist()}
    assert counts["LOSS"] == 10
    assert counts["GRADIENT_METRIC"] >= 10
    reader.close()


def test_incident_window_returns_all_event_types(tmp_path):
    """The causal story usually lives in lifecycle events, not in numbers."""
    _write_run(tmp_path, steps=30, layer_count=2)
    reader = DuckDBReader(tmp_path)
    window = reader.incident_window("run-w3", step=15, back=3, forward=1)
    steps = [s for s in window.column("step").to_pylist() if s is not None]
    assert min(steps) >= 12 and max(steps) <= 16
    types = set(window.column("event_type").to_pylist())
    assert {"LOSS", "GRADIENT_METRIC", "STEP_END"} <= types
    reader.close()


def test_series_extraction_shape(tmp_path):
    _write_run(tmp_path, steps=25, layer_count=2)
    reader = DuckDBReader(tmp_path)
    series = reader.series("run-w3", "train.loss")
    assert len(series) == 25
    assert all(isinstance(step, int) and isinstance(value, float) for step, value in series)
    assert [step for step, _ in series] == sorted(step for step, _ in series)

    by_layer = reader.series_by_layer("run-w3", "grad.layer_norm")
    assert sorted(by_layer) == [0, 1]
    reader.close()


def test_nan_survives_the_parquet_round_trip(tmp_path):
    """NaN is a product signal; storage must not silently null it."""
    writer = ParquetEventWriter(tmp_path, "run-nan")
    observer = TrainingObserver("run-nan", writer=writer, sampling=SamplingPolicy.uniform(1),
                                config=ObserverConfig(flush_interval_s=0.01))
    observer.metric("train.loss", float("nan"), step=1, event_type=EventType.LOSS)
    observer.nonfinite(1, metric_name="train.loss_nonfinite_count", count=1)
    observer.flush(timeout=5.0)
    observer.close()
    reader = DuckDBReader(tmp_path)
    values = reader.metrics("run-nan", ["train.loss"]).column("metric_value").to_pylist()
    assert len(values) == 1
    assert values[0] != values[0], "NaN must round-trip as NaN, not as null"
    reader.close()


def test_querying_a_nonexistent_run_returns_empty_not_an_error(tmp_path):
    reader = DuckDBReader(tmp_path)
    coverage = reader.coverage("does-not-exist")
    assert coverage.row_count == 0
    assert coverage.files_total == 0
    assert reader.metrics("does-not-exist", ["train.loss"]).num_rows == 0
    assert reader.events("does-not-exist").num_rows == 0
    assert reader.list_runs() == []
    reader.close()


def test_coverage_distinguishes_absent_from_unread(tmp_path):
    _write_run(tmp_path, steps=10, layer_count=1)
    reader = DuckDBReader(tmp_path)
    coverage = reader.coverage("run-w3")
    assert coverage.covers_step(5)
    assert not coverage.covers_step(999)
    assert coverage.to_dict()["files_readable"] >= 1
    reader.close()


def test_several_runs_coexist_in_one_dataset(tmp_path):
    _write_run(tmp_path, run_id="run-a", steps=5, layer_count=1)
    _write_run(tmp_path, run_id="run-b", steps=7, layer_count=1)
    reader = DuckDBReader(tmp_path)
    assert reader.list_runs() == ["run-a", "run-b"]
    assert reader.coverage("run-a").row_count != reader.coverage("run-b").row_count
    reader.close()


def test_metadata_is_stored_as_queryable_json(tmp_path):
    writer = ParquetEventWriter(tmp_path, "run-meta")
    observer = TrainingObserver("run-meta", writer=writer, sampling=SamplingPolicy.uniform(1),
                                config=ObserverConfig(flush_interval_s=0.01))
    observer.metric(
        "grad.global_norm", 1.5, step=1, event_type=EventType.GRADIENT_METRIC,
        metadata={"norm_type": "L2", "tensor_count": 12},
    )
    observer.flush(timeout=5.0)
    observer.close()
    reader = DuckDBReader(tmp_path)
    table = reader.events("run-meta", metric_name="grad.global_norm")
    payloads = table.column("metadata").to_pylist()
    assert payloads
    stored = json.loads(payloads[0])
    assert stored["norm_type"] == "L2"
    assert stored["tensor_count"] == 12
    reader.close()
