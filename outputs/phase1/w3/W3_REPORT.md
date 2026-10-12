# W3 REPORT — Parquet + DuckDB

**Wave:** W3
**Status:** **COMPLETE**
**Gate:** 1,000,000 events written and queried; crash-safe — **PASS**

---

## 1. What was built

```
src/training_debugger/storage/
  layout.py          explicit Arrow schema, partitioning, atomic file names
  parquet_writer.py  ParquetEventWriter (implements the observer's EventWriter)
  duckdb_reader.py   DuckDBReader + ReaderCoverage
tests/integration/test_storage.py       17 tests
tests/performance/benchmark_storage.py  the 1M-event benchmark
outputs/phase1/w3/STORAGE_BENCHMARK.md  generated results
```

**Parquet is the source of truth; DuckDB is only a query engine.** Trackio does
not appear here at all — Phase 0 established that its metric table stores metric
names as unindexed JSON keys and that every chart read is an unbounded
`fetchall()`, which disqualifies it as a telemetry store.

---

## 2. Measured results (1,000,000 events)

Environment: macOS 27.0.1, arm64 (Apple M2 Max), Python 3.9.6, torch 2.8.0,
numpy 2.0.2, pyarrow 21.0.0, duckdb 1.4.5. Full snapshot in the generated report.

### Write

| Metric | Value |
|---|---|
| Events written | **1,000,000** (verified against the Parquet row count) |
| Wall time | 27.5 s |
| **Throughput** | **36,314 events/s** |
| **Files written** | **10** |
| Rows per file (target / actual) | 100,000 / 100,000 |
| **Disk size** | **9.6 MB** |
| Bytes per event (compressed, zstd) | **10.1** |
| Peak RSS | 449 MB |
| Event-construction share of wall time | measured and reported separately |

**The small-file problem the brief asks about is answered concretely:** at a
100,000-row target, one million events produced **ten files**. A file-per-event
layout would have produced one million — a 100,000× reduction. The bound is real,
not nominal: `rows_per_file` is enforced by chunking inside the flush, which was
a bug fix (see §4).

The 10.1 bytes/event figure is credible rather than surprising: the data is
highly repetitive (5 distinct `metric_name`s, 8 layers, one rank), all
low-cardinality columns are dictionary-encoded, and zstd then compresses the
dictionary indices. This is the expected behaviour of columnar storage on
training telemetry and is the main reason Parquet was chosen over a row store.

### Query latency (median of 3)

| Query | Median | Rows |
|---|---|---|
| event-type counts (full scan, aggregate) | **2.2 ms** | 6 |
| metrics for one step | **13.4 ms** | 10 |
| incident window (±50 steps, all event types) | **12.8 ms** | 510 |
| metrics with `LIMIT 1000` | 26.5 ms | 1,000 |
| events by layer, one metric | 44.5 ms | 125,000 |
| events by rank (capped 10k) | 58.3 ms | 10,000 |
| full-run series for one metric | 95.5 ms | 100,000 |
| per-layer series for all layers | **692 ms** | 800,000 |

The last row is the honest outlier. `series_by_layer` materialises ~800k rows
through Python to build the `{layer: [(step, value)]}` mapping. It is a
**known inefficiency**, not a correctness problem: the grouping should happen in
DuckDB and return Arrow arrays grouped per layer. Recorded here rather than
hidden, and it is the first optimisation if the correlation engine needs many
per-layer reads.

### End-to-end observer throughput

Including schema validation, sampling, the bounded queue and the writer thread —
the real hot path:

| Metric | Value |
|---|---|
| Events emitted | 200,000 |
| Emit-path throughput | **25,367 events/s** |
| **Emit-path cost per event** | **14.8 µs** |
| Events dropped | **0** |
| Queue overflows | 0 |
| Writer failures | 0 |
| Internal errors | 0 |

**Context for that number:** the Phase 0 design target is ~52 records per step
per rank at 2 s/step, i.e. ~26 events/s per rank. At 200,000 events in ~8 s, the
hot path has roughly three orders of magnitude of headroom against the target
workload. The real constraint will be *per-step latency impact*, which is W4's
gate, not throughput.

---

## 3. Crash safety

The guarantee is deliberately narrow, matching the brief: **completed files
readable**, no transactional claim.

Mechanism: every batch is written to a `.parquet.tmp` file and then `os.replace`d
into place. Readers glob `*.parquet` only, so a process killed mid-write leaves a
temp file that is ignored rather than a truncated Parquet that makes the run
unreadable.

Verified by test:

| Scenario | Behaviour | Test |
|---|---|---|
| One part truncated to half its size | Detected, skipped, counted in `files_skipped`; the remaining files still query correctly | `test_a_corrupt_part_is_skipped_and_the_rest_stays_readable` |
| A stray `.tmp` file containing garbage | Invisible; `files_skipped == 0`; the file is left in place | `test_a_temp_file_left_by_a_crash_is_invisible` |
| A run directory that does not exist | Returns empty results and zero coverage, **not** an error | `test_querying_a_nonexistent_run_returns_empty_not_an_error` |

`ReaderCoverage` exists so a caller can distinguish **"no anomaly"** from **"no
data"** — Phase 0's most-cited failure mode for automated diagnosis. It reports
files total/readable/skipped, the step range present, event types, row count and
ranks.

---

## 4. Bugs found while building W3

| Bug | Impact | Fix |
|---|---|---|
| **`rows_per_file` bounded the buffer trigger, not the file** | One large batch (the observer's default batch is 256 events) produced a single arbitrarily large file; the file-size bound was fiction | `_flush_locked` now chunks the buffer into `rows_per_file`-sized groups |
| **Falsy-cache bug in the reader** | `_view_name` cached `""` for an empty run; `""` is falsy but not `None`, so `if view is None` missed it and produced `SELECT * FROM  WHERE ...` → `ParserException` | Cache an explicit `None`, and type the cache `Dict[str, Optional[str]]` |
| **`flush()` returned with a batch in flight** | Found by the 1M benchmark: 2 events were neither written nor dropped at the moment stats were read, because `len(queue) == 0` does not mean the writer thread is idle | `flush()` now waits for queue-empty **and** `_in_flight == 0`; added `accounting_identity()` and three tests asserting `written + dropped == emitted − sampled_out − validation_failures` |

The third is the most important: it was discovered by a *benchmark*, not by a
unit test, and the invariant it violated is the one class of bug this system
cannot tolerate — an event that silently vanishes. The identity is now asserted
under three regimes (healthy, failing writer, queue overflow).

---

## 5. Gate results

| Gate item | Status |
|---|---|
| Parquet writer works | **PASS** — 1M events, verified row count |
| DuckDB query layer works | **PASS** — all 7 required query shapes |
| 1M event storage benchmark completed | **PASS** |
| metrics for run / around step | **PASS** |
| events around anomaly / by rank / by layer | **PASS** |
| event type count | **PASS** |
| incident evidence window | **PASS** |
| file count bounded (no small-file explosion) | **PASS** — 10 files per 1M events |
| completed files readable after corruption | **PASS** |
| partial batch / writer crash / corrupt parquet handled | **PASS** |
| Trackio is not the telemetry truth | **PASS** — not imported anywhere in `storage/` |

---

## 6. Not verified in W3

* **Partitioned layout.** `partition_by_event_type=True` is implemented but the
  benchmark runs the default (unpartitioned). The file-count trade-off is
  therefore **not measured** in the partitioned case — only reasoned about.
* **Concurrent writers.** One writer per run is assumed. Two processes writing
  the same `run_id` would race on the file index; the atomic rename prevents
  corruption but not duplicate `ingest_seq`. Multi-process is W7+ work.
* **Compaction.** No compaction pass exists. At 10 files per 1M events it is not
  needed; the trigger would be file count in the thousands.
* **Sustained multi-hour writes.** The benchmark writes 1M events in one pass.
  Behaviour over many flushes with an evolving schema is not exercised.
* **Disk full at the OS level.** W1 tests `ENOSPC` from the writer's perspective;
  no genuinely full filesystem was created.
* **Cold-cache query latency.** All queries ran against a warm page cache.
