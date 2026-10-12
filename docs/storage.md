# Storage

**Parquet is the source of truth. DuckDB is the query engine. Trackio is not in
the telemetry path.**

## Layout

```
<root>/runs/<run_id>/events/[event_type=<TYPE>/]part-<index>-<seq>-<unique>.parquet
<root>/runs/<run_id>/manifest.json
```

| Property | Value |
|---|---|
| rows per file | 100 000 (target; enforced by chunking, not just by a buffer trigger) |
| compression | zstd, with dictionary encoding on low-cardinality columns |
| measured | 1 000 000 events → **10 files, 9.6 MB, 10.1 bytes/event** |

## Crash safety

Write to `.parquet.tmp`, then `os.replace`. Readers glob `*.parquet`, so a process
killed mid-write leaves an invisible temp file rather than a truncated Parquet that
makes the run unreadable. A corrupt part is detected by reading its footer,
skipped, and **counted** in `files_skipped`.

The guarantee is "completed files readable", not transactional semantics.

## Coverage

`ReaderCoverage` reports files total/readable/skipped, the step range present,
event types, row count and ranks, so a caller can distinguish **"no anomaly"** from
**"no data"** — the failure mode Phase 0 called out as the most common way an
automated diagnosis goes wrong.
