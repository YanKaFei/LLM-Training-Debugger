# Incident persistence

## Why it exists

Phase 1 incidents lived only in memory. Three things were therefore impossible:
explaining why an *old* run was judged anomalous, comparing incidents across runs,
and calibrating detectors against historical outcomes. All three are prerequisites
for anything built on top.

## Layout

```
<root>/runs/<run_id>/incidents/
  incidents/incidents-<n>-<id>.parquet
  incident_evidence/incident_evidence-<n>-<id>.parquet
  incident_hypotheses/incident_hypotheses-<n>-<id>.parquet
```

**Append-only, one file per write.** Parquet is immutable, so a revision is a new
row in a new part file, never a rewrite. An incident therefore has a *history*
rather than a current value, which is what `OPEN → UPDATED → RESOLVED → CLOSED`
requires: the earlier state is the evidence for why the later one was reached.

**Three tables, not one blob.** A single JSON column would make "show me every
incident where the learning rate was ruled out" inexpressible.
`incident_evidence` carries one row per evidence item with its bucket, level,
metric, baseline, threshold and window; `incident_hypotheses` carries one row per
scored hypothesis with its decomposition.

## Revisions

Every write produces `(incident_id, revision, state, timestamp)`. The read API
returns the **latest revision by default**; `full history` is available via
`history()`. `load(..., revision=N)` reads a specific revision.

## Provenance

Every evidence row carries:

| Column | Purpose |
|---|---|
| `evidence_id` | stable identity |
| `event_ids` | the events the claim rests on |
| `detector`, `detector_version` | what produced it |
| `rule_version` | **the thresholds in force** |
| `query_definition` | the statistical procedure, so a reviewer can recompute it |

`rule_version` is the column that makes an old incident explicable after a
threshold changes — which is exactly the question a postmortem asks.

## The test that matters

`test_restart_reconstruction_in_a_fresh_process` writes incidents in one
interpreter, exits, and reads them back **in a second interpreter**, asserting the
rendered output is byte-identical. An in-process round trip could pass while the
store still held a live object reference; a subprocess cannot.

## Crash safety and fail-open

Each part is written to `.parquet.tmp` and `os.replace`d, so a reader globbing
`*.parquet` never sees a half-written batch. A persistence failure is recorded in
`store.errors` and, by default, does not propagate: an analysis that already
succeeded must not be destroyed by a storage problem. `fail_open=False` is
available and tested for callers that want strictness.

## No database

Phase 0's trigger for Postgres was *several humans writing concurrently to the
incident store*. That remains untrue, so Parquet plus the existing DuckDB reader.
