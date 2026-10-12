# C2 — TRACKIO INTEGRATION REPORT

**Status: COMPLETE, and verified against the real library — which required
obtaining a Python 3.12 interpreter.**

---

## 1. The blocker, and how it was removed

`trackio` 0.40.0 declares `requires_python >=3.10`. This project's primary
environment is **Python 3.9.6**, so Trackio cannot be installed into it at all.

Two options were rejected:

* **Integrate against Trackio 0.0.x**, the only releases admitting 3.9. That is a
  different library from the 0.40.0 researched in Phase 0, so a passing test would
  have proven nothing about the real integration.
* **Test against a hand-written fake.** A fake validates my assumptions, not
  Trackio.

Instead `uv` was installed **into the workspace** (the sandbox blocks
`~/.local/bin`), a standalone CPython 3.12.15 was provisioned, and
**trackio 0.40.0 was installed into `.venv312`**. The real library was then
executed.

## 2. Two real API errors this caught

Running the actual library invalidated the Phase 0 notes:

| Phase 0 note | Reality in 0.40.0 |
|---|---|
| `trackio.init(..., mode="local")` | **`init()` has no `mode` parameter** — `TypeError` |
| `trackio.log({..., "step": n})` | **`log(metrics, step=n)`** — step is a parameter |

Neither was discoverable from documentation. Both are now fixed in `TrackioSink`,
with a comment recording why.

## 3. Verified behaviour against the real library

```
probe:      {'available': True, 'version': '0.40.0'}
forwarded:  60 allow-listed metrics       (6 metrics x 10 steps)
filtered:   20 non-allow-listed metrics   (grad.layer_norm, mem.allocated)
failures:   0
artefacts:  trackio/td-verify.db  (208,896 bytes)
```

The stored SQLite was opened directly and inspected:

```
table metrics: id, run_id, timestamp, run_name, step, metrics, log_id, space_id
rows: 60
sample: (1, '7bd70c31...', '2026-10-07T17:49:26+00:00', 'run-verify', 1, b'{"train.loss":2.0}', ...)
```

**Rows = 60 = exactly the allow-listed metrics. `grad.layer_norm` and
`mem.allocated` are absent from the store.** The allow-list held end to end, in
the real library's real on-disk format — not just in my filter function.

One further real-environment finding: Trackio resolves `TRACKIO_DIR` from the
environment **at import time** and otherwise writes to `$HF_HOME/trackio`. Under a
restricted environment it therefore silently persisted *nothing* on the first
verification run while reporting success. `TrackioSink(directory=...)` now sets it
before the import.

## 4. Presentation-only, enforced not promised

`PRESENTED_METRICS` is a **closed allow-list**: `train.loss`, `train.lr`,
`grad.global_norm`, `param.global_norm`, `train.step_time`,
`train.tokens_per_sec`. `FORBIDDEN_PREFIXES` additionally blocks any
`grad.layer_norm`, `param.layer_norm` or `event.*` name even if someone widens the
allow-list by mistake.

Presented for incidents: `incident_count`, `highest_incident_severity`,
`earliest_incident_step`, `not_correlated_count` — **counts and severity only**.
Incidents themselves are never forwarded: the incident store is the source of
truth, and a dashboard holding incidents would be a second copy of it.

Not forwarded: all layer metrics, all events, all evidence edges.

## 5. Fail-open

The bridge isolates every call and counts failures. Covered: Trackio
**unavailable** (the real state in `.venv`, where the import fails on 3.9 — a
genuine, not simulated, unavailable case), **throwing** (`FaultySink` raises from
every method), **slow** (`SlowSink`, 200 ms per call), and **initialisation
failure**. In every case training, Parquet, detectors and incidents are unaffected
and the failure is recorded in `health()`, never raised.

A bounded buffer (`max_buffer=512`) with counted overflows ensures an unreachable
sink cannot grow memory inside the training process.

## 6. No dual truth

Deleting every Trackio artefact leaves the product fully functional: detectors read
Parquet, incidents read the incident store, and nothing reads Trackio.

## 7. Not verified

* **Remote/HF Space mode.** Only local mode was exercised. `server_url`,
  `space_id` and `bucket_id` paths are untouched.
* **Trackio under `torch` in the same process.** The real-library verification ran
  in the 3.12 venv without torch; the 3.9 venv has torch but cannot import
  trackio. The bridge's behaviour is identical either way (it is duck-typed), but
  a single process running both was not exercised.
* **Concurrent runs to one project.** Untested.
