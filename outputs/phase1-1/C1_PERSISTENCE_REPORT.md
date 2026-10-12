# C1 — INCIDENT PERSISTENCE REPORT

**Status: COMPLETE.** 13 tests, including cross-process restart reconstruction.

---

## 1. Layout

```
<root>/runs/<run_id>/incidents/
  incidents/incidents-<n>-<id>.parquet
  incident_evidence/incident_evidence-<n>-<id>.parquet
  incident_hypotheses/incident_hypotheses-<n>-<id>.parquet
```

The brief's suggested layout was adopted unchanged, because the three-table split
is what makes provenance queryable: a single blob column makes "every incident
where the learning rate was ruled out" inexpressible. Append-only, one file per
write — Parquet is immutable, so a revision is a new part file and an incident has
a *history* rather than a current value.

No Postgres. Phase 0's trigger was concurrent human writers; still false.

## 2. Revisions

`(incident_id, revision, state, timestamp)`, append-only. Verified:
`revisions == [1, 2]` after one `update_state`, with `history[0].state == OPEN`
preserved. The reader API returns the latest revision by default; `history()` and
`load(..., revision=N)` give the full record.

## 3. Provenance

Every evidence row carries `evidence_id`, `event_ids`, `detector`,
`detector_version`, **`rule_version`**, and `query_definition` (the statistical
procedure). `rule_version` is what allows an old incident to be explained after a
threshold changes — the exact question a postmortem asks.

## 4. Restart reconstruction — the decisive test

`test_restart_reconstruction_in_a_fresh_process` builds and persists incidents in
one interpreter, **exits**, and reads them back in a **second interpreter**,
asserting byte-identical rendered output.

An in-process round trip could pass while the store still held a live object
reference; a subprocess cannot. **Result: identical.**

Also verified in-process: `render_incident(loaded) == render_incident(original)`,
evidence counts, `not_correlated` counts, hypothesis counts, `first_abnormal_event_id`,
`multiplicity` and `ordering_confidence` all preserved.

## 5. Crash safety, fail-open

Write to `.parquet.tmp` then `os.replace`, so a reader globbing `*.parquet` never
sees a partial batch. A persistence failure is recorded in `store.errors` and by
default does not propagate — an analysis that already succeeded must not be
destroyed by storage. Both fail-open and strict modes are tested.

## 6. Defects found and fixed during C1

| Defect | Symptom | Fix |
|---|---|---|
| `Hypothesis.to_dict` keys ≠ dataclass fields | `from_dict` raised `unexpected keyword argument 'supporting_evidence_ids'` | an explicit `_FIELD_ALIASES` map |
| `EventType`, `OrderingConfidence` unimported in `incident.py` | `NameError` during reconstruction | imports added |
| `earliest_abnormal.candidates` lacked `run_id`/`producer`/`layer_index` | reconstruction lost the ordering evidence | payload widened |
