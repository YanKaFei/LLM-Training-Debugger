# A4 — FOUR-VERDICT PERSISTED E2E

**Status: PASS — 4 of 4.**

```
A  expected=CONFIRMED_INCIDENT     observed=CONFIRMED_INCIDENT     matched=True   CLI exit 0
B  expected=CANDIDATE_ONLY         observed=CANDIDATE_ONLY         matched=True   CLI exit 0
C  expected=INSUFFICIENT_EVIDENCE  observed=INSUFFICIENT_EVIDENCE  matched=True   CLI exit 0
D  expected=NO_ANOMALY             observed=NO_ANOMALY             matched=True   CLI exit 0
```

Phase 1.11 achieved 3 of 4. The contract repair in A2 made `CANDIDATE_ONLY` reachable,
which closed the fourth.

## 1. Method

Four independent runs, each produced by the real `DetectorRunner`, the real
`EpisodeEngine` and the real `fuse()`, then written through the production persistence
path (`EpisodeStore`, `IncidentStore`, `AnalysisStateStore`, relation table). The real
CLI is then executed as a **separate process** for each.

**No verdict object is constructed and saved.**

| Scenario | Method | Labelled |
|---|---|---|
| A | REAL PYTORCH TRAINING | yes |
| B | EVENT-DRIVEN PIPELINE INTEGRATION | yes |
| C | EVENT-DRIVEN PIPELINE INTEGRATION | yes |
| D | REAL PYTORCH TRAINING | yes |

The `method` field is recorded **per scenario** in `four_verdict_e2e.json`, so the
event-driven cases are never passed off as training E2E.

## 2. What closed the gap

`CANDIDATE_ONLY` requires a **confirmed** `LOSS_SPIKE` episode, which requires two
observations within six steps, which the detector's 50-step cooldown made impossible.
With the cooldown no longer suppressing evidence, two distinct qualifying observations
reach the engine, the episode confirms, and fusion returns the candidate verdict —
while promotion stays OFF, so no incident is opened.

## 3. Artifacts

```
outputs/phase1-12a/four_verdict_e2e.json     (copy of cli_e2e_results.json)
outputs/phase1-12a/store/                    persisted episode / incident / analysis state
```

Each scenario records `run_id`, `episode_id`, `verdict`, `promotion_rule_id`,
`candidate_onset_step`, `confirmed_at_step`, the CLI command, exit code, stdout and
stderr.

## 4. Not verified

* **Read/write state-change concurrency.** The brief asks whether candidate evidence
  could be misread if an episode's state changes between write and read. Episodes are
  append-only with monotonic revisions, and the CLI reads the latest revision, so a
  later `CLOSED` revision correctly supersedes an earlier `SUSPECT` one — but no
  explicit **as-of-step** query semantic exists, and a test that reads mid-mutation was
  not written.
