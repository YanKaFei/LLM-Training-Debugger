# PHASE 1.12A REPORT

```
PHASE 1.12A STATUS:             PARTIAL
WORKSTREAM A (contract repair): PASS
WORKSTREAM B (observer race):   NOT REPRODUCED / NOT REPAIRED
READY_FOR_BLIND_VALIDATION:     NO
```

**No final holdout files were created.** No detector added, no gradient threshold
touched, no Phase 2 feature, no Agent.

`PHASE_1_REPORT.md` and `outputs/phase1-1..1-11/` are **unmodified**.

---

## 1. Workstream A — detector contract repair: **PASS**

### A1 — the defect, located in source

`_in_cooldown` is tested **before the `Anomaly` is constructed**, so a suppressed
observation never reaches `DetectorRunner`, never reaches `EpisodeEngine`, and cannot
contribute to persistence. Of the brief's four options, the answer is **(2) detector
anomaly emission** — the observation is discarded at the detector boundary.

Full trace in `A1_CONTRACT_TRACE.md`, measured on a sustained 20-step spike:

```
cooldown_suppresses_evidence = True   ->   1 observation  (step 90)
cooldown_suppresses_evidence = False  ->  20 observations (steps 90..109)
```

`LOSS_SPIKE` needs **2 within 6 steps**, so with one observation it closed as
`NOT_ANOMALY` and `CANDIDATE_ONLY` was unreachable.

### A2 — the fix

Evidence observation and alert deduplication are separated. `cooldown_suppresses_evidence`
defaults to `False`; a suppressed observation is still emitted and flagged
`cooldown_suppressed` so an alert-counting consumer can filter without losing evidence.

**Not** `cooldown_steps = 0` — deduplication was not removed, it was returned to the
layer that already does it: `EpisodeEngine` groups contiguous observations into one
episode and `fuse()` returns one verdict per anomaly type. Measured: 20 observations →
**1 episode → 1 verdict → 0 incidents**.

**Defect 2, found while testing this:** `EpisodeEngine` did not deduplicate, so
submitting the same observation twice manufactured persistence. Fixed at
`(step, layer)` granularity. The coarser `(step)` rule was tried first and broke
`test_cross_process_restart_readback`; the finer rule satisfies both.

### A3 — regression tests

```
417 -> 428 passing, 1 skipped   (+11)
```

All eight required tests present, using real project APIs. Two defects were found by
writing them.

### A4 — four-verdict persisted E2E: **4 of 4 PASS**

```
CONFIRMED_INCIDENT    PASS      CANDIDATE_ONLY         PASS
INSUFFICIENT_EVIDENCE PASS      NO_ANOMALY             PASS
```

Every one produced by the real detector → episode → fusion path, persisted through the
production stores, and read back by the **real CLI in a separate process**, all exiting
0. Method is labelled per scenario (`REAL PYTORCH TRAINING` vs
`EVENT-DRIVEN PIPELINE INTEGRATION`).

---

## 2. Workstream B — observer race: **NOT REPAIRED**

```
isolated repetitions  30   passes 30   failures 0
full-suite repetitions 5   passes  5   failures 0
```

**The Phase 1.11 failure did not reproduce in 35 attempts.** No root cause was
identified, no fix was applied, and no deterministic replacement test was written.

This is **not** a PASS. Absence of a reproduction does not clear a race that appeared
once. It is reported as `NOT REPRODUCED IN 35 ATTEMPTS`, which is weaker than PASS, and
the brief's `OBSERVER_STABILITY` claim is **not** asserted.

**A discarded measurement, reported:** an earlier loop in this phase ran the wrong path
(`tests/fault_injection/test_observer_faults.py::...`, which does not contain the test)
and printed `0/30`. Those were pytest **collection errors — "test not found"** — not
failures, and they are discarded with the reason recorded. A `0/30` figure that is
really a path typo would look exactly like a severe race if left unexplained.

Per the brief, a change without an identified race would be a guess, and with the test
passing 35/35 a guess could not even be validated against a reproduction. So none was
made.

---

## 3. Gate

| Gate | State |
|---|---|
| `LOSS_SPIKE` cooldown/evidence contract clarified | **PASS** |
| Confirmation can receive required observations | **PASS** |
| Duplicate observations are not counted as persistence | **PASS** — defect 2 fixed |
| `LOSS_SPIKE` promotion remains OFF | **PASS** |
| `CANDIDATE_ONLY` reachable through production components | **PASS** |
| Candidate recovery semantics remain correct | **PASS** |
| Four persisted CLI verdicts PASS | **PASS** — 4 of 4 |
| Independent-process CLI readback PASS | **PASS** |
| New regression tests added | **PASS** — +11 |
| Observer race root cause identified | **FAIL** — not reproduced |
| Deterministic queue-overflow test implemented | **FAIL** — not done |
| Observer 30 isolated repetitions PASS | **PASS** — 30/30 |
| Full suite 5 repetitions PASS | **PASS** — 5/5 |
| No new detector | **PASS** |
| No holdout | **PASS** |
| No Phase 2 | **PASS** |
| No Agent | **PASS** |

**15 PASS / 2 FAIL → PHASE 1.12A = PARTIAL.**

---

## 4. What moved

| Capability | Phase 1.11 | **Phase 1.12A** |
|---|---|---|
| persisted CLI verdicts | 3 of 4 | **4 of 4** |
| `LOSS_SPIKE` confirmability | structurally impossible | **reachable** |
| observation idempotency | absent | **enforced at `(step, layer)`** |
| evidence vs alerting | conflated | **separated and measured** |
| tests | 417 | **428** |

The evidence chain is now complete end to end for all four verdicts: real detector →
real episode → real fusion → real stores → **separate process → real CLI**.

## 5. Governing principle

> Phase 1.11 found that the fourth verdict was unreachable because a detector's cooldown
> made its own confirmation rule unsatisfiable.
>
> **Phase 1.12A repaired the contract — evidence now flows to the episode layer and
> deduplication stayed where it belongs — and all four verdicts now survive a write,
> a process boundary, and a CLI read. A second defect, observation double-counting, was
> found by the test that asserted it could not happen.**

Workstream A is closed. Workstream B is neither repaired nor cleared.

**No agent. No replay. No new diagnostic source. No causal claim.**
