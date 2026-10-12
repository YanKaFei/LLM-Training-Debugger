# DEVELOPMENT STATE

> Project memory. Read this first on resume, verify against the repository, then continue.
> Never rely on conversation context alone.

```
CURRENT_GOAL:            GITHUB PRE-RELEASE PREPARED — awaiting owner approval
PROJECT_LOCATION:        <repo> (dedicated checkout)
CURRENT_TASK:            Project relocated to a dedicated directory on 2026-10-11.
                         `git add -A` is now safe by construction. Nothing published.
                         Owner action: review outputs/rc/ then run the draft commands.
LAST_COMPLETED_TASK:     W1 COMPLETE -- temporal integrity. Root cause: the episode onset
                              was the first observation of the *contiguity run*, so a
                              benign WEAK/NORMAL-band excursion at step 75 (gap 3 <=
                              gap_tolerance 8) inherited the real step-80 injection and
                              the incident reported onset 75. The band had already called
                              those steps NORMAL, and a step the band calls NORMAL cannot
                              be the onset of an anomaly. Onset now anchors to the earliest
                              CONFIRMED **or UNCERTAIN** observation (UNCERTAIN retained so
                              genuine early warning survives); the raw first step is kept in
                              metadata.quickstart now reports onset 80. +3 tests.
                         W4 PARTIAL -- the flaky test's mechanism identified: it discarded
                              `flush()`'s return value and used timeout=2.0, so a timed-out
                              flush (events still in-flight, neither written nor dropped)
                              was reported as `unaccounted != 0` -- a misleading failure that
                              looks like data loss but is an incomplete drain. Fixed by
                              asserting completion first; +1 test documenting the
                              distinction. Stress 30/30. Race NOT reproduced -> UNRESOLVED.
                         A3/C2/C3 persisted invariants +9 tests (idempotency, independent
                              NaN protection, baseline restoration, all through the store)
                         D2 scale measured: 62,002 events at 2000 steps, 0 drops, linear
                         D3 CLOSED: a forced 484-event overflow reaches the verdict as
                              DEGRADED with a warning -- fail-open cannot mask loss
                         S1 re-diagnosed: root cause is NOT the frozen baseline. With the
                              baseline correctly tracking, l1 is 1.0 -> 1.7 -> 2.6 and then
                              a genuine ONE-STEP transient to 12.124 at step 179, back to
                              2.7 at step 180. A healthy run contained a real transient.
                         Baseline resume path REPAIRED: `protected_ratio_series` called
                              `suspect()`, which sets state=SUSPECT, and nothing ever set
                              it back -- but `update()` only absorbs while state is
                              NORMAL/CLOSED. `resume_after_steps` was DEAD CODE and the
                              reference froze permanently at the first suspicion.
                         A1 evidence_completeness COMPLETE (+10 tests)
                         A2 S1 false positive DIAGNOSED (frozen reference never resumes)
                         A1b fixed: from_observer_health read flat keys that do not exist,
                              so completeness silently always passed; now defaults to UNKNOWN
                         B1 persisted scorer EXECUTED from disk (thin denominator)
                         C1 smoke contracts converted to 10 regression tests
                         E1 LICENSE (Apache-2.0)
                         E2 README/QUICKSTART/ARCHITECTURE/CAPABILITIES/
                            KNOWN_LIMITATIONS/CONTRIBUTING/CHANGELOG/RELEASE_NOTES
                         E4 examples/quickstart.py RUNS end to end
                         E5 clean-venv editable install VERIFIED (exit 0, console script runs)
```

## ACTIVE_BLOCKERS

| # | Blocker | Severity | State |
|---|---|---|---|
| 1 | ~~`evidence_completeness`~~ | correctness | **CLOSED** (A1) |
| 2 | S1 healthy-run false positive — root cause is the frozen reference never resuming, because benign excursions occur closer together than `resume_after_steps=40`. Fix needs a measured design decision, not a threshold nudge | detector | **DIAGNOSED**, fix not applied |
| 3 | ~~persisted scorer~~ | scoring | **CLOSED** (B1) — but the development store holds only 14 runs and 0 relations, so quality is unmeasurable until the corpus is re-persisted |
| 4 | Historical 23 duplicates not split into product vs scorer error | audit | open |
| 5 | Observer accounting race unexplained | stability | UNRESOLVED |
| 8 | ~~S1~~ | detector | **CLOSED as a test-category error** — see D-011; the underlying 7.7 % healthy FP rate is a detector-quality finding carried under DETECTOR_RELIABILITY |
| 9 | ~~D2 disk growth~~ | measurement | **CLOSED** — 62,002 events, ~25 bytes/event, linear |
| 10 | Baseline resume fix cost precision/recall slightly; `resume_after_steps` now has a real effect | detector | **W2 sweeping**, pre-registered |
| 11 | ~~onset reported 5 steps before the fault~~ | correctness | **CLOSED** (W1) |
| 13 | ~~relation layer inverted~~ | relation | **CLOSED (v0.1.3)** — mechanism-constrained association; precision 1.0 recall 1.0 |
| 16 | Mechanism now observable at every stage via `param.global_norm`; the 2-step delay bound still assumes 1 step = 1 optimizer update, which gradient accumulation breaks | observability | **PARTIALLY CLOSED**, RFC written |
| 18 | Observationally equivalent histories could NOT be constructed; the abstention path under true ambiguity is unexercised | evaluation | open |
| 19 | **`*.global_norm == inf` is a reduction artifact**; the whole v0.1.3/1.4 mechanism chain keyed on it | correctness | **CLOSED (W0)** |
| 20 | Relation recall 0.167 on the challenge corpus; 20 missed consequences | observability | open |
| 21 | ~~optimizer witness changes nothing~~ | observability | **FROZEN** — paired identical twice (54 runs + 32 runs) |
| 22 | **C3 is blocked by the threshold-to-stability ratio**: the gradient magnitude needed to fire the aggregate detector also corrupts the model, so an invoked-but-independent case cannot be held to shape | observability | **ROOT-CAUSED**, open |
| 24 | ~~11-step latency~~ | diagnostic | **CLOSED** — it was baseline readiness (floor 50 + minimum_history 20 -> ready at 70); the detector is prompt once ready |
| 23 | C4: a real downstream overflow with finite parameters is missed 20x | coverage | open |
| 17 | ~~incident-level FN never incremented~~ | evaluation | **CLOSED (W7)** — `missed_primary_faults` surfaced |
| 14 | ~~42 vs 41 readable runs~~ | integrity | **CLOSED** — loader required an episode dir; zero-episode runs were dropped |
| 15 | ~~primary_recall always 1.0~~ | evaluation | **CLOSED** — `MISSED_PRIMARY` was never incremented; run-level recall is 0.50 |
| 12 | Observer flaky test: root mechanism identified (discarded `flush()` return value + short timeout), fixed; the underlying race was never reproduced | stability | **UNRESOLVED**, mechanism documented |
| 6 | ~~full-pipeline smoke~~ | integration | **CLOSED** — 8/8 |
| 7 | ~~LICENSE~~ | packaging | **CLOSED** |

## LAST_TEST_RESULTS

```
suite:     533 passing, 1 skipped
package:   training_debugger-0.1.7a1  (wheel + sdist)  version aligned 0.1.7a1 <-> v0.1.7-alpha
leaks:     0 local paths, 0 credentials in the built sdist (180 files searched)
RC:        wheel + sdist built; wheel verified in a clean venv
           relation safety: POSSIBLE_CONSEQUENCE_OF now DISABLED by default
smoke:     8/8 PASS
relations: v0.1.5 -- `*.global_norm == inf` is a REDUCTION ARTIFACT, not a non-finite
           tensor. Excluding it removed EVERY mechanism claim:
           50-run mixed TP 0 FP 0 FN 10 TN 40; adversarial TP 0 FP 0 FN 10 TN 35
           v0.1.4's 1.0/1.0 was measuring that artifact.
           OptimizerWitness implemented, default OFF, overhead +0.30% total / +0.97% p95
command:   .venv/bin/python -m pytest tests -q
smoke:     8 executed, 8 passed
healthy FP probe: 1/13 = 0.077 -> FAIL (gate 0.05)
corpus:    precision 0.964 recall 0.457 rank_corr 0.748 (was 0.973/0.468/0.860 before the resume fix)
install:   editable PASS; non-editable PASS (imports from site-packages)
```

## MODIFIED_FILES (this session)

```
docs/DEVELOPMENT_STATE.md          new
docs/AUTONOMOUS_BACKLOG.md         new
docs/DECISIONS.md                  new
docs/RELEASE_GATES.md              new
src/training_debugger/fusion/quality.py        new  (A1)
src/training_debugger/fusion/rules.py          data_quality threaded through fuse()
src/training_debugger/incidents/render.py      degraded stream no longer rendered as reassurance
tests/final_holdout/run_final_holdout.py       observer health -> verdict quality
tests/unit/test_semantics_and_reconciliation.py  +6 tests
tests/unit/test_abstention_renderer.py         updated to completeness-aware semantics
```

## NEXT_EXACT_ACTION

0. Read `outputs/alpha/corpus_after_resume_fix.txt` — the 680-run corpus measured WITH the
   baseline resume fix. If precision/recall regressed, the fix must be revisited.
1. Re-run the development corpus through the CURRENT pipeline so episodes, incidents and
   relations all persist under correct run ids (the Phase 1.12A run_id defect left the
   existing store with 14 of 169 runs and 0 relations).
2. Score that store -> real TP/FP/FN with denominators; split the historical 23.
3. Convert the eight smoke scenarios into regression tests (workstream C).
4. Write QUICKSTART / ARCHITECTURE / CAPABILITIES / KNOWN_LIMITATIONS / CONTRIBUTING /
   CHANGELOG, and verify a clean-environment install with a real command.
5. Then re-evaluate the Alpha gate.

## RELEASE_READINESS

```
RELEASE_CANDIDATE = READY
PRERELEASE_PREP   = READY  (package 0.1.7a1, tag v0.1.7-alpha; PUBLICATION NOT EXECUTED)
V0_1_7_ALPHA_READY = YES
V0_1_6_ALPHA_READY = YES
V0_1_5_ALPHA_READY = YES
V0_1_4_ALPHA_READY = YES
V0_1_3_ALPHA_READY = YES
V0_1_2_ALPHA_READY = YES
V0_1_1_ALPHA_READY = YES
ALPHA_RELEASE_READY = YES
ALPHA asserts installability and engineering correctness only.
DETECTOR_RELIABILITY = FAIL independently (recall 0.457; healthy FP 0.077).
observer race UNRESOLVED and disclosed with its risk boundary.
Outstanding research work: B2/B3/B4 (re-persist the corpus, score it, split the
historical 23) and D1 (observer race).
```


## PROJECT LOCATION (important)

```
<repo>      <- THE PROJECT (dedicated checkout)
<repo>                <- unrelated material only; no project files
```

The project was moved out of a shared working directory that held ~1.9 GB of unrelated
content (a 415 MB pnpm cache, a 1.1 GB unrelated project, a 208 MB unrelated project,
149 MB of research clones, a **personal contact database**, and proxy/VPN scripts).
`git add -A` in the old directory would have published all of it; in the new one it stages
22 project entries and nothing else. Verified with `git check-ignore`.

**If a future session starts in the old directory, it is in the wrong place.**
