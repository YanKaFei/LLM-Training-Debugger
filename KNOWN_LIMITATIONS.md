# KNOWN LIMITATIONS

Everything here is measured or explicitly unmeasured. Nothing is a guess.

## Correctness

1. **Healthy-run false positives — measured at 7.7 %, above the 5 % gate.** Over 13
   healthy runs on the smoke configuration, **1** produced a `CONFIRMED_INCIDENT`.
   Seed 61001 contains a genuine single-step gradient transient (joint aggregate
   `l1`: 2.635 -> **12.124** -> 2.728 at steps 178-180) that exceeds both
   `confirm_at = 11.5` and `magnitude_override = 8.0`, and confirms from **one**
   observation via the `aggregate_confirms` path. The detector is behaving to its own
   specification: **this is a detection-quality limitation, not a defect.** Reducing it
   means deciding whether a single observation should confirm from a transient, which
   trades directly against recall and must be measured — see `docs/DECISIONS.md` D-011
   and `outputs/alpha/BASELINE_RESUME_FIX_MEASUREMENT.md`.
2. **Baseline `resume_after_steps` was dead code until this session.** The freeze was
   permanent, because `suspect()` was the only state transition `protected_ratio_series`
   performed and `update()` only absorbs while the state is `NORMAL`. Repaired, at a
   measured cost: precision 0.973 -> 0.964, recall 0.468 -> 0.457, healthy-confirmed
   0.042 -> 0.056, rank-correlation 0.860 -> 0.748. Kept deliberately (D-010); the
   parameter now deserves a measured sweep rather than a nudge.
3. **Pre-injection onset.** The quickstart example injects at step 80 and the reported
   onset is **step 75** — five steps early. Under the acceptable-window rule this is a
   false-positive onset, not an early detection. Not yet diagnosed.
4. **No source-event-level dedup identity.** Idempotency is at `(step, layer)` scoped
   per `signal_key`. `Anomaly` carries no `event_id`, so two genuinely different
   observations sharing `(step, layer)` are indistinguishable. The honest claim is
   "idempotent at `(step, layer)` granularity".
5. **Historical duplicate count unexplained.** Phase 1.9's "23 duplicate top-level
   incidents" was produced by the *scorer* and has **not** been reproduced
   (`HISTORICAL_23_NOT_FULLY_REPRODUCED`). It is unknown how many were product
   duplicates versus scorer misclassifications.

## Stability

6. **Observer accounting race — `UNRESOLVED`.** One historical full-suite failure,
   never reproduced: 0 of 35 attempts (30 isolated, 5 full-suite) plus deterministic
   forced-overflow stress. Fail-open means a race would corrupt **counts**, not training
   numerics. Not claimed as fixed.
7. **Fail-open at the increased evidence volume — now measured.** Phase 1.12A raised
   detector observation volume 20×. The observer's raw-event load was measured
   **unchanged** (3722 → 3722) and detector observations saturate at 89, but fail-open
   behaviour at scale was re-verified: 62,002 events at 2,000 steps with **0 drops**, a
   linear ~25 bytes/event footprint, and a forced 484-event overflow that correctly
   surfaced as `DEGRADED` on the verdict rather than as a clean negative.

## Measurement

8. **No blind holdout has ever been run.** All quality figures are development data.
9. **Not one capability is validated.**
10. **Only 14 development runs are persisted under usable ids.** A Phase 1.12A `run_id`
   defect meant runs were stored under a path their own identifier could not find. The
   current store has 14 runs and **0 relations**, so relation scoring is
   `NOT ESTIMABLE`.
11. **Engineered for one process.** Multi-rank is untested; ordering paths exist but no
    real two-rank incident has been produced.
12. **No GPU telemetry.** Every measurement is CPU/MPS on an Apple M2 Max. No DCGM, HTA,
    Flight Recorder or NNsight.
13. **Observer overhead budgets are not met.** core +3.3 % at the largest workload
    against a < 3 % target; T1 passes at medium/large, fails on tiny.
14. **No replay**, so data-causation claims are capped at `HYPOTHESIS`.
