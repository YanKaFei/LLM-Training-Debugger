# KNOWN LIMITATIONS

Everything here is measured or explicitly unmeasured. Nothing is a guess.

Last audited against the code: v0.1.7-alpha, commit `ed3ea71`.

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
2. **The protected baseline freezes permanently unless the deviation clears.** The
   `resume_after_steps` transition was dead code; it is now wired, at a measured cost:
   precision 0.973 -> 0.964, recall 0.468 -> 0.457, healthy-confirmed 0.042 -> 0.056,
   rank-correlation 0.860 -> 0.748. Kept deliberately (D-010). An alternative that does
   not freeze at all — an EMA whose statistics are updated with clipped observations —
   is implemented as `baseline_mode="ema"` and is **not the default, because it was
   measured to be indistinguishable from `frozen` in every configuration tested**
   (D-040).
3. **No source-event-level dedup identity.** Idempotency is at `(step, layer)` scoped
   per `signal_key`. `Anomaly` carries no `event_id` (verified: the field does not
   exist), so two genuinely different observations sharing `(step, layer)` are
   indistinguishable. The honest claim is "idempotent at `(step, layer)` granularity".
4. **Historical duplicate count unexplained.** Phase 1.9's "23 duplicate top-level
   incidents" was produced by the *scorer* and has **not** been reproduced
   (`HISTORICAL_23_NOT_FULLY_REPRODUCED`). It is unknown how many were product
   duplicates versus scorer misclassifications.

## Stability

5. **Observer accounting race — `UNRESOLVED`.** One historical full-suite failure,
   never reproduced: 0 of 35 attempts (30 isolated, 5 full-suite) plus deterministic
   forced-overflow stress. Fail-open means a race would corrupt **counts**, not training
   numerics. Not claimed as fixed.
6. **Fail-open at increased evidence volume — measured.** Phase 1.12A raised detector
   observation volume 20×. Raw-event load was **unchanged** (3722 → 3722) and detector
   observations saturate at 89. Re-verified at scale: 62,002 events at 2,000 steps with
   **0 drops**, a linear ~25 bytes/event footprint, and a forced 484-event overflow that
   correctly surfaced as `DEGRADED` on the verdict rather than as a clean negative.

## Performance

7. **The observer's own overhead budget is not met, and the reason is the benchmark
   workload.** On the 4-layer gate workload the cost is `+1.635 ms/step (+52.2 %)`
   against a `< 3 %` target; the T1 layer hooks alone add `+45.7 %`. **But the cost is
   approximately fixed per step**, so the fraction collapses as the workload grows:

   ```
   tiny   (4L,  d32)   step  4.66 ms   +0.51 ms   +11.0 %
   medium (8L,  d128)  step 18.39 ms   +0.95 ms    +5.2 %
   large  (12L, d256)  step 44.76 ms   +0.19 ms    +0.4 %
   ```

   Above roughly a **17 ms step** the observer is inside `< 3 %`. On a 3–5 ms step no
   implementation could pass without falling below 0.14 ms. The transferable number is
   the absolute cost; the projection to production step times is **NOT VERIFIED** here.
   The large-model point is *lower* than the medium one, so the curve is not monotonic —
   that is measurement noise at that scale, reported as measured rather than smoothed.
8. **The two historical cost benchmarks disagreed and one is unusable.** A paired audit
   (`tests/performance/audit_observer_cost.py`) supersedes them. `benchmark_cost_ledger.py`
   reported a **negative residual** (accounted 3.86 ms against observed 0.41 ms), which
   cannot be a valid attribution, and reported a *minimum* where the other reported a
   *median*; the two differed by 5× on the same workload. The ledger is annotated as not
   usable for a decision.
9. **Event volume is not the linear cost driver it appears to be.** Removing 20 % of
   emitted events (`param.layer.norm`, which no detector consumes) saved `0.033 ms/step`
   where `events_removed x us_per_event` predicted `0.161 ms`. Micro-cost multiplied by
   count does not equal in-situ cost. Recorded as a correction to D-042 → D-043.

## Dependencies

10. **`duckdb` is bound below 1.5, and the bound is a workaround, not a preference.**
    duckdb 1.5.6 breaks `fetch_arrow_table()` with `SystemError: <built-in function
    __import__> returned a result with an exception set`, observed on continuous
    integration with duckdb 1.5.6, pyarrow 26.0.0 and numpy 2.5.4 on Python 3.11 and
    3.12. All 23 failures were at that single call in the two storage test modules; the
    other 560-odd tests passed and `import pyarrow` worked, so the fault is in duckdb's
    Arrow bridge rather than in pyarrow.

    It was invisible locally because duckdb 1.5.x declares `Requires-Python >= 3.10`, so
    a Python 3.9 environment resolves to 1.4.5 and never sees it. **Any user on Python
    3.10 or newer with an unpinned install therefore had a broken build.** The mechanism
    inside 1.5.6 was not reproduced here — 1.5 cannot be installed on Python 3.9 — so the
    bound rests on the continuous-integration evidence above.

    A CI job named *latest dependencies (informational)* deliberately ignores every upper
    bound and is allowed to fail. While it is red, the bound is still required; when it
    turns green, the bound should be removed.

## Measurement

11. **No blind holdout has ever been run.** All quality figures are development data. The
    final-holdout lock records `status: NOT_FROZEN` and the final report records
    `NOT DONE — NOT VERIFIED`; three attempts were invalidated by defects and the holdout
    was then deliberately not burned (D-005).
12. **Not one capability is validated.** `validated_capabilities` is empty, by design.
13. **Only 4–6 layer models have been exercised.** Every number in this repository comes
    from toy transformers. Nothing has been run at a scale where the fixed per-step cost
    would be negligible.
14. **Engineered for one process.** Multi-rank is untested; ordering paths exist
    (`world_size`, `local_rank` are carried on every event) but no real two-rank incident
    has been produced.
15. **No GPU telemetry.** Every measurement is CPU/MPS on an Apple M2 Max. No DCGM, HTA,
    Flight Recorder or NNsight.
16. **No replay**, so data-causation claims are capped at `HYPOTHESIS`.

## Capabilities added in v0.1.8 — what they cannot do

17. **Two-run divergence localization reports a step and a metric, never a cause.** It
    cannot name a kernel, a tensor or a line of code. Validated on one synthetic paired
    corpus (4 layers, 3 seeds, 12 pairs): 9/9 fault pairs localized to the exact injection
    step, 3/3 clean pairs abstained. **No blind holdout.**
18. **`scan` requires like-for-like runs.** It compares a batch against itself to name the
    outlier, and refuses when the batch is varied rather than one-of-these-is-different.
    Six *healthy* runs at three seeds returned `NO_SINGLE_OUTLIER`, correctly — but a user
    expecting a name receives an abstention instead.
19. **The batch scan reports divergences symmetrically.** A divergence between two runs is
    evidence about *both*; the tool cannot say which side moved, and says so.
20. **Per-module activation health covers only threshold-free states.** `DEAD`,
    `NONFINITE` and `SATURATED` are exact. **`vanishing` and `exploding` are deliberately
    not produced**, because calling a module's activations "vanishing" needs a notion of
    the right size that this project does not have for activations. A *sampled*
    observation cannot establish `DEAD` by absence, so that case returns `UNKNOWN` rather
    than a claim.
21. **The activation probe is off by default and costs +3.3 % when on.** It is opt-in for
    that reason. Emitting its metrics every step instead of on change cost +24.7 %; the
    capability was never expensive, the naive emission strategy was.
22. **Downsampling a detector's input is not safe.** Resampling `grad.layer.norm` to every
    5th step left detection intact — the detector still fired at the exact injection step —
    but collapsed incidents from 9 to 3, because the episode engine cannot reach
    `CONFIRMED` from a sparser series. **Density is required for confirmation, not for
    detection**, and a detector-level check does not catch the difference.
