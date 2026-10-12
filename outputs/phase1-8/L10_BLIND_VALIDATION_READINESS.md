# L10 — BLIND VALIDATION READINESS

# **READY_FOR_BLIND_VALIDATION = NO**

No `FINAL_HOLDOUT_MANIFEST` and no `FINAL_HOLDOUT_LOCK` were created.

---

## 1. Gate

| # | Requirement | State |
|---|---|---|
| 1 | baseline contamination regression tests added | **PASS** — 14 tests |
| 2 | stateful protected baseline implemented | **PASS** |
| 3 | sustained fault no longer self-normalises away | **PASS** — ratio holds at 5.00 vs decaying to 1.00 |
| 4 | ≥ 680-run development corpus re-evaluated | **PASS** |
| 5 | gradient precision ≥ 0.90 | **PASS** — 0.973 |
| 6 | gradient recall ≥ 0.80 | **FAIL** — 0.468 |
| 7 | healthy gradient confirmed rate ≤ 3 % | **FAIL** — 4.2 % |
| 8 | magnitude response monotonic / justified | **PASS** — 0 violations; low range bumpy, reported |
| 9 | `EpisodeStore` wired to real pipeline | **FAIL** |
| 10 | restart E2E | **FAIL** |
| 11 | CLI persisted-data E2E | **FAIL** |
| 12 | consequence-aware scorer executed | **FAIL** — built in 1.7, not run |
| 13 | numerics precision ≥ 0.95 | **NOT VERIFIED** |
| 14 | numerics recall ≥ 0.95 | **NOT VERIFIED** |
| 15 | duplicate root incident rate = 0 | **NOT VERIFIED** |
| 16 | earliest evidence measured | **FAIL** — evaluator built, never run |
| 17 | earliest evidence precision ≥ 0.92 | **NOT VERIFIED** |
| 18 | earliest evidence coverage ≥ 0.80 | **NOT VERIFIED** |
| 19 | pre-injection incidents = 0 | **PASS** (Phase 1.6) |
| 20–23 | development episode/incident precision and recall | **NOT VERIFIED** — L8 not run |
| 24 | healthy-run FP ≤ 3 % | **FAIL** — 4.2 % at the best-F1 point |
| 25 | smoke PASS | **FAIL** — not run |
| 26 | no final holdout created | **PASS** |
| 27 | no Phase 2 component | **PASS** |
| 28 | no Agent | **PASS** |

**9 PASS / 9 FAIL / 8 NOT VERIFIED.**

## 2. What was achieved

**The feature defect that Phase 1.7 identified is fixed.**

| | Phase 1.7 | **Phase 1.8** |
|---|---|---|
| confirmed precision | 0.942 | **0.973** |
| confirmed recall | 0.279 | **0.468** (+68 % relative) |
| healthy confirmed rate | 5.6 % | **4.2 %** |
| fault p50 aggregate | 8.77 | **13.66** |
| magnitude response ×1.5→×12 | flat ≤ 0.10 | 0.23 → 0.70 |

The **flat-then-cliff pathology is gone**. The response now rises across the whole
magnitude range instead of being blind from ×1.5 to ×12 and then jumping at ×25.

**Why the fix was correct:** the old feature was `value / trailing median`, and a
sustained fault dragged the median with it. The new reference **freezes at the first
hint of deviation** (`freeze_at_ratio = 1.3`, far below any confirmation threshold),
so the fault can no longer define normal. Verified: sustained ×5 holds at ratio
**5.00** for 150 steps, where the old feature reached 1.00.

## 3. Why it is still NO — and it is no longer a feature defect

```
healthy peak : p50 2.80   p95 15.38   p99 29.60   max 73.80
fault   peak : min 1.29   p50 13.66   p95 84.06
```

**The fault minimum (1.29) is still below the healthy median (2.80).** At ×1.5–×2 the
injected perturbation is smaller than a healthy run's own gradient variation, so it
is not distinguishable — and no feature computed from a single scalar level will
change that. This is a genuine capability boundary, not a calibration failure:

```
< x12      NOT RELIABLY DISTINGUISHABLE
x12        PARTIAL (0.70)
>= x20     DEVELOPMENT-SUPPORTED (0.95-1.00)
```

Recall ≥ 0.80 requires detecting faults down to ≈×3, which the evidence does not
support.

**The secondary blockers are unstarted work, not unknowns:** `EpisodeStore` is not
wired into the pipeline, so restart and CLI cannot be verified from persisted data;
the consequence and earliest-evidence evaluators exist and are tested but have never
been run over a corpus; and the earliest-evidence metric — the product's stated core
claim — remains unmeasured across seven phases.

## 4. What would move it

1. **Distinguish ×1.5–×2 faults from healthy variation using a second signal, not a better threshold.** The level feature is at its limit. The brief's own R3 proposal — downstream loss response as *corroborating* evidence — is the natural next step: a gradient excursion that is accompanied by a loss residual change is far more likely to be a fault, and at mild magnitudes the loss is where the fault's effect accumulates.
2. **Alternatively, accept the boundary and report it.** `>= x20` with precision 0.973 is a real, useful capability. Declaring `GRADIENT_EXPLOSION` as detecting only large perturbations, and abstaining below, is honest and would pass a precision-first gate.
3. **Then L4–L8**, in order, before any holdout.

## 5. Recommendation

**Do not spend a holdout.** The system now has one working, contamination-resistant
feature and a documented capability boundary at ≈×20. A blind holdout would measure
the same boundary and confirm the same recall ceiling. The work that could change the
answer — a second corroborating signal, or an explicit narrow capability claim — is
not done, and the unstarted integration work (L4–L7) would make the holdout measure a
pipeline whose evidence cannot survive a restart.
