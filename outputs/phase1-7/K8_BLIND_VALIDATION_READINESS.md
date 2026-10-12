# K8 — BLIND VALIDATION READINESS

# **READY_FOR_BLIND_VALIDATION = NO**

No `FINAL_HOLDOUT_MANIFEST` and no `FINAL_HOLDOUT_LOCK` were created.

---

## 1. Gate

| # | Requirement | State |
|---|---|---|
| 1 | K0 enlarged gradient corpus complete | **PASS** — 680 runs |
| 2 | ≥ 20 runs per retained magnitude | **PASS** — 40 per magnitude |
| 3 | boundaries re-derived after the authoritative-band fix | **PASS** |
| 4 | boundary frontier documented | **PASS** |
| 5 | uncertainty rates measured | **PASS** (Phase 1.6 figures; not re-measured here) |
| 6 | gradient confirmed precision ≥ 0.90 | **PASS** — 0.942 achievable |
| 7 | gradient confirmed recall ≥ 0.80 | **FAIL** — 0.279 at that precision; **0 qualifying points** |
| 8 | consequence-aware scoring complete | **PARTIAL** — built and tested, **not run** |
| 9 | numerics precision ≥ 0.95 | **NOT VERIFIED** |
| 10 | numerics recall ≥ 0.95 | **NOT VERIFIED** |
| 11 | duplicate root incident rate = 0 | **NOT VERIFIED** |
| 12 | earliest evidence precision ≥ 0.92 | **NOT VERIFIED** |
| 13 | earliest evidence coverage ≥ 0.80 | **NOT VERIFIED** |
| 14 | pre-injection incidents = 0 | **PASS** (Phase 1.6) |
| 15 | episode persistence wired into the pipeline | **FAIL** |
| 16 | restart works end to end | **FAIL** |
| 17 | CLI works end to end from persisted data | **FAIL** |
| 18–22 | development episode/incident precision, recall, FP rate | **NOT VERIFIED** — K6 not run |
| 23 | smoke PASS | **FAIL** — not run |
| 24 | no final holdout created | **PASS** |
| 25 | no Phase 2 capability | **PASS** |
| 26 | no Agent | **PASS** |

**9 PASS / 6 FAIL / 9 NOT VERIFIED.**

## 2. The blocking finding, and it is not a threshold problem

`K1` answered the question the brief posed — *where should the boundaries really be,
now that the band is authoritative?* — with:

```
qualifying operating points: 0
```

No `(A, B)` satisfies healthy confirmed ≤ 3 %, precision ≥ 0.90 and recall ≥ 0.80. The
best achievable point at a 3 % false-positive budget has **recall 0.163**.

The root cause is measurable and is a **feature defect, not a calibration defect**:

```
healthy peak l1_aggregate : p50  3.08   p95 17.07   p99 36.16   max 56.30
fault   peak l1_aggregate : min  1.35   p50  8.77   p95 46.73
P(CONFIRMED | magnitude):  flat at <= 0.10 from x1.5 to x12, then 1.00 at x25
```

The feature is `layer_norm[step] / trailing median of that layer`. A **sustained,
uniform** fault scaling drives the trailing median up with it, so the ratio returns to
≈ 1 and the fault becomes invisible. The cliff at ×20 is the point where the
per-step jump outruns the baseline before the baseline adapts.

**This invalidates the premise of this phase's calibration work**, which is itself the
result: re-deriving boundaries was the right question, and the answer is that the
feature cannot support the targets at any boundary.

## 3. What the next change must be

The failure mode is **baseline adaptation by the fault**, so the fix is a baseline the
fault cannot contaminate:

* a **frozen pre-fault baseline** rather than a rolling one;
* a **change-point** test on the level rather than a ratio against a moving median;
* a **rate-of-change** statistic, which a sustained shift does not erase.

This is a feature redesign, explicitly outside Phase 1.7's scope, and it must be
validated on this 680-run corpus before any boundary is re-derived again.

## 4. Also blocking

* **K4/K5**: the pipeline does not write to `EpisodeStore`, so neither restart nor CLI can be verified from persisted data.
* **K2/K3**: both evaluators are built and unit-tested but were never run over the corpus, so numerics and earliest-evidence quality remain unmeasured — including the mis-typed 0.314 figure, which is now known to be wrong in kind but not replaced by a number.
