# J8 — BLIND VALIDATION READINESS

# **READY_FOR_BLIND_VALIDATION = NO**

No `FINAL_HOLDOUT_MANIFEST` and no `FINAL_HOLDOUT_LOCK` were created, per the brief.

---

## 1. Gate

| # | Requirement | State |
|---|---|---|
| 1 | formal tests added | **PASS** |
| 2 | test count increased | **PASS** — 337 → **375** |
| 3 | gradient 3-way band covered | **PASS** |
| 4 | `l1_aggregate` covered | **PASS** — including layer-count invariance |
| 5 | consequence relations covered | **PASS** |
| 6 | standalone-NaN guarantee covered | **PASS** |
| 7 | consequence-aware scorer implemented | **FAIL** — J1 not done |
| 8 | numerics episode precision ≥ 0.95 | **FAIL** — 0.314 |
| 9 | numerics episode recall ≥ 0.95 | **PASS** — 1.000 |
| 10 | `LOSS_SPIKE` repaired **or** promotion disabled | **PASS** — disabled |
| 11 | H2 false incident ≤ 5 % if enabled | n/a — promotion off |
| 12 | episode persistence works | **PASS** |
| 13 | cross-process episode reconstruction works | **PASS** |
| 14 | relation persistence works | **PASS** |
| 15 | CLI abstention works | **PARTIAL** — implemented and unit-tested; integration **NOT VERIFIED** |
| 16 | healthy/fault UNCERTAIN rates reported | **PASS** |
| 17 | development episode precision ≥ 0.92 | **FAIL** — 0.411 |
| 18 | development episode recall ≥ 0.87 | **FAIL** — 0.353 |
| 19 | development incident precision ≥ 0.92 | **FAIL** — 0.680 |
| 20 | development incident recall ≥ 0.87 | **FAIL** — 0.405 |
| 21 | healthy-run false incident rate ≤ 3 % | **PASS** — **1.2 %** |
| 22 | earliest evidence precision ≥ 0.92 | **NOT VERIFIED** |
| 23 | pre-injection incidents = 0 | **PASS** |
| 24 | duplicate root incidents = 0 | **NOT VERIFIED** |
| 25 | smoke PASS | **FAIL** — not run (gated on J6) |
| 26 | no final holdout created | **PASS** |
| 27 | no Agent | **PASS** |
| 28 | no Phase 2 component | **PASS** |

**12 PASS / 9 FAIL / 3 NOT VERIFIED.**

## 2. The single blocking reason, and it is new

Phase 1.5 was blocked by **precision**: healthy false incidents at 28.6 %.

Phase 1.6 fixed that — **1.2 %**, and it is the first time this project has met any
detector-quality gate. But recall fell from 0.676 to **0.405** at the same time,
because the band, once it became authoritative, confirms only strong perturbations.

```
incident precision  0.680   (target >= 0.92)
incident recall     0.405   (target >= 0.87)
```

So the system is now **recall-limited**. The cause is identified and measurable: the
frozen boundaries 8.5 / 11.5 confirm only ≈×25 gradient perturbations, which the
magnitude sweep shows directly.

## 3. What would move it

1. **Re-derive the A/B boundaries now that the band is authoritative.** They were fitted when the band had no effect on confirmation, so they were never optimised against the quantity they now control. The brief froze them, correctly, to stop the phase moving the statistical target — but the J0 tests have now demonstrated the band was disconnected, which changes the situation. This is the highest-value single change.
2. **J1 consequence-aware scoring**, which is the only path to numerics precision ≥ 0.95.
3. **The smoke corpus**, then re-measure.
4. **`earliest evidence precision`, still never measured** across five phases.

## 4. Recommendation

**Do not spend a holdout.** The system is at a well-understood operating point on a
demonstrably wrong calibration: the boundaries were fitted against a code path that
did not use them. Re-deriving them on development data, re-measuring, and only then
freezing for a blind holdout, is the correct order.

The positive result should not be lost: **1.2 % healthy-run false incident rate and
0.030 false incidents per 1 000 steps are the first detector-quality gates this
project has passed.**
