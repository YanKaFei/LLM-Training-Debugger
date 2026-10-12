# K0 — GRADIENT CORPUS REPORT

**Status: COMPLETE.** 680 real training runs.

| Property | Value |
|---|---|
| Runs collected | **680** |
| Runs where the marginal detector fired | 610 |
| Healthy runs | **200** (144 with a detector fire) |
| Healthy kinds | stable, heavy_tail, scheduler, clipping, noisy — 20 each per config |
| Fault magnitudes | **12** (×1.5, 2, 2.5, 3, 4, 5, 7, 10, 12, 20, 25, 40) |
| Runs per magnitude | **20 per config = 40** (brief floor was 20) |
| Model configurations | **2** — 5-layer/d_model 40 style depth variants: 4L/d32 and 6L/d48 |
| Steps per run | 200, injection at step 110 |
| Wall time | 1103 s |

The brief required ≥ 20 runs per retained magnitude and ≥ 100 healthy runs. Both are
exceeded, and every magnitude survives: the ×4 / ×7 / ×12 non-monotonicity that
Phase 1.6 saw at 5 runs per magnitude was a small-sample artefact, and at 40 runs per
magnitude the curve is clean (0 monotonicity violations).
