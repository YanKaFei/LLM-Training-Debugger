# I10 — FINAL HOLDOUT READINESS

# **READY_FOR_FINAL_HOLDOUT = NO**

No final holdout files were created. No `FINAL_HOLDOUT_MANIFEST`, no
`FINAL_HOLDOUT_LOCK`, no holdout runs. Per the phase brief, that belongs to
Phase 1.6.

---

## 1. Gate

| # | Requirement | State |
|---|---|---|
| 1 | R0 tests added and passing | **PASS** — 337 |
| 2 | gradient uncertainty band implemented | **PASS** |
| 3 | joint gradient detector implemented | **PASS** — `l1_aggregate` |
| 4 | healthy gradient incident rate ≤ 3 % development | **FAIL** — 28.6 % all healthy |
| 5 | consequence attribution implemented | **PARTIAL** — relations built, **scoring not corrected** |
| 6 | numerics precision/recall ≥ 0.95 development | **FAIL** — 0.314 |
| 7 | `LOSS_SPIKE` passes or stays non-default | **FAIL** — neither; I2 not done |
| 8 | `LOSS_DIVERGENCE` passes or stays unsupported | **PASS** — remains UNSUPPORTED |
| 9 | throughput passes or stays unsupported | **PASS** — remains UNSUPPORTED |
| 10 | episode persistence works | **FAIL** — not implemented |
| 11 | cross-process reconstruction works | **FAIL** — episodes not persisted |
| 12 | abstention visible in CLI | **FAIL** — renderer exists, CLI not wired |
| 13 | development episode precision ≥ 0.92 | **FAIL** — 0.285 |
| 14 | development episode recall ≥ 0.87 | **FAIL** — 0.553 |
| 15 | development incident precision ≥ 0.92 | **FAIL** — 0.500 |
| 16 | development incident recall ≥ 0.87 | **FAIL** — 0.676 |
| 17 | healthy-run false incident rate ≤ 3 % | **FAIL** — 28.6 % |
| 18 | earliest evidence precision ≥ 0.92 | **NOT VERIFIED** |
| 19 | pre-injection confirmed incidents = 0 | **PASS** |
| 20 | duplicate top-level incidents = 0 | **NOT VERIFIED** |
| 21 | smoke corpus passes | **FAIL** — not run |
| 22 | no Phase 2 dependency added | **PASS** |
| 23 | Agent absent | **PASS** |

**7 PASS / 13 FAIL / 3 NOT VERIFIED.**

## 2. Reasons

Three independent blockers, any one of which is sufficient:

1. **Precision is 0.285 at the episode layer against a 0.92 target.** The band improved it from 0.195 but the remaining false positives come from detectors not yet repaired: `loss.robust_zscore` on H2 noisy healthy loss (I2 not done) and numerics consequence episodes (I1.3 scoring not done).

2. **Recall regressed to 0.676.** The band's `UNCERTAIN` region suppresses promotion, so the precision gain was paid for in recall. That is the correct trade in principle, but the operating point has not been shown to be the right one, and the uncertain rates needed to prove it are **not measured**.

3. **Episode persistence and CLI abstention are absent.** These are mechanical gates that were not reached. Running a holdout now would measure a system whose evidence cannot survive a restart and whose abstentions a user cannot see.

## 3. What would move it

In order, and only these:

1. **I1.3 — correct the scorer** for linked secondary manifestations. Until then numerics precision cannot improve no matter how good the attribution is.
2. **I2 — loss spike** residual envelope plus persistence, targeting H2 ≤ 5 %.
3. **I5 / I6 — persistence and CLI**; mechanical, and required by the gate.
4. **Re-measure**, and report the `UNCERTAIN` rates on healthy and fault runs before trusting any operating point.
5. Only then run the smoke corpus (I8) and, in a later phase, the frozen holdout.

## 4. Recommendation

**Do not spend a holdout yet.** Phase 1.5 moved episode precision 0.195 → 0.285 and halved the healthy false-incident rate; that is real progress and it is nowhere near the gate. I2 and I1.3 are the two changes that address the remaining false positives directly, and both are unimplemented.
