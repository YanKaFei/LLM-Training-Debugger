# A2 — S1 HEALTHY-RUN FALSE POSITIVE: ROOT CAUSE

**Status: DIAGNOSED. Fix requires a measured design decision — not applied blind.**

---

## 1. Reproduction

Exact scenario: healthy training, seed 61001, **no injection**, 200 steps, 4 layers.

```
anomalies by detector:  grad.robust_zscore x9   (nothing else fired)

step  81  layer=1     WEAK  band=NORMAL     value 0.1535  thr 0.123
step 131  layer=1     WEAK  band=NORMAL     value 0.1376  thr 0.132
step 152  layer=0     WEAK  band=NORMAL     value 0.2185  thr 0.176
step 167  layer=2     WEAK  band=NORMAL     value 0.3312  thr 0.246
step 179  layer=None  STRONG band=CONFIRMED value 2.5654  thr 1.652   <-- the incident
step 179  layer=3..5  WEAK  band=CONFIRMED
```

Episodes:

```
(GRADIENT_EXPLOSION, CLOSED, NOT_ANOMALY,      onset 81,  WEAK)   correctly dismissed
(GRADIENT_EXPLOSION, CLOSED, NOT_ANOMALY,      onset 131, WEAK)   correctly dismissed
(GRADIENT_EXPLOSION, CLOSED, NOT_ANOMALY,      onset 152, WEAK)   correctly dismissed
(GRADIENT_EXPLOSION, CLOSED, NOT_ANOMALY,      onset 167, WEAK)   correctly dismissed
(GRADIENT_EXPLOSION, CLOSED, CONFIRMED_ANOMALY, onset 179, STRONG) -> CONFIRMED_INCIDENT
```

The incident-opening anomaly is a **simultaneous multi-layer excursion at step 179**,
classified `CONFIRMED` because the joint aggregate crossed 11.5.

## 2. Root cause: an early false suspicion freezes the reference for the whole run

The protected baseline (Phase 1.8) freezes at the **first** hint of deviation
(`freeze_at_ratio = 1.3`) and resumes only after `resume_after_steps = 40` consecutive
quiet steps.

On this healthy run the benign excursions are **50, 21, 15 and 12 steps apart**:

```
81 -> 131 : 49 quiet steps  < 40?  NO  (just short)
131 -> 152: 20 quiet steps  < 40
152 -> 167: 15 quiet steps  < 40
167 -> 179: 12 quiet steps  < 40
```

**The baseline therefore never resumes.** It stays frozen at the level captured around
step 81 — early in training — while the model's genuine gradient norm grows as it fits.

By step 179 the *legitimate* late-training level is measured against an artificially
**low, early-training reference**, the aggregate exceeds 11.5, the band says `CONFIRMED`,
and a healthy run is reported as a gradient explosion.

**This is a direct design consequence of freeze-on-suspicion**, not a threshold mis-set:
the mechanism that fixed self-normalisation (Phase 1.8) introduces the opposite failure
when benign excursions recur more often than `resume_after_steps`.

## 3. Why no fix was applied in this pass

The obvious changes each carry a measurable cost, and choosing between them without
measuring would be over-fitting to one run:

| Candidate fix | Risk |
|---|---|
| lower `resume_after_steps` | shorter quiet requirement → a **sustained fault** can unfreeze and self-normalise again, re-opening the defect Phase 1.8 fixed |
| slow leak while frozen (the brief's "updates at a drastically reduced rate") | leak of 0.02/step absorbs a sustained ×5 fault to ~1.6× within 100 steps — **self-normalisation returns**; a smaller leak trades the other way |
| resume immediately on `SUSPECT → NORMAL` closure | principled — a dismissed excursion is evidence the baseline was right — but it changes what the baseline learns about *all* dismissed observations, and must be re-measured against the 680-run corpus |

The third is the most promising and is the recommended next action, but it requires a
full re-measurement of gradient precision and recall, not a one-line change verified
against a single run.

**No change was made.** The failing case is preserved as a regression target.
