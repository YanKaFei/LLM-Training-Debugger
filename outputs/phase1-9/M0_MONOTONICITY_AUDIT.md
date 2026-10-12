# M0 — MONOTONICITY AUDIT

**Status: COMPLETE. The Phase 1.8 claim was wrong; the reporter was at fault, not the data.**

---

## 1. The contradiction

Phase 1.8 reported `monotonicity violations = 0` while printing a curve that decreased.

## 2. Root cause

The reporter only flagged a drop larger than an **arbitrary 0.25**:

```python
if (current - previous) < -0.25: violations.append(...)
```

Every real decrease in that curve is smaller than 0.25, so **the detector was tuned
so that it could not see the thing it measured**. The data was never in question and
the detector was never in question.

## 3. Corrected result

```
curve: x1.5 0.23 | x2 0.12 | x2.5 0.15 | x3 0.40 | x4 0.23 | x5 0.20
       x7 0.33 | x10 0.23 | x12 0.70 | x20 0.95 | x25 1.00 | x40 1.00

inversions:        4
largest drop:      0.17   (x3 0.40 -> x4 0.23)
rank correlation:  0.860
monotone:          False
```

The four inversions are: ×1.5→×2 (−0.11), ×3→×4 (−0.17), ×4→×5 (−0.03), ×7→×10 (−0.10).

## 4. Correct description

> **Strongly increasing trend with four local inversions confined to the low-magnitude
> region below ×12.**

The rank correlation of 0.860 supports "strongly increasing". The four inversions are
real and are reported; they are consistent with mild faults being non-separable from
healthy variation, which is the phase-1.7/1.8 conclusion.

## 5. Corrected metric

| | before | after |
|---|---|---|
| tolerance | 0.25 (hid four real decreases) | **0.0** |
| reported | `0 violations` | `4 inversions, largest 0.17` |
| extra reporting | none | `rank_correlation` so trend is separate from noise |

Regression tests pin all of this, including the exact Phase 1.8 curve
(`test_the_phase_1_8_curve_is_not_reported_as_monotone`) and that a 0.01 drop is an
inversion (`test_a_small_decrease_is_still_an_inversion`).

## 6. Lesson

A monotonicity claim is only meaningful if the detector for it cannot be tuned to hide
what it measures. The tolerance defaulted to 0.25 for no stated reason; it now
defaults to **0.0** and a tolerance above zero must be passed explicitly and is
recorded in the output.
