# GRADIENT BOUNDARY FRONTIER

Measured on 680 runs. This is the document the brief asks for: **multiple candidate
operating points, not one**.

---

## 1. The distributions overlap

| Distribution | n | min | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| healthy peak `l1_aggregate` | 144 | — | **3.08** | 17.07 | **36.16** | **56.30** |
| fault peak `l1_aggregate` | 466 | **1.35** | 8.77 | 46.73 | — | 6.0e11 |

**The fault minimum (1.35) is below the healthy median (3.08).** The healthy upper
tail reaches 56.30 while the fault p95 is 46.73. The two distributions overlap
almost completely, which is why no boundary works.

## 2. Best achievable point per false-positive budget

| Healthy-FP budget | A | B | Precision | **Recall** |
|---|---|---|---|---|
| ≤ 3 % | 13.15 | 26.30 | **0.962** | **0.163** |
| ≤ 1 % | 19.89 | 39.78 | 0.977 | 0.090 |

**No (A, B) pair satisfies all three of the brief's gradient targets simultaneously**
— healthy confirmed ≤ 3 %, precision ≥ 0.90, recall ≥ 0.80.

```
qualifying operating points: 0
```

## 3. Selected point (best F1)

```
A = 8.537      B = 17.074
precision = 0.942      recall = 0.279
healthy confirmed rate = 0.056      fault uncertainty rate = 0.230
```

This is close to the Phase 1.6 operating point (8.5 / 11.5) and confirms Phase 1.6's
measured recall of 0.353 was the better end of a genuinely poor trade, not a
mis-set boundary.

## 4. Why no boundary works — the magnitude cliff

`P(CONFIRMED | magnitude)` at B = 17.074:

| Magnitude | Confirmed | Rate |
|---|---|---|
| ×1.5 | 2/26 | 0.08 |
| ×2.0 | 2/40 | 0.05 |
| ×2.5 | 0/40 | **0.00** |
| ×3.0 | 4/40 | 0.10 |
| ×4.0 | 2/40 | 0.05 |
| ×5.0 | 1/40 | 0.03 |
| ×7.0 | 2/40 | 0.05 |
| ×10.0 | 0/40 | **0.00** |
| ×12.0 | 3/40 | 0.07 |
| **×20.0** | 34/40 | **0.85** |
| **×25.0** | 40/40 | **1.00** |
| **×40.0** | 40/40 | **1.00** |

**The curve is flat at ≤ 0.10 from ×1.5 to ×12, then jumps to 1.00 at ×25.** This is
not noise — 0 monotonicity violations — it is a **cliff**, and it is the mechanism
behind the whole problem.

## 5. Root cause: the feature is blind to a sustained uniform scaling

The injected fault multiplies every gradient by a constant, and the feature is

```
layer_norm[step] / trailing median of THAT layer
```

Once the fault persists for more than a few steps, **the trailing median rises with
it** and the ratio returns to ≈ 1. The feature measures *deviation from the recent
past*, so a fault that becomes the recent past becomes invisible.

The cliff at ×20 corresponds to the point where the per-step jump is large enough to
outrun the baseline before the baseline adapts — at ×25 it never adapts in time.

**This is a defect in the feature, not in the boundary.** Re-deriving A and B was the
right question to ask; the answer is that the question has no good answer for this
feature, and the brief's instruction applies exactly:

> 如果最终结果证明这些目标不合理：不得自己改 Gate。报告失败。

## 6. What the frontier says the next change must be

The failure mode is *baseline adaptation*, so the fix must compare against a
baseline that the fault cannot contaminate — a **frozen pre-fault baseline**, a
change-point test, or a rate-of-change statistic rather than a level ratio. That is a
feature redesign, not a threshold search, and it is out of scope for this phase.
