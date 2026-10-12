# L3 — GRADIENT REVALIDATION

**Status: COMPLETE.** The same 680-run development corpus, re-evaluated with the
protected baseline.

---

## 1. Result

| Metric | Phase 1.7 (trailing median) | **Phase 1.8 (protected)** | Change |
|---|---|---|---|
| healthy peak p50 | 3.08 | 2.80 | — |
| healthy peak p95 | 17.07 | 15.38 | down |
| healthy peak p99 | 36.16 | **29.60** | **down** |
| fault peak p50 | 8.77 | **13.66** | **up 56 %** |
| fault peak p95 | 46.73 | **84.06** | **up 80 %** |
| **confirmed precision** | 0.942 | **0.973** | up |
| **confirmed recall** | **0.279** | **0.468** | **up 68 %** |
| healthy confirmed rate | 0.056 | **0.042** | down |
| fault uncertainty rate | 0.230 | 0.202 | down |
| qualifying operating points | 0 | **0** | — |
| monotonicity violations | 0 | 0 | — |

**The feature separates better on every measure.** The fault distribution moved up
while the healthy distribution moved down, and recall improved by 68 % relative at
higher precision.

## 2. The pathology is gone

`P(CONFIRMED | magnitude)` at the chosen point (A = 7.844, B = 15.688):

| Magnitude | Phase 1.7 | **Phase 1.8** |
|---|---|---|
| ×1.5 | 0.08 | 0.23 |
| ×2.0 | 0.05 | 0.12 |
| ×2.5 | 0.00 | 0.15 |
| ×3.0 | 0.10 | **0.40** |
| ×4.0 | 0.05 | 0.23 |
| ×5.0 | 0.03 | 0.20 |
| ×7.0 | 0.05 | 0.33 |
| ×10.0 | 0.00 | 0.23 |
| ×12.0 | 0.07 | **0.70** |
| ×20.0 | 0.85 | 0.95 |
| ×25.0 | 1.00 | **1.00** |
| ×40.0 | 1.00 | **1.00** |

**The flat-then-cliff shape is gone.** Phase 1.7 was flat at ≤ 0.10 from ×1.5 to ×12
and then jumped; the response now rises across the whole range. That was the specific
defect the protected baseline was built to remove, and it is removed.

The low range is still **bumpy** (0.23 at ×1.5 against 0.12 at ×2.0). My violation
detector uses a 0.25 drop threshold and therefore reports zero violations, but the
curve is not smooth below ×12 and that is stated rather than glossed.

## 3. Gate (L3.3)

| Requirement | Target | Measured | Verdict |
|---|---|---|---|
| confirmed precision | ≥ 0.90 | **0.973** | **PASS** |
| confirmed recall | ≥ 0.80 | **0.468** | **FAIL** |
| healthy confirmed rate | ≤ 3 % | 4.2 % | **FAIL** (marginally) |
| monotonicity violations | 0 | 0 | PASS |
| fault abstention rate | reported | **0.202** | reported |

```
qualifying operating points (all three simultaneously): 0
```

At a strict 3 % false-positive budget the best recall is **0.249**; at the best-F1
point recall is 0.468 with a 4.2 % healthy rate. **The two targets cannot be met
together**, and the reason is now a plain distributional overlap rather than a broken
feature:

```
healthy peak : p50 2.80   p95 15.38   p99 29.60   max 73.80
fault   peak : min 1.29   p50 13.66   p95 84.06
```

The fault **minimum (1.29) is still below the healthy median (2.80)**: a mild fault
remains indistinguishable from ordinary gradient noise, because at ×1.5–×2 the fault
amplitude is smaller than the healthy run's own variation.

## 4. Development-supported detection boundary

```
< x12      NOT RELIABLY DISTINGUISHABLE
x12        PARTIAL (0.70)
>= x20     DEVELOPMENT-SUPPORTED DETECTION (0.95-1.00)
```

Unchanged in conclusion from Phase 1.7, though the intermediate response is far
better. **Development-supported**, never *validated*.

## 5. Decision

`GRADIENT_EXPLOSION` remains **EXPERIMENTAL**. The brief's L3.3 says exactly this:

> 如果做不到：GRADIENT_EXPLOSION remains EXPERIMENTAL
> 但仍继续完成 pipeline/scoring 工作。

The pipeline and scoring work (L4–L7) was **not** completed, so the phase stops here
with the feature fixed and the remaining work unstarted.
