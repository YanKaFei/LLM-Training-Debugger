# C5 — DETECTOR CALIBRATION REPORT

**Status: COMPLETE. Result: the detectors are NOT precise enough for an agent.**

This is the most important output of Phase 1.1. For the first time the project has
precision and recall numbers instead of a test count, and they are worse than the
test suite implied.

---

## 1. Corpus

| Property | Value |
|---|---|
| Total runs | **66** (requirement: ≥ 50) |
| Steps per run | 200 |
| Fault classes | **11** |
| Healthy-by-construction runs | 24 |
| Fault-injected runs | 42 |
| Calibration / holdout | 44 / 22 |
| Wall time | 95 s |
| Model | real PyTorch 4-layer transformer, real autograd |

All 66 runs are **real training loops**. No hand-built event arrays. Ground truth
is written to `ground_truth.json`; the detector layer never reads it.

| Class | Runs sounded | Detector expectations |
|---|---|---|
| `H0_healthy` | **6/6 alarmed** | none |
| `H1_heavy_tail_gradients` | **6/6 alarmed** | none |
| `H2_noisy_healthy_loss` | 4/6 alarmed | none |
| `F7_lr_discontinuity_no_failure` | 5/6 alarmed | none |
| `F1_isolated_loss_spike` | 6/6 | `loss.robust_zscore` |
| `F2_sustained_divergence` | 6/6 | `loss.trend_slope`, `loss.robust_zscore` |
| `F3_gradient_explosion` | 6/6 | `grad.robust_zscore` |
| `F4_nan_loss` | 6/6 | `numerics.finiteness` |
| `F5_inf_gradient` | 6/6 | `numerics.finiteness` |
| `F6_throughput_degradation` | 6/6 | `throughput.regression` |
| **`F10_mixed_anomaly`** | **0/6** | gradient + loss + finiteness |

## 2. System-level result

| Split | Runs | TP | FP | TN | FN | Precision | Recall | FP / 1k steps |
|---|---|---|---|---|---|---|---|---|
| calibration | 44 | 24 | 14 | 2 | 4 | 0.632 | 0.857 | 1.59 |
| holdout | 22 | 12 | 7 | 1 | 2 | 0.632 | 0.857 | 1.59 |
| **all** | **66** | **36** | **21** | **3** | **6** | **0.632** | **0.857** | **1.59** |

## 3. Per-detector result

| Detector | TP | FP | FN | Precision | Recall | Median delay | FP / 1k steps |
|---|---|---|---|---|---|---|---|
| `grad.robust_zscore` | 6 | **50** | 6 | **0.107** | 0.500 | **−32 steps** | 3.79 |
| `loss.robust_zscore` | 12 | 7 | 6 | 0.632 | 0.667 | +3 steps | 0.53 |
| `loss.trend_slope` | 1 | 2 | 5 | 0.333 | 0.167 | **−25 steps** | 0.15 |
| `numerics.finiteness` | 12 | 2 | 6 | **0.857** | 0.667 | +2 steps | 0.15 |
| `throughput.regression` | 6 | 3 | 0 | 0.667 | **1.000** | +2 steps | 0.23 |

## 4. Findings, in order of severity

**F1. `grad.robust_zscore` has precision 0.107 — it is effectively an alarm that
is always on.** It fires on **6/6 `H0_healthy` runs**, the cleanest baseline in the
corpus, and raised 50 false positives against 6 true positives. The Phase 1
`A0_healthy` scenario passed, so this was invisible until now: the scenario's
gradient noise is milder than a real run's.

**F2. Detection delay is NEGATIVE for two detectors.** `grad.robust_zscore` reports
onset 32 steps *before* the injection window; `loss.trend_slope` 25 steps before.
This is not early detection. It means the detector's real fires happen in the
pre-injection region and the *first* such fire is being reported as the onset. It
is the same defect as F1 seen from the timing side.

**F3. `F10_mixed_anomaly` is a complete false negative — 0/6.** A run with a NaN
loss **and** a gradient explosion produces **no detector fire at all**, while the
same NaN in isolation (`F4`) is caught 6/6. This is a genuine, unexplained
interaction bug and the highest-value item to investigate next.

**F4. `loss.trend_slope` recall is 0.167.** It catches 1 of 6 sustained
divergences, its stated purpose.

**F5. Every healthy class alarms.** All four healthy classes produce false alarms
(`H0` and `H1` at 6/6). The system-level precision of 0.632 means roughly one in
three alarms is wrong.

**F6. Calibration and holdout are identical (0.632 / 0.857).** Consistent with the
absence of any tuning, but it also means the split currently provides no
independent evidence — with no tuning there is nothing for it to disagree with.

## 5. Honesty constraints stated in the report

* The corpus is 66 runs; with 42 fault-injected runs the 95 % Wilson half-width on
  a recall near 0.9 is roughly ±12 points. Per-class conclusions rest on 6 runs.
* Faults are injected at a **known single step** by a known mechanism. Real
  incidents arrive through data, hardware and configuration in ways this corpus
  does not model.
* **No threshold was tuned on either split.** The Phase 1 defaults were frozen
  before this harness existed; the split exists so a future tuning pass cannot
  consume the holdout.
* **Detector errors: zero.** Every detector completed on every run, so these
  numbers are not contaminated by silent failures.
