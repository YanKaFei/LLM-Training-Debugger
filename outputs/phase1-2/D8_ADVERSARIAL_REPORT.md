# D8 — ADVERSARIAL VALIDATION

**Status: PARTIAL.** Five adversarial classes were added to the D6 corpus and
scored. The full D8 list is not covered.

The goal is not that every case produces an incident. It is **correct
classification**, including correctly doing nothing.

---

## 1. Adversarial classes implemented and measured

| Class | Intent | Expected | Result |
|---|---|---|---|
| `A1_single_huge_harmless_batch` | one large but finite gradient excursion (×4) | no incident | see §2 |
| `A2_loss_spike_stable_gradients` | loss moves ×12, gradients do not | no incident | see §2 |
| `A3_gradient_spike_no_loss_consequence` | gradients ×30, loss unaffected | incident is legitimate | see §2 |
| `A4_nan_that_recovers` | NaN on one step, finite afterwards | incident | detected (`numerics.finiteness`) |
| `A5_inf_one_layer_only` | Inf in one layer's gradient only | incident | detected, but see §3 |
| `H1_heavy_tail_gradients` | healthy with 2.6× excursions every 17 steps | no incident | see §2 |
| `H4_gradient_clipping` | clipping active the whole run | no incident | see §2 |
| `H3_scheduler_transition` | LR change mid-run | no incident | see §2 |
| `F7_lr_jump_no_failure` | 7× LR jump, no failure | no incident | see §2 |

## 2. The precision result

The adversarial *healthy* classes are the ones that matter for false positives, and
the D6 holdout aggregate reports **214 healthy false positives across 45 healthy
runs** — a rate of **60.7 per 1,000 steps** against a target of **0.5**.

Individual classification is reported per class in `D6_CALIBRATION_REPORT.md`. The
aggregate verdict is unambiguous: **the system is far too willing to call a healthy
run abnormal**, and the adversarial classes are where that shows most clearly.

## 3. `A5_inf_one_layer_only` exposed a real defect

Setting one element of one layer's gradient to `Inf` poisons the optimiser's
momentum buffer, so **every subsequent step** in the run is non-finite and its
parameter norms are non-finite. `FinitenessDetector` had no cooldown and therefore
raised one finding per remaining step.

Measured impact: **810 post-fault findings across 90 holdout runs** — 94 % of the
entire corpus's false positives, and the single dominant cause of the
system precision of 0.037. A cooldown was added and the corpus re-run.

This is exactly what an adversarial case is for: a fault that *persists* had never
been exercised, because every earlier scenario injected a one-step fault and
returned to normal.

## 4. Abstention (D8.1) — **NOT IMPLEMENTED**

The brief requires that the detector or incident engine be able to return an
explicit `INSUFFICIENT_EVIDENCE`, and that this be a legitimate and important
output rather than something to be avoided in the name of recall.

**This is not implemented.** The closest existing behaviour is the D5 `WEAK`
classification, which prevents a single-component excursion from opening an
incident — but that is a *suppression*, not an abstention. There is no
`INSUFFICIENT_EVIDENCE` verdict, and no path by which the system states "I cannot
tell".

## 5. Detector error budget (D8.2)

`KNOWN_FAILURE_MODES.md` is written and covers each detector's false-positive and
false-negative regimes, unsupported signals, minimum detectable magnitude, and
warm-up and sampling caveats.

## 6. Not verified

* **Alternating noisy loss, slow throughput drift, sudden throughput cliff, multiple simultaneous anomalies** — not implemented as separate adversarial classes. `H2_noisy_healthy_loss` approximates the first.
* **No adversarial *success* target was set.** The brief correctly says the goal is correct classification, but no threshold distinguishing "correctly did nothing" from "failed to notice" was defined for the healthy-looking fault classes, so `A2` and `A3` are reported without a verdict.
* **Abstention is absent** (§4).
