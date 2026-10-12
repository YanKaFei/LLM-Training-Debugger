# KNOWN FAILURE MODES

Per-detector false-positive and false-negative regimes, unsupported signals, and the
minimum detectable magnitude. Derived from the D6 corpus (226 real training runs),
the 30-run gradient distribution study, and the D7 paired benchmark.

This file is the constraint set a future narration Agent must be given. Every
number here is measured; where a number is absent the entry says `NOT MEASURED`
rather than offering an estimate.

---

## `grad.robust_zscore` v2.0 (asymmetric envelope)

| Property | State |
|---|---|
| Detector version | `2.0` (was `1.0`) |
| Criterion | `value > min(median(window) × 3.0, quantile(window, 0.995) × 1.5)`, window excluding warm-up |

**False-positive regimes**
* Healthy heavy-tailed gradients. Measured **1.25 FP / 1k steps** on `H1` (occasional 2.6× excursions). This is **above the 0.5 target** and is the detector's principal known weakness.
* Warm-up, if the warm-up exclusion is disabled. With a contaminated baseline the detector fired at step 60 of a run whose fault was injected at step 80.
* Layer-level series. A single layer's own envelope is tight, so a lone layer excursion clears it easily. Mitigated by D5 fusion (classified `WEAK`, cannot open an incident) but **still present in the raw output**.

**False-negative regimes**
* Perturbations below roughly **2×**. Recall is 0.75 at ×2 and 1.00 at ×3, so the detection boundary is between them and is **not sharp**.
* Bursts shorter than the cooldown may be merged.

**Unsupported signals**: per-parameter gradients, activation magnitudes, second-order statistics. The detector reads only an L2 norm series.

**Minimum detectable magnitude**: between ×2 and ×3 of the trailing median. `NOT MEASURED` more precisely than that.

**Caveats**: `LAYER_MULTIPLICITY_STRONG = 0.4` was not swept. The envelope quantile 0.995 was not swept. Detrending is computed but no longer used for firing.

---

## `numerics.finiteness` v1.0

| Property | State |
|---|---|
| Criterion | any non-finite value in the observed metric |

**False-positive regimes**
* **Persistent non-finite conditions**, before the cooldown fix. A single `Inf` gradient poisons momentum, and every subsequent step is non-finite: **810 post-fault findings across 90 runs**, 94 % of the corpus's false positives. A cooldown is now applied, so one persistent condition yields one finding.
* Non-finite values in *unrelated* metrics in the same step are counted separately rather than as one event.

**False-negative regimes**
* A non-finite value that never reaches an emitted metric. Sampling at a cadence that skips the affected step will miss it entirely; this detector is P0-exempt in the sampling policy for that reason.
* Non-finite values in tensors the adapter does not compute a norm for.

**Unsupported signals**: it reads a value, not a series. It cannot say when a condition began, only that a step was non-finite.

**Minimum detectable magnitude**: any single non-finite element. This is the only detector with an exact boundary.

**Caveats**: `CERTAIN` classification means it may open an incident alone. This is correct for measured non-finiteness and is the only detector granted that right.

---

## `loss.robust_zscore` v1.0

| Property | State |
|---|---|
| Criterion | detrended robust z-score on a trailing window, plus a relative-change test |

**False-positive regimes**
* A loss spike that is real but benign (a large batch) is indistinguishable from a fault. Measured precision **0.700** on the D6 holdout.
* Post-fault duplicates under sustained divergence (4 in the holdout).

**False-negative regimes**
* Slow level shifts below the relative-change factor.
* `NOT MEASURED` for the boundary, because the loss magnitude sweep (0.5, 2.0, 8.0) was not analysed per magnitude.

**Unsupported signals**: it reads `train.loss` only. It cannot distinguish a data-caused spike from an optimisation-caused one.

**Minimum detectable magnitude**: `NOT MEASURED`.

**Caveats**: the 0.700 precision is below the 0.80 target.

---

## `loss.trend_slope` v1.0 — **KNOWN BROKEN**

| Property | State |
|---|---|
| Measured recall | **0.000** on a magnitude sweep of sustained divergence (1.006, 1.012, 1.03) |
| Measured precision | **0.000** |
| Pre-injection false positives | 16 |

**This detector does not work.** `LOSS_DIVERGENCE` is effectively unimplemented. It
fires before the fault (16 false positives) and never inside the acceptable window.
Any consumer of this signal must treat it as non-functional. Its stated purpose,
sustained worsening, requires a persistence-gated baseline-normalised slope that
was specified in D3.3 and not built.

---

## `throughput.regression` v1.0 — **KNOWN BROKEN**

| Property | State |
|---|---|
| Measured recall | **0.000** |
| Pre-injection false positives | 9 |
| False negatives | 3 |

Nine pre-injection fires and no true positive: the detector alarms during warm-up
and misses the injected step delay. **It does not work as a fault detector.**

Its `metadata.root_cause` is `None` and `scope` is `symptom_only` by design — it
deliberately does not attribute a slowdown to GPU, storage or network, since that
is v0.2's job with HTA and DCGM. That design decision is intact; the detection
itself is not.

---

## Cross-cutting caveats

| Caveat | Detail |
|---|---|
| **Sampling** | Default cadence skips steps; any fault confined to a skipped step is invisible except for P0 events. |
| **Warm-up** | All statistical detectors require `warmup_steps = 50` plus `min_samples = 20`. Nothing is reported in the first ~50 steps, by design. |
| **Run length** | The corpus is 200 steps per run. Behaviour on 100 000-step runs is **NOT VERIFIED**; the non-stationarity measured here (23 % warm-up shift) may behave differently at length. |
| **Model scale** | 4–6 layers, d_model 32–48. Multi-billion-parameter behaviour is **NOT VERIFIED**. |
| **Duplicates** | Before the finiteness cooldown, one fault could generate hundreds of findings. Cooldown is now applied to the finiteness detector only; `grad.robust_zscore` still produced 132 post-fault findings in the holdout, largely from optimiser poisoning. |
| **Abstention** | **No `INSUFFICIENT_EVIDENCE` output exists.** The system never states that it cannot tell. |
| **Single process** | Every measurement is single-rank. Multi-rank behaviour is **NOT VERIFIED**. |
