# E9 — DEVELOPMENT CORPUS REPORT

**Status: COMPLETE.** The Phase 1.2 holdout is formally demoted to development data.

---

## 1. The statistical rule (brief section 7)

The Phase 1.2 90-run holdout has been **examined, analysed, and used to choose the
Phase 1.3 design** — it is where the H2 noisy-loss alarm rate, the broken
`loss.trend_slope` and the finiteness duplicate explosion were all found. It can
therefore never again be quoted as an unbiased final estimate.

```
Phase 1.2 holdout  ->  DEVELOPMENT DATA (not a holdout)
```

Its Phase 1.2 numbers remain valid **as development numbers** and are not retracted.
They are no longer evidence of field performance.

## 2. Development corpora available

| Source | Runs | Use |
|---|---|---|
| Phase 1.2 calibration (4 layers, d_model 32) | 136 | rule design, threshold reasoning |
| Phase 1.2 holdout (6 layers, d_model 48) | 90 | demoted; failure analysis |
| Phase 1.2 adversarial classes | ~20 | red-team |
| Phase 1.3 development split (both shapes) | 169 | episode/fusion rule development |

Tuning was permitted on these. The Phase 1.3 episode rules
(`persistence_required`, `multiplicity_confirm`, `magnitude_override`,
`confirm_window`, `recovery_steps`, `gap_tolerance`) were set from the measured
distribution study and the Phase 1.2 failure analysis, **not** from the final
holdout.

## 3. Coverage achieved

Every class the brief asks for is present: stable healthy, heavy-tail healthy
gradient, noisy healthy loss, scheduler transition, gradient clipping, loss spike,
loss divergence, gradient explosion, NaN, Inf, throughput regression, mixed anomaly,
and benign LR discontinuity.

`F2_sustained_divergence` and `F6_throughput_drop` are present but their detectors
are disabled/unsupported, so those runs are expected to be missed and are counted as
false negatives.

## 4. Magnitude sweeps

| Fault | Magnitudes |
|---|---|
| `F3_gradient_explosion` | 2.5×, 4, 7, 12, 25, 50× |
| `F1_loss_spike` | 0.4, 1.5, 6.0 |
| `F2_sustained_divergence` | 1.008, 1.018, 1.04 compounding |

This gives a **detection boundary** rather than a single-magnitude demonstration.
Measured: all 30 `F3` runs across the sweep confirmed 29/30, so the ×2.5 end is
already detectable for gradient explosion — the boundary lies below the sweep.

## 5. Not verified

* Development results are **not reported as quality metrics** anywhere, by design — only the final holdout carries that claim.
* Rule tuning was reasoned from measurements rather than swept; no parameter search was run.
