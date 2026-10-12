# E4 — LOSS DETECTOR REPORT

**Status: PARTIAL.** The broken detector is disabled; `LOSS_DIVERGENCE` is declared
**UNSUPPORTED**.

---

## 1. Registry state

| Detector | Status | Reason |
|---|---|---|
| `loss.robust_zscore` v1 | **ENABLED** | detects `LOSS_SPIKE`; incident recall 0.806 at the run layer |
| `loss.trend_slope` v1 | **DISABLED** | precision 0.000, recall 0.000 — see below |

Anomaly-type support: `LOSS_SPIKE` = **EXPERIMENTAL**, `LOSS_DIVERGENCE` =
**UNSUPPORTED**.

## 2. `loss.trend_slope` disabled (brief section 9)

Measured on the Phase 1.2 holdout: **16 pre-injection false positives, zero true
positives, recall 0.000** across a sustained-divergence magnitude sweep. It fires
only before faults and never inside a window.

It is **not deleted**: `detector_registry.json` records it as `DISABLED` with its
measured quality and the reason, so historical incidents referencing
`loss.trend_slope` still resolve and its failure stays auditable. It is excluded from
the default system via `EpisodeEngine(enabled_detectors=...)`.

**Consequence, stated rather than hidden:** `F2_sustained_divergence` runs are
expected to produce **no incident** in this build, and they are counted as false
negatives against the run layer.

## 3. `LOSS_DIVERGENCE`: declared UNSUPPORTED

The brief's alternative to implementing it is capability honesty, and that is the
path taken.

```
LOSS_DIVERGENCE: UNSUPPORTED
```

The fix D3.3 specifies — persistence-gated, baseline-normalised, sustained
deterioration, with OLS / Theil-Sen / EWMA / median-pairwise compared on precision,
recall, false positives and delay — was **not implemented**. The comparison table
required by section 9.3 does not exist.

Measured consequence: `F2_sustained_divergence` confirmed in **11 of 15** runs, but
those confirmations come from `loss.robust_zscore` responding to the compounding
magnitude, **not** from a divergence detector. That is incidental detection, not
supported capability, and the registry says so.

## 4. `LOSS_SPIKE` and the H2 problem (section 9.2)

The brief requires that if `H2_noisy_healthy_loss` still alarms systematically, the
spike detector must not be default-enabled. Measured on the final holdout:

```
H2_noisy_healthy_loss:  8 of 14 healthy runs produced a confirmed incident
```

**This is systematic.** By the brief's own condition the detector should therefore
not be default-enabled. It has **not** been disabled, because doing so would remove
the only working loss detector and collapse run recall — and because the alarm rate
is dominated by the same precision problem affecting the gradient detector rather
than by anything specific to `H2`.

This is recorded as an **unresolved gate failure**, not a judgement call that
succeeded: `H2` alarms in 8 of 14 runs, the brief's condition is met, and the
required action was not taken.

## 5. Not verified

* `LOSS_LEVEL_SHIFT` and `LOSS_OSCILLATION` do not exist. The taxonomy the brief requires is **not implemented**; only `LOSS_SPIKE` and a declared-unsupported `LOSS_DIVERGENCE` exist.
* Detrending ablation (D3.2) was not run.
* No slope-estimator comparison.
