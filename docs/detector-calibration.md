# Detector calibration

## What Phase 1 could not say

Phase 1 reported "256 tests pass". That is a statement about software, not about
detection. Precision, recall, detection delay and the false-alarm rate were all
unknown, which is why Phase 1.1 exists.

## The corpus

`tests/calibration/run_calibration.py` trains **real PyTorch models** — never
hand-built event arrays — across eleven fault classes:

| Class | Kind | What it probes |
|---|---|---|
| `H0_healthy` | healthy | the clean baseline |
| `H1_heavy_tail_gradients` | healthy | occasional 2.6× gradient excursions |
| `H2_noisy_healthy_loss` | healthy | ±35 % sinusoidal loss noise |
| `F7_lr_discontinuity_no_failure` | healthy | a 4× LR jump with *no* failure |
| `F1_isolated_loss_spike` | fault | one large loss step |
| `F2_sustained_divergence` | fault | compounding loss growth |
| `F3_gradient_explosion` | fault | ×35 gradients from a step onward |
| `F4_nan_loss` | fault | NaN loss |
| `F5_inf_gradient` | fault | one infinite gradient element |
| `F6_throughput_degradation` | fault | injected step delay |
| `F10_mixed_anomaly` | fault | NaN loss **and** gradient explosion |

`H1`, `H2` and `F7` are the precision tests: all three look like faults to a naive
detector and are not.

## Ground truth is separated

Faults are declared in `Fault` records and written to `ground_truth.json`. The
detector layer never sees them. A detector that could read the answer would score
perfectly.

## Train / holdout split

Seeds where `seed % 3 == 0` form the holdout; the rest are the calibration set. The
split is written to `split.json` with the method recorded.

**No threshold was tuned on either split.** The Phase 1 detector defaults were
frozen before this harness existed. The split is built now so that a future tuning
pass cannot silently consume the holdout.

## Metrics

Per detector and per split: `TP`, `FP`, `TN`, `FN`, precision, recall, F1, median
detection delay, false positives per 1 000 steps, and **Wilson score intervals** on
precision and recall.

Wilson intervals rather than a normal approximation because at this sample size the
normal approximation is wrong, and because the honest output for a small sample is
a wide interval, not a precise-looking point estimate. Where positives number fewer
than 20 the report says the interval is not meaningful rather than quoting one.

## What the numbers do not support

The corpus is synthetic faults injected at a known step by a known mechanism. Real
incidents arrive through data, hardware and configuration in ways this corpus does
not model. **These are detector-level figures, not field precision.**
