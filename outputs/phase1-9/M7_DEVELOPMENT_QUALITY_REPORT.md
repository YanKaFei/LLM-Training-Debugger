# M7 — DEVELOPMENT QUALITY REPORT

**Status: COMPLETE.** 169 development runs, full pipeline with persistence, protected
baseline active.

---

## 1. System quality

| Layer | Phase 1.6 | **Phase 1.9** | Target |
|---|---|---|---|
| SIGNAL precision | 0.079 | 0.078 | — |
| SIGNAL recall | 0.906 | 0.906 | — |
| EPISODE precision | 0.411 | **0.358** | ≥ 0.92 |
| EPISODE recall | 0.353 | 0.341 | ≥ 0.87 |
| INCIDENT precision | 0.680 | **0.550** | ≥ 0.92 |
| INCIDENT recall | 0.405 | **0.440** | ≥ 0.87 |
| RUN precision | 0.972 | 0.956 | — |
| RUN recall | 0.412 | **0.506** | — |
| **healthy-run false incident rate** | 0.0119 | **0.0238** | ≤ 0.03 |
| **false incidents / 1k steps** | 0.0296 | **0.0592** | ≤ 0.3 |
| **pre-injection confirmed incidents** | 0 | **0** | = 0 |

**Recall improved** (incident 0.405 → 0.440, run 0.412 → 0.506) and **precision fell**
(incident 0.680 → 0.550). The protected baseline is more sensitive — it no longer
erases a sustained fault — and sensitivity costs precision, exactly as the 680-run
sweep predicted.

**Two gates PASS for the second consecutive phase:** healthy-run false incident rate
**2.38 %** (≤ 3 %) and false incidents **0.059 / 1k** (≤ 0.3). Pre-injection confirmed
incidents remain **0**.

## 2. Default system denominator (M7 / L8.1)

Only promotion-enabled capabilities are in the denominator above:

```
ENABLED + promotion ON : numerics.finiteness, grad.robust_zscore
ENABLED + promotion OFF: loss.robust_zscore (LOSS_SPIKE)
DISABLED               : loss.trend_slope, throughput.regression
```

`LOSS_SPIKE`, `LOSS_DIVERGENCE` and `THROUGHPUT_REGRESSION` therefore **do not drag
down** the default-system recall. Their state is reported separately, in the registry.

## 3. Uncertainty rates (M7.3)

| Population | NORMAL | UNCERTAIN | CONFIRMED | unbanded |
|---|---|---|---|---|
| healthy episodes | **98.0 %** | 1.4 % | 0.7 % | 0 % |
| fault episodes | 38.5 % | 3.9 % | 14.8 % | 42.8 % |

```
healthy abstention rate: 0.014
fault abstention rate:   0.039
```

The anti-cheat check passes: the precision result is not achieved by hiding healthy
runs in the abstention band. **42.8 % of fault episodes are `unbanded`** — they come
from the loss and numerics detectors, which produce no gradient band, and are
therefore outside the band accounting entirely.

## 4. Gradient quality, two denominators (M7.1)

| Figure | Value | Domain |
|---|---|---|
| full-spectrum precision | 0.550 | all injected magnitudes (incident layer) |
| full-spectrum recall | **0.440** | all injected magnitudes |
| declared-domain recall | **0.95 – 1.00** | ≥ ×20 only (from the 680-run sweep) |
| healthy confirmed rate | 2.38 % | — |
| healthy abstention rate | 1.4 % | — |
| fault abstention rate | 3.9 % | — |

Reported separately, as the brief requires, and **not** used to obtain a pass: the
Phase 1.x gates are judged on full-spectrum recall and **still FAIL**.

## 5. Numerics

```
observation precision: NOT VERIFIED
observation recall:    NOT VERIFIED
```

Per-capability numerics quality was not computed. The 0.314 figure is **retired** as
semantically invalid (M5 §4).

## 6. Verdict

```
episode precision >= 0.92   FAIL  (0.358)
episode recall    >= 0.87   FAIL  (0.341)
incident precision >= 0.92  FAIL  (0.550)
incident recall   >= 0.87   FAIL  (0.440)
healthy-run FP    <= 3%     PASS  (2.38%)
false incidents   <= 0.3/1k PASS  (0.059)
pre-injection     = 0       PASS
```

The phase's task was **not** to reach readiness (M7.3: *"不要强制 readiness"*). It was
to obtain a trustworthy complete measurement, and that is what this is.
