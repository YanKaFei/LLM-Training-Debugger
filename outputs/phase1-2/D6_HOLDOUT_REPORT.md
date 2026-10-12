# D6 — HOLDOUT REPORT

The holdout is **independently generated**: it uses a different model shape
(6 layers, d_model 48) from the calibration split (4 layers, d_model 32),
different seeds (4000-4039) and different fault magnitudes.

**No threshold was fitted on this split**, and no threshold was changed after
seeing any holdout result. The v2.0 envelope parameters came from a separate
30-run distribution study (seeds 2000-2029) disjoint from both splits.

## Holdout results

| Metric | Value | Gate | Verdict |
|---|---|---|---|
| Runs | 90 | — | — |
| Injected / healthy | 45 / 45 | — | — |
| True positives | 42 | — | — |
| False negatives | 3 | — | — |
| Pre-injection false positives | 25 | — | — |
| Post-fault false positives | 67 | — | — |
| Healthy false positives | 214 | — | — |
| Post-fault duplicates | 166 | — | — |
| Precision | 0.121 | >= 0.90 | FAIL |
| Recall | 0.933 | >= 0.85 | PASS |
| F1 | 0.214 | — | — |
| FP / 1k steps | 17.00 | <= 0.5 | FAIL |
| Median detection delay | 0 | — | — |
| Pre-injection incident rate | 0.422 | 0 | FAIL |

## Per-detector (holdout)

| Detector | TP | pre-FP | post-FP | FN | Precision | Recall | Verdict |
|---|---|---|---|---|---|---|---|
| `grad.robust_zscore` | 32 | 20 | 132 | 9 | 0.063 | 0.780 | FAIL |
| `loss.robust_zscore` | 21 | 0 | 4 | 0 | 0.700 | 1.000 | FAIL |
| `loss.trend_slope` | 0 | 16 | 0 | 16 | 0.000 | 0.000 | FAIL |
| `numerics.finiteness` | 22 | 0 | 24 | 0 | 0.478 | 1.000 | FAIL |
| `throughput.regression` | 1 | 9 | 0 | 3 | 0.100 | 0.250 | FAIL |

## Incident-level (cooldown-deduped)

| Metric | Value | Target | Verdict |
|---|---|---|---|
| Unique incidents raised | 117 | — | — |
| Incidents matching an acceptable window | 32 | — | — |
| Incident precision | 0.274 | — | — |
| Pre-injection incidents | 56 of 117 | 0 | **FAIL** |
| **EARLIEST_EVIDENCE_PRECISION** | **0.338** | >= 0.90 | **FAIL** |

`EARLIEST_EVIDENCE_PRECISION` is the fraction of runs whose **first**
incident-opening fire was graded a true positive. At 0.338 the system names a
defensible earliest abnormal event in roughly one run out of three.

## Verdict

**Four of the five holdout gates FAIL.** Recall passes. The machine-readable
form of this table is `detector_confusion_matrix.json`.
