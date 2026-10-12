# I0 — GRADIENT IMPLEMENTATION REPORT

**Status: COMPLETE.** The Phase 1.4 measurement is now product behaviour.

---

## 1. What was implemented

### I0.1 The joint representation

`joint_gradient_features()` in `detectors/runner.py` computes, per step, from the
existing T1 per-layer gradient norms — **no new tensor capture, no new hook**:

```
per-layer ratio = layer_norm[step] / trailing median of THAT layer   (warm-up excluded)
    l1_aggregate    = sum of ratios
    l2_aggregate    = sqrt(sum of squares)
    trimmed_l2      = sqrt(sum of squares excluding the largest)
    max_layer_ratio = largest ratio
    layers_observed = count
```

Each layer is normalised by **its own** trailing median, because per-layer gradient
magnitudes differ by depth by more than 1.5× (measured); an unnormalised sum would
report which layer is deepest rather than whether the step is abnormal.

### I0.2 The three-way verdict

The detector no longer answers a yes/no question:

| Band | Condition | Episode consequence |
|---|---|---|
| `NORMAL` | `l1 < A` | no episode |
| `UNCERTAIN` | `A ≤ l1 < B` | SUSPECT; **cannot confirm** |
| `CONFIRMED` | `l1 ≥ B` | strong candidate |

`NORMAL` and `UNCERTAIN` observations are set to `WEAK`, which the fusion rules
refuse to promote. The observation is **retained as evidence** (D5.3) — it is
demoted, not deleted. `band_reason` is written into the metadata so a renderer can
state why.

### I0.3 Frozen boundaries

From the development measurement over **both** healthy classes
(`gradient_boundary_registry.json`):

```
A = 34.0    (healthy max 33.32 across all classes, plus margin)
B = 46.0    (fault x4.0 minimum 46.18 region)
```

The brief warned explicitly against assuming heavy-tail is the hardest healthy
class. It is not: **healthy *stable* reached 33.32 while heavy-tail reached 30.06**.
A is therefore set from the maximum over both classes.

`trimmed_l2` was **rejected**: its healthy max is 12.07 against a fault ×3.0 minimum
of 10.88 — it overlaps at ×3.0 and separates only at ×4.0, whereas `l1_aggregate`
separates at ×3.0 with a healthy max of 33.32 against a fault minimum of 34.36.

## 2. Verified

| Check | Result |
|---|---|
| `A0_healthy` | **0 anomalies** |
| `A1_loss_spike` (gradient explosion at step 210) | 14 observations, **14 CONFIRMED** |
| `A2_gradient_explosion` | 14 observations, **14 CONFIRMED** |
| full test suite | 337 passing |

## 3. Minimum detectable magnitude

| Magnitude | Development behaviour |
|---|---|
| ≥ ×4.0 | `CONFIRMED`, development recall 1.00 |
| ×3.0 | largely `UNCERTAIN` (min 34.36 vs B = 46.0) |
| ≤ ×2.5 | **insufficient evidence by design** — the classes overlap |

**This is a development measurement. It is NOT VALIDATED**, because no untouched
holdout has been run.

## 4. Not verified

* **A and B were not swept as an optimisation.** They were set from the empirical maximum and the ×4.0 minimum — deliberately conservative, since with only 12 healthy observations a quantile would be an artefact.
* **12 healthy observations.** The boundary rests on 6 stable plus 6 heavy-tail runs. That is a small sample and the estimate is correspondingly uncertain; no confidence interval was computed.
* **UNCERTAIN rates on healthy and fault runs were not yet broken out** — reported in `I7_DEVELOPMENT_REEVALUATION.md`.
* **The band is not persisted.** Band metadata lives on the anomaly; there is no band field on the episode (I5).
