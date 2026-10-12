# R2 — JOINT GRADIENT REPORT

**Status: ANALYSIS COMPLETE, IMPLEMENTATION NOT DONE.**

---

## 1. What was asked

R2 requires the gradient problem to stop being treated as univariate: build a joint
feature representation from signals already collected (no new tensor capture),
measure separability, and introduce an explicit uncertainty band rather than forcing
a binary choice.

## 2. What was delivered

**R2.1 / R2.2 / R2.3 — the measurement.** Nine candidate deterministic statistics
were computed per step from the existing T1 per-layer gradient norms and the global
norm, each layer normalised by its own trailing baseline:

`global_norm_ratio` · `mean_layer_ratio` · `max_layer_ratio` · `l2_aggregate` ·
`l1_aggregate` · `fraction_above_1p5` · `fraction_above_2` · `top_k_mass` ·
`max_over_median` · `trimmed_l2`

They were evaluated on 48 real training runs across healthy heavy-tail, healthy
stable and six fault magnitudes, by threshold-independent AUC and by class overlap.
Results are in `GRADIENT_BOUNDARY_ANALYSIS.md`.

**The decisive result:** the univariate `global_norm_ratio` scores **AUC 0.25 at
×1.5** — worse than chance — while `l1_aggregate` reaches **0.96 at ×2.5**. The
gradient problem is measurably not univariate.

No learned model was used, and no new tensor capture was introduced, as required.

## 3. What was NOT delivered

| Requirement | Status |
|---|---|
| R2.1 joint feature representation wired into the detector | **NOT DONE** |
| R2.2 per-layer joint geometry as a detector input | **NOT DONE** |
| R2.3 separability measured | **DONE** |
| R2.4 explicit uncertainty band (`NORMAL` / `UNCERTAIN` / `CONFIRMED`) | **NOT DONE** |
| R2.4 A/B boundaries frozen from development data | **NOT DONE** |

The measurement that justifies the change exists; the change does not.

## 4. Consequence for Phase 1.4

The gradient detector is **unchanged from Phase 1.3**, so its measured episode
precision remains **0.181** and the `H1_heavy_tail_gradients` alarm rate remains
**11 of 14**.

Phase 1.4's R9.1 target — `H1` healthy confirmed-incident rate ≤ 5 % — is therefore
**NOT MET**, and cannot be met without the R2.4 implementation.

## 5. Why this is reported rather than approximated

The brief is explicit that a cleverer threshold is not the answer when evidence
overlaps:

> 不能通过复杂规则假装可分

Implementing the uncertainty band requires the A/B boundaries to be frozen from
development data and then re-measured — a change to detector, episode rules and
evaluation together. Doing it partially, or reporting the AUC as if it were a
detector improvement, would be exactly the substitution of analysis for
implementation that this project has repeatedly had to retract.
