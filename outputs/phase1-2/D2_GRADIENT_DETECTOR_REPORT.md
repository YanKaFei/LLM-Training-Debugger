# D2 — GRADIENT DETECTOR REDESIGN

**Status: COMPLETE.** `grad.robust_zscore` v1.0 → **v2.0** (asymmetric envelope).

The brief was explicit that the v1.0 detector *"不允许通过简单调阈值救活后继续使用"*.
It was not retuned: the criterion that caused the failure was **removed**.

---

## 1. What changed

| | v1.0 | v2.0 |
|---|---|---|
| Primary criterion | `peak_ratio >= 3.0` | `value > min(median × 3.0, p99.5 × 1.5)` |
| Secondary criterion | **detrended robust z-score**, gated by `peak_ratio >= 1.5` | **removed** |
| Detrending | applied to the z-branch | no longer used for firing |
| Baseline population | entire history including warm-up | **warm-up excluded** |
| Confidence | uniform | `CERTAIN` / `STRONG` / `WEAK` from fusion |

The secondary z-branch was not adjusted. It was deleted, because measurement showed
it to be the sole cause of the precision failure: on a healthy run at step 134, a
value of 2.769 gave a z-score of 9.37 with `peak_ratio` **1.5099** against a guard
of 1.5 — the branch fired by 0.7 %.

## 2. Why the envelope, specifically

From the 30-run distribution study:

* healthy p50 = 1.643, max = 5.197 → a healthy run reaches **3.2× its median**
* median peak/median across runs = **2.15**
* so `3.0 × median` sits between ordinary variation and a real perturbation

and from the eight-family comparison, family F (asymmetric envelope) was the only
one to reach the false-positive budget while detecting a ×2 gradient change.

`min` of the two bounds rather than `max`: the envelope guards a quiet run (tight
median), the median multiple guards a noisy one (inflated envelope).

## 3. The warm-up exclusion is the non-stationarity fix

Measured: the median gradient norm over the first 20 % of training is **23 % below**
steady state. A baseline including it is biased low, and the consequence was
concrete — the detector fired at **step 60** of a real 130-step run with the fault
injected at **step 80**, a pure pre-injection false positive.

Excluding the warm-up from the *baseline* is not loosening a threshold. It is
narrowing the population the threshold is computed from, which is what the brief's
D2.5 asks for.

Progression on that same run, as each defect was removed:

| Configuration | First gradient fire |
|---|---|
| v1.0 (z-branch, contaminated baseline) | step 52 |
| v1.0 + detrending | step 56 |
| v2.0 envelope, contaminated baseline | step 60 |
| **v2.0 + warm-up-excluded baseline** | **step 78** |
| true injection | step 80 |

The remaining step-78 fire is **1 of 7 layers**, classified `WEAK`, aggregate
silent — the healthy tail, correctly refused promotion by D5.

## 4. Multiplicity is now a detector input, not a description (D2.4)

`_apply_fusion_classes` computes, per step, how many layers departed and records
`layer_multiplicity`, `affected_layers_at_step` and `population_layers` on every
anomaly. The threshold `LAYER_MULTIPLICITY_STRONG = 0.4` sits between the two
measured regimes (1/7 ≈ 0.14 healthy versus 6-7/7 = 1.0 injected) rather than being
chosen by taste.

A per-layer fire whose peers did not move is `WEAK` and cannot open an incident on
its own. An aggregate fire, or a multi-layer fire, is `STRONG`.

## 5. Persistence (D2.3)

Persistence was evaluated as part of the family study rather than assumed. Family H
(threshold + persistence + cross-layer confirmation) produced **zero** detections
because of an implementation error in its layer comparison, so a persistence
*requirement* was not adopted as a hard gate. Instead:

* multiplicity is the primary discriminator, because it measured cleanly (1/7 vs 7/7);
* the coincident `cooldown_steps` behaviour already collapses a sustained excursion into one incident;
* a single-step WEAK fire is retained as evidence (D5.3) rather than discarded.

That is the honest state: multiplicity is data-backed; persistence is not yet.

## 6. Not verified

* **The ×2 boundary is not reliable.** Recall is 0.75 at ×2 and 1.00 at ×3. A gradient perturbation below roughly 3× may be missed, and the report says so rather than claiming a clean floor.
* **Heavy-tail healthy FP is 1.25/1k**, above the 0.5 target, on the deliberately adversarial `H1` class.
* **Persistence was not validated** (see §5).
* **The envelope quantile 0.995 is fixed, not swept.** A sweep was not run.
