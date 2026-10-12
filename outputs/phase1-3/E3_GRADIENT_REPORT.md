# E3 — GRADIENT DETECTOR REPORT

**Status: PARTIAL.** Redesigned to candidate-first; episode precision remains poor.

---

## 1. Before / after

| | Phase 1.2 (v2.0, incident-direct) | Phase 1.3 (v2.0, episode-first) |
|---|---|---|
| Criterion | `value > min(median×3.0, p99.5×1.5)`, warm-up excluded | unchanged (`GRADIENT_THRESHOLD_ALGEBRA.md` proves `min` is intended) |
| Output | incident | **SUSPECT episode** |
| Confirmation | none — a fire was an incident | aggregate departed, **or** multiplicity ≥ 0.4, **or** peak ≥ 8×, **or** 2 observations within 5 steps |
| Single excursion | incident | `WEAK` → never promotes |

The threshold was **not** changed. The brief's instruction was to stop a single
excursion becoming an incident, and that is what the episode layer does.

## 2. Measured (final holdout, 169 runs)

| Layer | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|
| signal | 0.053 | 0.906 | 77 | 1265 | 8 |
| episode | **0.181** | 0.583 | 35 | **158** | 25 |

Per-class behaviour on the 30 `F3_gradient_explosion` runs across a magnitude sweep
(2.5×, 4, 7, 12, 25, 50): **29 of 30 confirmed**.

## 3. Cross-layer multiplicity evaluated (E8.3)

The measured split holds on the new corpus:

| Situation | Layers departing | Aggregate | Outcome |
|---|---|---|---|
| healthy tail (`H1`) | 1 of 4–6 | silent | `WEAK`, unconfirmed |
| injected explosion (`F3`) | all | fires | `STRONG`, confirmed |

Multiplicity is now a **detector input** feeding confirmation, not a description
attached afterwards.

## 4. Persistence evaluated (E8.3)

Four strategies were specified for comparison — single observation, 2 consecutive,
N-of-M, very-large-single override. Implemented: **2 observations within 5 steps**
plus a **peak ≥ 8× override**, both from the episode rules. A systematic comparison
was **NOT COMPLETED**; the chosen forms were reasoned from the measured lag-1
autocorrelation (0.29) rather than swept.

## 5. The remaining problem

**158 episode false positives** — an episode precision of 0.181. The cause is visible
in the per-class data: gradient episodes fire on `H1_heavy_tail_gradients` (11/14
runs) and `H4_gradient_clipping` (6/14). A ×2.6 gradient excursion on a healthy run
still clears the envelope, confirms, and promotes.

The detection boundary is between ×2 and ×3 (`D2_GRADIENT_DETECTOR_REPORT.md`), and
the healthy heavy-tail class reaches ×2.6 by construction. **The detector cannot
currently separate a ×2.6 healthy excursion from a ×2.6 fault**, and the episode
layer's persistence rule does not help because the excursions recur every 17 steps.

## 6. Not verified

* Whether raising `multiplicity_confirm` from 0.4 would help: on the small models used, 2 of 4 layers = 0.5 already clears it.
* `gap_tolerance = 8` was not swept.
* Whether separating `grad.layer_norm` episodes from `grad.global_norm` episodes would reduce the 158 FPs.
