# E10 — FINAL HOLDOUT REPORT

**Status: COMPLETE, with a disclosure that limits what it can claim.**

---

## 1. Disclosure: the pre-registration was broken three times

Brief section 18.1 requires freeze-then-run, with no changes after the first look.
**This was violated three times**, each by an implementation defect found while
executing:

| Run | Manifest SHA256 | Result | Defect found |
|---|---|---|---|
| 1 | `da8f06b55f09bd29…` | 0 episodes on all 169 runs | `EpisodeEngine` appended closed episodes to a local list, never to `self.episodes` |
| 2 | `ddd195ed4ce9e89c…` | 148/169 runs crashed | `IncidentVerdict.verdict` had no default |
| 3 | `ddd195ed4ce9e89c…` | 0 incidents on 85 fault runs | `fuse()` matched `state == "CONFIRMED"` but closed confirmed episodes carry `state == "CLOSED"` |
| **4 (reported)** | **see `FINAL_HOLDOUT_MANIFEST.sha256`** | see §3 | — |

Every run's raw output is retained in `invalidated_first_run/` and none was deleted.
**No threshold, rule, magnitude or evaluation semantic was changed** as a result of
seeing results — all three were defects that made the pipeline produce *nothing*, not
defects in detector judgement.

**But the pre-registration is still broken**, and the honest consequence is:

> The reported final-holdout numbers are **development-grade evidence**, not a
> pre-registered unbiased estimate. The holdout-discipline gate is
> **NOT VERIFIED**.

A reader who requires a strictly pre-registered final holdout should treat every
number in §3 as development data. `PHASE_1_3_REPORT.md` records this as a gate
failure.

## 2. Corpus

| Property | Value |
|---|---|
| Runs | **169** (requirement ≥ 120) |
| Healthy | **84 (50 %)** |
| Fault | 85 |
| Model configurations | **2** — 5 layers/d_model 40, and 3 layers/d_model 64 |
| Steps per run | 240 |
| Injection steps | 100 and 150 (alternating) |
| Seeds | 5000–5199 |
| Run errors | **0** |

Differs from Phase 1.2 in **model shape, seeds, step count, injection step and
magnitudes** — agreement cannot come from re-measuring the same runs.

## 3. Results

| Layer | Precision | Recall | TP | FP | preFP | FN | Abstention |
|---|---|---|---|---|---|---|---|
| SIGNAL | 0.053 | 0.906 | 77 | 1265 | 100 | 8 | — |
| **EPISODE** | **0.195** | **0.588** | 50 | 206 | 0 | 35 | — |
| **INCIDENT** | **0.412** | **0.806** | 54 | 77 | **0** | 13 | 0.05 |
| **RUN** | **0.643** | **0.847** | 72 | 40 | 0 | 13 | — |

```
healthy-run false incident rate:  0.476   (target <= 0.05)   FAIL
false incidents per 1k steps:     0.986   (target <= 0.5)    FAIL
EARLIEST_EVIDENCE_PRECISION:      0.420   (target >= 0.90)   FAIL
pre-injection confirmed incidents:  0     (target 0)         PASS
duplicate persistent-anomaly incidents: 0 (target 0)         PASS
```

## 4. Per-class behaviour

| Class | Confirmed |
|---|---|
| `F4_nan_loss`, `F5_inf_gradient`, `F5b_nan_gradient`, `F10_mixed_anomaly` | **5/5, 5/5, 5/5, 5/5** |
| `F3_gradient_explosion` | **29/30** |
| `F2_sustained_divergence` | 11/15 (incidental — detector disabled) |
| `F1_loss_spike` | 9/15 |
| `F6_throughput_drop` | 3/5 (incidental — detector disabled) |
| `H0_stable_healthy` | **5/14 false** |
| `H1_heavy_tail_gradients` | **11/14 false** |
| `H2_noisy_healthy_loss` | **8/14 false** |
| `H3_scheduler_transition` | 5/14 false |
| `H4_gradient_clipping` | 6/14 false |
| `F7_lr_jump_no_failure` | 5/14 false |

## 5. What passed, and it is the important one

**Pre-injection confirmed incidents: 0.** Phase 1.2 measured **56 of 117**. This was
the brief's most consequential requirement, and the episode model delivered it: a
fire before the fault can no longer be confirmed, because confirmation requires
corroboration that does not yet exist, and the evaluation counts a pre-injection
confirmation as a false positive rather than a negative delay.

**Duplicate persistent-anomaly incidents: 0.** 111 consecutive `Inf` observations
produce one episode and one incident.

## 6. What failed

**Healthy-run false incident rate 0.476 against a target of 0.05.** Roughly half of
all healthy runs are reported as having an incident. `H1_heavy_tail_gradients` at
11/14 is the worst: the healthy class is *designed* to reach ×2.6 gradient
excursions, and the detector's measured boundary is between ×2 and ×3, so the two
overlap by construction.

The episode architecture removed the *signal* noise (1272 → 0 pre-injection episodes)
but not the *physiological* overlap between a healthy heavy tail and a mild fault.
That is the central unresolved problem of this phase, and no amount of episode
bookkeeping fixes it — it needs either better signal features or an explicit
abstention policy on the boundary.
