# PHASE 1.2 REPORT

```
PHASE 1.2 STATUS:              PARTIAL
PHASE 1 CLOSURE STATUS:        STILL PARTIAL
NARRATION_AGENT_READY:         NO
```

`PHASE_1_REPORT.md` and `outputs/phase1-1/` are **unmodified**. No gate was relaxed,
redefined, or measured against a substitute metric.

---

## 1. Release gate

| # | Gate item | Status | Evidence |
|---|---|---|---|
| 1 | H0 gradient false-positive pathology explained | **PASS** | the secondary z-branch fired at `peak_ratio = 1.5099` against a guard of 1.5 — by 0.7 % |
| 2 | gradient detector redesigned | **PASS** | v1.0 → **v2.0**, asymmetric envelope; the offending criterion deleted, not retuned |
| 3 | F10 0/6 root cause found | **PASS** | the scenario never ran; a Python-float NaN crashed `record_loss` and the harness scored the crash as a miss |
| 4 | F10 fixed | **PASS** | `numerics.finiteness` fires on F10 **3/3** traced seeds and 3/3 in the holdout |
| 5 | detector isolation invariant tested | **PASS** | D4.4 pairwise matrix |
| 6 | negative-delay metric semantics fixed | **PASS** | evaluation layer; pre-injection fire → FP with `delay = None` |
| 7 | no pre-injection incident on holdout faults | **FAIL** | **56 of 117** incidents; rate 0.422 |
| 8 | calibration / holdout independently generated | **PASS** | holdout is 6 layers / d_model 48 vs calibration 4 / 32; different seeds and magnitudes |
| 9 | thresholds fitted only on calibration | **PASS** | fitted on a **third** disjoint set (seeds 2000-2029); neither split consulted |
| 10 | >= 100 real training runs | **PASS** | **226** runs (136 calibration, 90 holdout) |
| 11 | fault magnitude sweep included | **PASS** | gradient 2/3/5/10/20/40×, loss 0.5/2/8, divergence 1.006/1.012/1.03 |
| 12 | system precision >= 0.90 | **FAIL** | **0.121** |
| 13 | system recall >= 0.85 | **PASS** | **0.933** |
| 14 | major detector precision >= 0.80 | **FAIL** | best is `loss.robust_zscore` at 0.700; `trend_slope` 0.000; `throughput` 0.100 |
| 15 | `numerics.finiteness` precision >= 0.95 | **FAIL** | **0.478** (improved from 0.026) |
| 16 | FP / 1k steps <= 0.5 | **FAIL** | **17.0** |
| 17 | `EARLIEST_EVIDENCE_PRECISION` >= 0.90 | **FAIL** | **0.338** |
| 18 | adversarial healthy cases tested | **PASS** | 9 adversarial/injected-healthy classes in the corpus |
| 19 | abstention supported | **FAIL** | no `INSUFFICIENT_EVIDENCE` output exists |
| 20 | performance benchmark uses paired/interleaved design | **PASS** | same-batch pairing, order randomised, frozen model |
| 21 | performance CI reported | **PASS** | bootstrap 95 % CI on paired differences |
| 22 | no negative physically-impossible overhead claim | **PASS** | median reported **only** where the CI excludes zero; medium is declared unresolvable |
| 23 | original Phase 1 overhead gate unchanged | **PASS** | core < 3 % / T1 < 8 %; both **FAIL** and reported as failing |
| 24 | no Phase 2 dependency introduced | **PASS** | no DCGM, HTA, Flight Recorder, NNsight, LMResiliency, replay, multi-rank |
| 25 | LLM Agent remains absent | **PASS** | zero agent code in `src/` |

**16 / 25 pass. Six detector-quality gates fail.** Detector quality gates failing
means `PHASE 1.2 = PARTIAL`, which is the outcome the brief specifies.

---

## 2. What Phase 1.2 established

**The repairs were real and they worked.**

| Metric | Phase 1.1 | Phase 1.2 | Direction |
|---|---|---|---|
| system recall | 0.857 | **0.933** | up |
| system precision | 0.632 | **0.121** | **down** |
| FP / 1k steps | 1.59 | 17.0 | down |
| `numerics.finiteness` precision | 0.857 | 0.478 | down |
| `numerics.finiteness` post-fault FPs | (not measured) | 810 → **24** after the cooldown fix | down |
| F10 mixed anomaly | **0/6** | **3/3** | fixed |
| negative detection delay | present | **eliminated** | fixed |
| cost-ledger negative overhead | present | **eliminated** | fixed |

The precision drop is the honest consequence of removing three measurement defects
that had been inflating it: a crashed run scored as a miss, a pre-injection fire
scored as an early detection, and 810 duplicate findings collapsed into one. Only
the third is a detector defect, and fixing it moved precision 0.037 → 0.121 while
recall held.

**System precision was never 0.632.** That figure was partly an artefact.

---

## 3. Defects found and fixed

| # | Defect | Consequence | Fix |
|---|---|---|---|
| 1 | F10 injected a Python-float NaN | the run crashed with `AttributeError` | NaN tensor |
| 2 | **A crashed run was scored as a detector miss** | an infrastructure failure became evidence about detectors; F10 was reported as a detector limitation for a whole phase | crashes excluded from scoring and reported separately |
| 3 | Detection delay from the raw first fire | pre-injection FPs reported as negative delays (reads as "fast") | D1 evaluation layer |
| 4 | v1.0 gradient z-branch gated at `peak_ratio >= 1.5` | fired on healthy data by 0.7 %; 50 FP vs 6 TP | criterion deleted |
| 5 | Baseline included the warm-up | 23 % biased-low baseline → fire at step 60 of an 80-injection run | warm-up excluded |
| 6 | `numerics.finiteness` had no cooldown | 810 post-fault findings; 94 % of corpus FPs | cooldown applied |
| 7 | `@dataclass` detached from `DetectorConfig` | `TypeError: DetectorConfig() takes no arguments` | decorator restored |
| 8 | Cost ledger measured OFF/ON in separate loops | negative overhead, physically impossible | paired interleaved design |
| 9 | `loss.trend_slope` recall 0.0 | `LOSS_DIVERGENCE` unimplemented | **NOT FIXED** — reported |

---

## 4. Performance (D7)

Paired, interleaved, same-batch, frozen model, order randomised, bootstrap 95 % CI on
the within-pair difference.

| Workload | Baseline | core delta | 95 % CI | Verdict |
|---|---|---|---|---|
| tiny (4L, d32) | 4.11 ms | +0.583 ms (**+14.19 %**) | (+0.594, +0.651) | resolvable — **FAIL** vs 3 % |
| medium (8L, d128) | 16.88 ms | +0.853 ms (+5.05 %) | (−0.156, +1.018) | **NO RESOLVABLE EFFECT AT THIS SCALE** |
| large (12L, d256) | 41.35 ms | +2.200 ms (**+5.32 %**) | (+1.406, +2.475) | resolvable — **FAIL** vs 3 % |

**No negative overhead is reported anywhere.** The medium interval crosses zero, so
the report declines to quote a sign rather than presenting +5 % as established —
which is what the Phase 1.1 ledger did wrongly in the other direction.

**The instrument now validates the component sum.** At the largest workload the
accounted components are `2.44 × 0.5 (grad) + 2.45 × 0.1 (param) + 4.89 × 0.1
(collector) + 36.7 µs × 10 (emit) ≈ 2.32 ms` against an observed paired delta of
**2.20 ms** — agreement within 5 %. This is the reconciliation the Phase 1.1 ledger
could not achieve, and it is what makes the ~0.8 ms "unattributed residual" from
Phase 1 explicable: it was largely instrument error, not missing cost.

The Phase 1 overhead gate is **unchanged and still fails**: core 5.32 % against 3 %,
T1 5.85 % against 8 %.

---

## 5. Not done, explicitly

| Item | State |
|---|---|
| **D3 loss detector redesign** | **PARTIAL.** `loss.trend_slope` recall is **0.000**; `LOSS_DIVERGENCE` is effectively unimplemented. The persistence-gated baseline-normalised slope D3.3 specifies was not built. `LOSS_OSCILLATION` does not exist. |
| **D8.1 abstention** | **NOT IMPLEMENTED.** No `INSUFFICIENT_EVIDENCE` output. |
| **`throughput.regression`** | **BROKEN.** Recall 0.250, 9 pre-injection false positives. Diagnosis not completed. |
| **D2.3 persistence gating** | Not adopted. Family H failed for an implementation reason, so multiplicity was used as the discriminator instead. |
| `LAYER_MULTIPLICITY_STRONG = 0.4`, envelope quantile 0.995 | Not swept. |
| Multi-rank, GPU, long runs, large models | **NOT VERIFIED.** |

---

## 6. Regression

**295 tests pass** (unchanged count, up from the Phase 1 baseline of 256 in
Phase 1.1). No original coverage was removed. New coverage added: evaluation
semantics, detector isolation, F10 regression, persistence, restart recovery,
cross-run comparison, correlation, calibration harness, Trackio failure isolation.

## 7. Governing principle

> Phase 1 proved we can detect something.
> Phase 1.1 proved something is not enough.
> **Phase 1.2 proved that when we called something abnormal, we were usually wrong
> — and that we had been measuring it wrong in our own favour.**

The detectors are better instrumented and one of them is genuinely redesigned. The
system is not yet trustworthy. **No agent, no new diagnostic source, no causal claim.**
