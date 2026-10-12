# D3 — LOSS DETECTOR REPORT

**Status: PARTIAL.** The measured behaviour is reported; the redesign D3.1–D3.3
specifies was **not completed**. This is stated plainly rather than dressed up.

---

## 1. Measured behaviour (holdout, D6 corpus)

| Detector | TP | FP(pre-inj) | FP(post) | FN | Precision | Recall | Median delay |
|---|---|---|---|---|---|---|---|
| `loss.robust_zscore` | 21 | 0 | 4 | 0 | **0.700** | **1.000** | +2 steps |
| `loss.trend_slope` | 0 | 16 | 0 | 16 | **0.000** | **0.000** | n/a |

## 2. `loss.trend_slope` is broken, and D6 is what shows it

Recall **0.0** across a magnitude sweep of sustained divergence (compounding
factors 1.006, 1.012, 1.03). It never fires on the fault it exists to detect, and
raises 16 pre-injection false positives instead.

Phase 1.1 already recorded a recall of 0.167 and this phase confirms it is worse
than that once the evaluation semantics are correct: the earlier 0.167 counted a
pre-injection fire as a "detection". Under D1 semantics, where a pre-injection fire
is a **false positive** and only a fire in the acceptable window counts, its recall
is **exactly zero**.

`LOSS_DIVERGENCE` is therefore effectively **not implemented**. The detector
exists, is wired in, and does nothing useful.

## 3. `loss.robust_zscore`

Precision 0.700 with recall 1.000 is the strongest detector in the corpus after
`numerics.finiteness` on recall, and it is close to the 0.80 precision target — but
it does not meet it. Its 4 post-fault false positives are duplicates from sustained
divergence, where the loss continues to breach after the first finding.

## 4. What D3 specified and what was not done

| Requirement | Status |
|---|---|
| D3.1 split `LOSS_SPIKE` / `LOSS_LEVEL_SHIFT` / `LOSS_DIVERGENCE` / `LOSS_OSCILLATION` | **NOT DONE** — `trend_slope` still serves every trend task |
| D3.2 validate that detrending reduces healthy FP without masking injected faults | **PARTIALLY DONE** — detrending is retained for the loss spike detector and its healthy FP rate is 0 on this corpus, but no ablation was run |
| D3.3 require **sustained** worsening for `LOSS_DIVERGENCE` (minimum persistence, magnitude, relative slope, baseline-normalised slope) | **NOT DONE** |

The brief's D3.3 diagnosis is corroborated by the data: a "brief positive slope"
criterion producing 16 pre-injection false positives and zero true positives is
exactly the failure mode of an unsustained trend test. The fix was specified and
not built.

## 5. Consequence for the phase gate

`loss.trend_slope` at precision 0.000 fails the "major detector precision >= 0.80"
gate outright, independently of the system-level numbers. **Phase 1.2 cannot be
COMPLETE on this basis**, and that is the correct outcome — the alternative would be
to exclude a broken detector from the report, which is precisely the kind of
omission this project exists to prevent.

## 6. Not verified

* Whether a persistence-gated, baseline-normalised slope would fix `LOSS_DIVERGENCE`. The design is specified; no implementation or measurement exists.
* `LOSS_OSCILLATION` detection. Nothing implements it.
* Whether the 4 loss duplicates are correctly a single incident — that depends on the incident engine's dedup, which was not re-measured.
