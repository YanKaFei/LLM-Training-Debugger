# L1 — GRADIENT FEATURE COMPARISON

**Status: PARTIAL.** Two feature families were implemented and measured; two were
reasoned about and not built.

---

## 1. What was measured

| Family | Status | Result |
|---|---|---|
| **A. Trailing-median ratio** (Phase 1.4–1.7) | measured | **eliminated** — self-normalises under a sustained fault |
| **B. Freeze-on-suspicion reference** (adopted) | measured | recall 0.279 → **0.468**, precision 0.942 → **0.973** |
| C. Change-point statistic | **NOT BUILT** | — |
| D. Rate-of-change + level | **NOT BUILT** | — |

Family A was not eliminated on AUC. It was eliminated on the criterion the brief
makes decisive:

> Does the feature remain abnormal after the fault persists?

It does not. `test_trailing_median_would_have_decayed` pins the arithmetic: once the
rolling window lies entirely inside the fault, `value / median` is **1.0** while the
truth is 5.0. A feature that erases its own signal cannot be rescued by any threshold.

## 2. Family B — why freeze-on-suspicion rather than fast-vs-slow

The brief's family B is "fast statistic vs slow statistic". The implementation is a
**stronger form of the same idea**: rather than a slower horizon that still
eventually absorbs the fault, the reference **stops updating entirely** at the first
hint of deviation. A slow horizon only delays self-normalisation; freezing prevents
it.

Verified: a sustained ×5 fault holds at ratio **5.00** from step 50 to step 199, where
the old feature had decayed to 1.00 by step 299.

## 3. Why families C and D were not built

The brief's own warning applies to C:

> Phase 1.2 的 CUSUM 对旧 signal 产生大量 FP。

Phase 1.2 measured CUSUM at **147 false positives per 1 000 steps** on healthy data.
The brief is right that this does not condemn every change-point method, but building
and calibrating one properly is a phase of work, and family B solved the identified
failure mode. D (rate-of-change) was not built because a derivative alone sees only
the onset and not the persistence — which is precisely the property that had to be
fixed.

## 4. Comparison criteria (L1.1 / L1.2)

| Criterion | A | **B** |
|---|---|---|
| healthy/fault separation | fault p50 8.77 vs healthy p95 17.07 — overlapping | fault p50 **13.66** vs healthy p95 15.38 |
| response across magnitude | **flat ≤ 0.10 from ×1.5 to ×12** | rises 0.12 → 0.40 → 0.70 → 1.00 |
| remains abnormal under persistence | **NO** | **YES** |
| model-shape stability | OK (mean aggregate, layer-count-invariant) | OK |
| warm-up behaviour | classified against a thin baseline | abstains (`None`) |
| self-normalisation | **rapid** | none |
| monotonicity violations | 0 | 0 |

Family B is better on every measured criterion. It is **not** sufficient —
`qualifying operating points: 0` — but the failure mode it was built to fix is fixed.

## 5. Not verified

* **`freeze_at_ratio` and `resume_after_steps` were not swept.**
* **AUC was not computed per feature** this phase; separation is reported via the distribution quantiles and the magnitude response instead.
* **Families C and D are untested**, so no claim is made that B is the best of the four — only that it is the best of the two built.
