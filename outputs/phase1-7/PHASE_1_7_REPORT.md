# PHASE 1.7 REPORT

```
PHASE 1.7 STATUS:              PARTIAL
READY_FOR_BLIND_VALIDATION:    NO
PHASE 1 ORIGINAL CLOSURE:      STILL PARTIAL
AGENT READINESS:               NO
```

**No final holdout files were created**, per the brief.

`PHASE_1_REPORT.md` and `outputs/phase1-1..1-6/` are **unmodified**.

---

## 1. The phase's finding

The brief posed a specific question:

> Now that the band is actually authoritative, where should the operating boundaries
> really be?

**Answer: there is no operating point that works.**

Measured on **680 real training runs** (200 healthy, 40 per magnitude across 12
magnitudes, 2 model configurations):

```
qualifying operating points: 0
```

No `(A, B)` achieves healthy confirmed ≤ 3 % **and** confirmed precision ≥ 0.90
**and** confirmed recall ≥ 0.80. The best point at a 3 % false-positive budget:

```
A = 13.15   B = 26.30   precision = 0.962   recall = 0.163
```

Phase 1.6's 8.5 / 11.5 sits near the best achievable F1 (precision 0.942, recall
0.279). **It was not mis-set.** The trade itself is bad.

## 2. Why, and this is the substantive result

```
healthy peak l1_aggregate : p50  3.08   p95 17.07   p99 36.16   max 56.30
fault   peak l1_aggregate : min  1.35   p50  8.77   p95 46.73

P(CONFIRMED | magnitude):  x1.5 0.08 | x2 0.05 | x2.5 0.00 | x3 0.10 | x4 0.05
                           x5 0.03 | x7 0.05 | x10 0.00 | x12 0.07
                           x20 0.85 | x25 1.00 | x40 1.00
```

The fault **minimum** (1.35) is below the healthy **median** (3.08). The magnitude
curve is **flat at ≤ 0.10 from ×1.5 to ×12 and then jumps to 1.00 at ×25** — a cliff,
not noise (0 monotonicity violations at 40 runs per magnitude).

**Root cause — a feature defect, not a calibration defect.** The feature is
`layer_norm[step] / trailing median of that layer`. A **sustained uniform** fault
scaling drives the trailing median up with it, so the ratio returns to ≈ 1 and the
fault becomes invisible. The ×20 cliff is where the per-step jump outruns the
baseline before it adapts.

This **invalidates the premise of the recalibration** — and that is the finding.
Asking where the boundaries should be was the right question; the answer is that the
feature cannot support the targets at any boundary.

## 3. What was delivered

| Item | State |
|---|---|
| **K0** corpus expansion | **COMPLETE** — 680 runs, 40/magnitude, 200 healthy, 2 configs |
| **K1** boundary re-derivation + frontier | **COMPLETE** — 126 points graded; 0 qualifying |
| **K2** consequence-aware scoring | **IMPLEMENTED + TESTED**, not run over the corpus |
| **K3** earliest-evidence evaluator | **IMPLEMENTED + TESTED**, not run over the corpus |
| **K4** pipeline persistence | **NOT DONE** |
| **K5** CLI end to end | **NOT DONE** |
| **K6** development re-evaluation | **NOT RUN** — gated on K4/K5 |
| **K7** smoke | **NOT RUN** |
| **K8** readiness audit | **COMPLETE** — NO |

The **0.314 numerics precision is now known to be a mis-typed number**: it mixed true
secondary manifestations with duplicate root incidents. The corrected figure has not
been measured, because K4 was not built, and replacing it with an estimate would
repeat the error in a new form.

## 4. Development-supported detection boundary

```
< x12      NOT RELIABLY DISTINGUISHABLE from healthy heavy-tail behaviour
x12..x20   UNCERTAIN
>= x20     DEVELOPMENT-SUPPORTED DETECTION
```

**Development-supported**, never *validated*. Far above the ×3 the Phase 1.4 analysis
implied; the difference is that Phase 1.4 measured the immediate post-injection
response while this corpus measures the run's peak, which the baseline adaptation
dominates.

## 5. The operating point was deliberately not changed

No qualifying point exists, so moving A and B would substitute one bad trade for
another and would move the statistical target without improving the system. The brief
forbids exactly that, and states the required behaviour for this situation:

> 如果最终结果证明这些目标不合理：不得自己改 Gate。报告失败。

The gate is reported as failed.

## 6. Tests

```
before:  375
added:    15
total:   390 passing, 1 skipped
```

New coverage: six observation classes, primary true positive, **secondary
manifestation is not a false positive**, duplicate root incident is an error,
out-of-window and wrong-family consequences, healthy-run observations, ground-truth
schema round trip, earliest-evidence true/false/UNKNOWN, wrong family is UNKNOWN not
FALSE, coverage guard against all-UNKNOWN, confirmation delay separated from onset
error, healthy runs excluded.

## 7. Governing principle

> Phase 1.6 proved the band could suppress false alarms once it was authoritative, and
> that the old boundaries had been fitted against a code path that did not use them.
>
> **Phase 1.7 spent 680 runs re-deriving those boundaries and found that no boundary
> works — because the feature measures deviation from a trailing baseline that a
> sustained fault scaling contaminates. The problem was never the threshold.**

**No agent. No replay. No new diagnostic source. No causal claim.**
