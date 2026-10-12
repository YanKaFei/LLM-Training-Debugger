# K1 — GRADIENT RECALIBRATION REPORT

**Status: COMPLETE.** The boundaries were re-derived on the authoritative band.

---

## 1. The question, answered

Phase 1.7 asked, as the brief directs:

> Now that the band is actually authoritative, where should the operating boundaries
> really be?

**Answer: there is no operating point that meets the targets.** The previous 8.5 /
11.5 were not mis-set — they sit near the best achievable trade, and that trade is
bad because the **feature**, not the threshold, is the problem.

## 2. Method

`A` cannot change precision or recall, because both `NORMAL` and `UNCERTAIN` are
non-promotable; it only partitions observations between them. `B` drives every
quality metric. 126 candidate `(A, B)` pairs were graded on 610 runs with a marginal
detector fire.

## 3. Result

```
qualifying operating points (healthy <= 3%, precision >= 0.90, recall >= 0.80):  0
monotonicity violations:                                                         0
```

| Target | Best achieved | Verdict |
|---|---|---|
| healthy confirmed rate ≤ 3 % | 3 % achievable (at recall 0.163) | PASS only by sacrificing recall |
| confirmed precision ≥ 0.90 | **0.962** achievable | achievable |
| confirmed recall ≥ 0.80 | **0.163** at that precision | **FAIL** |

Precision ≥ 0.90 is reachable. Recall ≥ 0.80 is reachable at ×25+. **Both together
are not**, because the healthy upper tail (max 56.30) sits above where a ×3–×12 fault
lands.

## 4. Development-supported detection boundary

```
< x12      NOT RELIABLY DISTINGUISHABLE from healthy heavy-tail behaviour
x12..x20   UNCERTAIN
>= x20     DEVELOPMENT-SUPPORTED DETECTION
```

**Development-supported**, never *validated*: no blind holdout has been run.

This is far worse than the ×3 the Phase 1.4 analysis implied, and the difference is
explained: Phase 1.4 measured the feature's response shortly after injection, when
the baseline had not yet adapted. The 680-run corpus measures the run's *peak*
response, which is dominated by the adaptation effect described in
`GRADIENT_BOUNDARY_FRONTIER.md` §5.

## 5. Decision

**The operating point was not changed.** No qualifying point exists, so changing A
and B would substitute one bad trade for another and would move the statistical
target without improving the system. The brief forbids exactly that. The finding is
reported instead:

```
GRADIENT_EXPLOSION detection target: NOT ACHIEVABLE with the current feature
reason: the feature measures deviation from a trailing baseline that a sustained
        uniform fault scaling contaminates
```
