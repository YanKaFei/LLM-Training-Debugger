# E11 — PERFORMANCE STATUS

**Status: UNCHANGED from Phase 1.2.** The observer was not rewritten, per brief
section 23.

---

## 1. Method

Unchanged: the Phase 1.2 paired/interleaved benchmark
(`tests/performance/benchmark_paired.py`). Same batch through both arms, arm order
randomised per pair, frozen model, within-pair differences, bootstrap 95 % CI.

## 2. Last measured result (Phase 1.2, not re-run)

| Workload | Baseline | core delta | 95 % CI | Verdict |
|---|---|---|---|---|
| tiny (4L, d32) | 4.11 ms | +0.583 ms (**+14.19 %**) | (+0.594, +0.651) | resolvable — **FAIL** |
| medium (8L, d128) | 16.88 ms | +0.853 ms (+5.05 %) | (−0.156, +1.018) | **NO RESOLVABLE EFFECT** |
| large (12L, d256) | 41.35 ms | +2.200 ms (**+5.32 %**) | (+1.406, +2.475) | resolvable — **FAIL** |

T1 total at the largest workload: **+5.85 %**.

**No negative overhead is reported anywhere.** Where the CI crosses zero the report
declines to quote a sign.

## 3. Original gate — unchanged and still failing

```
core observer < 3%     ->  5.32 %   FAIL
T1 total      < 8%     ->  5.85 %   PASS at large, FAIL on tiny (17.9 %)
```

The gate was **not** modified for Phase 1.3, and Phase 1.3 does not claim
performance closure. Detector reliability and performance closure are reported as
two different states, per brief section 23.

## 4. Cost of the episode layer

The episode engine runs **offline** over detector output, not in the training loop,
so its cost does not appear in the per-step overhead at all. Measured wall time:
169 runs × 240 steps completed in 279 s including training, detection, episode
grouping and fusion — i.e. the episode layer's cost is not measurable against the
training cost at this corpus size.

## 5. Not verified in Phase 1.3

* **The benchmark was not re-run.** The Phase 1.2 numbers are quoted. Nothing in Phase 1.3 changed the observer or the adapter hot path, so a re-run was not expected to differ — but "not expected to differ" is not a measurement, so the numbers above are labelled as Phase 1.2's.
