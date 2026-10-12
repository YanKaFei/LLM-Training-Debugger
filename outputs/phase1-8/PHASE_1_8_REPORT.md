# PHASE 1.8 REPORT

```
PHASE 1.8 STATUS:              PARTIAL
READY_FOR_BLIND_VALIDATION:    NO
PHASE 1 ORIGINAL CLOSURE:      STILL PARTIAL
AGENT READINESS:               NO
```

**No final holdout files were created**, per the brief.

`PHASE_1_REPORT.md` and `outputs/phase1-1..1-7/` are **unmodified**.

---

## 1. The feature defect is fixed

Phase 1.7's finding was that the gradient feature erased the fault it was meant to
expose: a sustained uniform scaling dragged the trailing median with it, so the ratio
decayed toward 1. **No threshold could fix that**, and 680 runs proved it.

Phase 1.8 replaced the reference with a **state-aware, freeze-on-suspicion baseline**:

```
NORMAL      -> learns
SUSPECT     -> FREEZES; the candidate regime is never absorbed
CONFIRMED   -> stays frozen
RECOVERING  -> stays protected
CLOSED      -> resumes only after the recovery policy completes
```

Verified directly: a sustained ×5 fault holds at ratio **5.00** for 150 steps, where
the old feature had reached 1.00.

## 2. Revalidation on the same 680 runs

| Metric | Phase 1.7 | **Phase 1.8** |
|---|---|---|
| confirmed precision | 0.942 | **0.973** |
| confirmed recall | **0.279** | **0.468** (+68 % relative) |
| healthy confirmed rate | 5.6 % | **4.2 %** |
| fault peak p50 | 8.77 | **13.66** |
| healthy peak p99 | 36.16 | **29.60** |
| qualifying operating points | 0 | **0** |

`P(CONFIRMED | magnitude)` went from **flat ≤ 0.10 across ×1.5–×12, then a cliff to
1.00 at ×25**, to a monotone rise: 0.23, 0.12, 0.15, 0.40, 0.23, 0.20, 0.33, 0.23,
**0.70 at ×12**, 0.95 at ×20, 1.00 at ×25 and ×40.

**That specific pathology is gone.** The detector is no longer blind across a 8×
range of fault magnitudes.

## 3. Why it is still NO — and the reason has changed again

```
healthy peak : p50 2.80   p95 15.38   p99 29.60   max 73.80
fault   peak : min 1.29   p50 13.66   p95 84.06
```

**The fault minimum (1.29) is below the healthy median (2.80).** At ×1.5–×2 the
injected perturbation is *smaller than a healthy run's own variation*. Recall ≥ 0.80
would require detecting those, and a single scalar level statistic cannot.

```
< x12      NOT RELIABLY DISTINGUISHABLE
x12        PARTIAL (0.70)
>= x20     DEVELOPMENT-SUPPORTED (0.95-1.00)
```

This is a **capability boundary**, stated as such, not a calibration failure. The
brief's instruction applies: where distributions overlap, **abstain**, and do not
claim a capability the evidence does not support.

## 4. What was NOT delivered

| Item | State |
|---|---|
| **L1** families C (change-point) and D (rate-of-change) | **NOT BUILT** |
| **L4** `EpisodeStore` wired into the real pipeline | **NOT DONE** |
| **L5** CLI E2E from persisted data | **NOT DONE** |
| **L6** consequence-aware scorer executed | **NOT DONE** |
| **L7** earliest-evidence evaluator executed | **NOT DONE** |
| **L8** development quality re-evaluation | **NOT RUN** |
| **L9** smoke | **NOT RUN** |

The evaluators for L6 and L7 were built and unit-tested in Phase 1.7 and
**remain unexecuted**, because running them needs the pipeline integration (L4) that
was not built. So numerics quality is still unmeasured — and the 0.314 figure remains
known to be mis-typed without a replacement.

## 5. Tests

```
before:  390
added:    14
total:   404 passing, 1 skipped
```

New coverage: the sustained-scaling property, the **defect itself pinned as a
regression**, state semantics for all five states, resume-after-false-suspicion,
warm-up abstention, and joint-feature integration. One superseded test was updated:
`test_aggregate_excludes_the_warmup_when_asked` asserted that including the warm-up
*changed* the verdict, which was true of the trailing median and is false of a
freezing reference — the assertion was replaced by the property that now matters
(both choices agree, so the verdict does not depend on an arbitrary window boundary).

## 6. Performance

Unchanged and explicitly verified: the protected baseline is computed **offline over
an emitted series** in the detection path. No observer or adapter change, **no new
training-path synchronisation, no new raw tensor capture** — the feature reads only
the T1 per-layer gradient norms already emitted. No fresh benchmark was required and
none is claimed.

## 7. Governing principle

> Phase 1.7 proved a rolling baseline can erase the very fault it is supposed to
> detect.
>
> **Phase 1.8 made the baseline state-aware: it learns while healthy, freezes when
> suspicious, and never lets the suspected fault redefine normal. Recall rose 68 %
> and the pathology disappeared — and the remaining limit is now a genuine overlap
> between a mild fault and healthy variation, which no threshold can resolve.**

**No agent. No replay. No new diagnostic source. No causal claim.**
