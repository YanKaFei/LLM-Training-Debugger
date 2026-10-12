# J6 — DEVELOPMENT QUALITY REPORT

**Status: COMPLETE.** 169 development runs. This is **development** data, not a
holdout.

---

## 1. The headline: the band had never actually been in control

Two development re-runs after implementing the gradient band produced **byte-identical
numbers**. That was the diagnostic: the band set `confidence_class`, but the
`GRADIENT_EXPLOSION` episode rule has `aggregate_confirms=True`, so **whenever the
global norm detector fired the episode confirmed regardless of its band**. A healthy
run whose global norm twitched was confirmed exactly as before the band existed.

Three defects were found and fixed by writing the J0 tests and diagnosing the corpus:

| # | Defect | Consequence |
|---|---|---|
| 1 | `l1_aggregate` was a **sum** over layers | scaled with model depth; the 4-layer-fitted boundary was 50 % wrong on the 6-layer configuration. 16 healthy episodes reached CONFIRMED. Fixed to the layer-count-invariant **mean**; 34/46 as a 4-layer sum is exactly 8.5/11.5 as a mean, so a 4-layer run is classified identically. |
| 2 | An `UNCERTAIN` observation closed as `NOT_ANOMALY` | the abstention the band exists to express was reported as a positive statement that nothing happened |
| 3 | **The band was not authoritative for confirmation** | the band had **no effect at all** on incidents |

## 2. Result, before and after the band became authoritative

| Layer | band ineffective | **band authoritative** | Target |
|---|---|---|---|
| SIGNAL precision | 0.079 | 0.079 | — |
| SIGNAL recall | 0.906 | 0.906 | — |
| EPISODE precision | 0.285 | **0.411** | ≥ 0.92 |
| EPISODE recall | 0.553 | **0.353** | ≥ 0.87 |
| INCIDENT precision | 0.500 | **0.680** | ≥ 0.92 |
| INCIDENT recall | 0.676 | **0.405** | ≥ 0.87 |
| RUN precision | 0.718 | **0.972** | — |
| RUN recall | 0.718 | **0.412** | — |
| **healthy-run false incident rate** | 0.286 | **0.0119** | ≤ 0.03 |
| **false incidents / 1k steps** | 0.710 | **0.0296** | ≤ 0.3 |
| pre-injection confirmed incidents | 0 | **0** | = 0 |

**Two gates now PASS that never passed before:**
`healthy-run false incident rate 1.2 %` (target ≤ 3 %) and
`false incidents / 1k steps 0.030` (target ≤ 0.3).

**Recall collapsed.** Incident recall 0.676 → 0.405. The operating point is now
precision-limited no longer — it is **recall-limited**.

## 3. Diagnostic findings (not in the ledger, shown to explain what happened)

```
diagnostic: anomaly type of the 24 healthy-run incidents
  GRADIENT_EXPLOSION 24 ; all others 0

diagnostic: anomaly type of the 6 healthy-run incidents after the fix
  GRADIENT_EXPLOSION  6
```

Every healthy false incident came, and still comes, from the gradient detector. Loss
and numerics contribute **none**.

## 4. Uncertainty accounting (J5)

| Population | NORMAL | UNCERTAIN | CONFIRMED |
|---|---|---|---|
| healthy episodes | **98.6 %** | 0.7 % | 0.7 % |
| fault episodes | 41.6 % | 4.3 % | 11.3 % |

`HEALTHY_ABSTENTION_RATE` = **0.7 %**, `FAULT_ABSTENTION_RATE` = **4.3 %**.

The anti-cheat check passes: the precision gain did **not** come from pushing
everything into abstention. The healthy UNCERTAIN rate is 0.7 %, so the system is
not hiding behind the middle band — it is genuinely classifying healthy runs as
NORMAL.

## 5. Development-supported detection boundary

Confirmed-incident rate by injected magnitude:

| Magnitude | Confirmed |
|---|---|
| loss spike 0.4 / 1.5 / 6.0 | 0/5, 0/5, 0/5 |
| gradient ×2.5 | **0/5** |
| gradient ×4.0 | 1/5 |
| gradient ×7.0 | **0/5** |
| gradient ×12.0 | 2/5 |
| gradient ×25.0 | **5/5** |
| gradient ×50.0 | **5/5** |
| divergence 1.008 / 1.018 / 1.04 | 1/5, 0/5, 1/5 |
| throughput 0.006 | 0/5 |

**Development-supported minimum reliably detectable gradient magnitude ≈ ×25** with
the band authoritative — far above the ×4 the boundary analysis suggested, and
qualitatively different from the ×3 that motivated the design.

The ×7.0 result (0/5) sitting between ×4 (1/5) and ×12 (2/5) is **not monotone** and
is a warning that the per-magnitude sample is 5 runs and the estimate is unstable.

This is labelled a **development-supported boundary**, never a *validated* one.

## 6. Verdict

```
episode precision >= 0.92   FAIL  (0.411)
episode recall    >= 0.87   FAIL  (0.353)
incident precision >= 0.92  FAIL  (0.680)
incident recall   >= 0.87   FAIL  (0.405)
healthy-run FP    <= 3%     PASS  (1.2%)
false incidents   <= 0.3/1k PASS  (0.030)
earliest evidence >= 0.92   NOT VERIFIED
pre-injection     = 0       PASS
duplicate roots   = 0       NOT VERIFIED
```

```
READY_FOR_BLIND_VALIDATION = NO
```

The reason is now the **opposite** of Phase 1.5's: the system is no longer
over-alarming, it is under-detecting. The frozen boundaries 8.5 / 11.5 confirm only
strong perturbations, and the brief forbids retuning them in this phase.
