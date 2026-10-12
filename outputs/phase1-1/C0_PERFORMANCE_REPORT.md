# C0 — PERFORMANCE CLOSURE REPORT

**Status: PARTIAL — the unattributed residual was NOT closed.**

Phase 1 left ~0.8 ms/step unattributed at the largest workload. C0 set out to close
it. **It did not, and the ledger instrument itself is the reason.** That is
reported here rather than papered over.

---

## 1. Overhead (Phase 1 measurement, unchanged and not revised)

Steady state, 50-step policy warm-up excluded, 60 measured steps × 5 repeats,
minimum of per-repeat medians:

| Workload | Baseline | core | T1 total | **T1 incremental** |
|---|---|---|---|---|
| tiny (4L, d32) | 4.41 ms | +13.3 % | +17.9 % | **≈ 0.00 ms** |
| medium (8L, d128) | 17.28 ms | +5.0 % | +5.4 % | **≈ 0.02 ms** |
| large (12L, d256) | 41.62 ms | +3.3 % | +3.3 % | **≈ 0.00 ms** |

Budgets: core < 3 % (**FAIL**, 3.3 % at large), T1 < 8 % (**PASS** at medium and
large, **FAIL** on tiny at 17.9 %).

The T1 incremental column is reported for explanation only. **The gate is T1
*total* against the baseline, and it is not redefined to incremental to obtain a
PASS.**

---

## 2. The cost ledger, and why it did not close the gap

`tests/performance/benchmark_cost_ledger.py`, output in
`outputs/phase1-1/performance/OBSERVER_COST_LEDGER.md`.

| Workload | Baseline | Observed core overhead | Accounted | Residual | **Dispersion** |
|---|---|---|---|---|---|
| tiny | 4.83 ms | +0.69 ms (+14.3 %) | 3.67 ms | −2.98 ms | **±0.24 ms** |
| medium | 19.04 ms | −0.29 ms (−1.5 %) | 3.97 ms | −4.26 ms | **±1.52 ms** |
| large | 43.05 ms | +2.60 ms (+6.0 %) | 5.13 ms | −2.53 ms | **±3.71 ms** |

**This table is not a successful reconciliation, and it should not be read as
one.** Three problems are visible in it:

1. **A negative observed overhead at medium** (−0.29 ms) is physically impossible.
   The debugger cannot make training faster.
2. **The residual is negative at every workload**, meaning the sum of parts
   exceeds the whole.
3. **At the largest workload the dispersion (±3.71 ms) exceeds the effect being
   measured (+2.60 ms).** The measurement cannot resolve the quantity it is
   supposed to explain.

### Root causes identified

| # | Cause |
|---|---|
| 1 | The `accounted` column applies component costs measured on the 8L d128 model to all three workloads. The second revision measured them per workload, but `collector_compute` is still measured by constructing a fresh `TrainingObserver` + `LayerMetricCollector` **inside the timed lambda**, so it is inflated by construction cost rather than measuring `compute()` alone. |
| 2 | `off`-mode and instrumented-mode baselines were measured in separate repeat loops. Run-to-run machine variation between loops (±1.5 to ±3.7 ms at medium/large) is the same order as the effect, so the difference of medians is dominated by drift. |
| 3 | `null` writer + a 50 000-event queue means the observer's writer-thread cost is not represented, while the per-event `emit()` micro-cost is multiplied by an assumed 10 events/step — an assumption, not a measurement. |

**The honest conclusion: component micro-costs are measured and trustworthy
(§3); the whole-step reconciliation is not.** Closing C0 requires an interleaved
A/B measurement design (alternate `off` and instrumented steps inside one loop) and
a `collector.compute()` benchmark that does not include construction. Neither was
completed in Phase 1.1.

### The ~0.8 ms residual: **NOT VERIFIED**

It remains unattributed. It is **not** explained by measurement error, because the
dispersion at the largest workload (±3.71 ms) is larger than the residual itself,
which means the residual is not distinguishable from noise at this precision.

---

## 3. Component micro-costs (these ARE trustworthy)

Median / p95 / MAD, measured on a 12-layer d256 model unless noted. Cadences from
`DEFAULT_POLICY_PHASE1`.

| Component | Cost | Cadence | Per-step |
|---|---|---|---|
| `global_gradient_norm` | 2.44 ms | every 2nd step | 1.22 ms |
| `global_parameter_norm` | 2.45 ms | every 10th step | 0.245 ms |
| `LayerMetricCollector.compute` | 4.89 ms | every 10th step | 0.489 ms |
| `observer.emit` (full) | 36.7 µs | ~10 events/step | 0.367 ms |
| loss `.item()` | ~9 µs | every step | 0.009 ms |
| `memory_snapshot` | ~1.5 µs | every 10th step | 0.00015 ms |
| `validate_event` | ~6 µs | per event | included above |
| `new_event_id()` (ULID) | ~1.3 µs | per event | included above |
| `to_dict` | ~4 µs | per event | included above |

### Synchronisation audit (C0.3)

Every candidate that can force a device sync was checked:

| Site | Syncs? | Verdict |
|---|---|---|
| `global_gradient_norm` | **NO** — `torch._foreach_norm` returns one tensor, one `.item()` | already fixed in Phase 1 |
| `global_parameter_norm` | **NO** — same | already fixed |
| `LayerMetricCollector.compute` | **NO** — `_foreach_norm` over all layers, then one host transfer | already fixed (35 540 → 4 890 µs) |
| `record_loss` | **YES — SYNCING** | **DEFER not possible**: the loss must be compared against NaN for the finiteness detector, which requires a host value. Cost is ~9 µs/step, so it is retained and documented rather than removed. |
| `memory_snapshot` | allocator query, no device sync on MPS | retained |
| `.cpu()`, `.numpy()`, `torch.cuda.synchronize()` | **absent from the hot path** | verified by grep over `adapters/` and `observer/` |

One syncing site remains: `record_loss`. It is **DECLARED, not hidden**, and its
cost (9 µs, 0.2 % of a 4 ms step) does not justify losing NaN detection.

### Python fixed cost (C0.4)

Measured individually, not estimated: event construction ≈ 5 µs, enum conversion
≈ 0.4 µs, ULID ≈ 1.3 µs, two clock reads ≈ 0.15 µs, `to_dict` ≈ 4 µs,
`validate_event` ≈ 6 µs, `would_emit` ≈ 0.3 µs, queue insert ≈ 0.6 µs. Per-event
Python overhead totals ≈ 18 µs, i.e. **≈ 0.18 ms/step at 10 events/step** — about
7 % of the measured per-step cost at the tiny workload.

### Gating preserved (C0.5)

`parameter_norm` every 10 steps means **9 steps compute nothing**. This is enforced
by `would_emit()` before the reduction, not after. Verified in W4 and unchanged.

---

## 4. What C0 established, and what it did not

**Established:** every component's individual cost, that no un-audited
synchronisation exists on the hot path, that the one remaining sync is declared and
cheap, and that sampling gates computation.

**Not established:** how those components sum to the observed whole-step overhead.
**C0 is PARTIAL and the ~0.8 ms residual is NOT VERIFIED.**
