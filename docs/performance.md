# Performance

All numbers are measured on this machine and reproducible via the scripts named.
Environment is captured in every report (`src/training_debugger/env_info.py`).

## Storage — 1 000 000 events

| Metric | Value |
|---|---|
| write throughput | 36 314 events/s |
| files | 10 |
| disk | 9.6 MB (10.1 bytes/event, zstd) |
| query: event-type counts | 2.2 ms |
| query: metrics for one step | 13.4 ms |
| query: incident window (±50 steps) | 12.8 ms |
| query: per-layer series, all layers | **692 ms** (known slow path) |
| end-to-end observer emit | 25 367 events/s, **14.8 µs/event**, 0 dropped |

`tests/performance/benchmark_storage.py`

## Overhead — steady state, past the policy warm-up

| Workload | baseline | core metrics | + T1 layer hooks |
|---|---|---|---|
| tiny (4L, d32) | 4.41 ms | +13.3 % | +17.9 % |
| medium (8L, d128) | 17.28 ms | +5.0 % | +5.4 % |
| large (12L, d256) | 41.62 ms | +3.3 % | +3.3 % |

**Budgets: core < 3 %, T1 < 8 %. Verdict: core FAILS (3.3 % at the largest
workload), T1 PASSES at medium and large (3.3–5.4 %) and fails only on the tiny
workload.** T1's incremental cost over core is 0.00–0.02 ms — the layer hooks are
effectively free.

`tests/performance/benchmark_overhead_scaling.py`

## Where the cost is, and what was fixed

| Component | Before | After | Cause |
|---|---|---|---|
| `global_gradient_norm` | 826 µs | 169 µs | `.item()` per tensor: one device sync per parameter |
| `global_parameter_norm` | 880 µs | 178 µs | same |
| `LayerMetricCollector.compute` | 35 540 µs | 4 890 µs | same bug, 149 syncs per collected step |
| `observer.emit` | 54.8 µs | 36.7 µs | up to ten `dataclasses.replace` calls |

The reductions are now at the **memory-bandwidth floor**: 4.89 ms ≈ 2.44 (grads) +
2.45 (params) for ~4 M elements. A gradient norm cannot be computed faster than it
can be read.

## Measurement discipline

Two early measurements were **wrong and are retracted**:

1. A +7.4 % figure came from measuring inside the policy's 50-step full-resolution
   warm-up. Corrected to +3.3 %.
2. A median-of-medians estimator produced a physically impossible result
   (instrumented faster than baseline). Replaced with the minimum of per-repeat
   medians — contention can only add time.

## Not verified

* **GPU.** Everything here is CPU/MPS on an Apple M2 Max. CUDA behaves differently.
* **Production scale.** The largest workload is 12 layers at d_model 256; the
  percentage at a multi-second step is a projection, not a measurement.
* **~0.8 ms of the largest-workload overhead is not yet attributed.**
