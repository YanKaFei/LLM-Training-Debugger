# D7 PERFORMANCE REPORT

**Method:** interleaved paired A/B. For each step the *same batch* is executed
by both arms, arm order is randomised per pair, and the difference is taken
**within the pair**. The model is frozen (`lr = 0`), so both arms perform
identical arithmetic. This removes the drift that invalidated the Phase 1.1
ledger.

## Paired result

| Workload | Baseline | core delta (95 % CI) | core % | + T1 delta (95 % CI) | T1 total % | T1 incremental |
|---|---|---|---|---|---|---|
| tiny (4L, d32) | 4.11 ms | +0.583 ms (+0.594, +0.651) | +14.19 % | +0.572 ms (+0.691, +1.105) | +13.94 % | -0.001 ms |
| medium (8L, d128) | 16.88 ms | +0.853 ms (-0.156, +1.018) | +5.05 % | +0.952 ms (+0.061, +1.254) | +5.64 % | +0.049 ms |
| large (12L, d256) | 41.35 ms | +2.200 ms (+1.406, +2.475) | +5.32 % | +2.417 ms (+2.048, +3.344) | +5.85 % | +0.037 ms |

## Verdicts

| Workload | core | T1 total |
|---|---|---|
| tiny (4L, d32) | resolvable effect | resolvable effect |
| medium (8L, d128) | NO RESOLVABLE EFFECT AT THIS SCALE | resolvable effect |
| large (12L, d256) | resolvable effect | resolvable effect |

**A negative overhead is never reported.** Where the 95 % CI crosses zero the
report says NO RESOLVABLE EFFECT AT THIS SCALE rather than quoting a sign the
data does not support.

## Gate

| Budget | Target | Measured (largest workload) | Verdict |
|---|---|---|---|
| core observer | < 3 % | +5.32 % | FAIL |
| T1 layer hooks (total) | < 8 % | +5.85 % | PASS |

The original Phase 1 gate is unchanged. T1 incremental cost is reported for
explanation only and is **not** substituted for the total.

## Microbenchmark (separate system — NOT summed into the above)

| Component | median | MAD |
|---|---|---|
| `construct` | 6.270 µs | 0.647 µs |
| `emit` | 36.854 µs | 0.687 µs |
| `grad_norm` | 609.625 µs | 0.834 µs |
| `new_event_id` | 3.708 µs | 0.208 µs |
| `param_norm` | 631.854 µs | 20.083 µs |
| `validate` | 6.750 µs | 0.125 µs |
| `would_emit` | 0.584 µs | 0.042 µs |

These numbers locate cost. Adding them up to predict the macrobench delta is the
error that produced the Phase 1.1 ledger's 5.13 ms "accounted" figure against a
2.60 ms observed effect.

## Environment

```json
{
  "cpu_count": 12,
  "extra": {
    "benchmark": "paired_interleaved"
  },
  "git_commit_sha": null,
  "git_dirty": null,
  "machine": "arm64",
  "os": {
    "release": "27.0.0",
    "system": "Darwin",
    "version": "Darwin Kernel Version 27.0.0: Tue Aug 11 21:20:40 PDT 2026; root:xnu-13432.1.9~1/RELEASE_ARM64_T6020"
  },
  "packages": {
    "duckdb": "1.4.5",
    "numpy": "2.0.2",
    "pyarrow": "21.0.0"
  },
  "platform": "macOS-27.0.1-arm64-arm-64bit",
  "processor": "arm",
  "python_implementation": "CPython",
  "python_version": "3.9.6",
  "torch": {
    "cuda_available": false,
    "cuda_version": null,
    "mps_available": true,
    "num_threads": 8,
    "torch_version": "2.8.0"
  }
}
```
