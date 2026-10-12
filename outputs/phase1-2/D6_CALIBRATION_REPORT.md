# D6 CALIBRATION AND HOLDOUT REPORT

**Corpus:** 226 real training runs across two independently generated
splits. **Wall time:** 316 s.

## Independent generation

| Split | Model | Seeds | Runs |
|---|---|---|---|
| calibration | 4 layers, d_model 32 | 3000-3059 | 136 |
| holdout | 6 layers, d_model 48 | 4000-4039 | 90 |

The holdout uses a **different model shape**, different seeds and different
fault magnitudes, so agreement between the splits is independent evidence
rather than a restatement of the same runs.

**Threshold fitting rule.** The v2.0 envelope parameters came from a separate
30-run distribution study (seeds 2000-2029) disjoint from both splits. Neither
split was consulted while choosing them, and no threshold was changed after
seeing any holdout result.

## System-level detection quality

| Split | Runs | TP | FP(pre-inj) | FP(post) | Healthy FP | FN | Precision | Recall | F1 | FP/1k | Median delay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calibration | 136 | 57 | 36 | 85 | 255 | 7 | 0.132 | 0.891 | 0.229 | 13.82 | 0 |
| holdout | 90 | 42 | 25 | 67 | 214 | 3 | 0.121 | 0.933 | 0.214 | 17.00 | 0 |

## Gate assessment (holdout)

| Gate | Target | Measured | Verdict |
|---|---|---|---|
| system precision >= 0.90 | 0.9 | 0.121 | FAIL |
| system recall >= 0.85 | 0.85 | 0.933 | PASS |
| FP / 1k steps <= 0.5 | 0.5 | 17.000 | FAIL |

Per-detector precision >= 0.80:

| Detector | Split | Precision | Recall | Median delay | FP/1k | Verdict |
|---|---|---|---|---|---|---|
| `grad.robust_zscore` | holdout | 0.063 | 0.780 | 0 | 29.94 | FAIL |
| `loss.robust_zscore` | holdout | 0.700 | 1.000 | 2 | 1.73 | FAIL |
| `loss.trend_slope` | holdout | 0.000 | 0.000 | n/a | 5.00 | FAIL |
| `numerics.finiteness` | holdout | 0.478 | 1.000 | 0 | 5.45 | FAIL |
| `throughput.regression` | holdout | 0.100 | 0.250 | 3 | 11.25 | FAIL |

## Pre-injection incidents (D6.6)

* **calibration**: 28 of 64 fault runs produced an incident-opening fire before the
  fault could exist (rate 0.438).
* **holdout**: 19 of 45 fault runs produced an incident-opening fire before the
  fault could exist (rate 0.422).

Target: **0**. A pre-injection incident is a defect even when the fault is later
detected, because the earliest-abnormal-event claim would name the wrong step.

## Detection boundary by magnitude

| Fault | Magnitude | Split | Detected | Median delay |
|---|---|---|---|---|
| `A4_nan_that_recovers` | None | calibration | 4/4 | 0.0 |
| `A4_nan_that_recovers` | None | holdout | 3/3 | 0 |
| `A5_inf_one_layer_only` | None | calibration | 4/4 | 0.0 |
| `A5_inf_one_layer_only` | None | holdout | 3/3 | 0 |
| `F10_mixed_anomaly` | None | calibration | 0/4 | n/a |
| `F10_mixed_anomaly` | None | holdout | 0/3 | n/a |
| `F1_loss_spike` | 0.5 | calibration | 3/3 | 0 |
| `F1_loss_spike` | 0.5 | holdout | 2/2 | 0.0 |
| `F1_loss_spike` | 2.0 | calibration | 3/3 | 0 |
| `F1_loss_spike` | 2.0 | holdout | 2/2 | 0.0 |
| `F1_loss_spike` | 8.0 | calibration | 3/3 | 0 |
| `F1_loss_spike` | 8.0 | holdout | 2/2 | 0.0 |
| `F2_sustained_divergence` | 1.006 | calibration | 0/3 | n/a |
| `F2_sustained_divergence` | 1.006 | holdout | 0/2 | n/a |
| `F2_sustained_divergence` | 1.012 | calibration | 0/3 | n/a |
| `F2_sustained_divergence` | 1.012 | holdout | 0/2 | n/a |
| `F2_sustained_divergence` | 1.03 | calibration | 0/3 | n/a |
| `F2_sustained_divergence` | 1.03 | holdout | 0/2 | n/a |
| `F3_gradient_explosion` | 2.0 | calibration | 1/3 | 5 |
| `F3_gradient_explosion` | 2.0 | holdout | 2/2 | 3.0 |
| `F3_gradient_explosion` | 3.0 | calibration | 1/3 | 3 |
| `F3_gradient_explosion` | 3.0 | holdout | 2/2 | 1.5 |
| `F3_gradient_explosion` | 5.0 | calibration | 2/3 | 0.0 |
| `F3_gradient_explosion` | 5.0 | holdout | 0/2 | n/a |
| `F3_gradient_explosion` | 10.0 | calibration | 0/3 | n/a |
| `F3_gradient_explosion` | 10.0 | holdout | 1/2 | 0 |
| `F3_gradient_explosion` | 20.0 | calibration | 0/3 | n/a |
| `F3_gradient_explosion` | 20.0 | holdout | 0/2 | n/a |
| `F3_gradient_explosion` | 40.0 | calibration | 0/3 | n/a |
| `F3_gradient_explosion` | 40.0 | holdout | 0/2 | n/a |
| `F4_nan_loss` | None | calibration | 4/4 | 0.0 |
| `F4_nan_loss` | None | holdout | 3/3 | 0 |
| `F5_inf_gradient` | None | calibration | 4/4 | 0.0 |
| `F5_inf_gradient` | None | holdout | 3/3 | 0 |
| `F5b_nan_gradient` | None | calibration | 4/4 | 0.0 |
| `F5b_nan_gradient` | None | holdout | 3/3 | 0 |
| `F6_throughput_drop` | 0.004 | calibration | 4/4 | 0.0 |
| `F6_throughput_drop` | 0.004 | holdout | 3/3 | 0 |

## Infrastructure failures

None. Every run completed and every detector returned.

## Environment

```json
{
  "cpu_count": 12,
  "extra": {
    "calibration": {
      "injection_step": 120,
      "runs": 226,
      "splits": {
        "calibration": {
          "d_model": 32,
          "layers": 4
        },
        "holdout": {
          "d_model": 48,
          "layers": 6
        }
      },
      "steps": 200
    }
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
