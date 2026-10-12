# CAPABILITIES

Generated from `detector_registry()`. **Nothing in this build is `VALIDATED`.**

```
validated_capabilities = []
```

Two axes are reported and they are not the same:

```
detector status: ENABLED / EXPERIMENTAL / DISABLED     -- does it run?
capability:      IMPLEMENTED / EXPERIMENTAL / VALIDATED / UNSUPPORTED  -- can it be trusted?
```

## Per-module activation health

| | |
|---|---|
| Capability | `DEAD` / `NONFINITE` / `SATURATED` per **named module**, from exact observations only |
| Evidence | a frozen output projection was named and dated at step 1 with OBSERVED confidence 1.0; its healthy sibling unaffected |
| Source | `adapters/pytorch/activations.py`, detector `activation.health` v1.0 |
| Cost | **+3.3 %** with edge-triggered emission (+24.7 % if emitted every step) |
| Default | **probe OFF**. The detector is registered and costs nothing without it |
| Limitation | `vanishing` / `exploding` are **not** produced — they need a reference this project does not have for activations. A sampled observation cannot establish DEAD |
| Status | **EXPERIMENTAL** — one synthetic model, no blind holdout |

## Two-run divergence localization

| | |
|---|---|
| Capability | `compare <ref> <suspect>` reports the first step at which two runs stop agreeing |
| Evidence | 9/9 fault pairs localized to the exact injection step; 3/3 clean pairs abstained. Paired corpus, real events, 4-layer config, 3 seeds |
| Source | `src/training_debugger/comparison/divergence.py`, rule `v0.1.8-1.0` |
| Reproduce | `tests/experiments/paired_divergence_corpus.py` then `compare` |
| Limitation | names a **step and a metric**, not a kernel or a tensor. Excludes timing metrics by default. Untested on models larger than 6 layers |
| Status | **EXPERIMENTAL** — measured on one synthetic paired corpus, no blind holdout |

## Supported, with limits

| Anomaly type | Capability | Evidence |
|---|---|---|
| `GRADIENT_EXPLOSION` | **EXPERIMENTAL** | episode precision 0.181 (Phase 1.3), recall 0.58 |
| `NAN_DETECTED` | **EXPERIMENTAL** | episode precision 0.314 (Phase 1.3, retired as mis-typed) |
| `INF_DETECTED` | **EXPERIMENTAL** | shares the finiteness detector |
| `LOSS_SPIKE` | **EXPERIMENTAL**, promotion **OFF** | H2 noisy healthy loss alarmed 8/14; candidates are recorded but no incident is opened |

## Unsupported

| Anomaly type | State | Why |
|---|---|---|
| `LOSS_DIVERGENCE` | **UNSUPPORTED** | `loss.trend_slope` measured precision 0.000, recall 0.000 — it fires only *before* faults. Detector disabled, retained for audit. |
| `THROUGHPUT_REGRESSION` | **UNSUPPORTED** | recall 0.250 with 9 pre-injection false positives; compares raw step time without workload normalisation. Detector disabled. |
| `LOSS_LEVEL_SHIFT` | **NOT IMPLEMENTED** | |
| `LOSS_OSCILLATION` | **NOT IMPLEMENTED** | |

## The honest headline

```
injected fault detected at the incident layer : ~44 %
healthy run falsely alarmed                   : ~2.4 %
earliest claim made                           : 28 % of fault runs
earliest claim correct, when made             : 75 %
```

The system is **precise about what it confirms and silent about most of what it does
not**. That is deliberate: where healthy and faulty distributions overlap it returns
`INSUFFICIENT_EVIDENCE` rather than guessing.

**Minimum development-supported gradient perturbation: roughly ×20.** Below ×12 the
classes are not separable on the signals already collected. This is a development
figure from a synthetic scaling benchmark and **does not extrapolate** to a physical
"20× limit" on real training incidents.
