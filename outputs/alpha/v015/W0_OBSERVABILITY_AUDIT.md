# W0 — WHAT IS ACTUALLY OBSERVED

Audit of `param.global_norm` and `grad.global_norm`: source, dtype, reduction, timing.

---

## 1. Source and computation

`src/training_debugger/adapters/pytorch/metrics.py`

```python
value = _total_from_norms(per_tensor_l2_norms(tensors))     # FP32 L2, fused
if value != value or value == float("inf"):
    nonfinite = sum(tensor_nonfinite_count(t) for t in tensors)
    if nonfinite:
        value = float("nan")          # <-- non-finite ELEMENTS exist
    return NormResult(value, NORM_TYPE_L2, tensors, elements, nonfinite)
return NormResult(value, NORM_TYPE_L2, tensors, elements, 0)
```

**Both** `global_gradient_norm` and `global_parameter_norm` share this logic, and the
mapping is **definitional, not probabilistic**:

```
*.global_norm == inf   ->  nonfinite_count == 0  ->  every element is FINITE
*.global_norm == nan   ->  nonfinite_count > 0   ->  non-finite elements present
```

Timing: `record_parameters()` runs **after** the optimizer phase of the step, so the
value is post-update.

## 2. The five cases, measured

| Case | Observed `param.global_norm` | `nonfinite_count` | Element truth |
|---|---|---|---|
| **1** params finite, FP32 reduction overflows | `inf` | **0** | all finite |
| **2** params actually contain `Inf` | `nan` | 4160 | non-finite |
| **3** params actually contain `NaN` | `nan` | 4160 | non-finite |
| **4** `step()` called, parameters unchanged | value unchanged | 0 | all finite |
| **5** `step()` skipped by `GradScaler` | value unchanged | 0 | all finite |

Measured directly:

```
CASE 1  elements finite=True   global_parameter_norm=inf  nonfinite_count=0
CASE 2  global_parameter_norm=nan  nonfinite_count=4160
CASE 3  global_parameter_norm=nan  nonfinite_count=4160
```

Cases 4 and 5 are **indistinguishable** from each other and from ordinary steps on the
value alone: a skipped update leaves the norm where it was. That is why the optimizer
witness of W2 is necessary, and why `parameters_changed` is never inferred.

## 3. The defect this audit found

**`param.global_norm == inf` was being auto-derived as a non-finite observation.**

`FinitenessDetector` tested `isfinite(value)` only. It never consulted
`nonfinite_count`, and `DetectorContext` carries no event metadata, so it *could not*.

Consequences, both measured:

* **v0.1.4** added `param.global_norm` to the finiteness watch list. A pure
  norm-reduction overflow with perfectly healthy parameters therefore produced a
  "non-finite parameter" observation — a false midstream witness.
* More seriously, the **antecedent** of v0.1.3/v0.1.4's mechanism chain was
  `grad.global_norm == inf`, which by the mapping above means *the reduction overflowed
  and every element was finite*. **The whole mechanism chain was keyed on a reduction
  artifact.**

The brief's principle, applied literally:

> **A non-finite norm is not necessarily a non-finite tensor.**

## 4. The repair

`FinitenessDetector` now classifies `inf` on a `*.global_norm` metric as its own kind:

```
kind = "norm_reduction_overflow"
```

It is still reported, with a reason that names it a reduction artifact. The observation
kind is carried onto the episode (`metadata["anomaly_kind"]`) and
`is_nonfinite_observation()` excludes it from being mechanism evidence.

## 5. Effect on the corpora — the honest headline

With the artifact excluded, the relation layer produced:

```
ORIGINAL 50-run mixed corpus : 20 relations, 0 consequence claims
ADVERSARIAL 45-run corpus    :  20 order-only, 0 consequence claims

A8X genuine propagation  ->  MISSED
A4  genuine propagation  ->  MISSED
A2 / A8Y independent     ->  correctly refused
```

**Zero fabricated associations and zero true positives.** The v0.1.4 result of
precision 1.0 / recall 1.0 was measuring the artifact; with it removed, the existing
telemetry contains **no direct observation of the optimizer boundary at all**. This is
what makes the witness necessary rather than optional, and it is reported rather than
hidden behind a re-tuned rule.
