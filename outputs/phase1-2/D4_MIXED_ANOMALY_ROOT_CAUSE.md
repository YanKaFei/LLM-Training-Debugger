# D4 — F10 MIXED ANOMALY: ROOT CAUSE

**Status: RESOLVED. There was no detector interaction bug.**

---

## 1. The reported symptom

Phase 1.1 reported `F10_mixed_anomaly` at **0/6** — a run with a NaN loss *and* a
gradient explosion produced no detector fire at all — while `F4_nan_loss` in
isolation was caught **6/6**.

The brief's inference was reasonable: *"说明存在 interaction bug"*. The same input
in isolation works, the combination does not, therefore the combination breaks
something.

## 2. The actual cause

The F10 injection was:

```python
if fault_type == "F10_mixed_anomaly" and step == INJECTION_STEP:
    loss = float("nan")          # a Python float, not a torch.Tensor
```

`training_debugger.adapters.pytorch.adapter.record_loss` then executed:

```python
value = float(loss.detach().to(torch.float32).item())
# AttributeError: 'float' object has no attribute 'detach'
```

**The run died at the injection step.** It never produced a gradient explosion,
never produced a NaN gradient, and never reached a detector.

`F4_nan_loss` used `loss = loss * float("nan")` — which keeps the tensor — and
therefore worked. That single character of difference between the two scenarios is
the entire explanation for 6/6 versus 0/6.

## 3. Why it was reported as a detector failure

Two independent defects had to coincide:

| # | Defect | Effect |
|---|---|---|
| 1 | The F10 injection produced a non-tensor loss | the run crashed |
| 2 | The harness recorded `fired_detectors: []` plus a `run_error`, and **the scoring function ignored `run_error`** | the crash was scored as a detector **false negative** |

Defect 2 is the more serious. Defect 1 is a test-authoring slip; defect 2 converts
an infrastructure failure into evidence about detector quality, and it did so
silently. All six F10 runs carry
`AttributeError: 'float' object has no attribute 'detach'`, and every one was
counted against detector recall.

**Phase 1.1's quoted system recall of 0.857 was therefore computed with six crashed
runs counted as misses.**

## 4. Fixes

1. **Injection corrected** to `loss = loss * float("nan")` — a NaN tensor, in both the trace harness and the calibration harness.
2. **Crashed runs are excluded from scoring**, never counted as misses, and are reported in a dedicated *Infrastructure failures* section. `aggregate()` excludes them and reports `excluded_run_errors`.
3. A regression test asserts that a NaN loss **and** a gradient explosion both produce their own detection — the two detectors must not interfere.

## 5. Verification

With the corrected injection, traced over 3 seeds:

```
F10_mixed_anomaly  fired = ['grad.robust_zscore', 'loss.robust_zscore', 'numerics.finiteness']
F4_nan_loss        fired = ['grad.robust_zscore', 'numerics.finiteness']
F3_gradient_explosion fired = ['grad.robust_zscore', 'loss.robust_zscore']
```

`numerics.finiteness` fires on F10 in **3/3**. The mixed fault is detected.

## 6. D4.3 invariant — upheld and tested

> One detector must never prevent another detector from seeing its own valid input.

The invariant was **not** violated in this case, because no detector ever ran. It
is nonetheless now tested directly: `tests/unit/test_detector_isolation.py`
exercises the D4.4 pairwise matrix (loss+gradient, NaN+gradient, throughput+loss,
LR+loss, NaN gradient+throughput) and asserts that each detector's own finding
survives the presence of the others.

## 7. D4.4 pairwise interaction matrix

| Combination | Result |
|---|---|
| loss spike + gradient explosion | both detectors report their own finding |
| NaN loss + gradient explosion | `numerics.finiteness` and `grad.robust_zscore` both fire |
| throughput drop + loss spike | neither suppresses the other |
| LR jump + loss spike | neither suppresses the other |
| NaN gradient + throughput drop | neither suppresses the other |

## 8. What this teaches

The brief's hypothesis was well-formed and wrong, and it was wrong because a
harness reported a crash as a measurement. The instruction *"不要先 patch，必须定位"*
— locate before patching — is what surfaced it. Patching the detector would have
left defect 2 in place, and it would have corrupted every subsequent measurement.
