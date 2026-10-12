# W3 — CHALLENGE CORPUS

54 runs, 9 cases × 2 configurations × 3 seeds. All real PyTorch training.

| Case | Designed to test | Observed |
|---|---|---|
| **C1** true propagation, optimizer invoked | mechanism link forms | **consequence claim (6/6)** ✓ |
| **C2** gradient anomaly, update explicitly skipped | refuse | **refused** ✓ |
| **C3** invoked + *independent* NaN | refuse despite `invoked` | **refused** ✓ — but see below |
| **C4** finite parameters, real downstream overflow | do not deny propagation just because the count is 0 | **MISSED** ✗ |
| **C5** norm reduction overflow only | only report the artifact | **refused** ✓ |
| **C6** invoked with no gradients | refuse | **refused** ✓ |
| **C7** gradient accumulation | use the real optimizer boundary | **consequence claim (3/6)** ✓ |
| **C8** delayed manifestation | do not deny a fact for lack of an in-window signal | **consequence claim (3/6)** ✓ |
| **C9** two optimizers, one skipped | keep statuses separate | **refused** ✓ |

## C3 could not be constructed as specified — and that matters

C3 required: *gradient anomaly observed → optimizer invoked → independent NaN one step
later*. It is **not constructible with the current detector**. Measured:

```
C3 episodes:  LOSS_SPIKE train.loss @61      (the injection)
              NAN_DETECTED train.loss @62    (the independent NaN)
              GRADIENT_EXPLOSION grad.global_norm @71   <-- AFTER the NaN
```

The gradient detection latency is ~11 steps, so a downstream failure within the
2-step propagation window **cannot have an observed gradient antecedent at all**. C3 is
therefore refused for the correct *outcome* but not by the discrimination the case was
designed to exercise. Raising the injection from ×20 to ×100 did not fix it.

**Consequence:** the witness's ability to separate "invoked and propagated" from
"invoked and independent" is **NOT VERIFIED**, because the case that would test it cannot
be built here. This is reported rather than claimed.

## C4 is a genuine coverage gap

A real downstream overflow with finite parameters produces **no** non-finite midstream
observation, so no mechanism is available and the consequence is missed (20 FN across
the corpus). Consistent with the brief's own warning against denying propagation merely
because `parameter_nonfinite_count == 0`.
