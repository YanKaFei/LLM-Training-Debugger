# v0.1.6-alpha — OPTIMIZER WITNESS: END-TO-END DISCRIMINABILITY TRIAL

```
V0_1_6_ALPHA_READY = YES
PRODUCTION DECISION = KEEP_RESEARCH_ONLY
IDENTIFIABILITY = INSUFFICIENT
DETECTOR RELIABILITY = EXPERIMENTAL   (unchanged)
READY_FOR_BLIND_VALIDATION = NO
```

```
tests 531 passing, 1 skipped · smoke 8/8 · quickstart PASS
installation PASS (editable + non-editable) · CLI PASS · cross-process restart PASS
```

---

## The question, and the answer

> Does the Optimizer Boundary Witness provide independent, effective and generalizable
> new evidence for propagation judgement?

**No — measured, not asserted.** Paired on 54 byte-identical runs, with the witness given
its strongest fair reading:

```
                         TP   FP   FN   TN    precision   recall
WITNESS_DISABLED          4    0   20   30      1.000     0.167
WITNESS_ENABLED           4    0   20   30      1.000     0.167

recovered true 0 · new false 0 · lost true 0 · changed abstentions 0
```

**Every case is identical with the witness on and off.**

## Why it cannot help

A mechanism claim still requires an **observed antecedent inside the propagation window**,
and the gradient detector's aggregate latency is **~11 steps**. A downstream failure close
enough to be mechanically attributable therefore has **no observed antecedent at all** —
so the witness clause never fires. The witness observes a boundary the rule cannot reach.

## What actually discriminates (and was already available)

```
C1  param.global_norm == nan, nonfinite_count 4160  -> real corruption  -> claim
C3  param.global_norm absent                        -> never corrupted  -> refuse
C5  param.global_norm == inf, nonfinite_count 0     -> reduction artifact -> refuse
```

The signal that works is the **non-finite element observation**, not the optimizer
boundary. The witness tells us *when an update was attempted*; it does not tell us whether
that update *caused* a later value.

## C3 could not be constructed — and that is a finding

C3 (gradient anomaly observed → optimizer invoked → independent NaN one step later) is
**not constructible here**. Measured ordering:

```
LOSS_SPIKE        train.loss            @61   (injection)
NAN_DETECTED      train.loss            @62   (independent NaN)
GRADIENT_EXPLOSION grad.global_norm     @71   <-- AFTER the NaN
```

Raising the injection ×20 → ×100 did not change this. The witness's ability to separate
"invoked and propagated" from "invoked and independent" is therefore
**`NOT VERIFIED`**, not demonstrated.

## C4 is a genuine coverage gap

A real downstream overflow with **finite** parameters produces no non-finite midstream
observation, so no mechanism is available and the consequence is missed (20 FN). The brief
warned against exactly this: *"不能因为 parameter_nonfinite_count 为零就否认所有可能传播"* —
and the system currently does.

## Compatibility

```
SGD PASS · Adam PASS · Gradient Accumulation PASS · Multiple Optimizers PASS
REAL AMP: NOT VERIFIED — no CUDA device on this machine; the skip path is tested at the
          harness level only and no AMP compatibility claim is made.
```

## Performance

```
witness increment: total +0.30% · median -0.11% (noise floor) · p95 +0.97%
event volume: one record per optimizer step · device syncs introduced: none
```

**Not claimed:** no GPU benchmark exists here, and the standing finding that observer
overhead exceeds the **<3 % core target** at the largest workload was **not** re-measured
and is **not** resolved.

## Decision

```
KEEP_RESEARCH_ONLY
```

Not `INTEGRATE_OPT_IN_EXPERIMENTALLY`: the measured benefit is exactly zero, and the one
case built to expose the witness's limits could not be constructed, so its discriminating
power is unverified. Integrating would repeat v0.1.4's error.

Not `REJECT_AS_DIAGNOSTIC_EVIDENCE`: the witness never produced a false association in any
configuration and its contract is honest about what it does not know. It is **insufficient**,
which is a different finding from **wrong**.

**Conditions to revisit:** (1) an antecedent must become observable inside the propagation
window; (2) C3 must become constructible; (3) C4 must be addressed.

`validated_capabilities = []` · `OptimizerWitness` remains default-disabled and is not
wired into promotion.

## Governing principle, applied

> Observing that an optimizer step was invoked is useful timing evidence. It is not proof
> that parameters changed — `parameters_changed` is `None` on every record, because
> proving it needs a scan this module may not do. Even a verified parameter change would
> not, by itself, prove a later anomaly was caused by it.

The round's value was **decisive negative evidence**: a witness that works, costs almost
nothing, and demonstrably adds nothing to the decision it was built to improve.
