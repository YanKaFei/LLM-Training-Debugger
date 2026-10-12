# v0.1.5-alpha — OPTIMIZER BOUNDARY WITNESS

```
V0_1_5_ALPHA_READY = YES
MECHANISM BENEFIT = NOT VERIFIED (the witness is not integrated into production)
PRODUCTION INTEGRATION RECOMMENDATION = MORE EVIDENCE REQUIRED
DETECTOR RELIABILITY = EXPERIMENTAL
READY_FOR_BLIND_VALIDATION = NO
```

```
tests        531 passing, 1 skipped   (516 -> 531, +15)
smoke        8/8 PASS
quickstart   PASS
installation PASS (editable + non-editable)
CLI          PASS
```

---

## W0 — the audit found a defect that invalidated v0.1.4's headline

`*.global_norm` maps definitionally, not probabilistically:

```
inf  ->  nonfinite_count == 0  ->  every element FINITE (the FP32 reduction overflowed)
nan  ->  nonfinite_count > 0   ->  non-finite elements present
```

`FinitenessDetector` tested `isfinite(value)` alone and `DetectorContext` carries no event
metadata, so it could not consult the count. **The antecedent of v0.1.3's and v0.1.4's
mechanism chain was `grad.global_norm == inf` — a float32 reduction artifact with every
element finite.**

> v0.1.4's relation precision 1.0 / recall 1.0 was measuring an artifact.

**Repair.** `inf` on a `*.global_norm` metric is now its own kind,
`norm_reduction_overflow`, still reported with a reason naming it an artifact, carried
onto the episode, and excluded by `is_nonfinite_observation()` from being mechanism
evidence. A test that had encoded the misconception
(`test_A3b_inf_is_observed`) was corrected, and the synthetic harness that emitted
`inf` alongside `nonfinite_count = 1` — a pair the real function never produces — was
fixed to emit `nan`.

## W1/W2 — the event contract and the witness

`OptimizerWitness`, **disabled by default**, built on the public
`register_step_pre_hook` / `register_step_post_hook` API. The three facts are kept
separate and never collapsed:

```
ATTEMPTED   the training code asked for an update
INVOKED     optimizer.step() was entered            (hook)
SKIPPED     the update was deliberately not applied (caller-supplied reason)
UNKNOWN     could not be established                (and UNKNOWN is never False)
```

`parameters_changed` is **always `None`**. Proving "invoked but unchanged" needs a
parameter scan, which this may not do — and the hooks fire even with no gradients, so
invocation alone proves nothing applied.

An optimizer without hook support attaches `False` and reports `UNKNOWN`. No confident
conclusion is manufactured for an unsupported optimizer.

## W3 — training step semantics

Covered by 14 new tests: SGD, Adam, **gradient accumulation** (`training_step_id` and
`optimizer_step_id` are different concepts and are recorded as such), explicit skip,
multiple optimizers with independent status, hook attach/detach, provenance, and a test
asserting that **enabling the witness changes no loss value**.

`PROPAGATION_DELAY_STEPS = 2` is therefore no longer offered as evidence of an optimizer
update anywhere.

## W4 — mechanism evaluation

```
ORIGINAL 50-run   TP 0  FP 0  FN 10  TN 40   precision N/E  recall 0.000
ADVERSARIAL 45    TP 0  FP 0  FN 10  TN 35   precision N/E  recall 0.000
                  false association 0.000   negative rejection 1.000
```

**Zero fabricated associations and zero true positives.** With the reduction artifact
excluded the existing telemetry contains **no direct observation of the optimizer
boundary**, so the two genuine propagations (A8X, A4) are now missed and the two
independent faults (A2, A8Y) are correctly refused.

This is the honest outcome the brief anticipated: *"如果现有观测确实不足，请在报告中提出下一
阶段所需的最小、低开销、明确授权的新增观测方案"*. It is reported as a **coverage
collapse**, not hidden by re-tuning.

## W5 — overhead, paired and interleaved

```
metric      off       on        delta     delta%
total_s     0.3704    0.3715    +0.0011   +0.30%
median_ms   3.0725    3.0692    -0.0034   -0.11%
p95_ms      3.2678    3.2995    +0.0317   +0.97%
event volume: off 0 records, on 120 (one per optimizer step)
```

7 interleaved repeats with alternating order. The median delta is **negative**, which
means the effect is at the noise floor of this harness; the p95 delta of +0.97 % is the
more conservative reading. Either way it is **well inside the 3 % core target**.

**This does not resolve the standing overhead finding.** The witness adds no new
parameter reduction and no new device sync — it observes a boundary the training loop
crosses anyway — but the separate, pre-existing result that observer overhead exceeds
the <3 % core target at the largest workload is untouched and is not re-measured here.

## W6 — regression

`531 passing, 1 skipped` (+15). Smoke 8/8, quickstart, four-verdict CLI, cross-process
recovery, editable and non-editable installation all re-run and passing.

## Recommendation

```
PRODUCTION INTEGRATION RECOMMENDATION = MORE EVIDENCE REQUIRED
```

The witness is implemented, tested, defaults **off**, and measured as cheap. It is **not**
wired into production promotion, for three reasons, each of which is a measurement rather
than an opinion:

1. **It has never been evaluated end-to-end on a corpus.** W4 measured the relation layer
   with the artifact removed, not with the witness supplying a replacement signal.
2. **The wrong-norm was a bigger defect than the missing witness.** A witness bolted onto
   a rule that still reads `inf` as non-finite would inherit the same error.
3. **It cannot yet distinguish propagation from injection.** The witness observes the
   boundary; it does not observe that a *particular* downstream value came through it.

**Next step, and the honest reason to stop here:** run the adversarial corpus with the
witness enabled as the midstream evidence in place of `param.global_norm == inf`, and
compare A8X/A4 recall against A2/A8Y false associations. Until that number exists,
integrating would repeat the v0.1.4 mistake of shipping a rule whose evidence had not been
validated against the cases designed to break it.
