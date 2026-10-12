# v0.1.4-alpha — CAUSAL IDENTIFIABILITY & ADVERSARIAL VALIDATION

```
V0_1_4_ALPHA_READY = YES
MECHANISM_GENERALIZATION = LIMITED (was indistinguishable; now identified for this corpus)
DETECTOR_RELIABILITY = EXPERIMENTAL
READY_FOR_BLIND_VALIDATION = NO
```

```
tests        516 passing, 1 skipped
smoke        8/8 PASS
quickstart   PASS
installation PASS (editable + non-editable)
CLI          PASS
```

---

## W1 — the v0.1.3 100 % was signature-dependent

**`SYNTHETIC_GENERATION_SIGNATURE_DEPENDENCE` CONFIRMED.** With identical timing, scope
and target, the rule's decision flipped on one field:

```
antecedent non-finite (gradient norm inf)  -> POSSIBLE_CONSEQUENCE_OF
antecedent FINITE (gradient scaled x20)    -> PRECEDES
```

The M1/M3 result therefore rested on how M1 was *generated*, not on a general law.

**Finite tensor / infinite norm are not distinguished.** Measured:

```
gradient TENSORS all finite : True
true norm (float64)         : 2.451e+30
float32 norm reduction      : inf
```

The **reduction overflows while the tensor does not**, and the observer records only the
norm. So the telemetry cannot distinguish "finite tensor, overflowing reduction" from
"non-finite tensor". Reported as a limitation, not papered over.

## W2/W3 — the adversarial corpus, and what it broke

45 real training runs across 9 cases. The first run of the corpus **falsified v0.1.3**:

```
A2  (independent fault, suppressed update)   5 x POSSIBLE_CONSEQUENCE_OF  <- FABRICATED
A8Y (independent fault, suppressed update)   5 x POSSIBLE_CONSEQUENCE_OF  <- FABRICATED
```

**Root cause, found by measurement rather than reasoning:**

```
A8X (REAL propagation)  step 80  param.global_norm = inf      <- the update overflowed
A8Y (suppressed update) step 80  param.global_norm = 22.37    <- the update did NOT
A2  (suppressed update) step 80  param.global_norm = 22.37    <- the update did NOT
```

The chain is gradient → **optimizer update** → **parameters** → loss. The observer
**records** `param.global_norm`, the rule **ignored** it, and the finiteness detector did
not even watch it. The midstream witness was available and unused.

**Two defects fixed:**

1. `FinitenessDetector.metric_names` now includes `param.global_norm` — widening an
   existing detector's watch list to a metric the observer **already records**. Not a new
   detector, not a new telemetry source.
2. `EpisodeEngine` grouped by `(detector, rank)` while keying by metric, so both metrics
   landed in one group, the key came from `group[0]`, and `metric_name` was overwritten
   by the last member — an episode keyed on `grad.global_norm` reported
   `param.global_norm`. **The witness replaced the upstream observation instead of
   joining it.** Grouping is now metric-scoped at both levels.

**Rule change:** E3 now requires a non-finite observation at **every** stage of the chain,
not merely an upstream one. `RELATION_RULE_VERSION = v0.1.4-3.0`.

## W2.1 — observational equivalence: tested, and NOT achievable here

The brief's A8 asks for two cases with identical observable fields and different causal
truth. **The pair could not be made truly equivalent**, and that is the finding:

> A genuine propagation leaves a persistent footprint — non-finite parameters, then
> non-finite gradients and losses at every subsequent step. An independent one-step
> injection does not. `param.global_norm` is *inf* at the injection step for A8X and
> finite for A8Y.

So this corpus does **not** demonstrate the abstention case; it demonstrates that the two
histories are **separable with the telemetry already recorded**. A truly equivalent pair
would require an intervention that leaves no trace, which this environment cannot
construct. Reported as `OBSERVATIONAL_EQUIVALENCE = NOT ACHIEVED`, not as PASS.

## W3/W6 — results, per corpus, never merged

```
ORIGINAL 50-run mixed-fault
  TP 10   FP 0   FN 0   TN 40      precision 1.000   recall 1.000
  positive_coverage 1.000   negative_rejection 1.000
  false_association 0.000   true_abstention 0.250

ADVERSARIAL 45-run
  TP 10   FP 0   FN 0   TN 35      precision 1.000   recall 1.000
  positive_coverage 1.000   negative_rejection 1.000
  false_association 0.000   true_abstention 0.571
```

`true_abstention` is now a **separate quantity** from negative rejection: it counts runs
where an order-only `PRECEDES` was recorded, which is an explicit refusal to call
something a consequence. v0.1.3 reported `coverage 1.000` and `abstention 0.800`
simultaneously, which was only possible because "no claim" was being counted as both a
success and an abstention.

Per case on the adversarial corpus:

```
A1  suppressed update, no downstream       0 claims        correct refusal
A2  suppressed update + INDEPENDENT fault  PRECEDES only   FIXED (was 5 fabricated)
A3  finite anomaly + later fault           PRECEDES only   correct
A4  genuine propagation, delayed           POSSIBLE_CONS.  correct (recovered)
A5  large anomaly, no downstream failure   0 claims        correct refusal
A6  standalone numerical fault             0 claims        correct refusal
A7  skipped update + unrelated fault       PRECEDES only   correct
A8X genuine propagation                    POSSIBLE_CONS.  correct
A8Y independent fault, identical metrics   PRECEDES only   FIXED (was 5 fabricated)
```

## W5 — optimizer witness

RFC delivered. Recommendation: `post_update_nonfinite_detected` reusing the parameter
norm already computed, plus a per-optimizer-step `step_applied` boolean — **no new
reduction, no new sync**. The three candidate facts are kept distinct; collapsing them
into one `optimizer.update_overflow` would lose the "no update happened" case that
mixed precision produces routinely. **AUTHORISATION REQUIRED**; not implemented.

## W7 — regression

Five pre-existing tests encoded the v0.1.3 rule and were updated to carry the midstream
witness, each with its reasoning recorded. Adding that witness also means a single
propagation now records **more** relations (gradient→parameters, gradient→loss,
parameters→loss), so three assertions that pinned exact counts were changed to assert
semantics — with a comment saying why, so the change is auditable rather than silent.

## Honest summary

**What improved:** the fabricated associations are gone (10 → 0 across both corpora),
and the reason is mechanistic — the propagation is now required to be observable at every
stage rather than inferred from two step indices. A genuine defect was found in the
episode layer that had been silently discarding the metric identity the rule needed.

**What is NOT established:** `MECHANISM_GENERALIZATION = LIMITED`. The result holds on
these two corpora, both synthetic and both generated by the same harness. Observationally
equivalent histories were **not** constructed, so the abstention path under true ambiguity
remains unexercised. The two-step delay bound still encodes an unverified assumption
about step/update alignment, which gradient accumulation would break.

## Next product decision

Authorise the optimizer-boundary witness, or accept the delay-bound inference and
document its scope. Everything else in this round is closed.
