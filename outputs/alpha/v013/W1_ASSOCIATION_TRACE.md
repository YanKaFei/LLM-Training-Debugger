# W1 — M1 / M3 ASSOCIATION TRACE

Produced **before** any change to `link_consequences()`, as required.

---

## The brief's six questions, answered

### 1. Did M1 ever produce a Gradient Episode?

**No — never, in any of the 10 runs.** And that is the root cause.

```
grad.fill_(1e28)
  -> recorded grad.global_norm  =  1e28 * sqrt(N_params)  >  float32 max (3.4e38)
  -> the OBSERVED norm is already `inf`
  -> grad.robust_zscore sees a non-finite value and returns None
  -> GRADIENT_EXPLOSION never fires
```

The injection was intended as "a huge but **finite** gradient". It is finite in the
*tensor*, but the **norm the observer records** overflows. The gradient detector is a
*scale* detector and correctly declines to classify a non-finite value.

**The gradient evidence was never missing. It was mis-typed and mis-scoped.**

### 2. If it existed, when did it form and close?

It never formed as a gradient episode. What formed instead, per run:

```
step 80  numerics.finiteness  grad.global_norm            value=inf  CERTAIN
step 81  numerics.finiteness  train.loss                  value=nan  CERTAIN
step 81  numerics.finiteness  train.loss_nonfinite_count  value=1.0  CERTAIN
step 131 numerics.finiteness  train.loss                  value=nan
step 181 numerics.finiteness  train.loss                  value=nan
```

**The mechanism is fully observable in the raw anomalies: the gradient went non-finite
at step 80, one step before the loss did.**

### 3. Why was that episode ineligible for association?

Three compounding reasons, all in the episode layer:

1. **Signal-key collapse.** `_observe` built the key as `"{detector}:rank{rank}"`.
   `numerics.finiteness` multiplexes *every* metric through one detector id, so the
   gradient-`inf`, the loss-`nan` and the counter landed in **one episode**, which
   recorded `metric_name = train.loss_nonfinite_count`. **The metric identity — the
   mechanism evidence — was destroyed at the episode boundary.**
2. **Type.** That episode is typed `NAN_DETECTED`, which is in `CONSEQUENCE_TYPES`.
3. **Eligibility.** The antecedent set was built from `PRIMARY_TYPES` only, so with a
   `NAN_DETECTED` antecedent there was **no antecedent at all** and the loop `continue`d.

### 4. Was it wrongly querying only the current active episode?

**No.** The linker received episodes in any state, and M3's antecedent was `CLOSED` yet
still linked. Lifecycle was not the obstacle. The hypothesis was reasonable but
**wrong** — which is precisely why the trace is required before coding: a "fix" for
episode availability would have changed nothing.

### 5. Why did M3's independent faults satisfy the rule?

```
GRADIENT_EXPLOSION  confirmed at step 80   (finite, x20 — a real scale anomaly)
NAN_DETECTED        onset    at step 120   (independently injected)
gap = 40 <= max_gap_steps = 60             layers overlap
-> POSSIBLE_CONSEQUENCE_OF
```

The rule was **temporal proximity + scope overlap and nothing else**. Two independently
injected faults 40 steps apart are indistinguishable from a fault and its consequence —
exactly the temporal-adjacency-as-causation the project forbids.

### 6. Which existing fields could support or refute a mechanistic link?

| Field | M1 | M3 | Discriminating? |
|---|---|---|---|
| step / candidate_onset_step | 80 -> 81 (**gap 1**) | 80 -> 120 (**gap 40**) | **yes, decisively** |
| `metric_name` | `grad.global_norm` -> `train.loss*` | `grad.global_norm` -> `train.loss*` | **yes** — M1's antecedent is non-finite, M3's is a finite *scale* anomaly |
| non-finite flag | antecedent `inf` | antecedent finite | **yes** |
| affected_layers | overlap | overlap | no |
| gradient_band | — | `CONFIRMED` | no |

**W4 answer: the existing telemetry IS sufficient**, once metric identity survives to the
episode layer. No new hot-path instrumentation is required.

---

## W4 observability audit

```
AVAILABLE_MECHANISM_EVIDENCE
  metric identity (per anomaly; now per episode too)
  non-finite flag (the observation value itself)
  step, per-layer index, affected_layer scope
  the ordering: which metric family went non-finite first

MISSING_MECHANISM_EVIDENCE
  no direct observation of the optimizer update boundary
  no parameter-integrity sample taken between step and step+1
  no dtype / scaler state
  => the *exact* mechanism is inferred, not observed

IDENTIFIABLE_CASES
  M1  non-finite value appears upstream at t, downstream at t+1..t+2   -> IDENTIFIABLE
  M2  non-finite value downstream only, nothing upstream              -> IDENTIFIABLE
  M3  upstream anomaly FINITE, downstream non-finite much later        -> IDENTIFIABLE
  M4  finite upstream anomaly, no downstream non-finite                -> IDENTIFIABLE
  M5  nothing                                                          -> IDENTIFIABLE

NON_IDENTIFIABLE_CASES
  an upstream anomaly that is finite but whose magnitude ALREADY overflowed the
  parameter update, with the downstream non-finite value delayed past the propagation
  bound. Nothing recorded distinguishes that from an independent downstream fault.
  -> must abstain (PRECEDES or no relation), never claim a consequence.

MINIMUM FUTURE OBSERVATION (proposed, NOT implemented this round)
  one post-update parameter-finiteness sample per step -- already emitted as
  `param.global_norm`; the gap is that it is not consulted as a mechanism witness.
  A true mechanism witness would be a single `optimizer.update_overflow` boolean,
  which is O(1) per step and needs explicit authorisation to add.
