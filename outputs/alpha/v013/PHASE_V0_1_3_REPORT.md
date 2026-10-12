# v0.1.3-alpha — MECHANISM-AWARE EVIDENCE ASSOCIATION

```
V0_1_3_ALPHA_READY = YES
DETECTOR_RELIABILITY = EXPERIMENTAL   (not upgraded)
READY_FOR_BLIND_VALIDATION = NO
```

```
tests        515 passing, 1 skipped   (490 -> 515, +25)
smoke        8/8 PASS
quickstart   PASS
installation PASS (editable + non-editable)
CLI          PASS (reads the mixed corpus from a separate process)
```

---

## W1 — M1 root cause

**M1 never produced a Gradient Episode, in any of the 10 runs**, and that is the cause:

```
grad.fill_(1e28)
  -> recorded grad.global_norm = 1e28 * sqrt(N) > float32 max (3.4e38)
  -> the OBSERVED norm is already `inf`
  -> grad.robust_zscore sees a non-finite value and returns None
  -> GRADIENT_EXPLOSION never fires
```

The injection was meant as "huge but finite". It is finite in the *tensor*, but the
**norm the observer records** overflows. The gradient detector is a *scale* detector and
correctly declines to classify a non-finite value.

**The evidence was never missing — it was mis-typed and mis-scoped.** The
`numerics.finiteness` detector caught it, but it multiplexes every metric through one
detector id, and `_observe` keyed episodes on `detector:rank` alone. So the
gradient-`inf`, the loss-`nan` and the counter collapsed into **one episode** whose
`metric_name` became `train.loss_nonfinite_count`. The mechanism signal was destroyed at
the episode boundary, and because that episode is typed `NAN_DETECTED` — a
*consequence* type — `link_consequences` found **no antecedent at all**.

**The brief's question 4 was tested and found false.** The linker was *not* querying only
active episodes; M3's antecedent was `CLOSED` and still linked. A "fix" for episode
availability would have changed nothing. This is why the trace is mandatory before coding.

## W3 — M3 false-association root cause

```
GRADIENT_EXPLOSION  confirmed at 80   (finite, x20 — a real scale anomaly)
NAN_DETECTED        onset    at 120   (independently injected)
gap 40 <= max_gap_steps 60, layers overlap -> POSSIBLE_CONSEQUENCE_OF
```

The rule was **temporal proximity + scope overlap and nothing else**. Two independently
injected faults 40 steps apart are indistinguishable from a fault and its consequence.

## W2 / W3 — the repair

**W2, evidence availability.** `numerics.finiteness` episodes are now metric-scoped. A
non-finite gradient and a non-finite loss are different signals and become different
episodes. Scoped deliberately: splitting `grad.robust_zscore` would break aggregate-vs-layer
confirmation, which *depends* on those sharing an episode, so per-metric detectors are
untouched.

**W3, mechanism-constrained association.** The decision now requires a full evidence
hierarchy, and the tier decided is recorded on every relation:

```
E1 temporal eligibility      -> otherwise no relation at all
E2 scope compatibility       -> otherwise no relation at all
E3 mechanistic propagation   -> POSSIBLE_CONSEQUENCE_OF
   (the antecedent must be a NON-FINITE value on an UPSTREAM metric within
    PROPAGATION_DELAY_STEPS = 2, i.e. one optimizer update)
E4 contradiction             -> otherwise independent, free to open its own incident

E1+E2 only                   -> PRECEDES   (order, never consequence)
```

`PROPAGATION_DELAY_STEPS = 2` is a **mechanism** bound, not a tuned window: one training
step is one optimizer update, so forward propagation surfaces within the next step or
two. `metric_class()` derives upstream/midstream/downstream from the metric names the
observer already emits — gradient -> optimizer -> parameters -> loss is the direction a
non-finite value can actually travel.

`reconcile_incidents` was narrowed to match: **only a consequence claim may group.**
Grouping on mere `PRECEDES` would repeat the same temporal-adjacency error one layer up.

## W4 — is the existing telemetry sufficient?

```
AVAILABLE: metric identity, non-finite flag, step, layer scope, which metric
           family went non-finite FIRST
MISSING:   no direct observation of the optimizer update boundary; no dtype/scaler
           state; the exact mechanism is INFERRED, not observed

IDENTIFIABLE:     M1 M2 M3 M4 M5 — all five
NON_IDENTIFIABLE: a finite upstream anomaly whose magnitude already overflowed the
                  update, with the downstream non-finite delayed past the propagation
                  bound. Nothing recorded distinguishes that from an independent
                  downstream fault -> MUST abstain.
```

**Answer: sufficient for this corpus, once metric identity survives to the episode
layer.** No new hot-path instrumentation was added. A true mechanism witness
(`optimizer.update_overflow`, O(1) per step) is proposed as the next research question,
not implemented.

---

## W5 — re-evaluated on the frozen 50-run corpus

```
                        consequence  order_only  antecedent_obs  supported
M1 gradient->nonfinite       10          10           10            10
M2 standalone nonfinite       0           0            0             0
M3 independent dual           0          10            0             0
M4 gradient + loss noise      0           0            0             0
M5 healthy                    0           0            0             0
```

```
RELATION        TP 10   FP 0   FN 0     precision 1.000   recall 1.000
M1 end-to-end relation recall            1.00   (10/10)
M1 conditional recall (antecedent obs)   1.00   (10/10)
M3 fabricated associations               0/10   <- the gate's top priority
M3 correctly independent                10/10
false association rate                   0.000
correct abstention rate                  0.800
```

**From v0.1.2's `TP 0 / FP 10 / FN 10 / precision 0.0 / recall 0.0`.** Relation precision
and recall both moved 0.0 -> 1.0 simultaneously, with **zero fabricated associations** —
achieved by *refusing* to claim consequence where only order was observed, not by
loosening anything.

## W7 — scoring unit audit

**Incident-level recall had the same structural bug as v0.1.1's `primary_recall`:**
`incident_ls.fn` was never incremented, so a totally-missed fault contributed
`{tp: 0, fp: 0, fn: 0}` and **vanished from the denominator**. Now:

* incident-level recall is `NOT ESTIMABLE` — no independent truth exists for how many
  incidents a fault should produce;
* `missed_primary_faults` reports the misses as an explicit count that always equals
  RUN-level `fn`, so nothing can disappear unnoticed.

Both corpora are scored separately, never mixed:

```
42-run single-fault           43-run note: runs 42 readable 42 excluded 0
  EPISODE   elig 97   TP 29 FP 60      precision 0.326   recall N/E
  INCIDENT  elig 25   TP 18 FP  7      precision 0.720   recall N/E
  RUN       elig 42   TP 18 FP  1 FN 18  precision 0.947   recall 0.500
  RELATION  elig  0                    NOT ESTIMABLE
  missed_primary_faults 18

50-run mixed-fault
  EPISODE   elig 100  TP 30 FP 30      precision 0.500   recall N/E
  INCIDENT  elig 40   TP 20 FP 20      precision 0.500   recall N/E
  RUN       elig 50   TP 20 FP  0 FN 20  precision 1.000   recall 0.500
  RELATION  elig 10   TP 10 FP  0 FN  0  precision 1.000   recall 1.000
  missed_primary_faults 20   order_only 20
```

## W6 — regression

**+25 tests.** Five pre-existing tests encoded the v0.1.2 rule
(*"a finite gradient 38-40 steps before a NaN is a `POSSIBLE_CONSEQUENCE_OF`"*) — the
exact behaviour measured as 10/10 fabricated. Each was **corrected with its reasoning in
the docstring and strengthened**, not weakened: e.g. `test_gradient_fault_then_nan_creates_a_relation`
became `test_gradient_fault_then_nan_is_precedence_not_consequence` and now additionally
asserts the relation is *not* a consequence claim. Four grouping fixtures were rebuilt to
carry real mechanism evidence rather than being relaxed.

## Known limitations

1. **`PRECEDES` relations are recorded but do not group.** 20 exist on the mixed corpus;
   they are order-only and correctly leave episodes independent.
2. **The mechanism is inferred, not observed.** No direct optimizer-boundary observation
   exists; `PROPAGATION_DELAY_STEPS = 2` encodes the inference.
3. **The non-identifiable case must abstain.** A finite upstream overflow with a delayed
   downstream failure cannot be distinguished from independence on current telemetry.
4. **M2/M3/M4/M5 all produce zero relations**, but M3 produces 10 order-only ones; the
   distinction between "correctly independent" and "order-only" is preserved in scoring.
5. **RUN recall is 0.500 on both corpora** — the ×2–×12 gradient boundary is unchanged
   and remains the binding constraint.
6. **Healthy false-incident rate 1/13 = 7.7 %**, above the 5 % gate.
7. **Observer race `UNRESOLVED`**; corrected flush semantics, no reproduction.
8. **The historical 23 is `HISTORICAL_23_NOT_FULLY_REPRODUCED`.**
9. **Pre-injection confirmed incidents = 0** on the single-fault corpus.

## Next research question

**Is a single O(1)-per-step optimizer-boundary witness enough to convert the inferred
mechanism into an observed one, and to resolve the non-identifiable case?** Concretely: a
boolean `optimizer.update_overflow` recorded at the update boundary would let the
association rule cite an *observed* propagation rather than inferring one from a
two-step delay, and would make the finite-overflow-with-delayed-failure case
decidable. That requires explicit authorisation to extend data collection on the
training hot path, and was **not** done this round.
