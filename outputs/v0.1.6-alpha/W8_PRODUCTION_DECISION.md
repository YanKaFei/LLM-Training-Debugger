# W8 — PRODUCTION DECISION

```
KEEP_RESEARCH_ONLY
```

## On what evidence

The witness **works**. It observes the optimizer boundary correctly, keeps
`ATTEMPTED` / `INVOKED` / `SKIPPED` / `UNKNOWN` distinct, never fabricates a conclusion
for an unsupported optimizer, costs ~0.3 % total / +0.97 % p95, adds no device sync and
no parameter scan, and is verified not to change a single loss value.

**It also changes nothing about the decision it was built to improve.** Paired on 54
byte-identical runs, with the witness given its strongest fair reading:

```
recovered true associations : 0
new false associations      : 0
lost true associations      : 0
changed abstention decisions: 0
```

Every case is identical with the witness on and off.

## Why

The clause never gets to fire. A mechanism claim still requires an **observed antecedent
within the propagation window**, and the gradient detector's aggregate latency is ~11
steps — so a downstream failure close enough to be mechanically attributable has **no
observed antecedent at all**. The witness observes a boundary that the rule cannot reach.

## Why not `INTEGRATE_OPT_IN_EXPERIMENTALLY`

The criterion is that the witness provide "a recomputable diagnostic benefit on new
counterexample data". The measured benefit is **exactly zero**. And the one case designed
to expose its limits — C3, invoked-with-independent-NaN — **could not be constructed**,
so its discriminating power is `NOT VERIFIED` rather than demonstrated. Integrating on
that basis would repeat v0.1.4's error of shipping a rule whose evidence had not been
validated against the cases built to break it.

## Why not `REJECT_AS_DIAGNOSTIC_EVIDENCE`

The witness does not systematically mislead. It never produced a false association in any
configuration, and its contract is honest about what it does not know. It is simply not
*sufficient*, which is a different finding from being *wrong*.

## Conditions for revisiting

1. **An antecedent must be observable within the propagation window.** Either the
   gradient detector's latency must fall, or the window must be justified against the
   measured latency rather than assumed.
2. **C3 must be constructible.** Until an invoked-but-independent case can be built, the
   witness's discriminating power cannot be tested at all.
3. **C4 must be addressed.** A real downstream overflow with finite parameters is
   currently missed 20 times; the witness does not help, because there is nothing at the
   boundary to observe.

## Capability status

No capability is upgraded. `validated_capabilities = []`.
`DETECTOR_RELIABILITY = EXPERIMENTAL`. `READY_FOR_BLIND_VALIDATION = NO`.
`OptimizerWitness` stays disabled by default and is not wired into promotion.
