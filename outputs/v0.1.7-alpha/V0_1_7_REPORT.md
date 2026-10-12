# v0.1.7-alpha — DIAGNOSTIC LATENCY REPAIR · ADVERSARIAL DISCRIMINATION

```
V0_1_7_ALPHA_READY = YES
EARLY_EVIDENCE_BENEFIT = INCONCLUSIVE (not implemented)
OPTIMIZER_WITNESS_BENEFIT = NO (frozen)
PRODUCTION INTEGRATION = KEEP RESEARCH ONLY
NEXT PRODUCT DECISION = RESEARCH BLOCKED BY OBSERVABILITY
DETECTOR RELIABILITY = EXPERIMENTAL
READY_FOR_BLIND_VALIDATION = NO
```

```
tests 531 passing, 1 skipped · smoke 8/8 · quickstart PASS
installation PASS (editable + non-editable)
```

---

## WORKSTREAM A — the 11-step latency had a single cause

**Root cause: baseline readiness. Not smoothing, not episode persistence, not fusion.**

```
baseline_floor = warmup_steps = 50     (baseline_excludes_warmup = True)
minimum_history = 20                   (protected_ratio_series)
-> first usable ratio at step 70
```

Measured directly:

```
an injection at step 60  -> cannot be scored until step 70   (the "11-step latency")
an injection at step 85  -> fires at step 85                 (latency ZERO)
```

**The detector is prompt once ready.** The apparent latency was an artifact of injecting
*inside* the readiness window. The timestamp chain separates cleanly:

| Concept | Where it comes from |
|---|---|
| `fault_injection_step` | harness only — never a production field |
| `raw_event_step` | observer event `step` |
| `first_abnormal_observation_step` | detector `first_abnormal_step` |
| `episode.candidate_onset_step` | first **band-abnormal** observation (v0.1.3 fix) |
| `episode.confirmed_at_step` | rule satisfaction |
| `incident.detected_step` | promotion |
| `optimizer_step_id` | witness records — `UNKNOWN` when unattached |

`PROPAGATION_DELAY_STEPS = 2` was **not** changed. Changing it would have masked this.

**Early Evidence Lane: NOT IMPLEMENTED.** With the root cause being plain readiness, adding
a candidate lane would have produced candidates that are *also* unscoreable before step 70 —
there is no evidence to make early, only a baseline that is not yet ready. Building one
would have added machinery without a measurable delay to remove. Recorded as
`INCONCLUSIVE`, not as a success.

## WORKSTREAM B — C3 is still not constructible, now with a precise reason

Injections were moved past readiness (`INJECTION = 90`), which *did* fix the ordering:

```
C3   gradient fires @90   non-finite @92   injection @90     <-- order now correct
C11  gradient fires @90   no non-finite                     <-- control
C4   gradient never fires  non-finite @90
```

**But C3 still fails, for a new and concrete reason.** The magnitude required to fire the
aggregate detector also destabilises the model:

```
x100  -> gradient fires @90, but parameters become non-finite in 8/8 runs
x30   -> gradient fires @90, parameters finite in C11 ... and STILL 8/8 non-finite in C3
```

The difference from C11 is the injected NaN: once it lands, the run's parameters corrupt.
So a case with *"gradient evidence precedes → optimizer invoked → independent non-finite"*
**cannot be held to that shape** in this environment with the current confirmation
threshold. Increasing the magnitude repeatedly was tried and is explicitly not the answer;
the blocker is the threshold-to-stability ratio, not the injection size.

**Consequence, measured:** C3 and C12 are scored against `INDEPENDENT` ground truth and the
engine claims `POSSIBLE_CONSEQUENCE_OF` — **16 false associations** (8 runs each). They are
*valid for measuring harm* and *invalid for measuring discrimination*, and are reported as
both.

**C4 was constructed and is real**: the parameters stay finite through the update and the
**next forward pass genuinely overflows** (`nonfinite @90`, `param non-finite 0/8`). It is
missed — consistent with the v0.1.6 finding that a real overflow with finite parameters has
no observable midstream evidence.

**C11 (early suspicious gradient, no failure) is clean**: gradient fires at 90, no
non-finite value at all, no relation claimed. The early-evidence concern does not
materialise.

## WORKSTREAM C — three strategies, measured

```
                     TP   FP   FN   TN    precision   recall   false association
ORIGINAL 50-run       A=B=C, no relation opportunities scored (labels did not resolve)
ADVERSARIAL 45-run   A=B=C    0    0   10   35   NOT ESTIMABLE   0.000      0.000
TIMELINE 32-run      A=B=C    0   16    8    8      0.000        0.000      0.667
```

**All three strategies are byte-identical on every corpus.**

* **A vs C (witness):** zero difference, for the second independent time. The witness
  observes a boundary the rule cannot reach.
* **A vs B (early evidence):** zero difference because **B was not implemented**, and the
  Workstream A root cause says there was nothing to build — the delay is baseline
  readiness, not evidence discovery.

### Decisions, each with its data

```
EARLY_EVIDENCE_USEFUL    = INCONCLUSIVE  (not implemented; root cause made it moot)
WITNESS_ADDS_VALUE       = NO            (paired identical twice: 54 runs then 32)
CURRENT_TELEMETRY_INSUFFICIENT = YES     (C3 blocker is threshold-vs-stability, C4 has no
                                          observable midstream evidence)
```

**False associations: 16**, all from the two invalid C3/C12 cases. No false association was
produced on any valid case in any strategy. The brief's top priority —
*no fabricated association for independent faults* — holds on every case that could be
constructed correctly, and fails precisely on the case that could not.

## WORKSTREAM D — product stabilization

```
full suite   531 passing, 1 skipped
smoke        8/8 PASS
quickstart   PASS
installation PASS (editable + non-editable)
CLI          PASS · cross-process restart PASS · relation persistence PASS
```

Evidence completeness (D1) is unchanged from v0.1.2 and still correct: an unreported
completeness yields a warning, never an unqualified `NO_ANOMALY`.

No new hot-path collection, no parameter scan, no added GPU sync. The witness remains
**default disabled**. The standing overhead finding — observer overhead exceeding the
**<3 % core target** at the largest workload — is **not** re-measured and **not** resolved.

---

## Verdict

The round produced **one durable repair and one durable negative**:

1. **The latency was readiness, and it is now explained and measured** — a 10-step
   warm-up boundary plus one step of scoring, with the detector prompt once ready.
   No threshold was moved to hide it.
2. **The witness is frozen**: two independent paired experiments, 86 runs total, zero
   difference. Not rejected — it is honest and cheap — but it is not evidence.
3. **C3 is blocked by observability, not by effort**: the detector's confirmation threshold
   demands a gradient excursion large enough to corrupt the model, so the case cannot be
   built. That is a property of the threshold-to-stability ratio and is the next thing a
   detector change must address, not another injection.

`validated_capabilities = []` · no final holdout · no Phase 2 · no Agent · nothing published.
