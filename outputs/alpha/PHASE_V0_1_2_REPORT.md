# v0.1.2-alpha — EVALUATION INTEGRITY & MIXED-FAULT VALIDATION

```
V0_1_2_ALPHA_READY = YES
DETECTOR_RELIABILITY = EXPERIMENTAL   (not upgraded)
```

A debugger cannot be more trustworthy than the evaluation system used to measure it.
This release fixed the measurement. It found one severe product defect in the process.

```
tests        490 passing, 1 skipped   (477 -> 490, +13)
smoke        8/8 PASS
quickstart   PASS
installation PASS (editable and non-editable)
```

---

## WS1 — Evaluation semantics: **repaired**

Two defects, both of which made the v0.1.1 scoreboard unreadable.

### The unit was wrong

**42 runs produced 97 "classification records".** The unit was the **episode**, and
**32 of 42 runs hold more than one episode**, so an episode count was being read as a
run count. The scorer now reports the accounting explicitly:

```
runs 42 | episodes classified 97 | runs with more than one episode 32
```

### `primary_recall` could not report a miss

`MISSED_PRIMARY` was **never incremented anywhere**, so
`primary_recall = TP / (TP + 0) = 1.0` whenever anything at all was detected. The metric
was structurally incapable of reporting failure.

**Corrected, with one denominator per level:**

```
LEVEL       elig   TP   FP   FN    TN  precision    recall   abstain
SIGNAL         0    0    0    0   n/a        N/E       N/E       N/E
EPISODE       97   29   60    0     8      0.326       N/E       0.0
INCIDENT      25   18    7    0   n/a      0.720       1.0       0.0
RUN           42   18    1   18     5      0.947       0.5     0.286
RELATION       0    0    0    0   n/a        N/E       N/E       N/E
```

**The real run-level recall is 0.50, not 1.00.** Episode-level recall is marked
`NOT ESTIMABLE` rather than left at a meaningless 1.0: the truth names the injected
fault, not how many episodes it should yield, so a missed fault appears there as an
*extra* episode (a precision error), never a missing one.

### Fail-closed

`validate_scoring_inputs` refuses to score when an injected run lacks a primary fault
type, an acceptable window or an injection step — the three conditions that silently
produced `precision = 0.0` earlier in this project. The result is `SCORING_INVALID` with
**no numbers emitted**.

## WS2 — Corpus integrity: **closed, and it was a loader bug**

```
ground truth 42 | readable 41 | excluded 1
dev-healthy-L4d32-s80002 -- healthy, ZERO episodes
```

`persisted_run_ids` required `episodes/episodes/` to exist. A healthy run produces **no
episodes**, so it has no such directory and was silently excluded — even though its
analysis state was persisted and complete.

**That run is the strongest evidence of correct abstention**, and dropping it biased
every healthy-run statistic. Fixed: a run is readable if it has *any* store artifact.

```
ground truth 42 | readable 42 | excluded 0
```

## WS3 — Mixed-fault corpus: **built**

50 real training runs, 2 model configurations, 5 seeds each, covering five deliberately
distinct scenarios:

| | scenario | expectation |
|---|---|---|
| M1 | gradient fault → **genuinely** non-finite loss | a relation should exist |
| M2 | standalone non-finite loss | no relation |
| M3 | gradient fault + a **deliberately independent** numerical fault | **no relation** |
| M4 | gradient anomaly + ordinary loss noise | no relation |
| M5 | healthy heavy-tail gradients | no relation |

**M1's chain is real, not asserted:** gradients are filled with a huge but **finite**
`1e28`; `optimizer.step()` overflows the parameters and the *next* forward pass is
non-finite.

```
50 runs | 90 episodes | 10 relations persisted
ground truth written to a SEPARATE tree; production code cannot read the labels
relations read back from disk in an independent process: 10 across 10 runs
```

## WS4 — Relation evaluation: **the relation layer is inverted**

```
M1  REAL consequence chain      0 relations   <- TRUE POSITIVE MISSED
M2  standalone non-finite       0 relations   <- correct
M3  two INDEPENDENT faults     10 relations   <- FABRICATED ASSOCIATION
M4  gradient + loss noise       0 relations   <- correct
M5  healthy                     0 relations   <- correct

RELATION  eligible 20  TP 0  FP 10  FN 10  precision 0.0  recall 0.0
```

**Precision 0.0 and recall 0.0 at once.** Every fabricated relation is
`POSSIBLE_CONSEQUENCE_OF` at `gap_steps = 40`, inside `max_gap_steps = 60`. The one
genuine causal chain produced nothing, because once the parameters overflow every
subsequent gradient is non-finite, so no *gradient* episode remained to link from.

**Root cause:** `link_consequences` decides on temporal proximity and scope overlap. It
has no notion of mechanism. Two independently injected faults 40 steps apart are
indistinguishable from a fault and its consequence — which is exactly the "temporal
adjacency as causation" the brief forbids.

**This matters more than a detector miss.** A missed detection is a silent non-event. A
fabricated relation is an **assertion**: it tells the user a numerical instability is
plausibly downstream of a gradient fault when the two were injected independently.
**No fix was applied** — a mechanism-aware rule is a design change that must be measured
against this corpus, which now exists for that purpose.

## Earliest evidence, recomputed from persisted data

```
eligible_fault_runs 36   claims_made 24   correct 24   false 0   abstained 12
precision  1.000     coverage 0.667
median onset error 0.0    p90 onset error 0
median confirmation delay 0.0
pre-injection confirmed incidents 0        (gate requires 0)
```

**Corpus caveat, stated rather than buried:** 4 of this corpus's 6 scenarios are strong
faults, so precision 1.0 and coverage 0.667 are **corpus-specific and must not be read as
general detector quality.** The 12 abstentions are the ×2 and ×5 gradient faults, which
remain indistinguishable from healthy variation.

## Alpha regression

```
490 tests passing, 1 skipped      quickstart PASS
smoke 8/8 PASS                    installation PASS (editable + non-editable)
four-verdict CLI readback PASS    cross-process restoration PASS
```

## What was NOT done

* **The historical 23 is still `HISTORICAL_23_NOT_FULLY_REPRODUCED`.** The original
  169-run corpus is gone; nothing here is presented as an audit of it.
* **No detector change, no threshold tuning.** The ×2–×12 gradient boundary is
  unchanged and remains the binding constraint.
* **`DETECTOR_RELIABILITY` stays `EXPERIMENTAL`.** `validated_capabilities = []`.
* **The relation defect is measured, not fixed.**

## Governing principle, applied

v0.1.2 was supposed to be a measurement release. It turned out to be one: correcting the
denominators moved run-level recall from a fictitious **1.00** to a real **0.50**, and
building the corpus that could test relations revealed they are **anti-correlated with
the truth**. Neither finding was visible while the scoreboard was quietly wrong.
