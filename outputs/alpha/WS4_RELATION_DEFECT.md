# WS4 — THE RELATION LAYER IS INVERTED

**This is the headline finding of v0.1.2, and it is a measured defect, not a
calibration gap.**

---

## 1. The experiment

The v0.1.1 corpus injected one fault type per run, so **0 relations were persisted** and
the relation path had never been measured. v0.1.2 built a **mixed-fault corpus** with
five deliberately distinct scenarios, 10 real training runs each, 2 model
configurations, 50 runs total:

| | scenario | expectation |
|---|---|---|
| **M1** | gradient fault → **genuinely** non-finite loss | a relation *should* exist |
| **M2** | standalone non-finite loss, no gradient fault | no relation |
| **M3** | gradient fault **+ a deliberately independent** numerical fault | **no relation** |
| **M4** | gradient anomaly + ordinary loss noise | no relation |
| **M5** | healthy heavy-tail gradients | no relation |

**M1's chain is real, not asserted.** The gradients are filled with a huge but **finite**
value (`1e28`); `optimizer.step()` overflows the parameters, and the *next* forward pass
produces a non-finite loss. The gradient fault demonstrably causes the numerical
manifestation.

## 2. What happened

```
M1  REAL consequence chain      0 relations   <- TRUE POSITIVE MISSED
M2  standalone non-finite       0 relations   <- correct
M3  two INDEPENDENT faults     10 relations   <- FABRICATED ASSOCIATION
M4  gradient + loss noise       0 relations   <- correct
M5  healthy                     0 relations   <- correct
```

Every one of the **10 fabricated** relations is `POSSIBLE_CONSEQUENCE_OF`, at
`gap_steps = 40` — inside the linker's `max_gap_steps = 60`.

**The one genuine causal chain produced no relation at all.**

## 3. Scored

```
RELATION   eligible 20   TP 0   FP 10   FN 10   precision 0.0   recall 0.0
```

**Precision 0.0 and recall 0.0 simultaneously.** Not a trade-off — an inversion. The
linker is anti-correlated with the truth on the only corpus that can test it.

## 4. Root cause

`link_consequences` decides on **temporal proximity and scope overlap**. It has no notion
of mechanism, so:

* two independent faults 40 steps apart look exactly like a fault and its consequence —
  hence M3's 10 false positives;
* M1's real chain produced only `NAN_DETECTED` episodes (once the parameters overflow
  every subsequent gradient is non-finite), so there was no *gradient* episode left to
  link from — hence the 10 false negatives.

**This is precisely what the brief forbids:** treating time adjacency as evidence of
consequence. The relation type is honestly named `POSSIBLE_CONSEQUENCE_OF` and no causal
type exists — but a *possible* consequence still has to be better than "40 steps later",
and here it is worse than chance.

## 5. Why this matters more than a detector miss

A missed detection is a silent non-event. A **fabricated** relation is an assertion:
it tells the user that a numerical instability is plausibly the downstream consequence of
a gradient fault, when the two were injected independently. That is the failure mode this
project was built to avoid, and it is now measured rather than assumed.

## 6. What is not claimed

No fix was applied. Choosing a smarter association rule (mechanism-based, layer-scoped,
or requiring the non-finite value to be *derivable* from the gradient excursion) is a
design change that must be measured against this corpus — the corpus now exists for
exactly that purpose. **Relation quality is `FAIL` and is reported as such.**

The historical 23 duplicates remain `HISTORICAL_23_NOT_FULLY_REPRODUCED`; nothing here is
presented as an audit of that older corpus.
