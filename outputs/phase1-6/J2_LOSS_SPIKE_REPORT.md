# J2 — LOSS SPIKE REPORT

**Status: COMPLETE, by the route the brief permits.** `LOSS_SPIKE` default promotion
is **OFF**.

---

## 1. What was done

The brief gave two acceptable outcomes and forbade a third consecutive round of
leaving the same action unexecuted:

```
repair
OR
disable default promotion
```

**The second was taken.** `PROMOTION_DISABLED_TYPES = {"LOSS_SPIKE"}` in
`fusion/rules.py`, with an explicit `promotion_enabled()` check and the disabled set
recorded in `fusion_rule_registry.json`.

This is promotion off, **not** detector off:

* the detector still runs;
* candidates are still recorded as episodes;
* fusion returns `CANDIDATE_ONLY` and **names the episode**;
* the renderer shows it;
* the registry records why.

So the failure stays visible and measurable rather than being hidden by removal, and
a detector known not to work yet cannot pollute default incident quality.

## 2. Verified

```
promotion_enabled("LOSS_SPIKE") -> False
fuse() on a confirmed LOSS_SPIKE episode -> CANDIDATE_ONLY, episodes_used = ['e1']
```

## 3. Measured effect on healthy runs

**None.** The `H2_noisy_healthy_loss` false-incident rate went from 8/14 (Phase 1.3)
to **1/14** in the current development run — but the diagnostic shows the change came
from the gradient band becoming authoritative, not from disabling `LOSS_SPIKE`.

Every healthy false incident, before and after, is `GRADIENT_EXPLOSION`. Loss
contributes **zero**.

That is an honest and slightly awkward result: the brief's J2 targets a real problem
that had already been solved by a different change, so disabling promotion cost
nothing and gained nothing measurable on this corpus. It is retained because the
Phase 1.3 measurement (8/14) was real and the repair was never built.

## 4. Not done

| Requirement | State |
|---|---|
| residual envelope / quantile / MAD comparison | **NOT DONE** |
| spike semantics (large local residual + short duration + recovery) | **NOT DONE** |
| development gate (precision ≥ 0.85, recall ≥ 0.80, H2 ≤ 5 %) | **NOT EVALUATED** |

`LOSS_SPIKE` is therefore **EXPERIMENTAL** with promotion **OFF**, which the brief
states is a legitimate result and explicitly not to be forced on for recall.
