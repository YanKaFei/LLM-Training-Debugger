# PHASE 1.12B REPORT

```
PHASE 1.12B STATUS:          PARTIAL
EVIDENCE_CHAIN_CLOSURE:      FAIL
OBSERVER_STABILITY:          UNRESOLVED
DETECTOR_RELIABILITY:        FAIL
READY_FOR_BLIND_VALIDATION:  NO
```

**No final holdout.** No detector added, no gradient threshold touched, no Phase 2
feature, no Agent.

`PHASE_1_REPORT.md` and `outputs/phase1-1..1-12a/` are **unmodified**.

---

## 1. What was delivered

### B0 — semantic contract audit: **COMPLETE**

**B0.1.** `CANDIDATE_ONLY` conflated three unrelated facts. Split into two independent
dimensions with the external API unchanged:

```
evidence_state:        SUSPECT / CONFIRMED / CLOSED
promotion_disposition: ELIGIBLE / DISABLED / WITHHELD
reason_code:           PROMOTED | CONFIRMED_BUT_PROMOTION_DISABLED
                       CONFIRMED_BUT_NO_RULE_MATCHED | EVIDENCE_INSUFFICIENT
                       OBSERVATIONS_DISMISSED | NO_OBSERVATIONS
```

A confirmed `LOSS_SPIKE` now reports `CONFIRMED` + `DISABLED` and the CLI states
*"This is NOT a statement that the evidence was insufficient"* — instead of implying
the system could not tell.

**B0.2.** Verified the dedup key `(step, layer)` is scoped per `signal_key`
(`detector:rank`) within a run: retried observations count once; distinct same-step
observations and different anomaly types are preserved; restarted processing does not
double-count. **Limitation recorded:** there is **no source-event-level identity** —
`Anomaly` carries no `event_id` — so the honest claim is *"idempotent at
`(step, layer)` granularity"*, not "fully deduplicated".

### B1 — incident reconciliation: **COMPLETE (mechanism)**

The three situations are separated **first**, which is the point:

| | Handling |
|---|---|
| **A** independent anomalies | not merged |
| **B** plausible consequence | one `IncidentGroup`, `has_proven_root_cause = False` |
| **C** duplicate emission | reported as an idempotency defect, not grouped |

Only `POSSIBLE_CONSEQUENCE_OF` and `PRECEDES` attach. `UNKNOWN_RELATION` and
`CO_OCCURS_WITH` leave episodes **independent** — co-occurrence is not association.
`CAUSED_BY` / `ROOT_CAUSE_OF` remain absent. Every episode id survives grouping.

Verified: linked → 1 group; standalone NaN → independent; unrelated NaN → both
independent; duplicate emission → flagged; no relation → no merge; idempotent.

### Tests

```
428 -> 443 passing, 1 skipped   (+15)
```

One older test asserting the previous `CANDIDATE_ONLY` wording was **updated**, not
deleted, with the reason recorded.

---

## 2. What was NOT delivered

| Item | State |
|---|---|
| **B2** persisted consequence scorer | **NOT DONE** |
| **B3** full pipeline smoke | **NOT DONE** |
| **B4** persisted development evaluation | **NOT DONE** |
| **B5** resource-boundary tests | **NOT DONE** |

No code was written for B2–B4, nothing was executed against the corpus, and no number
is estimated in their place.

---

## 3. The historical 23 — `NOT VERIFIED`, and reframed

The mechanism to re-measure exists and the three situations are now distinguishable,
but the re-measurement was **not run**.

What can be said from the code, and matters: Phase 1.9's count of **23** was produced by
the *scorer*, which classified an observation as a duplicate when the same anomaly was
also opened as its own top-level incident. **It counted classification outcomes, not
necessarily 23 incidents the product created.** The brief asks precisely this
distinction and the answer requires the corpus run that was not performed. Producing a
split without measuring it would be the fabrication the brief forbids.

---

## 4. Gate

| Gate | State |
|---|---|
| B0 semantics audited and tested | **PASS** |
| Incident association/reconciliation implemented | **PASS** |
| Independent NaN protected | **PASS** |
| No inferred causal claim | **PASS** |
| Persisted relations consumed by scorer | **FAIL** |
| Current numerics metrics measured | **FAIL** |
| Duplicate-root count re-measured | **FAIL** |
| Full pipeline smoke executed and passed | **FAIL** |
| Earliest evidence evaluated from persisted data | **FAIL** |
| Readback evidence references resolve | **PASS** (measured in 1.12A, unchanged) |
| New regression tests pass | **PASS** — 443 |

```
EVIDENCE_CHAIN_CLOSURE   = FAIL   (5 of 11 fail)
OBSERVER_STABILITY       = UNRESOLVED
DETECTOR_RELIABILITY     = FAIL   (unchanged; episode recall 0.341)
READY_FOR_BLIND_VALIDATION = NO
```

---

## 5. The most consequential open risk

Phase 1.12A's `cooldown_suppresses_evidence = False` multiplies evidence volume — 20
observations where there was 1, on the synthetic loss spike. The **incident** layer is
protected (measured: 20 → 1 episode → 1 verdict → 0 incidents), but the **observer**
layer is not:

```
does the bounded queue remain bounded?          NOT VERIFIED
is drop accounting still correct?               NOT VERIFIED
is fail-open preserved at the new volume?       NOT VERIFIED
evidence volume growth on a real corpus         NOT VERIFIED
```

The queue is bounded and drops by priority, so a 20× volume increase can push a run from
"no drops" into "drops" — **silently truncating the very evidence chain this phase exists
to make trustworthy**. Phase 1.1 measured drop behaviour, but not at this volume. This is
recorded rather than assumed benign.

---

## 6. What moved

| Capability | Phase 1.12A | **Phase 1.12B** |
|---|---|---|
| verdict semantics | `CANDIDATE_ONLY` conflated 3 facts | **2 dimensions + 6 reason codes** |
| incident grouping | absent | **implemented, 3 cases separated** |
| causal claims | forbidden in the type system | **forbidden, and grouping asserts no root cause** |
| tests | 428 | **443** |

## 7. Governing principle

> Phase 1.12A proved evidence can flow without being suppressed by cooldown.
>
> **Phase 1.12B separated the three things that had been called "duplicate", refused to
> merge independent faults, and made the system say "the evidence is confirmed but this
> capability is not cleared" instead of implying it could not tell.**

The scoring, smoke and evaluation steps were not reached, and the observer's resource
behaviour under the new evidence volume is unmeasured.

**No agent. No replay. No new diagnostic source. No causal claim.**
