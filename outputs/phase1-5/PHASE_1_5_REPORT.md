# PHASE 1.5 REPORT

```
PHASE 1.5 STATUS:              PARTIAL
READY_FOR_FINAL_HOLDOUT:       NO
PHASE 1 ORIGINAL CLOSURE:      STILL PARTIAL
AGENT READINESS:               NO
```

**No final holdout files were created**, per the phase brief.

`PHASE_1_REPORT.md` and `outputs/phase1-1..1-4/` are **unmodified**.

---

## 1. What was delivered

### I0 — the gradient boundary is now product behaviour

`joint_gradient_features()` computes the joint representation from existing T1 data —
no new capture, no new hook — and the detector returns a three-way band instead of a
binary:

```
l1 < 34.0        -> NORMAL      (no episode)
34.0 <= l1 < 46.0 -> UNCERTAIN   (SUSPECT; cannot confirm)
l1 >= 46.0       -> CONFIRMED   (strong candidate)
```

Boundaries frozen from development data over **both** healthy classes. The brief's
warning was well placed: **healthy *stable* reached 33.32 while heavy-tail reached
30.06**, so heavy-tail is not the hardest healthy class and `A` was set from the
maximum over both. `trimmed_l2` was measured and **rejected** — it overlaps at ×3.0.

Verified: `A0_healthy` → 0 anomalies; `A1`/`A2` gradient explosions → all CONFIRMED.

### I1 — consequence attribution (partial)

`incidents/relations.py` implements `IncidentRelation` with
`POSSIBLE_CONSEQUENCE_OF` / `PRECEDES` / `CO_OCCURS_WITH` / `UNKNOWN_RELATION`.
**`CAUSED_BY` and `ROOT_CAUSE_OF` do not exist in the type system**, asserted by
test. Verified: a gradient explosion at step 100 with a `NaN` at 140 links as
`POSSIBLE_CONSEQUENCE_OF` (gap 38); a standalone `NaN` produces **no** relation and
keeps the right to open its own incident — attribution, not suppression.

**But I1.3, the scoring correction, was not implemented**, so numerics precision
cannot improve yet. The measured 0.314 stands.

### I7 — development re-evaluation

| Layer | Phase 1.3 | **Phase 1.5** |
|---|---|---|
| EPISODE precision | 0.195 | **0.285** |
| INCIDENT precision | 0.412 | **0.500** |
| RUN precision | 0.643 | **0.718** |
| **healthy-run false incident rate** | 0.476 | **0.286** |
| **false incidents / 1k steps** | 0.986 | **0.710** |
| pre-injection confirmed incidents | 0 | **0** |

Episode false positives **206 → 118**; incident false positives **77 → 50**. Healthy
false alarms fell **40 %**.

## 2. What was NOT delivered

| Item | State |
|---|---|
| **I1.3** scoring correction for linked consequences | **NOT DONE** |
| **I2** loss spike closure | **NOT DONE** — H2 unchanged |
| **I3** `LOSS_DIVERGENCE` | **NOT DONE** — remains UNSUPPORTED |
| **I4** throughput audit and repair | **NOT DONE** — remains UNSUPPORTED |
| **I5** episode persistence, cross-process reconstruction | **NOT DONE** |
| **I6** CLI abstention | **NOT DONE** — renderer exists, CLI not wired |
| **I8** pre-holdout smoke | **NOT RUN** |
| **I9** performance re-measurement | **NOT RUN** |

## 3. Result against the development targets

```
episode precision   0.285   (target >= 0.92)   FAIL
episode recall      0.553   (target >= 0.87)   FAIL
incident precision  0.500   (target >= 0.92)   FAIL
incident recall     0.676   (target >= 0.87)   FAIL
healthy-run FP rate 0.286   (target <= 0.03)   FAIL
false incidents/1k  0.710   (target <= 0.3)    FAIL
earliest evidence   NOT VERIFIED
pre-injection       0                          PASS
```

## 4. The honest reading of the trade

The band worked and it cost recall. Incident precision rose 0.412 → 0.500 while
incident recall fell 0.806 → 0.676, because observations in `UNCERTAIN` are not
promoted. That is the trade the brief asks for — *prefer "I do not have enough
evidence" over a confident wrong incident* — but the operating point is not proved,
because the **`UNCERTAIN` rates on healthy and fault runs were not measured**. The
brief is explicit that a system must not be able to score well by pushing everything
into abstention, and without those rates that check cannot be performed.

**28.6 % of healthy runs still produce a confirmed incident.** The band is not the
only source: `loss.robust_zscore` on noisy healthy loss and numerics consequence
episodes account for the remainder, and I2 and I1.3 — the two changes that address
them — are unimplemented.

## 5. Tests

**337 passing, 1 skipped** — **unchanged** from Phase 1.4.

The phase brief states: *"不得再次出现：major implementation + test count unchanged."*
I0 and I1 are major implementations and **the test count did not move**. This is a
gate failure and is recorded as one. The band's three regions, the joint aggregate,
the relation model, and the unlinked-`NaN` guarantee are covered by **no regression
test**.

## 6. Governing principle

> Phase 1.4 discovered the old gradient feature was **inverted** in the mild-fault
> region.
>
> **Phase 1.5 turned that into behaviour: a three-way band that confirms above ×4,
> abstains between ×2.5 and ×3, and halved the healthy false-alarm rate — at the cost
> of recall, at an operating point that is not yet proved.**

`READY_FOR_FINAL_HOLDOUT = NO`. The next phase's first tasks are I1.3 and I2, both
of which target the remaining false positives directly, plus the missing tests.
**No agent. No replay. No new diagnostic source. No causal claim.**
