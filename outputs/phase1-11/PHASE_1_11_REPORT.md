# PHASE 1.11 REPORT

```
PHASE 1.11 STATUS:              PARTIAL
ENGINEERING_CLOSURE:            FAIL
DETECTOR_RELIABILITY:           FAIL
READY_FOR_BLIND_VALIDATION:     NO
NARRATION_AGENT_READY:          NO
```

**No final holdout files were created.**

`PHASE_1_REPORT.md` and `outputs/phase1-1..1-10/` are **unmodified**. Every number
below was produced in this phase; Phase 1.9's figures appear only where explicitly
labelled historical.

---

## 1. Workstream A — evidence integration

### A1 — CLI four-verdict E2E: **3 of 4. GATE NOT MET.**

| Scenario | Method | Expected | Observed | Matched |
|---|---|---|---|---|
| A | **REAL PYTORCH TRAINING** | `CONFIRMED_INCIDENT` | `CONFIRMED_INCIDENT` | YES |
| B | EVENT-DRIVEN PIPELINE INTEGRATION | `CANDIDATE_ONLY` | `NO_ANOMALY` | **NO** |
| C | EVENT-DRIVEN PIPELINE INTEGRATION | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | YES |
| D | **REAL PYTORCH TRAINING** | `NO_ANOMALY` | `NO_ANOMALY` | YES |

All four ran the **real CLI in a separate process** against the persisted store, and
all four exited 0. No verdict object was constructed and saved to manufacture a pass.

**Three of the four required verdicts are now demonstrated from real persisted data**,
up from one in Phase 1.10. The brief's gate requires four.

### A2–A5 — not started

The brief's serial rule (A1 → A2 → …) and A1's failure mean A2–A5 were not begun. No
code was written and no measurement exists.

### The defect A1 exposed

`LOSS_SPIKE` cannot confirm, and the cause is a **design inconsistency between two
independently specified components**:

```
loss.robust_zscore   applies cooldown_steps = 50  ->  ONE observation per spike
LOSS_SPIKE rule      requires 2 observations within 6 steps
                     -> never satisfied -> NOT_ANOMALY -> verdict NO_ANOMALY
```

The confirmation rule demands persistence that the detector's own cooldown makes
impossible. This is the same class of defect Phase 1.6 found with the gradient band:
two halves of a rule set that were never checked against each other. It is reported
rather than tuned around, because reaching `CANDIDATE_ONLY` by adjusting the injection
would hide it.

---

## 2. Workstream B — observer reliability: **UNRESOLVED**

```
isolated:    3 of 3 PASS
full suite:  1 FAIL observed
fix:         NOT ATTEMPTED
```

The brief forbids repairing this with sleeps or weakened assertions. A real fix needs
the race identified among queue-overflow eviction, producer/consumer handoff, flush
completion, writer shutdown ordering and cross-thread counter visibility. That
investigation was not carried out, so **`FLAKY_TEST_STATUS = UNRESOLVED`** — reported
rather than retried until green.

---

## 3. Tests

```
before:  412  (Phase 1.10)
added:     5  (Phase 1.10's restart readback suite)
total:   417 passing, 1 skipped
```

**This phase added no tests.** A1 produced an executable end-to-end script
(`tests/experiments/four_verdicts_e2e.py`) and machine-readable results, but no new
regression test, and A2–A5 were not started. That is a real shortfall against the
brief's test requirement and is stated rather than glossed.

---

## 4. Gate summary

| Gate | State |
|---|---|
| 4 verdicts real persisted-data E2E | **FAIL** — 3 of 4 |
| CLI independent-process readback | **PASS** — all four exited 0 |
| Episode / Incident / Relation references resolve | **PASS** for A, C, D; B has no incident by design |
| Production incident reconciliation implemented | **NOT DONE** — A2 not started |
| Independent NaN not suppressed | **NOT VERIFIED** |
| Scorer consumes persisted relations | **NOT DONE** — A3 not started |
| Numerics current metrics measured | **NOT VERIFIED** |
| Earliest evidence re-derived from persisted data | **NOT VERIFIED** |
| Full pipeline smoke PASS | **NOT RUN** |
| Flaky observer test repaired | **FAIL** — UNRESOLVED |
| Full suite repeated without unexplained failure | **NOT DONE** |
| No final holdout | **PASS** |
| No Phase 2 | **PASS** |
| No Agent | **PASS** |
| No new detector | **PASS** |
| No threshold retuning | **PASS** |

```
ENGINEERING_CLOSURE   = FAIL
DETECTOR_RELIABILITY  = FAIL   (episode recall 0.341, last measured Phase 1.9)
READY_FOR_BLIND_VALIDATION = NO
```

---

## 5. What moved

| Capability | Phase 1.10 | **Phase 1.11** |
|---|---|---|
| verdicts demonstrable from persisted data | 1 of 4 | **3 of 4** |
| independent-process CLI readback | one run | **four scenarios, all exit 0** |
| labelled event-driven vs training E2E | n/a | distinguished and disclosed per scenario |
| `LOSS_SPIKE` confirmability | unknown | **diagnosed as structurally impossible** |

For the first time the system produced `CONFIRMED_INCIDENT` and
`INSUFFICIENT_EVIDENCE` **from persisted evidence, through a separate process, rendered
by the real CLI** — which is what "the user can query the evidence chain" means.

---

## 6. Governing principle

> Phase 1.9 proved the system can **write** evidence.
> Phase 1.10 proved it can **recover** episodes and the protected baseline.
>
> **Phase 1.11 proved three of four verdicts can be queried from persisted evidence by
> a separate process — and found that the fourth is unreachable because a detector's
> cooldown makes its own confirmation rule unsatisfiable.**

The readback path is open for three verdicts. It is not open for the fourth, and the
reason is a rule-set inconsistency rather than a missing feature.

**No agent. No replay. No new diagnostic source. No causal claim.**
