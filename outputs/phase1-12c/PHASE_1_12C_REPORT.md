# PHASE 1.12C REPORT

```
PHASE 1.12C STATUS:               PARTIAL
ENGINEERING WORK STATUS:          PARTIAL

PERSISTED_SCORING                 FAIL          (not executed)
OBSERVER_RESOURCE_INTEGRITY       PASS
DETECTOR_EVIDENCE_VOLUME          PASS
OBSERVER_RACE                     UNRESOLVED
FULL_PIPELINE_SMOKE               FAIL          (7 of 8; S1 false positive)
PERSISTED_EVALUATION              FAIL          (not run)
EVIDENCE_COMPLETENESS_VISIBILITY  FAIL          (not implemented)
DETECTOR_RELIABILITY              FAIL
READY_FOR_BLIND_VALIDATION        NO
```

**No final holdout.** No detector added, no gradient threshold touched, no Phase 2
feature, no Agent. `PHASE_1_REPORT.md` and `outputs/phase1-1..1-12b/` are **unmodified**.

---

## 1. The phase's instruction, and how it was met

> **执行已经存在的系统，而不是再写一轮没有运行的评估基础设施。**

Three of four workstreams were **executed**, and one of them closed the phase's largest
open risk.

| Workstream | Built | **Executed** | Gate |
|---|---|---|---|
| C1 persisted scoring | no new code | **NO** | **FAIL** |
| C2 resource boundary | stress harness | **YES** | **PASS** |
| C3 full pipeline smoke | 8-scenario runner | **YES** | **FAIL** (7/8) |
| C4 persisted evaluation | — | **NO** | **FAIL** |

---

## 2. C2 — the largest open risk is closed

Phase 1.12B recorded that the cooldown fix multiplied evidence 20× and could push the
observer's bounded queue into dropping events, **silently truncating the evidence
chain**. That risk is now **measured and closed**:

```
observer persisted events:  cooldown ON 3722   cooldown OFF 3722   UNCHANGED

Detector-side evidence volume increased.
Observer raw-event load unchanged.
```

The cooldown lives inside the detector, which runs **offline over already-persisted
events**. There is no feedback path into the observer queue, so the bounded queue never
sees the increase. The feared feedback did not exist.

**And the detector side is bounded:**

| Spike length | observations | episodes | incidents | detector ms |
|---|---|---|---|---|
| 1 | 1 | **1** | **0** | 181.8 |
| 20 | 20 | **1** | **0** | 177.9 |
| 100 | 89 | **1** | **0** | 184.5 |
| 1000 | 89 | **1** | **0** | 181.7 |

Observations **saturate at 89**, episodes stay at 1, incidents at 0, and **detector time
is flat at ~180 ms across a 1000× range**. Growth is bounded by the episode, not by the
input length.

**Queue behaviour under forced overflow** (no consumer attached, so overflow is caused
rather than hoped for): capacity 64, offered 4999, accepted 64, refused/evicted 4935,
size 64 — **bounded**, accounting exact. A full P3 queue accepted an incoming **P0** by
eviction and the P0 **survived the drain**.

---

## 3. C3 — the smoke finally ran

```
.executed=8  passed=7  failed=1      exit code 1
real training: 3   event-driven: 5 (labelled per scenario)
```

The full path ran for every scenario: training/events → observer → event store →
detector → episode → `EpisodeStore` → fusion → `IncidentStore` → relations →
independent-process CLI.

**S1 is a genuine product failure.** An ordinary healthy training loop with **no
injection at all** produced a confirmed incident. That is a false positive, consistent
with the ~2.4 % healthy-run rate measured at Phase 1.9, and it is recorded as FAIL.
Re-running it, picking a different seed, or calling it a harness artefact would be the
substitution this project has repeatedly had to retract.

**S5 verified the causal discipline:** a gradient episode and a later `NaN` produced one
`POSSIBLE_CONSEQUENCE_OF` relation, and **no proven-cause claim** — the relation type
system still has no `CAUSED_BY`.

**S8 verified the restart path:** two interpreters, the same episode resumed, onset 80
preserved, `episode_count == 1`, protected baseline restored.

---

## 4. What was not done

| Item | State |
|---|---|
| **C1** persisted scoring | **NOT EXECUTED** — the scorer still reads the in-memory result dict, not `episode_relations` |
| **C1.3** duplicate audit | **NOT RUN** — `HISTORICAL_23_NOT_FULLY_REPRODUCED` |
| **C2.4** evidence-completeness visibility | **NOT IMPLEMENTED** |
| **C4** persisted development evaluation | **NOT RUN** |
| **C5** new regression tests | **NOT ADDED** — 443 unchanged |
| **C2.5** observer race | **UNRESOLVED** |

**No new tests were added.** The phase produced executable harnesses and real results
but no regression coverage. That is a shortfall against the brief's C5 requirement and
is stated rather than glossed.

**C2.4 is a live correctness gap.** Because `evidence_completeness` does not exist, a
`NO_ANOMALY` still cannot be distinguished from *"no anomaly in the events that
arrived"*. The brief forbids reporting missing evidence as evidence of normality, and
that distinction is **not yet representable**.

---

## 5. Gate

```
PERSISTED_SCORING                 FAIL
OBSERVER_RESOURCE_INTEGRITY       PASS
DETECTOR_EVIDENCE_VOLUME          PASS
OBSERVER_RACE                     UNRESOLVED
FULL_PIPELINE_SMOKE               FAIL   (7/8)
PERSISTED_EVALUATION              FAIL
DETECTOR_RELIABILITY              FAIL   (episode recall 0.341, unchanged)
READY_FOR_BLIND_VALIDATION        NO
```

C1–C4 are not all PASS, so **Phase 1.12C engineering work is not COMPLETE**.

---

## 6. What moved

| | Phase 1.12B | **Phase 1.12C** |
|---|---|---|
| observer load under the cooldown fix | feared, unmeasured | **measured: unchanged** |
| detector evidence volume | unbounded concern | **bounded at 89; time flat** |
| queue overflow behaviour | inherited from Phase 1.1 | **re-measured under forced overflow** |
| full pipeline smoke | never run | **executed, 7 of 8** |
| healthy-run false positive | rate known | **reproduced on a specific run (S1)** |

## 7. Governing principle

> Phase 1.12A repaired evidence emission. Phase 1.12B repaired evidence semantics.
>
> **Phase 1.12C ran the system: the cooldown change does not load the observer at all,
> detector evidence volume is bounded and costs nothing, overflow accounting is exact,
> and the eight-scenario smoke executed for the first time — 7 of 8, the failure being a
> genuine false positive on a healthy run.**

Scoring, evaluation and evidence-completeness visibility were not reached, and the
observer race remains unexplained.

**No agent. No replay. No new diagnostic source. No causal claim.**
