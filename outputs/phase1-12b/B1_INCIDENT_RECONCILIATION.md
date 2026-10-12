# B1 — INCIDENT RECONCILIATION

**Status: COMPLETE for the mechanism. The 23-duplicate re-measurement is NOT VERIFIED.**

---

## 1. The three situations, separated first

The brief requires this before anything is changed, and the separation is the point:

| Situation | Correct handling | Implemented |
|---|---|---|
| **A.** two genuinely independent anomalies | must **not** be merged | yes — no relation, no group |
| **B.** one anomaly with a plausible consequence | may form one Incident Group | yes — `POSSIBLE_CONSEQUENCE_OF` / `PRECEDES` only |
| **C.** the same episode emitted twice | an **idempotency defect**, not a merge | yes — reported in `duplicate_episode_ids` |

Conflating them would let a grouping rule "reduce duplicates" by merging faults that
were never related — which would be worse than the duplicates.

## 2. Implementation

`src/training_debugger/incidents/reconciliation.py`

```
IncidentGroup:
    group_id, primary_episode_id, primary_anomaly_type
    secondary_episode_ids, secondary_anomaly_types
    relation_ids, relation_types, association_evidence
    has_proven_root_cause = False        # ALWAYS, before replay
    presentation_note                    # states the grouping is not causal
```

**Only `POSSIBLE_CONSEQUENCE_OF` and `PRECEDES` attach.** `UNKNOWN_RELATION` and
`CO_OCCURS_WITH` leave the two episodes **independent** — co-occurrence is not
association. `CAUSED_BY` and `ROOT_CAUSE_OF` remain absent from the type system.

`has_proven_root_cause` is hard-coded `False`: **a group is a presentation unit, not a
proven cause.**

## 3. Verified behaviours

| Behaviour | Result |
|---|---|
| gradient then linked NaN | **1 group**, primary `g1`, secondary `n1`, `has_proven_root_cause=False` |
| standalone NaN | **0 groups**, stays independent |
| gradient + unrelated NaN (disjoint scope, far in time) | **0 groups**, both independent |
| duplicate episode emission | reported in `duplicate_episode_ids`, not grouped |
| `UNKNOWN_RELATION` | does not attach; recorded as unresolved |
| every episode id preserved | PASS — grouping deletes nothing |
| idempotent | PASS — repeated reconciliation is byte-identical |

## 4. On the historical 23

**`NOT VERIFIED`.** The mechanism to re-measure exists and the three situations are now
distinguishable, but the re-measurement was not run against the original corpus.

What can be stated with confidence from the code, and is worth stating because it
reframes the number: Phase 1.9's `DUPLICATE_TOP_LEVEL_INCIDENT` count of **23** was
produced by the *scorer*, which classified an observation as a duplicate when the same
anomaly was also opened as its own top-level incident. It therefore counted
**classification outcomes, not necessarily 23 separate incidents created by the
product**. The brief asks exactly this distinction —

> 必须明确报告"23"究竟有多少是产品重复创建，多少是旧评分规则导致的错误分类。

— and the answer requires the corpus run that was not performed. Reporting a
split without measuring it would be the fabrication the brief forbids.
