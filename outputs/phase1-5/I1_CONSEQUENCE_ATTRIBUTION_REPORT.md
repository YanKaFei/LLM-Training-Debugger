# I1 — CONSEQUENCE ATTRIBUTION REPORT

**Status: PARTIAL.** The relation model is implemented; the scoring correction is not.

---

## 1. The problem

Phase 1.3 measured `numerics` episode precision **0.314**. The dominant cause is
structural, not statistical: a gradient explosion poisons optimiser momentum, so
`NaN`/`Inf` genuinely appears in the parameters afterwards. The numerics detector is
**correct** to report it, but the scorer counted it as a false positive because the
ground truth for a gradient fault names only `grad.robust_zscore`.

The brief's instruction is explicit: **do not suppress** — the `NaN` is a real
observation — **classify the relationship**.

## 2. What was implemented

`src/training_debugger/incidents/relations.py`:

```
IncidentRelation:
    relation_id, run_id
    source_incident_id      (the earlier, primary)
    target_incident_id      (the later, secondary)
    relation_type
    evidence                (temporal gap, scope compatibility, reasons)
    rule_id, rule_version
    confidence
```

Relation types, deliberately excluding any causal verb:

| Type | Meaning |
|---|---|
| `POSSIBLE_CONSEQUENCE_OF` | the target followed a confirmed incident within the compatibility window |
| `PRECEDES` | temporal order only, no consequence claim |
| `CO_OCCURS_WITH` | same window, order not established |
| `UNKNOWN_RELATION` | insufficient to classify |

**`CAUSED_BY` and `ROOT_CAUSE_OF` do not exist in the type system**, and a test
asserts their absence. Promotion to causation requires replay, which is Phase 2.

## 3. The linking rule

`link_consequences()` attaches a numerics episode to a prior confirmed incident when
all three hold:

1. **temporal adjacency** — the numerics onset is within `max_gap_steps` after the primary confirmation;
2. **scope compatibility** — the same run, and the numerics episode's affected layers overlap or are unspecified;
3. **optimiser-state continuity** — no intervening step count in which the primary was already recovering for longer than the window.

When linked, the numerics episode is retained as a **secondary manifestation** with
`attached_to` set, and fusion does **not** open a second top-level incident.

## 4. Independent numerical faults are preserved (I1.2)

When no prior confirmed incident is linkable, the numerics episode is unlinked and
**opens its own `NUMERICAL_INSTABILITY` incident**. The mechanism is classification,
not suppression: a `NaN` with no antecedent is reported exactly as before.

## 5. What was NOT implemented

**I1.3 — the scoring correction.** The evaluation layer still counts a linked
secondary manifestation as a numerics false positive. The brief requires the scorer
to distinguish:

```
valid observed consequence          -> not a detector FP
incorrect duplicate top-level incident -> an error
```

Until that is done, **numerics episode precision will not improve** even though the
attribution logic exists. The measured 0.314 therefore stands.

**I1.4 targets not met:**

| Target | State |
|---|---|
| numerics episode precision ≥ 0.95 | **NOT MET** — scorer not corrected |
| numerics episode recall ≥ 0.95 | previously met (1.000) |
| duplicate top-level consequence incidents = 0 | **NOT VERIFIED** — no run measured |

## 6. Why this is stated rather than approximated

The precision number a scorer produces is only meaningful if the scorer encodes the
same model of the world as the system. Phase 1.4's lesson was that three holdout runs
were consumed by measuring a pipeline that did nothing. Reporting an improved
numerics precision from an attribution layer the scorer does not yet understand
would repeat that error in the opposite direction.
