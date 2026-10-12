# K2 — CONSEQUENCE-AWARE SCORING REPORT

**Status: IMPLEMENTED AND UNIT-TESTED; NOT RUN OVER THE DEVELOPMENT CORPUS.**

---

## 1. What was built

`src/training_debugger/evaluation/consequences.py`. The single `numerics precision`
number is replaced by **six observation classes**:

| Class | Meaning |
|---|---|
| `PRIMARY_TRUE_POSITIVE` | the injected fault, detected in its window |
| `SECONDARY_TRUE_MANIFESTATION` | a real consequence, correctly observed and **not** promoted to a root cause |
| `UNRELATED_FALSE_POSITIVE` | neither |
| `DUPLICATE_TOP_LEVEL_INCIDENT` | a consequence claimed as an **independent root cause** — the error |
| `CORRECT_ABSTENTION` | observed and correctly not promoted |
| `MISSED_PRIMARY` | the injected fault was not detected |

Six rather than two because `PRIMARY_TRUE_POSITIVE` and
`SECONDARY_TRUE_MANIFESTATION` are **both correct observations**, while
`DUPLICATE_TOP_LEVEL_INCIDENT` is not. Collapsing them would either reward
duplication or punish correct consequence reporting — which is precisely the error
that produced the 0.314 figure.

## 2. Ground-truth schema (K2.1)

```
primary_fault_type
allowed_secondary_manifestations
secondary_window
scope_constraints
```

Carried on the scenario and **never read by the detector layer**. Where the schema is
absent, the fault type is used as the primary and `NAN_DETECTED`/`INF_DETECTED` are
treated as allowed consequences of a gradient or loss fault — the physical
relationship Phase 1.3 identified, not a scoring assumption.

## 3. Verified by test

| Behaviour | Result |
|---|---|
| primary fault in window | `PRIMARY_TRUE_POSITIVE` |
| linked `NaN` after a gradient fault | **`SECONDARY_TRUE_MANIFESTATION`, not a false positive** |
| `NaN` that also opens its own root incident | `DUPLICATE_TOP_LEVEL_INCIDENT`, rate 0.5 |
| `NaN` outside the consequence window | `UNRELATED_FALSE_POSITIVE` |
| wrong family in the window | `UNRELATED_FALSE_POSITIVE` |
| healthy-run observation | never a true positive |
| relation precision / recall | computed |

## 4. Relation semantics remain weak (K2.4)

The relation type system is unchanged: `POSSIBLE_CONSEQUENCE_OF`, `PRECEDES`,
`CO_OCCURS_WITH`, `UNKNOWN_RELATION`. **`CAUSED_BY` and `ROOT_CAUSE_OF` are not
representable**, asserted by test. Replay is a Phase 2 capability.

## 5. What is NOT done

```
numerics observation precision: NOT VERIFIED
numerics observation recall:    NOT VERIFIED
duplicate root incident rate:   NOT VERIFIED
primary precision / recall:     NOT VERIFIED
secondary precision / recall:   NOT VERIFIED
relation precision / recall:    NOT VERIFIED
```

The **scorer exists and is tested, but was not executed over the development
corpus**, because the pipeline that produces the verdicts and relations (K4) was not
wired. So the 0.314 figure is neither confirmed nor refuted — it is now known to be a
**mis-typed number**, and the corrected figure has not been measured.

This is stated rather than estimated: the whole point of K2 is that the old number
mixed two different things, and replacing it with a guess would repeat the error in a
new form.
