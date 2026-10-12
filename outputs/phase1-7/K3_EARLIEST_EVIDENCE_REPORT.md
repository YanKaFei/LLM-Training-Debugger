# K3 — EARLIEST EVIDENCE REPORT

**Status: EVALUATOR IMPLEMENTED AND UNIT-TESTED; NOT RUN OVER THE DEVELOPMENT CORPUS.**

Six phases reported this as the product's core claim and never measured it. The
evaluator now exists; the measurement does not.

---

## 1. Definition (K3.1)

Graded on the **earliest defensible abnormal episode onset**, never the earliest raw
detector fire. A fire that precedes the fault is a false positive, not an early
detection.

## 2. The strict condition (K3.3)

A claim is `TRUE` only when **all three** hold:

1. the episode is confirmed;
2. it belongs to the **expected anomaly family**;
3. its onset lies inside the acceptable window.

Finding a healthy fluctuation at step 40 for a fault at step 80 is therefore a
**false earliest claim** even when the family matches.

A confirmed episode of the **wrong** family is graded `UNKNOWN`, not `FALSE`: it is
neither a true claim nor a false one, because the system has no evidence about this
fault. Counting it as `FALSE` would penalise correctly declining to claim.

## 3. Reported separately (K3.2)

```
EARLIEST_EVIDENCE_PRECISION
EARLIEST_EVIDENCE_COVERAGE
absolute onset error (median)
signed onset error (median)
confirmation delay (median)
UNKNOWN rate
ABSTENTION rate
pre-injection confirmed incident count
```

## 4. Coverage guards against answering UNKNOWN to everything (K3.4)

`UNKNOWN` is permitted and is **not** a false positive, which alone would let a system
score perfect precision by never claiming anything. `EARLIEST_EVIDENCE_COVERAGE` is
therefore reported beside precision and both are compared against the targets
together. A test asserts that an all-`UNKNOWN` population scores precision `None` and
coverage `0.0`.

## 5. Development targets

```
earliest evidence precision >= 0.92
earliest evidence coverage  >= 0.80
pre-injection confirmed incidents = 0
```

**None is measured.**

## 6. Verified by test

| Behaviour | Result |
|---|---|
| onset in window | `TRUE`, precision 1.0, coverage 1.0 |
| onset before the fault | `FALSE`, pre-injection count 1 |
| wrong family | `UNKNOWN`, precision `None`, coverage 0.0 |
| insufficient-evidence outcome | `UNKNOWN`, not `FALSE` |
| all-UNKNOWN population | precision `None`, coverage 0.0 |
| confirmation delay separated from onset error | PASS |
| healthy runs excluded | PASS |

## 7. Why it was not run

Running the evaluator requires the development corpus to carry, per run, the
confirmed episodes with band metadata, the verdicts, and the relations — produced by
a pipeline that wires `EpisodeStore` into the run and writes relations (K4). That
integration was not built this phase, so the evaluator ran only against constructed
scenarios in tests.

```
EARLIEST_EVIDENCE_PRECISION: NOT VERIFIED
EARLIEST_EVIDENCE_COVERAGE:  NOT VERIFIED
onset error:                 NOT VERIFIED
confirmation delay:          NOT VERIFIED
pre-injection incidents:     NOT VERIFIED
```
