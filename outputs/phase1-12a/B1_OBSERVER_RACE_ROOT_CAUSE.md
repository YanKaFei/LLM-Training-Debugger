# B1 — OBSERVER RACE ROOT CAUSE

**Status: NOT IDENTIFIED.** The race was not reproduced in 35 controlled repetitions.

---

## 1. Reproduction attempts

| Context | Repetitions | Passes | Failures |
|---|---|---|---|
| isolated (`tests/unit/test_observer.py::test_accounting_identity_holds_under_queue_overflow`) | 30 | **30** | **0** |
| full suite | 5 | **5** | **0** |

## 2. A discarded measurement, reported rather than hidden

An earlier run in this phase executed a **wrong path**
(`tests/fault_injection/test_observer_faults.py::...`, which does not contain the
test) and reported `0/30`. Those were pytest **collection errors — "test not found" —
not test failures**, and the result is discarded. It is recorded here because a
`0/30` figure that is really a path typo would look exactly like a severe race if left
unexplained.

## 3. What this means

Phase 1.11 recorded one full-suite failure in a single run. This phase's 5 full-suite
runs and 30 isolated runs produced **no failure at all**.

That is **not** proof the race is absent. It means:
* the failure is rare enough not to appear in 35 attempts, or
* it depends on machine load, and this phase's runs did not reproduce that load, or
* the Phase 1.11 failure had a different cause that was not captured.

The brief requires that a single unexplained failure yields
`OBSERVER_STABILITY = FAIL`. No failure was observed here, so that clause is not
triggered — but **absence of a reproduction is not a repair**, and no root cause was
identified.
