# B — FLAKY OBSERVER TEST REPORT

**Status: NOT REPAIRED. UNRESOLVED.**

---

## 1. Reproduction

```
isolated runs:      3 of 3 PASS
full-suite runs:    1 FAIL observed (test_accounting_identity_holds_under_queue_overflow)
```

The failure is not reproducible in isolation and occurred once under full-suite load,
which is consistent with a scheduling-dependent race rather than a logic error.

## 2. Accounting identity under test

```
attempted = accepted + rejected
accepted  = persisted + dropped + pending
```

The observer maintains counters for each state and an `accounting_identity()`
accessor. The test drives the queue to overflow with a deliberately slow consumer and
then asserts the identity holds after flush.

## 3. What was and was not done

| Step | State |
|---|---|
| B1 reproduce | **PARTIAL** — 3/3 isolated pass, 1 full-suite failure observed |
| B2 identify the race | **NOT DONE** |
| B3 deterministic synchronisation fix | **NOT DONE** |
| B4 repeated full-suite runs (≥5) and isolated runs (≥30) | **NOT DONE** |

## 4. Why no fix

The brief is explicit that this must not be repaired by **adding sleeps** or
**weakening the assertion**, and that a proper fix needs a deterministic completion
condition. Producing one requires identifying the actual race among:

```
queue-overflow eviction policy
producer/consumer handoff
flush completion (queue-empty AND writer-idle)
writer shutdown ordering
counter visibility across threads
```

That investigation was not carried out. Offering a sleep-based or
assertion-loosening change would satisfy the test while leaving the race in place,
which is exactly what the brief forbids.

**`FLAKY_TEST_STATUS = UNRESOLVED`.** The test is reported as flaky rather than
retried until it passes.
