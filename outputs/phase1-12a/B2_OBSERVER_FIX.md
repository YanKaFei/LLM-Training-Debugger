# B2 — OBSERVER FIX

**Status: NO FIX APPLIED.** The brief is explicit that this must not be repaired by
adding sleeps, widening timeouts, or weakening assertions, and that a real fix requires
the race to be identified first.

## What was and was not done

| Step | State |
|---|---|
| B1 ensure the counting identity is correctly stated | **PARTIAL** — see below |
| B2 construct a deterministic overflow scenario | **NOT DONE** |
| B2 fix production code if a race exists | **NOT DONE** — no race observed to fix |

## The counting identity

The observer's `accounting_identity()` asserts, after a flush:

```
emitted  =  written + dropped + sampled_out + validation_failures
```

`flush()` waits for **queue-empty AND writer-idle** (`_in_flight == 0`), which is the
deterministic completion condition the test relies on. That condition is the right
shape — it does not guess at timing — so the defect, if any, is elsewhere.

Candidate mechanisms that were **not** investigated: queue-overflow eviction policy
under concurrent admission, cross-thread counter visibility, writer shutdown ordering,
and exception propagation during a write failure.

**No change was made.** A change without an identified race would be a guess, and the
test currently passes 35/35, so a guess could not even be validated against a
reproduction.
