# C1 — PERSISTED SCORING

**Status: NOT DONE — NOT VERIFIED.**

## What was not done

The consequence scorer was not modified to read `episode_relations` from disk, and was
not executed against the persisted corpus.

```
persisted_scoring = NOT EXECUTED
primary TP/FP/FN          NOT VERIFIED
secondary TP/FP/FN        NOT VERIFIED
relation TP/FP/FN         NOT VERIFIED
numerics observation      NOT VERIFIED
numerics top-level        NOT VERIFIED
duplicate incident audit  NOT VERIFIED
```

## Consequence for the historical 23

The duplicate audit was not run, so:

```
historical scorer count:  23   (Phase 1.9, historical)
reproduced historical count:   HISTORICAL_23_NOT_FULLY_REPRODUCED
actual product duplicates:     NOT VERIFIED
scorer misclassifications:     NOT VERIFIED
unresolved associations:       NOT VERIFIED
```

This is the most significant unclosed item in the phase, because it is the one place
where the historical number and the current product behaviour are known to be described
by the same word ("duplicate") while meaning different things.

## What *is* now in place

`reconcile_incidents()` (Phase 1.12B) distinguishes the three situations the brief
requires — independent anomaly, plausibly-linked manifestation, duplicate emission —
and is unit-tested. The **mechanism to produce the audit exists**; the audit itself was
not executed.
