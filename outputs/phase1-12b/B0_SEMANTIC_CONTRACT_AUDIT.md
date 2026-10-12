# B0 — SEMANTIC CONTRACT AUDIT

**Status: COMPLETE.** Both suspected defects were confirmed and repaired.

---

## B0.1 `CANDIDATE_ONLY` conflated two different facts

Phase 1.12A's contract repair made `CANDIDATE_ONLY` reachable — and in doing so exposed
that it meant two unrelated things:

```
confirmed episode, promotion DISABLED   <- evidence is fine, capability is not cleared
confirmed episode, no rule matched      <- evidence is fine, but only WEAK
unresolved episodes                     <- the evidence really is insufficient
```

Reporting all three as "candidate" invites a user to read *"we could not tell"* when the
truth is *"we can tell, and this build is not cleared to raise an incident"*.

**Repair: two independent dimensions**, with the external `Verdict` API unchanged so no
existing consumer breaks.

```
evidence_state:        SUSPECT / CONFIRMED / CLOSED
promotion_disposition: ELIGIBLE / DISABLED / WITHHELD
reason_code:           PROMOTED
                       CONFIRMED_BUT_PROMOTION_DISABLED
                       CONFIRMED_BUT_NO_RULE_MATCHED
                       EVIDENCE_INSUFFICIENT
                       OBSERVATIONS_DISMISSED
                       NO_OBSERVATIONS
```

Measured on a confirmed `LOSS_SPIKE`:

```
verdict:       CANDIDATE_ONLY
reason_code:   CONFIRMED_BUT_PROMOTION_DISABLED
evidence_state: CONFIRMED
disposition:   DISABLED
reason:        the LOSS_SPIKE episode is CONFIRMED; promotion for this experimental
               capability is switched off in this build ... This is not a statement
               that the evidence was insufficient.
```

The CLI renderer now states this explicitly instead of implying missing evidence.

## B0.2 Observation deduplication key

**What the key is:** `(step, layer_index)`, held in `_OpenEpisode.observed_pairs`, scoped
to one `signal_key` (`detector_id:rank`) within one run. Verified behaviour:

| Case | Result |
|---|---|
| same source observation retried | **counted once** |
| distinct legitimate observations at the same step/layer | **preserved** |
| different anomaly types at the same step/layer | **preserved** — they are different `signal_key`s, hence different episodes |
| restarted processing | **no double counting** for identical `(step, layer)`; new *layers* at an already-seen step are correctly accepted |

**Limitation, recorded rather than papered over:** there is **no source-event-level
identity**. `Anomaly` carries `detector`, `metric_name`, `step`, `rank` and `layer_index`
but **no `event_id`**, so the dedup key cannot be traced to the `TrainingEvent` that
produced it. Two genuinely different observations that happen to share
`(step, layer)` within one detector and rank would be indistinguishable.

So the correct claim is **"idempotent at `(step, layer)` granularity"**, not "fully
deduplicated". Reaching source-event identity would require the detector to carry the
originating `event_id` through to the anomaly — a schema change not undertaken here.
