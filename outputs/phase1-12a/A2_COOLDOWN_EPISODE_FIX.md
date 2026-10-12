# A2 — COOLDOWN / EPISODE FIX

**Status: COMPLETE.** Evidence and alerting are separated; deduplication was not removed.

---

## 1. The change

`DetectorConfig.cooldown_suppresses_evidence: bool = False`

When `False`, a qualifying observation during cooldown is **still emitted**, and its
metadata carries `cooldown_suppressed: True` so an alert-counting consumer can filter
while the evidence chain stays complete. The legacy behaviour is retained behind
`True` so the defect stays reproducible.

The `loss.robust_zscore` detector honours the flag at both its fire sites.

**Why not simply set `cooldown_steps = 0`:** that would have removed the *only*
deduplication the detector had, without proving anything about flooding. The flag keeps
deduplication where it belongs and demonstrably does not flood — see §3.

## 2. Idempotency, found while testing this

`test_same_observation_cannot_count_twice` exposed a second, independent defect:
**`EpisodeEngine` did not deduplicate observations**, so submitting the same step twice
manufactured the persistence a confirmation rule requires.

Fixed at `(step, layer)` granularity — not `(step)`:

* the *same* `(step, layer)` is the same observation and is ignored;
* the *same step* for a **different layer** is a genuinely distinct observation, which
  is exactly what a restarted process supplies when it resumes mid-step.

The coarser `(step)` rule was tried first and **broke
`test_cross_process_restart_readback`**, because the restarted process's new layers were
discarded; the finer rule satisfies both. Persistence still counts distinct *steps*, so
many layers at one step cannot inflate the count.

## 3. No flooding — measured, not assumed

| Property | Result |
|---|---|
| observations for a 20-step spike | 20 (was 1) |
| episodes | **1** |
| verdicts | **1** |
| incidents | **0** (LOSS_SPIKE promotion remains OFF) |

`test_many_observations_do_not_flood_the_alert_layer` asserts this, and
`test_cooldown_does_not_duplicate_incidents` asserts that twenty observations do not
become twenty incidents.

## 4. Candidate semantics preserved

| Situation | Behaviour |
|---|---|
| one qualifying observation | `SUSPECT` episode, `CANDIDATE_ONLY`, names its episode |
| the fluctuation resolves | `NOT_ANOMALY` — **no stale active candidate** |
| two observations within 6 steps | episode `CONFIRMED`, but promotion stays OFF → still `CANDIDATE_ONLY` |

`test_candidate_recovery_clears_the_verdict` pins the second row: an episode that
resolves must not leave a live candidate behind.

## 5. `LOSS_SPIKE` status unchanged

```
LOSS_SPIKE              EXPERIMENTAL
default promotion       OFF
```

This phase repaired **reachability of confirmation and candidate state**. It does not
claim reliable production detection for `LOSS_SPIKE`, and promotion stays disabled.
`test_loss_spike_promotion_remains_disabled` asserts it.
