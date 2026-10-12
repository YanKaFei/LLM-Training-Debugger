# P0 — RESTART READBACK REPORT

**Status: PASS.** The evidence chain is now readable in the reverse direction.

---

## 1. The gap this closes

Phase 1.9 proved the system could **write** evidence. Nothing proved it could **read it
back**. The specific inconsistency the brief forbids is:

```
Episode  = SUSPECT
Baseline = newly initialized
```

A persisted episode whose detector reference had been silently rebuilt would mean the
same run judged by two different baselines.

## 2. Design choice (brief P0.2)

| Option | Verdict |
|---|---|
| **A. Persist a detector-state snapshot** | **adopted as the source of truth** |
| **B. Rebuild deterministically from persisted events** | retained as a **verifier**, not the mechanism |

**B cannot be the only mechanism**, and the reason is decisive: a truncated run's
baseline depends on *where processing stopped*. Rebuilding from the full event stream
would produce the **finished** baseline — the contaminated one — rather than the
reference actually in force at the moment of the crash. That is precisely the failure
mode being guarded against.

A is cheap: one reference and a dispersion per observed layer. So the guarantee is real
for negligible cost, and B remains available to detect a corrupted snapshot rather than
trusting it.

**This is not training replay.** No model, optimiser or data-loader state is restored,
and nothing claims a training run can be resumed. Only the **analysis** state needed to
continue judging an already-emitted event stream is persisted.

## 3. What is restored (P0.1)

`AnalysisState` + `EpisodeStore` between them restore:

```
run_id, episode_id, incident_id
episode revision
candidate_onset_step, confirmed_at_step, last_abnormal_step
supporting_event_ids
detector versions, fusion rule version, episode rule version
gradient protected baseline: reference, dispersion, state, frozen, reason, freeze_step
incident lifecycle and relations
```

`AnalysisStateStore` is append-only, one file per revision, temp-then-`os.replace`,
readers skip and count an unreadable part, fail-open with errors surfaced.

## 4. The cross-process test (P0.3)

`test_cross_process_restart_readback` runs two **separate interpreters**:

```
process A   one observation of a fault at step 80  -> SUSPECT, confirmed_at = None
            persists episode revision 1 + analysis state (revision 1)
            exits

process B   loads OPEN episodes from the store
            loads analysis state
            EpisodeEngine.resume(...)
            consumes the remaining three observations
            -> the SAME episode CONFIRMED
```

Asserted:

| Property | Result |
|---|---|
| same `run_id` | PASS |
| same `episode_id` | PASS — not a new episode |
| same `candidate_onset_step` (80) | PASS |
| correct `signal_count` (4, each observation counted once) | PASS |
| **no duplicate episode** | PASS — `episode_count == 1` |
| **protected baseline restored to the pre-fault reference** | PASS — every reference ≈ 1.0, not a fresh initialisation |

Additional tests: the snapshot reference is taken from the **pre-fault** level; the
snapshot round-trips exactly through the store; episodes reload byte-identically; and
`resume()` ignores closed episodes.

## 5. A defect this work exposed

Running the real corpus revealed that `_persist_run` computed the store path as
`"{fault_type}-{seed}"` while `engine.consume()` had grouped the episodes under
`fault.fault_type`. **Episodes carried one run id and were stored under another** — a
directory entry that cannot be found by the identifier inside it. Fixed: the store now
uses the same run id the episodes carry.

## 6. Gate

```
P0 Restart readback E2E  ->  PASS   (5/5 tests, two real interpreter processes)
P0.1 Protected baseline restored  ->  PASS
P0.2 Idempotency: no duplicate episode on resume  ->  PASS
```

**P0 PASS.** Proceeded to P1.
