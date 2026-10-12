# PHASE 1.10 REPORT

```
PHASE 1.10 STATUS:              PARTIAL
ENGINEERING_CLOSURE:            FAIL
DETECTOR_RELIABILITY:           FAIL
READY_FOR_BLIND_VALIDATION:     NO
```

**No final holdout files were created.**

`PHASE_1_REPORT.md` and `outputs/phase1-1..1-9/` are **unmodified**.

---

## 1. What the phase proved

### P0 — the evidence chain is now readable in reverse: **PASS**

Phase 1.9 proved the system could write evidence. P0 proves it can **recover it**.

`AnalysisState` persists the **protected gradient baseline** — reference, dispersion,
state, frozen flag, reason, freeze step — alongside the episode. The brief's forbidden
state (`Episode = SUSPECT`, `Baseline = newly initialized`) is now structurally
impossible.

`test_cross_process_restart_readback` runs **two separate interpreters**:

```
process A   one observation at step 80  -> SUSPECT, confirmed_at = None
            persists episode revision 1 + analysis state, exits

process B   loads open episodes + analysis state
            EpisodeEngine.resume(...)
            consumes three more observations
            -> the SAME episode CONFIRMED
```

Asserted: same `run_id`, same `episode_id` (**not a new episode**), same onset 80,
`signal_count` 4 with each observation counted once, `episode_count == 1`, and every
restored baseline reference ≈ the pre-fault level rather than a fresh initialisation.

**Design note.** Option B (rebuild from persisted events) is deliberately *not* the
mechanism: a truncated run's baseline depends on **where processing stopped**, so
rebuilding from the full stream would produce the *contaminated* reference — the exact
failure being guarded against. A snapshot is the source of truth; B is retained as a
verifier.

**No training replay is claimed.** No model, optimiser or data-loader state is
restored, and nothing here says a training run can be resumed.

### P1 — CLI readback: **PARTIAL**

The `verdicts` command now **prefers persisted evidence** over recomputation and reads
the episode store in a fresh process:

```
command:  ...cli.main --root outputs/phase1-3/store verdicts F1_loss_spike
exit:     0
source:   .../runs/F1_loss_spike/episodes/episodes/*.parquet
```

But only **`NO_ANOMALY`** was observed. `CONFIRMED_INCIDENT`, `CANDIDATE_ONLY` and
`INSUFFICIENT_EVIDENCE` are implemented and unit-tested and were **not** produced from
persisted corpus data, because the run chosen carried no confirmed episode.

The brief's P1 gate requires all four verified against real persisted data.
**Unit-tested is not what P1 asks for, so the gate is not met.**

## 2. Work stopped at the P1 gate, per the brief

> 如果当前步骤没有通过 Gate，不要跳到下一步。

P2 (consequence attribution and scoring), P3 (full-pipeline smoke), P4 (flaky test)
and P5 (development evaluation) were **not started**. No code was written for them, no
command was executed, and nothing is estimated in their place.

## 3. A defect this phase found

`_persist_run` computed the store path as `"{fault_type}-{seed}"` while
`engine.consume()` grouped episodes under `fault.fault_type`. **Episodes carried one
run id and were stored under another** — a directory entry that cannot be found by the
identifier inside it. This is exactly the class of inconsistency P0 exists to catch,
and it was found by *reading the store back* rather than by any unit test. Fixed.

## 4. Tests

```
before:  412
added:     5
total:   417 passing, 1 skipped
```

New: cross-process restart readback, snapshot taken from the pre-fault level, exact
snapshot round trip, episodes reload byte-identically, `resume()` ignores closed
episodes.

**The flaky test is unfixed.** `test_accounting_identity_holds_under_queue_overflow`
passes 3/3 in isolation and failed once in a full-suite run, consistent with a
queue-overflow timing dependency. The brief forbids fixing it by adding sleeps or
weakening the assertion; a proper fix needs a deterministic completion condition, and
that investigation was not done. It is reported as unresolved rather than retried away.

## 5. Gate summary

| Gate | State |
|---|---|
| Restart readback E2E | **PASS** |
| Protected baseline restored | **PASS** |
| Episode/Incident idempotency on resume | **PASS** |
| CLI persisted-data E2E | **PARTIAL** — exit 0, 1 of 4 verdicts observed |
| Four verdicts rendered from persisted data | **FAIL** |
| IncidentRelation consumed by scorer | **NOT DONE** |
| Production duplicate-root issue addressed | **NOT DONE** — 23 duplicates stand |
| Independent NaN not wrongly suppressed | **NOT VERIFIED** |
| Numerics current quality measured | **NOT VERIFIED** |
| Full pipeline smoke | **NOT RUN** |
| Flaky observer test repaired | **FAIL** |
| Full test suite | **PASS** — 417 passing |
| Earliest evidence from persisted data | **NOT RUN** |
| Per-capability quality | **NOT VERIFIED** |
| No new detector introduced | **PASS** |
| No final holdout | **PASS** |
| No Agent | **PASS** |
| No Phase 2 feature | **PASS** |

```
ENGINEERING_CLOSURE   = FAIL   (P1 partial, P2–P5 not done)
DETECTOR_RELIABILITY  = FAIL   (unchanged from 1.9: episode recall 0.341)
READY_FOR_BLIND_VALIDATION = NO
```

## 6. Governing principle

> Phase 1.9 proved the system can **write** evidence.
>
> **Phase 1.10 proved it can read it back — a second process resumes the same episode,
> restores the same protected baseline, and does not duplicate the episode — and then
> stopped at the CLI gate, because only one of the four verdicts could be produced from
> real persisted data.**

The reverse direction is open for episodes and baselines. It is not yet open for
verdicts, relations or scoring.

**No agent. No replay. No new diagnostic source. No causal claim.**
