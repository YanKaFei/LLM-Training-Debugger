# J3 — EPISODE PERSISTENCE REPORT

**Status: COMPLETE.** Append-only persistence and genuine cross-process continuation.

---

## 1. Storage

`src/training_debugger/persistence/episode_store.py`, layout mirroring the incident
store:

```
<root>/runs/<run_id>/episodes/
  episodes/            episode_id, revision, state, outcome, detector, versions,
                       candidate_onset_step, confirmed_at_step, last_abnormal_step,
                       recovered_at_step, signal_count, affected_layers,
                       affected_ranks, supporting_event_ids, confidence,
                       gradient_band, signature, payload (lossless)
  episode_revisions/   the state history
  episode_relations/   consequence links
```

Append-only, one file per write, temp-file then `os.replace`, reader skips and counts
an unreadable part. `fail_open=True` by default: a persistence failure is recorded in
`store.errors`, not raised, and `fail_open=False` is available and tested.

**`gradient_band` is persisted**, because the band is the evidence for *why* an
episode was not promoted — without it a restart loses the abstention rationale.

## 2. Verified

| Behaviour | Result |
|---|---|
| `SUSPECT` then `CONFIRMED` is two revisions | PASS |
| latest revision loaded by default | PASS |
| historical revision loadable | PASS |
| `open_episodes()` returns unclosed only | PASS |
| band survives persistence | PASS |
| relations persist and reload byte-identically | PASS |
| missing episode raises `KeyError` | PASS |

## 3. Cross-process continuation (J3.2)

`test_cross_process_episode_continuation`:

```
process A:  create SUSPECT episode at step 80, persist, exit
process B:  load open episodes, continue to step 82,
            promote the SAME episode, persist revision 2
```

Asserted: `episode_id == "ep_x"` (**not** a new episode), `candidate_onset_step == 80`
preserved, `confirmed_at == 82`, `signal_count == 3`, history
`["SUSPECT", "CONFIRMED"]`, and the primary process can read what the second wrote.

This is the requirement Phase 1.5 stated and did not meet.

## 4. Relation persistence (J3.3)

`append_relations()` / `load_relations()`. A `POSSIBLE_CONSEQUENCE_OF` relation
survives a restart and reloads byte-identically, so an incident's attached secondary
manifestation is reconstructable after the process that created it has exited.

## 5. Not done

* **The episode store is not wired into the live pipeline.** `run_final_holdout.py`
  writes episodes to Parquet artefacts; it does not call `EpisodeStore`. The store is
  implemented and tested but not yet the production path.
* **No revision-compaction or retention policy.** Every revision is a file.
* **Incidents and episodes are not linked by foreign key** beyond the relation table.
