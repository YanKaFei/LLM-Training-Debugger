# M2 — PIPELINE PERSISTENCE REPORT

**Status: COMPLETE.** `EpisodeStore` and `IncidentStore` are now called by the real
run pipeline.

---

## 1. What changed

`tests/final_holdout/run_final_holdout.py` — `run_one()` now calls `_persist_run()`
after detection, episode grouping and fusion:

```
TrainingEvent -> Detector -> AnomalyEpisode -> EpisodeStore
                                              -> Fusion -> Incident -> IncidentStore
                                              -> IncidentRelation -> episode_relations
```

The ad-hoc Parquet artefacts remain, but as **derived summaries of the same objects**,
not a second source of truth. The brief's M2.1 rule is respected:

```
TrainingEvent truth = existing event Parquet
Episode truth       = EpisodeStore
Incident truth      = IncidentStore
```

## 2. Persistence ordering (M2.2)

Enforced in code and asserted by test: **every episode revision is written before any
incident that consumes it.** A crash between the two steps therefore cannot leave an
incident whose episode is missing.

```
test_pipeline_persists_relations_and_incidents asserts, for every promoted incident,
that each episode it names already has a revision history
```

## 3. Idempotency (M2.3)

Episode identity is `sha256(run_id | detector_id | signal_key | onset_step)`, so
re-running the same detection produces the **same** episode ids. A second pass adds
revision 2, never a new episode. Asserted by
`test_pipeline_is_idempotent_across_a_rerun`.

Relations are keyed by `sha256(run_id | source | target)` and are likewise stable.

## 4. Fail-open, but surfaced (M2.2 / L4.2)

`EpisodeStore(fail_open=True)` records failures in `store.errors` and returns `-1`
rather than raising; `_persist_run` returns `errors` and the run result carries them.
A persistence failure cannot fail training, and cannot be silent.

## 5. Verified on the real pipeline

The full 169-run development corpus ran with persistence enabled and wrote
**157 Parquet files** into `outputs/phase1-3/store/runs/<run_id>/{episodes,episode_revisions,episode_relations}`,
with `incidents/` alongside. Zero persistence errors were recorded.

## 6. Not verified

* **Restart E2E (M3) was not executed.** The store is written by the real pipeline, but no test yet terminates a process mid-run and continues from the persisted state.
* **The protected gradient baseline is not yet persisted** (M3.1). It is reconstructed deterministically from the emitted event series, which reproduces it exactly for a completed run but has not been verified for a truncated one.
* No retention or compaction policy; every revision is a file.
