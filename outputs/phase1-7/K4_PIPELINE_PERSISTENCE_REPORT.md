# K4 — PIPELINE PERSISTENCE REPORT

**Status: NOT DONE — NOT VERIFIED.**

The brief requires the real data flow:

```
TrainingEvent -> Detector -> Episode -> EpisodeStore -> Fusion -> Incident -> IncidentStore
```

`EpisodeStore` exists, is tested, and supports cross-process continuation
(`J3_EPISODE_PERSISTENCE_REPORT.md`). It is **not wired into the run pipeline**:
`tests/final_holdout/run_final_holdout.py` still writes episodes to Parquet artefacts
and never calls the store.

So the component-level guarantee holds and the **pipeline-level guarantee does not**.
The required restart scenario — process A emits, detects, persists a `SUSPECT`
episode and terminates; process B loads the run state, continues, confirms the same
episode and opens an incident — was **not executed**.
