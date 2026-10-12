# E1 — EPISODE PERSISTENCE REPORT

**Status: PARTIAL.** Episodes persist to Parquet artefacts; they are **not** written
to the incident store, and cross-process episode reconstruction is **NOT VERIFIED**.

---

## 1. What persists

`episode_results.parquet` — one row per episode, written by
`tests/final_holdout/run_final_holdout.py`:

```
run_id, fault_type, split, injected, magnitude,
episode_id, detector_id, anomaly_type, signal_key,
state, outcome,
candidate_onset_step, confirmed_at_step, last_abnormal_step,
signal_count, affected_layer_count, cross_layer_multiplicity,
confidence_class, peak_ratio, confirmation_delay_steps
```

`incident_results.parquet` — one row per fusion verdict, carrying the provenance the
brief requires:

```
verdict, anomaly_type, candidate_onset_step, confirmed_at_step,
promotion_rule_id, promotion_rule_version,
episodes_used, episodes_rejected, abstained_episodes, signals_used,
confidence, severity
```

`run_results.parquet` and `final_holdout_results.parquet` — one row per run.

## 2. Fusion provenance survives persistence

Every incident row names:

* `promotion_rule_id` and `promotion_rule_version` — **which rule** promoted it;
* `episodes_used` — the episodes that justified it;
* `episodes_rejected` — episodes of the same type that were dismissed;
* `abstained_episodes` — episodes that could not be resolved;
* `signals_used` — the signal keys involved.

So "why did this become an incident?" is answerable from the stored row without a
language model. This is the mechanism the brief's section 13.1 requires.

## 3. Deterministic reconstruction

| Property | State |
|---|---|
| `AnomalyEpisode.to_dict()` / `from_dict()` round trip | implemented; derived fields (`duration_steps`, `confirmation_delay_steps`, `cross_layer_multiplicity`) are recomputed rather than stored |
| `AnomalyEpisode.signature()` | stable identity from `(run_id, detector_id, signal_key, onset_step)`, deliberately excluding timing and observation count so a re-run over a slightly different number of steps still matches |
| Episode persistence into the incident store | **NOT IMPLEMENTED** |
| Cross-process episode reconstruction test | **NOT VERIFIED** |

## 4. What is missing, stated plainly

The Phase 1.1 incident store (`src/training_debugger/persistence/`) persists
*incidents* with revisions and provenance, and its cross-process restart test passes.
It does **not** know about episodes: there is no `episode` table, no episode
revision, and no migration path from an episode to the incident that consumed it.

Consequence: an episode can be re-derived from the run's Parquet events plus the
frozen rules, but it cannot be *read back* as a durable object. Phase 1.2's headline
requirement — "restart reconstruction works" — therefore holds for incidents and
**not** for episodes.
