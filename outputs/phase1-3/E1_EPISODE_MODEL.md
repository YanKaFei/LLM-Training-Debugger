# E1 — EPISODE MODEL

**Status: COMPLETE.** `src/training_debugger/episodes/`.

---

## 1. The architectural change

```
BEFORE:  metric -> detector -> incident
AFTER:   raw signal -> candidate anomaly -> ANOMALY EPISODE -> fusion -> verdict
```

Phase 1.2's failures were all the same failure: a detector fire was interpreted as
an incident immediately, so a persistent `Inf` became 111 findings and a single
healthy excursion became a gradient explosion.

## 2. The object

`AnomalyEpisode` carries every field the brief requires: identity
(`episode_id`, `run_id`, `detector_id`, `detector_version`, `rule_version`,
`anomaly_type`, `signal_key`), the two distinct moments
(`candidate_onset_event_id`/`candidate_onset_step` and
`confirmed_event_id`/`confirmed_at_step`), the tail
(`last_abnormal_event_id`/`last_abnormal_step`,
`recovered_event_id`/`recovered_at_step`), the population
(`affected_layers`, `affected_ranks`, `signal_count`), the magnitude
(`peak_value`, `peak_step`, `baseline`, `threshold`, `peak_ratio`), the
classification (`severity`, `confidence`, `confidence_class`), and provenance
(`supporting_event_ids`, `observation_steps`, `metadata`).

Derived, never stored: `duration_steps`, `confirmation_delay_steps`,
`cross_layer_multiplicity`.

## 3. State machine (E2)

```
NORMAL -> SUSPECT -> CONFIRMED -> RECOVERING -> CLOSED
             |
             +-------> NORMAL          (the transition the brief insists on)
```

| Transition | Trigger |
|---|---|
| `NORMAL → SUSPECT` | the first abnormal observation |
| `SUSPECT → CONFIRMED` | a confirmation rule fires (see §4) |
| `SUSPECT → NORMAL` | ``confirm_window`` steps with no corroboration → `outcome = NOT_ANOMALY` |
| `CONFIRMED → RECOVERING` | ``recovery_steps`` without a breach |
| `RECOVERING → CLOSED` | episode ends |

`EpisodeOutcome` is the terminal classification: `CONFIRMED_ANOMALY`,
`NOT_ANOMALY`, or `INSUFFICIENT_EVIDENCE`. **`SUSPECT → NORMAL` is what makes
"nothing happened" expressible**, and it is the mechanism by which 71 healthy signal
fires became 0 incidents.

## 4. Confirmation rules (E3/E4/E5/E6)

Per anomaly type, in `EPISODE_RULES`. Any one suffices:

| Route | Applies to | Rationale |
|---|---|---|
| `CERTAIN` observation | numerics | a measured `NaN` cannot be a statistical accident; delaying it would be dishonest |
| aggregate series departed | gradient | a real explosion moved the global norm 15.7×; healthy tails did not move it at all |
| cross-layer multiplicity ≥ 0.4 | gradient | measured split: healthy 1 of 7 (0.14), injected 6–7 of 7 (0.86–1.0) |
| peak ratio ≥ magnitude override | gradient, loss | a very large single event does not need corroboration |
| persistence: 2 within 5 steps | gradient, loss | a single step cannot be resolved from noise (lag-1 autocorrelation 0.29) |

## 5. Why cooldown does not define identity

The brief is explicit: *cooldown 只能降低日志量，cooldown 不得定义异常身份*.

Phase 1.2's cooldown was inside the detector, so it **hid observations from the
grouper** — which is exactly why one poisoned optimiser could still produce hundreds
of findings, each separated by more than the cooldown. Identity is now **contiguity
of observation** (``gap_tolerance`` steps), and the detectors emit every observation.
A persistent `Inf` from step 80 to 190 is one episode because the observations are
contiguous, not because a timer suppressed them.

## 6. Grouping granularity

Grouping is by ``(detector_id, rank)``, **not** per layer. Grouping per layer would
turn a 7-layer explosion into 7 episodes. Layers accumulate into
`affected_layers` on the single episode, which is what makes cross-layer
multiplicity available to fusion as *evidence* rather than as description.

## 7. Not verified

* **`gap_tolerance` was not swept.** 8 steps for gradient, 15 for divergence, chosen from the observation cadence rather than measured.
* **Multi-rank grouping is untested.** Every run is single-rank.
* **Episode identity across a restart** uses the `(run_id, detector_id, signal_key, onset_step)` signature; persistence of episodes is not yet wired to the incident store (E1 persistence report covers what exists).
