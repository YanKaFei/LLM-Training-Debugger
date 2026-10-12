# R0 — TEST COVERAGE REPORT

**Status: COMPLETE.** The R0 gate is satisfied before any detector change.

```
before:  295 tests
added:    42 tests
total:   337 tests passing, 1 skipped
```

Phase 1.3 added an entire architectural layer — `AnomalyEpisode`, fusion, the
registry, abstention — and the test count did not move. Behaviour was verified by
direct execution, which leaves no regression protection. That state is now closed.

---

## 1. New test files

| File | Tests | Covers |
|---|---|---|
| `tests/unit/test_episode_fusion.py` | 36 | episode lifecycle, identity, serialisation, registry, fusion, promotion provenance, deduplication, cross-process reconstruction |
| `tests/unit/test_abstention_renderer.py` | 6 | all four verdict renderings, causal-language guard, determinism |

## 2. R0.1 requirement coverage

| Required test | Where |
|---|---|
| single WEAK signal → SUSPECT → NORMAL | `test_single_weak_signal_goes_suspect_then_normal` |
| persistent gradient candidate → one episode | `test_persistent_gradient_candidate_becomes_one_episode` |
| persistent Inf → one episode | `test_persistent_inf_is_one_episode` |
| persistent NaN → one episode | `test_persistent_nan_is_one_episode` |
| episode recovery | `test_episode_recovery_and_close` |
| episode close | `test_episode_recovery_and_close` |
| `candidate_onset != confirmed_at` | `test_candidate_onset_differs_from_confirmed_at_for_statistical_types` |
| 111 Inf observations → one episode → one incident | `test_111_inf_observations_produce_one_episode_and_one_incident` |
| episode serialization | `test_episode_serialisation_round_trip` |
| episode identity / signature | `test_episode_signature_ignores_observation_count`, `..._distinguishes_onset` |
| episode cross-process reconstruction | `test_cross_process_episode_reconstruction` |
| registry ENABLED / EXPERIMENTAL / DISABLED | `test_registry_has_all_three_statuses_available` |
| old detector ID resolution | `test_old_detector_ids_remain_readable` |
| fusion CERTAIN | `test_fusion_certain_promotes_alone` |
| fusion STRONG | `test_fusion_strong_promotes_alone` |
| fusion concordant WEAK | `test_fusion_concordant_weak_promotes` |
| single WEAK does not promote | `test_single_weak_does_not_promote` |
| `INSUFFICIENT_EVIDENCE` | `test_insufficient_evidence_when_nothing_resolves` |
| `NO_ANOMALY` | `test_no_anomaly_when_every_observation_was_dismissed` |
| candidate-only rendering | `test_candidate_only_is_rendered_with_its_scope` |
| abstention rendering | `test_insufficient_evidence_is_rendered_explicitly` |
| promotion provenance | `test_promotion_provenance_is_complete` |
| rejected episode provenance | `test_rejected_episode_provenance_is_recorded` |
| incident deduplication | `test_many_episodes_of_one_type_still_yield_one_incident` |
| disabled detector stays disabled | `test_disabled_detector_produces_no_episode_by_default` |
| disabled detector still observable | `test_disabled_detector_can_still_be_observed_when_explicitly_requested` |
| no causal language in output | `test_renderer_never_uses_causal_language` |

**Not covered — `episode persistence` and `consequence linkage`.** Both require
functionality that does not exist (§3, §4). They are gaps, not omissions.

## 3. Episodes are not persisted

`test_cross_process_episode_reconstruction` writes episode **dicts** to JSON in one
interpreter and reconstructs them in a second. It does **not** exercise the incident
store, because episodes are not written there: there is no episode table and no
migration path from an episode to the incident that consumed it. The Phase 1.1
incident store's restart guarantee therefore covers incidents, **not** episodes.

## 4. Two real defects found by writing these tests

Writing the tests surfaced two implementation bugs that direct execution had missed:

| Defect | Consequence | Fix |
|---|---|---|
| `NAN_DETECTED` / `INF_DETECTED` were absent from `promote.strong.single` | a numerics episode classified `STRONG` rather than `CERTAIN` matched **no promotion rule** and could never open an incident | anomaly types added to the rule |
| `CANDIDATE_ONLY` recorded no episodes | the renderer could not show what had been observed and declined, defeating the purpose of surfacing abstention | confirmed-but-unpromoted episodes are now recorded |

Both are exactly the class of silent hole that a 0-test architecture layer permits.

## 5. Gate

```
new architecture test coverage exists   -> PASS
```

Detector optimisation may now proceed.
