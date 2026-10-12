# A3 — REGRESSION TEST REPORT

**Status: COMPLETE.**

```
before:  417
added:    11
total:   428 passing, 1 skipped
```

All in `tests/unit/test_loss_contract.py`, using real project APIs —
`DetectorRunner`, `EpisodeEngine`, `fuse`, `detector_registry` — not mocks.

---

## 1. Required tests

| Required | Test |
|---|---|
| `test_loss_spike_cooldown_does_not_starve_episode` | present |
| `test_single_observation_creates_candidate` | `test_single_observation_creates_a_candidate_not_a_confirmation` |
| `test_candidate_only_is_reachable` | `test_candidate_only_is_reachable_through_production_components` |
| `test_candidate_recovery_clears_active_verdict` | present |
| `test_two_distinct_observations_can_confirm` | present |
| `test_same_observation_cannot_count_twice` | present |
| `test_loss_spike_promotion_remains_disabled` | present |
| `test_cooldown_does_not_duplicate_incidents` | present |

Plus three the defects demanded: `test_cooldown_starves_the_episode_confirmation_rule`
(pins the defect), `test_many_observations_do_not_flood_the_alert_layer` (proves the
fix is not a flood), and `test_suppressed_observations_are_flagged_not_hidden`.

## 2. Two defects found by writing these tests

| # | Defect | How found | Fix |
|---|---|---|---|
| 1 | cooldown starves the episode confirmation rule | A1 source trace | `cooldown_suppresses_evidence` flag |
| 2 | **`EpisodeEngine` counts the same observation twice** | `test_same_observation_cannot_count_twice` | idempotency at `(step, layer)` |

Defect 2 is the more serious: it means persistence could be manufactured by feeding the
same observation twice, which is precisely the property a confirmation rule must not
have. It was invisible until a test asserted it.
