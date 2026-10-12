# J0 — TEST REPORT

**Status: COMPLETE.** The gate is satisfied: tests were added **before** any further
change.

```
before:  337
added:    34
total:   371 passing, 1 skipped
```

---

## 1. New regression files

| File | Tests | Covers |
|---|---|---|
| `tests/unit/test_gradient_band_and_relations.py` | 27 | the joint aggregate, the three-way band, band boundaries, relations |
| `tests/integration/test_episode_persistence.py` | 7 | episode revisions, band survival, relation persistence, cross-process continuation |

## 2. Required coverage

| Requirement | Test |
|---|---|
| gradient NORMAL region | `test_gradient_normal_region` |
| gradient UNCERTAIN region | `test_gradient_uncertain_region` |
| gradient CONFIRMED region | `test_gradient_confirmed_region` |
| `l1_aggregate` calculation | `test_l1_aggregate_is_the_mean_of_per_layer_ratios` |
| per-layer own trailing median | `test_each_layer_uses_its_own_trailing_median_not_a_shared_scale` |
| boundary equality `l1 == A` | `test_normal_boundary_equality_is_uncertain_not_normal` |
| boundary equality `l1 == B` | `test_confirmed_boundary_equality_is_confirmed` |
| warm-up exclusion in the baseline | `test_aggregate_excludes_the_warmup_when_asked` |
| single weak signal → no incident | `test_single_weak_gradient_signal_never_confirms` |
| uncertain → abstention | `test_uncertain_gradient_episode_produces_abstention_not_an_incident` |
| confirmed → promotion | `test_confirmed_gradient_episode_promotes` |
| relation creation | `test_gradient_fault_then_nan_creates_a_relation` |
| `POSSIBLE_CONSEQUENCE_OF` | above |
| `PRECEDES` / `CO_OCCURS_WITH` / `UNKNOWN_RELATION` | `test_relation_types_available` |
| `CAUSED_BY` unavailable | `test_causal_relation_types_are_not_representable` |
| `ROOT_CAUSE_OF` unavailable | above |
| fault + later NaN → relation | `test_gradient_fault_then_nan_creates_a_relation` |
| standalone NaN → no relation, independent | `test_standalone_nan_creates_no_relation_and_stays_independent` |
| candidate-only preserves ids | `test_candidate_only_preserves_episode_ids` |
| `INSUFFICIENT_EVIDENCE` preserves ids | `test_insufficient_evidence_preserves_episode_ids` |
| renderer receives candidate episodes | `test_candidate_only_is_rendered_with_its_scope` (Phase 1.4) |
| episode persistence | `test_suspect_then_confirmed_is_two_revisions` |
| cross-process continuation | `test_cross_process_episode_continuation` |

## 3. Two correctness bugs found by writing the tests

### a. An `UNCERTAIN` observation closed as `NOT_ANOMALY`

The brief maps the bands explicitly:

```
NORMAL    -> no episode
UNCERTAIN -> SUSPECT / INSUFFICIENT_EVIDENCE
CONFIRMED -> strong candidate
```

The engine closed **every** unconfirmed `SUSPECT` episode as `NOT_ANOMALY`, so an
`UNCERTAIN`-band observation — the abstention the band exists to express — was
reported as a positive statement that nothing happened. Fixed: the closing outcome
now depends on whether any observation carried `gradient_band == "UNCERTAIN"`.

### b. An abstention named no episode

`fuse()` returned `INSUFFICIENT_EVIDENCE` with `abstained_episodes` empty, so the
renderer had nothing to display and the abstention was unactionable. Fixed: the
unconfirmed episodes and their signal keys are now recorded.

Both are exactly the class of silent hole a zero-test architecture layer permits,
and both were found by writing the test rather than by running the code.

## 4. Third defect — found by diagnosing the development corpus

Not a test discovery but a measurement one, and the brief permits fixing it:

**`l1_aggregate` was a SUM over layers, so it scaled with model depth.** The
boundaries were fitted on 4-layer models and applied unchanged to the 6-layer
configuration in the development corpus, inflating the feature by 50 %. Measured
consequence: **16 healthy episodes above the CONFIRMED boundary**, against a fitted
healthy maximum of 33.32.

Fixed: the aggregate is now the **mean** of the per-layer ratios, which is
layer-count-invariant. The frozen operating point is preserved exactly — 34.0 / 46.0
as a 4-layer sum is 8.5 / 11.5 as a mean — so a 4-layer run is classified
identically to before. Three regression tests pin the invariance.
