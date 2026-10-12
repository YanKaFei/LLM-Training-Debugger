# B — PERSISTED SCORING

**Status: EXECUTED from disk. Thin denominator; historical 23 NOT reproduced.**

---

## 1. What changed

`src/training_debugger/evaluation/persisted.py` — the scorer now reads the stores
directly instead of the run script's in-memory result dict:

```
EpisodeStore.episodes           -> episodes
EpisodeStore.episode_relations  -> relations
IncidentStore.incidents         -> incidents
analysis_state revisions        -> completeness of the persisted record
```

Ground truth is read **only** in this module and never reaches the detector, the episode
engine, fusion or the relation decision.

## 2. Execution — real commands, real exit codes

```
outputs/phase1-12c/store   7 runs read from disk   1 persisted relation read
outputs/phase1-3/store    14 runs read from disk   0 persisted relations read
```

### `outputs/phase1-12c/store` (smoke runs, current schema)

| Class | Count |
|---|---|
| `UNRELATED_FALSE_POSITIVE` | 7 |
| `DUPLICATE_TOP_LEVEL_INCIDENT` | **5** |
| `CORRECT_ABSTENTION` | 6 |

```
primary precision                 0.000
primary recall                    NOT ESTIMABLE   (no run carried a label matching its episodes)
secondary precision               0.000
secondary recall                  NOT ESTIMABLE
relation precision                0.000
relation recall                   NOT ESTIMABLE
duplicate top-level incident rate 0.278
```

### `outputs/phase1-3/store` (development corpus, **old schema**)

| Class | Count |
|---|---|
| `PRIMARY_TRUE_POSITIVE` | 4 |
| `UNRELATED_FALSE_POSITIVE` | 111 |
| `DUPLICATE_TOP_LEVEL_INCIDENT` | **13** |
| `CORRECT_ABSTENTION` | 96 |

```
primary precision                 0.031
primary recall                    1.000
relation precision                NOT ESTIMABLE
relation recall                   0.000
duplicate top-level incident rate 0.058
```

## 3. Honest reading

**The scorer genuinely reads relations from disk.** That is the B1 deliverable and it is
met.

**But the measurement is thin and its denominator is not the corpus:**

* the development store holds only **14 runs**, not the 169 the corpus contained — the
  Phase 1.12A `run_id` defect meant runs were written under a path that could not be
  looked up by the identifier inside them, so most were not persisted under a usable id;
* that store predates relation persistence, so **0 relations could be read** and every
  relation metric is `NOT ESTIMABLE`;
* primary precision **0.031** is dominated by 111 unrelated false positives, which is
  what happens when a thin ground-truth mapping meets many episodes per run.

Neither figure should be quoted as the system's current quality. They are a proof that
the **path works**, on data that is not adequate to measure quality.

## 4. The historical 23

```
historical scorer count:      23   (Phase 1.9)
reproduced historical count:  HISTORICAL_23_NOT_FULLY_REPRODUCED
current, both stores:         18   (5 + 13)  -- scorer classifications, NOT re-measurement
actual product duplicates:    NOT VERIFIED
scorer misclassifications:    NOT VERIFIED
unresolved associations:      NOT VERIFIED
```

**13 + 5 = 18 is not a re-measurement of 23.** It is a different number of runs, from a
different schema, scored by a different scorer. Reporting it as a reduction would be the
substitution the brief forbids.

Producing the real audit requires re-running the development corpus through the current
pipeline so that episodes, incidents and **relations** all persist under correct run ids,
then scoring that. That is the next action, and it was not reached.

## 5. Numerics

```
observation precision / recall: NOT VERIFIED
top-level precision / recall:   NOT VERIFIED
```

The per-capability breakdown was not computed. **0.314 remains retired and unreplaced.**
