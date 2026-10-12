# v0.1.1-alpha — W2 / W3 / W4 RESULTS

---

# W2 — `resume_after_steps` pre-registered sweep

Design frozen in `W2_PRE_REGISTRATION.md` **before** the run. 216 training runs, trained
once and cached; every candidate scored on identical evidence, so the comparison is
**paired**.

| `resume_after_steps` | TP | FP | FN | precision | recall | healthy FP | abstain | delay |
|---|---|---|---|---|---|---|---|---|
| **20** | 44 | 6 | 52 | 0.880 | 0.458 | 0.050 | 0.031 | 0 |
| 30 | 43 | 6 | 53 | 0.878 | 0.448 | 0.050 | 0.042 | 0 |
| **40 (default)** | 43 | 6 | 53 | 0.878 | 0.448 | 0.050 | 0.042 | 0 |
| 60 | 43 | 6 | 53 | 0.878 | 0.448 | 0.050 | 0.042 | 0 |
| never | 43 | 6 | 53 | 0.878 | 0.448 | 0.050 | 0.042 | 0 |

**Paired comparison vs the default (40):**

```
20    gained 1, lost 0, discordant pairs 1, sign-test p = 1.0
30    gained 0, lost 0, discordant pairs 0
60    gained 0, lost 0, discordant pairs 0
never gained 0, lost 0, discordant pairs 0
```

## Result

```
qualifying candidates: NONE
selection:             none — no candidate satisfies the pre-registered gates
```

Two findings, both reportable:

1. **The parameter is essentially inert.** `30`, `60` and `never` are **byte-identical**
   to `40`. Only `20` differs, by exactly **one** confirmation out of 96 fault runs — a
   single discordant pair, sign-test **p = 1.0**. There is no statistically meaningful
   difference among the candidates.
2. **No candidate qualifies**, and the binding constraint is **recall (0.448–0.458 vs a
   0.85 gate)**, not this parameter. Precision is 0.878–0.880, close to the 0.90 gate.

Per the pre-registration, **no candidate is selected** and the parameter keeps its
documented default of `40`. The maximum-precision candidate was **not** chosen and the
sweep was **not** re-run with a friendlier magnitude set. `DETECTOR_RELIABILITY` remains
`EXPERIMENTAL`, and this is **not** a calibration success — it is a measured negative
result that closes the question raised by the Phase 1.13 dead-code repair.

---

# W3 — development corpus re-persistence

**Stable run identity**, a pure function of the scenario so a re-run reproduces it
exactly:

```
{split}-{scenario}-L{layers}d{d_model}-s{seed}
e.g. dev-grad_large-L4d32-s80000
```

**42 runs persisted** with `TrainingEvents`, `AnomalyEpisodes`, `AnalysisState`,
`Incidents` and `IncidentRelations`. Ground truth is written to a **separate tree**
(`outputs/alpha/corpus/ground_truth/`) and never enters a run directory.

## Measured, from disk

```
runs readable from the store : 41 of 42   (1 run produced no episodes, so no store dir)
episodes persisted           : 97
incidents persisted          : 25
relations persisted          : 0
```

| scenario | runs | episodes | incidents | verdicts |
|---|---|---|---|---|
| healthy | 6 | 12 | **1** | 5 NO_ANOMALY, 1 CONFIRMED_INCIDENT |
| loss_spike ×20 | 6 | 13 | 6 | 6 CONFIRMED_INCIDENT |
| grad_small ×2 | 6 | 16 | **0** | 6 NO_ANOMALY |
| grad_mid ×5 | 6 | 15 | **0** | 6 NO_ANOMALY |
| grad_large ×20 | 6 | 13 | 6 | 6 CONFIRMED_INCIDENT |
| grad_extreme ×40 | 6 | 15 | 6 | 6 CONFIRMED_INCIDENT |
| nan_loss | 6 | 13 | 6 | 6 CONFIRMED_INCIDENT |

**The detection boundary is stark and reproducible on a fresh corpus:**

```
×2  -> 0/6      ×5  -> 0/6      ×20 -> 6/6      ×40 -> 6/6
```

Mild gradient faults are **not distinguishable from healthy variation** by any signal
collected. The system abstains rather than guessing, which is the designed behaviour.

## Persisted scoring, executed from disk

```
PRIMARY_TRUE_POSITIVE          18
UNRELATED_FALSE_POSITIVE       20
DUPLICATE_TOP_LEVEL_INCIDENT    4
CORRECT_ABSTENTION             55

primary precision              0.4286
primary recall                 1.0        <- SEE BELOW
duplicate top-level rate       0.0412
secondary / relation           NOT ESTIMABLE
```

**`primary_recall = 1.0` must not be read as "every fault was found."** There are 36
injected runs and only 18 true positives; `grad_small` and `grad_mid` were 0/6 each. The
recall denominator is therefore **not** the injected-run count. Until that definition is
pinned down, the **raw counts above are the trustworthy figures** and the ratio is
reported as suspect rather than quoted.

**Relations: `NOT ESTIMABLE`.** The corpus injects exactly one fault type per run, so no
run contains two distinct anomalies that could be linked, and **0 relations were
persisted**. Scoring the relation path needs mixed-fault runs (a gradient fault plus a
downstream non-finite), which this corpus does not contain.

## The historical 23

```
HISTORICAL_23_NOT_FULLY_REPRODUCED
```

The original 169-run corpus is gone: only 14 runs survived under usable ids and it holds
**0 relations**. The 23 classifications cannot be re-derived, and reconstructing them
across a different corpus, schema and scorer would produce a number, not a reproduction.
Per the brief, this is stated plainly rather than back-filled.

## A defect found while scoring — twice

`primary precision` read **0.0** on a corpus that plainly detects ×20/×40/NaN/`LOSS_SPIKE`.
Two causes, both silent:

1. `build_scenarios_from_store` dropped the `ground_truth` key, so the scorer fell back
   to `primary_anomaly_type("grad_large")` — a scenario label it cannot resolve —
   and every primary scored as a miss.
2. `classify_observation` needs `acceptable_window` to place the onset in the expected
   interval. Without it, `primary_match` is true but `in_window` is false, so **every
   correct detection is filed as `UNRELATED_FALSE_POSITIVE`**.

Both are now guarded: the loader emits a `scoring_warning` when an injected run has no
`acceptable_window`, so this cannot silently zero a scoreboard again. **A scorer that
reports 0.0 without an error is more dangerous than one that reports nothing.**

---

# W4 — observer reliability

## The flaky mechanism, identified

`test_accounting_identity_holds_under_queue_overflow` called `observer.flush(timeout=2.0)`
and **discarded the return value**, then asserted `unaccounted == 0`.

`flush` returns `False` when it hits its deadline with work outstanding:

```python
if deadline is not None and time.monotonic() > deadline:
    break          # returns with events still queued or in flight
```

An event held in the writer's batch is counted in `emitted` but in **neither** `written`
nor `dropped`, so it appears as `unaccounted`. Under full-suite load, two seconds was not
always enough — and the test then reported **an accounting violation that looks like data
loss but is an incomplete drain.** In isolation the flush always finished within two
seconds, which is why it passed 3/3 there.

**Fixed** by asserting the flush completed first (`timeout=30.0`, return value checked),
which separates "the drain did not finish" from "an event vanished". A second test
documents the distinction explicitly.

## Stress

```
isolated repetitions: 30   passes 30   failures 0
```

```
OBSERVER_RACE = UNRESOLVED
```

The underlying race was **never reproduced** — 0 of roughly 70 attempts across two
sessions, plus a slow-writer attempt to force the in-flight window deliberately. The
mechanism above explains a *misleading failure*, not a lost event. The accounting
identity itself held in every configuration tried, including `flush(timeout=0.0)` with a
slow writer (`written + dropped == expected`, exactly).

Not described as fixed. Disclosed with its risk boundary: fail-open means a race would
corrupt **counts**, not training numerics, and completeness is now surfaced on every
verdict rather than silent.
