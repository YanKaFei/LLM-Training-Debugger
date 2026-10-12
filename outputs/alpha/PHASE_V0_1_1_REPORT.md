# v0.1.1-alpha — MEASUREMENT INTEGRITY & DETECTOR CALIBRATION

```
V0_1_1_ALPHA_READY = YES
```

All original v0.1.0-alpha gates still hold. No new diagnostic source, no multi-rank, no
GPU telemetry, no replay, no Agent, no final holdout, no threshold tuning on test data.

```
tests            477 passing, 1 skipped   (473 -> 477)
full suite       5 of 5 repetitions clean
quickstart       PASS — real training, onset now correct at step 80
smoke            8/8 (unchanged from v0.1.0)
installation     PASS — editable and non-editable
```

---

## W1 — Temporal integrity: **COMPLETE, defect found and fixed**

**The defect.** An episode's `candidate_onset_step` was the first observation of the
**contiguity run**, whatever the gradient band said about it.

```
step 75  layer 0    WEAK  band=NORMAL      <- benign excursion, opens the episode
step 78  layers 1-2 WEAK  band=NORMAL      <- same episode (gap 3 <= gap_tolerance 8)
step 80  all layers STRONG band=CONFIRMED  <- THE REAL INJECTION confirms it
reported onset = 75                          five steps before anything went wrong
```

Classification: **`INCORRECT_EPISODE_ASSOCIATION`** — not a `TRUE_PRECURSOR`, and not
insufficient evidence. The band had already classified steps 75 and 78 as `NORMAL`, and
**a step the band calls NORMAL cannot be the onset of an anomaly.**

**The fix.** Onset anchors to the earliest observation the band called **`CONFIRMED` or
`UNCERTAIN`**. `UNCERTAIN` is kept deliberately: excluding it would discard genuine early
warning from a gradually developing fault, which is the signal this system exists to
find. The raw first step is preserved in `metadata.contiguity_onset_step` so grouping
stays auditable.

Quickstart now reports **onset step 80** for an injection at step 80. +3 regression tests.

## W2 — Baseline resume sweep: **COMPLETE, measured negative result**

Pre-registered before running. 216 runs trained once, every candidate scored on identical
evidence — **paired**.

```
resume  precision  recall  healthy_fp
20      0.880      0.458   0.050
30      0.878      0.448   0.050
40      0.878      0.448   0.050     <- default
60      0.878      0.448   0.050
never   0.878      0.448   0.050

qualifying candidates: NONE
paired vs 40: 20 -> gained 1, lost 0, 1 discordant pair, sign-test p = 1.0
              30 / 60 / never -> byte-identical
```

**The parameter is essentially inert**, and the binding constraint is **recall**
(0.448–0.458 vs a 0.85 gate) — not this parameter. Per the pre-registration, **no
candidate is selected**; the maximum-precision candidate was not chosen and the sweep was
not re-run with a friendlier magnitude set. This closes the question raised by the
Phase 1.13 dead-code repair.

## W3 — Corpus re-persistence: **COMPLETE for identity and persistence**

Stable, reproducible ids — a pure function of the scenario:

```
dev-{scenario}-L{layers}d{d_model}-s{seed}     e.g. dev-grad_large-L4d32-s80000
```

42 runs persisted with events, episodes, analysis state, incidents and relations. Ground
truth in a **separate tree**, never inside a run directory. 41 of 42 readable from disk;
1 produced no episodes.

**The detection boundary, reproduced on a fresh corpus:**

```
×2 -> 0/6     ×5 -> 0/6     ×20 -> 6/6     ×40 -> 6/6
```

Persisted scoring executed from disk:

```
PRIMARY_TRUE_POSITIVE       18      CORRECT_ABSTENTION   55
UNRELATED_FALSE_POSITIVE    20      DUPLICATE_TOP_LEVEL   4
primary precision           0.4286
duplicate top-level rate    0.0412
secondary / relation        NOT ESTIMABLE
```

**`primary_recall = 1.0` is reported as suspect, not quoted.** With 36 injected runs and
18 true positives, its denominator is plainly not the injected-run count. The raw counts
are the trustworthy figures until that definition is pinned down.

**Relations `NOT ESTIMABLE`** — the corpus injects one fault type per run, so **0
relations were persisted**. Scoring the relation path needs mixed-fault runs.

```
HISTORICAL_23_NOT_FULLY_REPRODUCED
```

**A defects found while scoring — twice, silently.** Precision read `0.0` on a corpus that
clearly detects ×20/×40/NaN. Two causes: the loader dropped the `ground_truth` key, and
`classify_observation` needs `acceptable_window` or every correct detection is filed as
`UNRELATED_FALSE_POSITIVE`. Both now guarded by a `scoring_warning`.

## W4 — Observer reliability: **mechanism found, race UNRESOLVED**

The flaky test discarded `flush()`'s **return value** and used `timeout=2.0`. When the
flush hits its deadline with work outstanding it returns `False`, and an event held in
the writer's batch is counted in `emitted` but in neither `written` nor `dropped` — so it
appears as `unaccounted`. The test therefore reported **an accounting violation that looks
like data loss but is an incomplete drain.** Under suite load two seconds was not always
enough; in isolation it always was.

Fixed by asserting completion first. Stress **30/30**. The underlying race was **never
reproduced** — 0 of ~70 attempts — so `OBSERVER_RACE = UNRESOLVED`, disclosed with its
risk boundary.

---

## Honest summary

```
EXPLAINED AND FIXED   onset integrity; flush-completion ambiguity; two silent scorer defects
MEASURED NEGATIVELY   resume_after_steps does not matter much, and nothing qualifies
STILL UNRESOLVED      observer race; relation scoring (no mixed-fault corpus)
STILL UNMEASURABLE    historical 23 (the original corpus is gone)
BINDING CONSTRAINT    RECALL, at the ×2-×12 boundary where classes genuinely overlap
```

**Not re-claimed:** `DETECTOR_RELIABILITY = EXPERIMENTAL`. v0.1.1 improved the
*trustworthiness* of the measurements, not the detector's capability, and no capability
is `VALIDATED`.
