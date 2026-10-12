# EVALUATION REPORT

Every number here is measured. Where a denominator does not exist, the value is
`NOT ESTIMABLE`; where a measurement does not exist, `NOT VERIFIED`.

## Scoring units

Five levels, each with its own population. Mixing them was a defect (v0.1.2) and is now
prevented by construction.

```
SIGNAL    one observation from one detector      NOT ESTIMABLE (no independent truth)
EPISODE   one AnomalyEpisode                     precision only; recall NOT ESTIMABLE
INCIDENT  one confirmed incident                 precision only
RUN       one training run                       precision and recall
RELATION  one persisted IncidentRelation         precision and recall
```

Episode-level *recall* is `NOT ESTIMABLE`: ground truth names the injected fault, not how
many episodes it should produce, so a miss appears there as an *extra* episode.

## Development corpora

### 42-run single-fault corpus

```
EPISODE   elig 97   TP 29  FP 60          precision 0.326   recall N/E
INCIDENT  elig 25   TP 18  FP  7          precision 0.720   recall N/E
RUN       elig 42   TP 18  FP  1  FN 18   precision 0.947   recall 0.500
RELATION  elig  0                         NOT ESTIMABLE
missed_primary_faults 18
```

### 50-run mixed-fault corpus

```
EPISODE   elig 100  TP 30  FP 30          precision 0.500
RUN       elig 50   TP 20  FP  0  FN 20   precision 1.000   recall 0.500
missed_primary_faults 20
```

## Detection boundary (development-supported)

```
gradient scaling  x2 -> 0/6    x5 -> 0/6    x20 -> 6/6    x40 -> 6/6
```

Mild faults are **not distinguishable from healthy variation** by any signal collected.
The system abstains rather than guessing. This is a *development* figure from a synthetic
scaling benchmark and **does not extrapolate** to real training incidents.

## Earliest evidence (persisted data, 42-run corpus)

```
eligible_fault_runs 36   claims_made 24   correct 24   false 0   abstained 12
precision 1.000   coverage 0.667   median onset error 0.0   p90 onset error 0
pre-injection confirmed incidents 0
```

**Corpus caveat:** 4 of these 6 scenarios are strong faults, so these figures are
**corpus-specific** and are not general detector quality.

## Relation layer

`POSSIBLE_CONSEQUENCE_OF` is **disabled by default**. It was measured to be
undiscriminable from an independently injected fault in every case that could be
constructed (C3/C12 produced 16 claims against `INDEPENDENT` ground truth), so the
production default records order only.

```
                              TP   FP   FN   TN    precision   recall
50-run mixed  (linker off)     0    0   10   40   NOT ESTIMABLE   0.000
45-run adversarial (off)       0    0   10   35   NOT ESTIMABLE   0.000
```

An **optimizer-boundary witness** was implemented, tested and measured as adding
**exactly zero** discrimination across two paired experiments (86 runs). It is
**research-only and default disabled**.

## Performance

```
observer core overhead      FAILS the < 3 % target at the largest workload
witness increment           +0.30 % total / +0.97 % p95   (not a fix for the above)
storage                     1,000,000 events -> 10 files, 9.6 MB, ~10 bytes/event
```

**No GPU benchmark exists.** All figures are CPU/MPS on Apple Silicon and do not
represent GPU workloads.

## What has never been done

- **No blind holdout has ever been run.** Every figure above is development data.
- **Not one capability is validated.**
