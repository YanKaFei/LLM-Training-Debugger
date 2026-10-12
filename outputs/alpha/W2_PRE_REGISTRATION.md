# W2 — PRE-REGISTRATION: `resume_after_steps` SWEEP

**Written before the sweep was run. Not edited afterwards.**

---

## 1. Why

Phase 1.13/v0.1.0 found that `resume_after_steps` was **dead code**: the freeze was
permanent. Making it live cost precision 0.973 → 0.964, recall 0.468 → 0.457,
healthy-confirmed 0.042 → 0.056, rank-correlation 0.860 → 0.748 on the 680-run corpus.

The parameter now has a real effect for the first time. Choosing its value by nudging it
until a preferred number returns is the over-fitting the brief forbids. This is a
**pre-registered** sweep instead: the candidates, the comparison method and the scoring
are fixed here, before any result is seen.

## 2. Frozen design

```
candidates   resume_after_steps in {20, 30, 40, 60, 10**6 ("never")}
             everything else held at DetectorConfig() defaults
             A/B band bounds NOT swept (that would be threshold tuning)

configs      (4 layers, d_model 32), (6 layers, d_model 48)
healthy      60 seeds per config       = 120 runs
fault        8 magnitudes x 12 seeds x 1 config = 96 runs
             magnitudes (1.5, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 40.0)
             injection step 110, 200 steps

total training runs: 216, trained ONCE and cached
evaluation: every candidate scored on the SAME cached evidence  -> paired
```

**Paired by construction.** Every candidate sees byte-identical evidence, so differences
are attributable to the parameter and not to seed noise. Uncertainty is reported as the
standard error of the paired difference, and as a sign test on discordant pairs.

## 3. Scoring

Primary (reported, not optimised):
```
confirmed precision          TP / (TP + FP)
confirmed recall             TP / (TP + FN)
healthy false-incident rate  healthy runs that opened an incident / healthy runs
fault abstention rate        fault runs with no confirmed episode / fault runs
```

Mechanism (reported):
```
reference contamination      frozen runs whose reference never re-anchors, as a fraction
confirmation delay           confirmed_at_step - injection_step, median over TP
magnitude response           P(confirmed | magnitude) per magnitude
```

## 4. Decision rule, fixed in advance

A candidate is **acceptable** only if it satisfies all of:

```
confirmed precision       >= 0.90
confirmed recall          >= 0.85
healthy false-incident    <= 0.05
```

**Tie-break among acceptable candidates:** highest recall, then lowest healthy
false-incident rate. **If no candidate is acceptable — which the current 0.457 recall
makes overwhelmingly likely — no candidate is selected**, the parameter keeps a
documented default, and `DETECTOR_RELIABILITY` stays `EXPERIMENTAL` with the measured
trade-off reported.

**Explicitly not permitted:** selecting the maximum-precision candidate and calling the
sweep a success; re-running with a different magnitude set because the first set did not
separate; using the final holdout (it does not exist and must not be created).
