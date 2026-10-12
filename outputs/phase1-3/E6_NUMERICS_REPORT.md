# E6 — NUMERICS DETECTOR REPORT

**Status: PARTIAL.** Episode model implemented and working; precision target missed.

---

## 1. Episode model (brief section 11)

NaN and Inf need no statistical threshold, so they do not get one:

```
NaN or Inf  ->  OBSERVED  ->  CERTAIN  ->  confirms on first observation
```

Delaying confirmation of a measured `NaN` would be theatre: nothing can make the
measurement false. `EpisodeRule.certain_confirms = True` and
`persistence_required = 1`.

## 2. One persistent poison, one episode — verified

The brief's central numerics requirement:

```
step 80  gradient = Inf
step 81  gradient = Inf
   ...
step 190 gradient = Inf
    ->  1 episode, not 111
```

Verified directly: **111 consecutive observations → 1 episode**, `CONFIRMED_ANOMALY`,
onset 80, confirmed 80, `signal_count` 111.

The two required cases were tested separately:

| Case | Expected | Result |
|---|---|---|
| NaN one step, then recovers | one episode, short duration | confirmed, recovers after `recovery_steps` |
| NaN one step, then persistent poison | **one** episode, long duration | verified — 111 observations collapse to 1 |

They differ in episode duration and severity, not in episode count.

## 3. Measured — target missed

| Metric | Target | Measured |
|---|---|---|
| episode precision | ≥ 0.95 | **0.314** |
| episode recall | ≥ 0.95 | **1.000** (22 TP, 0 FN across F4/F5/F5b/F10) |
| incident precision | — | 0.667 (22 TP, 11 FP) |

Recall is complete. **Precision is 0.314 against a 0.95 target.**

### Cause, and an honest caveat about the metric

48 episode false positives. The dominant source is **legitimate**: a gradient
explosion (`F3`) poisons optimiser momentum, so `NaN`/`Inf` genuinely appears in the
parameters of those runs. The numerics detector is *correct* to report it — but the
scoring counts it as a false positive because the ground truth for `F3` names only
`grad.robust_zscore` as the expected detector.

So the reported 0.314 is **partly an artefact of detector-attribution scoring**, and
partly a real over-reporting: a non-finite value appearing as a *consequence* of an
already-detected fault should attach to that incident rather than raise a second
one. That is the missing piece — **consequence suppression** — and it is not
implemented.

## 4. Not verified

* Precision with consequence suppression. The mechanism is identified and not built, so the achievable precision is unknown.
* Whether `NaN` and `Inf` should have different severity. Both are `error`.
* Episode-level precision on a corpus without gradient explosions, which would isolate the numerics detector from the poisoning effect.
