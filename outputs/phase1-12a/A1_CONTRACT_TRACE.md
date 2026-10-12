# A1 — CONTRACT TRACE

**Status: COMPLETE.** The defect was located in the source, not inferred from config.

---

## 1. What the cooldown actually suppresses

`BaseDetector._in_cooldown` / `_mark_fired` (`detectors/base.py:424`):

```python
def _in_cooldown(self, key, step):
    previous = self._last_fired_step.get(key)
    if previous is None:
        return False
    return (step - previous) < self.config.cooldown_steps
```

And in `detectors/loss.py`:

```python
if self._in_cooldown(key, step):
    continue            # <-- skips BEFORE the Anomaly is constructed
self._mark_fired(key, step)
```

**Answer to the brief's four options: (2) detector anomaly emission.**

The `continue` executes *before* the `Anomaly` object exists. No `Anomaly` means
`DetectorRunner` never sees it, which means `EpisodeEngine` never receives it. The
observation is not "sampled out", not "held for later", and not "recorded but not
alerted" — **it is discarded at the detector boundary**.

## 2. Full trace

```
TrainingEvent  train.loss = 20.0 at step 90
      |
      v
DetectorRunner  ->  LossSpikeDetector.detect()
      |                 qualifying observation?  YES  (20.0 >= 2.0 x window median)
      v
candidate       ->  _in_cooldown("loss.robust_zscore:rank?", 90)
      |                 first fire -> False  ->  EMITTED, _mark_fired(90)
      v
TrainingEvent  train.loss = 20.0 at step 91
      |
      v
DetectorRunner  ->  LossSpikeDetector.detect()
      |                 qualifying observation?  YES
      v
cooldown        ->  _in_cooldown(..., 91) -> (91 - 90) < 50 -> True
      |                 -> `continue` -> SUPPRESSED, no Anomaly constructed
      |                 reason: cooldown_steps = 50
      v
EpisodeEngine   ->  receives ONE observation
      |
      v
episode state   ->  SUSPECT; LOSS_SPIKE rule requires 2 within 6 steps -> NOT SATISFIED
      |                 confirm_window elapses -> closes as NOT_ANOMALY
      v
Fusion          ->  no confirmed episode -> verdict NO_ANOMALY
      |
      v
persisted       ->  NO_ANOMALY   (CANDIDATE_ONLY unreachable)
```

Measured on a sustained 20-step spike:

```
cooldown_suppresses_evidence = True   ->  1 observation  (steps [90])
cooldown_suppresses_evidence = False  -> 20 observations (steps [90..109])
```

## 3. The design error

Two concepts had been conflated:

| Concept | Correct owner |
|---|---|
| **evidence observation** — every qualifying observation | detector → episode |
| **alert deduplication** — one notification per event, not per observation | episode grouping + fusion |

Deduplication already exists downstream and is *better* there: `EpisodeEngine` groups
contiguous observations into **one** episode, and `fuse()` returns **one** verdict per
anomaly type. Suppressing at the detector therefore removed evidence the episode needed
while adding no deduplication the downstream layers lacked.

This is the same class of defect Phase 1.6 found with the gradient band: two halves of
a rule set, specified independently and never checked against each other.
