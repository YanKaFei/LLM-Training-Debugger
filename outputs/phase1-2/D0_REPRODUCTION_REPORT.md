# D0 — REPRODUCTION REPORT

**Status: all four failures reproduced. One of them was not what it appeared to be.**

The failing state was frozen **before** any code was changed
(`outputs/phase1-2/baseline/FREEZE_MANIFEST.json`): there is no VCS in this
project, so the frozen artefact is the manifest of Python/Torch versions, detector
versions, rule version, every threshold and the seed set.

Reproduction was done by `tests/experiments/trace_failures.py`, which re-runs the
Phase 1.1 scenarios with full step-level tracing and **modifies no detector**.

---

## 1. Frozen state

| Item | Value |
|---|---|
| Python / Torch | 3.9.6 / 2.8.0 |
| Platform | macOS-27.0.1 arm64 |
| `rule_version` | `phase1.1-1.0` |
| Detector versions | all `1.0` |
| Seeds | 1000–1005 |

## 2. Failure 1 — H0 healthy gradient false positives: **REPRODUCED**

Reproduced on 3/3 seeds. The mechanism was then located exactly, not inferred:

```
seed 1000, step 134 (no fault in the run)
  value                2.7690
  window_peak          1.8339     -> peak_ratio 1.5099
  grad_z_min_peak_factor        1.5
  z_score              9.3714     -> above spike_z = 6.0
  => the SECONDARY z-branch fired
```

The v1.0 detector had a primary criterion (`peak_ratio >= 3.0`, which correctly did
not fire) and a secondary z-criterion gated by `peak_ratio >= 1.5`. The healthy run
reached **1.5099** — clearing its own guard by **0.7 %**.

So the false positives were not a threshold that needed nudging. The z-branch was
a second, weaker detector bolted onto the first, and its guard sat inside the
healthy distribution.

## 3. Failure 2 — F10 mixed anomaly 0/6: **NOT A DETECTOR BUG**

This was the highest-priority item in the phase, on the stated reasoning that "the
same NaN in isolation is caught 6/6, so something about the combination suppresses
it". **That reasoning rested on a false premise.**

Tracing the execution path showed the F10 scenario **never ran**:

```
F10 injection:  loss = float("nan")        <- a Python float, not a tensor
record_loss():  float(loss.detach()...)    <- AttributeError: 'float' has no attribute 'detach'
```

The calibration harness caught the exception and recorded `fired_detectors: []`
with a `run_error`. All six F10 seeds carry exactly this:
`AttributeError: 'float' object has no attribute 'detach'`.

**The scoring function then ignored `run_error` and counted the crash as a detector
false negative.** So "F10 is 0/6" was an infrastructure failure reported as a
detector limitation, and Phase 1.1's quoted recall of 0.857 was computed with six
crashed runs counted as misses.

With the injection corrected to a **NaN tensor**, F10 fires
`numerics.finiteness` on **3/3** traced seeds. There was no interaction bug.

Two defects are fixed as a result, and they are separate:

1. The injection produced a non-tensor loss (harness defect).
2. **A crashed run was scored as a detector miss** (evaluation defect) — the more
   serious of the two, because it silently converts an infrastructure failure into
   evidence about detector quality.

## 4. Failure 3 — negative detection delay: **REPRODUCED**

`grad.robust_zscore` reported onset at step 134 with an injection step of 166 in
the Phase 1.1 layout, giving a delay of −32. Reproduced, and explained: the delay
was computed as `first_fire − expected_effect_start` with no check that the fire
came *after* the effect could exist. Fixed in D1 by redefining the metric.

## 5. Failure 4 — performance measurement instability: **REPRODUCED**

The Phase 1.1 ledger produced a **negative** observed overhead at medium
(−0.29 ms) and a residual (−4.26 ms) whose magnitude exceeded the effect. A
negative overhead is physically impossible, so the instrument was disqualified
rather than reported. Repaired by the paired design in D7.

---

## 6. What D0 establishes

Every failure was reproduced from a frozen state before any change, and one of
them turned out to have a different cause than the phase brief assumed. That is
the value of reproducing before repairing: had F10 been "fixed" by patching a
detector, the actual defect — a crashed run being scored as a detector miss —
would have survived into every later measurement.
