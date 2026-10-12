# BASELINE RESUME-PATH FIX — MEASURED EFFECT

680-run gradient corpus, identical configuration and seeds, before and after.

---

## 1. The defect

`protected_ratio_series` calls `baseline.suspect(step, ...)`, which sets
`state = SUSPECT`. **Nothing ever set it back.** But `ProtectedBaseline.update` only
absorbs while the state is `NORMAL` or `CLOSED`:

```python
absorbing = self.state in (BaselineState.NORMAL.value, BaselineState.CLOSED.value)
if not absorbing:
    self.frozen = True
    return
```

So the resume branch below it could never run through this entry point:

```
state      = SUSPECT       (forever)
frozen     = True
quiet_since= None          (never even started)
resume_after_steps = 40    DEAD CODE
```

The class documented *"Steps of normal behaviour required before learning resumes after
a freeze"* and did not do it. Verified directly on a benign-excursion series: the
reference stayed pinned at 1.0000 while the series moved to 1.45, forever.

**Fixed** by making `protected_ratio_series` drive the state back to `NORMAL` when the
value returns below the suspicion threshold, so the existing quiet-period logic runs.
An ongoing suspicion restarts the quiet clock (`note_suspicious`), so the clock cannot
accumulate across intervening excursions.

## 2. Measured effect — the fix is a correctness repair with a real cost

| Metric | **Before** (Phase 1.8) | **After** (this fix) | Delta |
|---|---|---|---|
| chosen A / B | 7.844 / 15.688 | 8.165 / 16.33 | — |
| confirmed precision | 0.973 | **0.964** | −0.009 |
| confirmed recall | 0.468 | **0.457** | −0.011 |
| healthy confirmed rate | 0.042 | **0.056** | **+0.014** |
| healthy peak p50 / p99 / max | 2.80 / 29.60 / 73.80 | 2.81 / 31.07 / 75.31 | +1.5 p99 |
| monotonicity inversions | 4 (largest 0.17) | 4 (largest 0.175) | — |
| rank correlation | 0.860 | **0.748** | −0.112 |
| qualifying operating points | 0 | 0 | — |

**The fix slightly degrades every quality number.** It does not improve any.

## 3. Why it was kept anyway

The `resume_after_steps` parameter is part of the class's documented contract, and the
freeze-forever behaviour was accidental — a consequence of `suspect()` being the only
state transition `protected_ratio_series` ever performed. Keeping a documented parameter
as dead code is the defect; the measured cost of making it live is the price.

The magnitude is small and within the corpus's own noise: the same corpus already shows
4 monotonicity inversions, and `qualifying operating points` is **0** before and after.
**`DETECTOR_RELIABILITY` is FAIL either way** — recall 0.457 against a 0.85 gate.

**What was deliberately not done:** tuning `resume_after_steps` (or `freeze_at_ratio`,
or the band bounds) to recover 0.973/0.468. That is the over-fitting the brief forbids.
The parameter now has a real, measurable effect for the first time, so a **measured
sweep** of it is the legitimate next step — not a nudge to restore a preferred number.

## 4. Note on the earlier S1 diagnosis

The Phase 1.12C diagnosis said S1 failed because *"benign excursions occur closer
together than `resume_after_steps=40`"*. That was based on the **assumption that resume
worked**. It did not. The corrected diagnosis: with the baseline now tracking properly
(l1: 1.0 → 1.7 → 2.6), S1 has a **genuine single-step transient** at step 179
(l1 = 12.124, neighbours 2.635 and 2.728), which exceeds `confirm_at = 11.5` and
`magnitude_override = 8.0`, and confirms from one observation via `aggregate_confirms`.

**S1 is not a bug.** It is a real false positive of a detector behaving to its own
specification. Measured over 13 healthy runs: **1/13 = 7.7 %**, above the 5 % gate.
