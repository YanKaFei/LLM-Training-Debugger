# UPGRADE — two-run divergence localization

Driven by `docs/research/EXTERNAL_LANDSCAPE.md`. Source version unchanged
(`v0.1.7-alpha` / `0.1.7a1`); this is a capability addition, not a new release number.

```
tests        555 passing, 1 skipped    (533 -> 555, +22)
smoke        8/8 PASS
quickstart   PASS
package      wheel + sdist built; `compare` verified inside the installed wheel
```

---

## What changed and why

The external review produced one finding that mattered more than any feature list:

> OpGuard (OSDI '26) gets its precision from comparing **two runs** and taking the first
> bit-level mismatch as the pivot. Its user testimony: *"the loss precision is too low and
> gives us no clue where to start."*

This project compared only **within** one run. `compare_runs` already existed but answered
*"did this metric get worse on average?"* — a median-level regression, never a step. **The
CLI did not expose it at all.**

### The new capability

`compare <reference> <suspect>`:

```
align two runs' series  ->  d[t] = b[t] - a[t]
estimate the scale of d from its OWN early region   (the runs agree there)
report the first step where |d| leaves that scale AND stays out
```

**Shared noise cancels in the difference**, which is why this localises where a single-run
threshold cannot. Measured:

```
FIRST DIVERGENCE: step 90  metric grad.global_norm
  A = 0.450342   B = 67.5513   difference = +67.1
  6200.9x the agreement scale, sustained for 61 step(s)
```

### Validation

A new paired corpus with **real events persisted** and a known injection step
(`tests/experiments/paired_divergence_corpus.py`, 12 pairs):

```
fault pairs   9/9 localized to the EXACT injection step (90)
clean pairs   3/3 correctly abstained (NO_DIVERGENCE)
```

### Wired into diagnosis

```bash
python -m training_debugger.cli.main --root <root> diagnose <suspect> --against <reference>
```

Renders the pivot, the per-metric pivots in step order, and the incidents split into
**at-or-after** (candidates for what changed) and **before** (which the pivot cannot
explain, because they came first). The renderer states in the output that this is an
ordering, not a cause.

---

## The finding I did not expect: the mechanism chain emerges by itself

Wiring the pivot into `diagnose --against` produced something better than I intended.
Real output, one pair:

```
  per-metric pivots
    step 90  grad.global_norm         +67.1
    step 90  grad.layer_norm          +64.2
    step 90  grad.layer_norm_max      +64.2
    step 90  grad.layer_norm_mean     +14.13
    step 91  train.loss               +0.7754      <- loss follows one step later
    step 92  grad.layer_norm_argmax_layer -1
    step 93  param.global_norm        +2.773       <- parameters follow two steps later
    step 95  param.layer_norm         +2.47
```

**That is gradient -> loss -> parameters, in real step offsets, measured rather than
assumed.** v0.1.4 tried to establish this chain with a `PROPAGATION_DELAY_STEPS = 2`
inference and a threshold gate; v0.1.6 then measured that the inference was keyed on a
float32 reduction artifact.

The differential version needs **no window assumption at all**. The offsets are whatever
they turn out to be, and shared noise cancels, so the ordering is evidence rather than a
rule. This is a stronger footing for the mechanism claim than anything the single-run
detector produced, and it was not the reason I built it.

**It is still not causation.** It says the gradient moved first; it does not say the loss
moved *because* the gradient did. A third run with a different fault would be needed to
test that, and the renderer says so.

## Two defects found by running it, not by reasoning

### 1. A zero-variance agreement region disabled the comparison

Two runs that agree **exactly** in the early region produce `d[t] = 0` throughout, so the
MAD is `0` and an early guard skipped the metric entirely — when identical-up-to-here is
the **strongest** possible divergence signal. Fixed with a floor **relative to the metric's
own level** (agreement to within 1 % counts as agreement), so metrics of very different
magnitudes are treated consistently.

### 2. Timing metrics were reported as algorithmic divergence

On real data it reported `train.step_time` at **step 58** when the fault was injected at
**step 90**. Wall-clock timings differ from the first measurement — machine load, allocator
state, thread scheduling.

> **A timing difference is not an algorithmic divergence.**

This is the same class of error as *"a non-finite norm is not a non-finite tensor"*, which
already cost this project one retraction. A determinism classification now excludes
`train.step_time`, `train.optimizer_time_ms`, `train.samples_per_sec`,
`train.tokens_per_sec` and `mem.*` by default; `--include-nondeterministic` opts in and the
report says which metrics were excluded and why.

### 3. A reproducibility defect in every existing corpus

No corpus wrote raw `TrainingEvent`s to Parquet — only episodes and incidents. Nothing
event-level was reproducible from disk. The new paired corpus writes events, which is what
made defect 2 visible at all.

---

## Honesty boundaries, stated in the product

- The reason string ends: **"a pivot for investigation, NOT a root cause: it names a step
  and a metric, not a kernel or a tensor."**
- Abstention is first-class: too few common steps → `INSUFFICIENT_EVIDENCE`; agreement
  throughout → `NO_DIVERGENCE`.
- No interpolation. Only steps present in both runs are used; an interpolated value would
  be a fabricated observation.
- A single-sample blip is not a divergence — persistence is required.
- No capability is upgraded to `VALIDATED`. `validated_capabilities = []` unchanged.

## Status

```
EXPERIMENTAL — measured on one synthetic paired corpus (4-layer, 3 seeds, 12 pairs).
No blind holdout. Untested above 6 layers. Names a step and a metric, not a cause.
```

## What this does and does not fix

**Does:** turns "an anomaly happened somewhere" into "these two runs stopped agreeing at
step 90 on `grad.global_norm`", with zero false pivots on the clean controls.

**Does not:** reach OpGuard's precision. Their pivot is a *kernel*, theirs covers logic
errors, races and hardware faults (65 % of their 20 production cases) that no metric-level
method can see. This narrows the gap; it does not close it, and the README now says so.
