# Per-module activation health — the gap a competitor exposed

NeuralDBG classifies `Vanishing` / `Exploding` / `Dead` / `Saturated` **per named
module**. This project observed gradient norms per layer and **nothing about
activations**. That was a real gap, found by reading a competitor rather than by
guessing, and it is observational — it needs no causal claim.

---

## 1. Design rule: only threshold-free states

This repository has retracted twice for inventing a threshold. So the health states are
split by what can be established **exactly**:

```
EXACT — no threshold, no window, no reference
    DEAD          every element is exactly zero        (zero_fraction >= 1.0)
    NONFINITE     an element is NaN or infinite        (nonfinite_count > 0)
    SATURATED     |x| reaches the dtype's own maximum  (abs_max >= float32 max)

NOT PRODUCED
    vanishing / exploding — these need a notion of the right size for a module, and
    this project does not have one. The magnitudes are emitted as statistics
    (act.abs_mean, act.global_norm) and left to the reader.
```

A competitor reports *"activation_regime_shift cascaded through [module list]"* at 0.72
confidence. This reports *"blocks.2.mlp.2: all 4096 elements are exactly zero at step 1"*,
which is a fact.

## 2. Validated on real training

A module's output projection was zeroed and frozen. Real forward passes, 6 steps:

```
module              step   zero_frac   abs_max    state
blocks.0.mlp.2      1      0.000       0.03667    healthy
blocks.2.mlp.2      1      1.000       0          dead
blocks.2.mlp.2      3      1.000       0          dead

detector: activation.health   step=1   kind=dead   level=OBSERVED   conf=1.0
```

The dead module is named, dated, and reported at **OBSERVED confidence 1.0**. Its healthy
sibling is unaffected.

## 3. Sampling can only weaken a claim, never strengthen it

The probe can bound its cost with a strided sample. That changes what may be concluded:

```
nonfinite_count > 0           -> still NONFINITE      (finding one element is proof)
zero_fraction >= 1.0, sampled -> UNKNOWN, not DEAD    (a sample of zeros proves nothing
                                                       about the whole tensor)
```

A sampled observation cannot establish DEAD by absence. This is the same discipline as
`*.global_norm == inf` not implying a non-finite tensor.

## 4. The cost, and the mistake in how it was emitted

Measured, paired and interleaved:

```
                             total     delta     act events
probe off                    0.3148 s  baseline       0
on, emit every step          0.3927 s  +24.7 %     1600
on, edge-triggered (default) 0.3252 s   +3.3 %       20
```

**+24.7 % looked like the price of reading activations. It was not.** Isolating the
layers:

```
no hooks            0.3080 s
hook that counts    0.3144 s   +2.1 %    the hook is cheap
hook that stores    0.3171 s   +3.0 %    holding the reference is cheap
compute, emit=False 0.3196 s   +2.7 %    the statistics are cheap
sampled vs full     0.4046 vs 0.4071     sampling saved nothing
```

**The computation costs +2.7 %. Emitting five metrics per module per step cost a further
+23.5 %** — 1,600 events over an 80-step run with four modules. Sampling could not help
because the cost was never in the reduction.

Edge-triggered emission — emit only when a module's health state **changes** — took
1,600 events to 20 and the overhead to **+3.3 %**. A module healthy for a thousand steps
does not need a thousand events saying so.

## 5. What this honestly adds and does not

**Adds:** per-module visibility into activations, which the project had none of. A dead
output projection, a saturated layer, a non-finite activation — each is now named, dated
and corroborated at confidence 1.0.

**Does not add:** any causal claim. `activation.health` reports a state, not a
consequence, and `vanishing`/`exploding` are deliberately absent.

**Cost position:** +3.3 % is **on top of a baseline that already misses its own <3 %
target**. That pre-existing failure is not fixed or excused by this, and the probe remains
**off by default**.

## 6. Status

```
tests  584 passing, 1 skipped   (570 -> 584, +14)
smoke  8/8 · quickstart PASS · wheel builds · 0 local paths in the sdist

ACTIVATION_PROBE          : IMPLEMENTED, default OFF
ACTIVATION_HEALTH_DETECTOR: REGISTERED, enabled (fires only on exact findings)
VALIDATED_CAPABILITIES    : []  (unchanged)
```
