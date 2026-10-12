# T1 layer-hook cadence — and a correction to my own audit

The audit concluded *"the largest single remaining lever is event volume."* **Measured, it
is not.** This records the change, the saving that did not arrive, and the more important
thing the experiment found instead.

```
tests  584 passing, 1 skipped   (unchanged)
```

---

## 1. The change

Event volume on a 4-layer model, per step:

```
grad.layer_norm        8.00   <- feeds grad.robust_zscore (layer_series)
param.layer_norm       8.00   <- consumed by NOTHING
None (step_start/end)  2.03
aggregates (max/mean/p99) 3.00
single-value metrics   7.00
                       ----
                       32.0
```

`param.layer.norm` was **25 % of all event volume and had no consumer**. Downstream it
appears only in the presentation sink's `FORBIDDEN_PREFIXES`.

`LayerMetricCollector` now takes `observability_cadence` (default **5**) governing layer
metrics that no detector reads, separate from `cadence`, which governs the metrics the
detection path reads. Emit at the rate the consumer needs.

```
cadence=1   32.0 events/step   param.layer_norm 8.00/step
cadence=5   25.6 events/step   param.layer_norm 1.60/step   (-20 %)
grad.layer_norm           8.00/step in both  <- detection path untouched
```

## 2. The saving did not arrive

Paired, 7 repeats, arms alternating inside each repeat:

```
cadence  median     events/step   emit us/ev   added
1        4.3930 ms  38.4          20.93        baseline
5        4.3599 ms  30.7          21.55        -0.033 ms
```

```
events removed        7.7 per step
predicted saving      7.7 x 20.9 us = 0.161 ms/step
measured saving       0.033 ms/step          <- 20 % of the prediction
```

**Event volume is not the linear cost driver the per-event micro-cost implies.** The
audit multiplied `events/step` by `us/event` and called the product the overhead; the
measured whole-step effect is five times smaller. That is the same error the cost ledger
made with its negative residual -- *micro-cost x count does not equal in-situ cost* -- and
I made it again in the audit's own conclusions.

**Correction: `D-042` says event volume is the largest remaining lever. It is not
established.** The per-event cost, or something that overlaps with compute, dominates.

## 3. What the experiment found instead — and it matters more

The quality harness resampled the **detector's own input** (`grad.layer_norm`) to test the
downsampling idea. Against the paired corpus with known injection steps:

```
cadence=1   6/9 gradient fires, every one at step 90, 9 incidents
cadence=5   6/9 gradient fires, every one at step 90, 3 incidents
```

**Detection survives. Confirmation does not.**

* The detector still fires at the exact injection step -- latency is unchanged, so the
  first-crossing evidence is intact.
* Incidents collapse from 9 to 3: the episode engine cannot reach `CONFIRMED` from a series
  five times sparser, so the finding never becomes an incident.

**Density is required for confirmation, not for detection.** Those are different
requirements on the same series, and only one of them shows up in a detector-level test.

This is why `grad.layer_norm` stays at full cadence and only the unconsumed
`param.layer_norm` is downsampled. Had the change been driven by the cost model alone --
"fewer events, less overhead" -- it would have quietly cost two thirds of the incidents,
and a detector-level check would not have caught it.

## 4. What this leaves

```
SAFE AND SHIPPED      observability_cadence=5 for param.layer_norm   (-20 % events)
MEASURED SAVING       0.033 ms/step -- within noise, ~1 %
NOT ESTABLISHED       that event volume is the dominant remaining lever
ESTABLISHED           downsampling the detector's input costs 2/3 of incidents
                      while leaving detection latency untouched
```

The honest position: this change is safe, it removes genuinely unconsumed work, and it
does **not** measurably improve performance. It should not be reported as a performance
win. The next move on overhead is a re-measurement of where the time actually goes, not
another volume reduction.
