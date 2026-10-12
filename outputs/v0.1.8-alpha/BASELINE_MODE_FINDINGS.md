# A — EMA BASELINE: implemented, measured, **and it buys nothing**

Plus a **correction to my own v0.1.7 root-cause analysis**.

---

## 1. What was asked and what was built

The external review (ZClip, arXiv 2504.02507 §2.3.4) suggested replacing freeze-on-suspicion
with an EMA whose statistics are updated with **clipped** observations. It also showed a
25-step warm-up where this project uses 50 + 20 = 70.

Built: `AdaptiveBaseline` / `adaptive_ratio_series` in `detectors/baseline.py`, selectable
via `DetectorConfig.baseline_mode` (`"frozen"` **default**, `"ema"` opt-in). Threaded
through `_layer_ratio_series_protected` and `joint_gradient_features`.

```
READINESS (production configuration)
  frozen : first usable ratio at step 70   (floor 50 + minimum_history 20)
  ema    : first usable ratio at step 26   (warmup 25)
  -> 44 fewer blind steps
```

## 2. Measurement 1 — on the paired corpus, identical

```
PAIRED CORPUS (injection at step 90)
  frozen : fired 6/9 fault runs, median latency 0, clean incidents 0/3
  ema    : fired 6/9 fault runs, median latency 0, clean incidents 0/3
```

Identical, because step 90 is **after both** readiness windows. The corpus cannot separate
them — the difference only appears for a fault *inside* the blind window.

## 3. Measurement 2 — the decisive test, and it refutes my own diagnosis

Injection at **step 60**, inside the frozen window:

```
  baseline_mode=frozen (current)               -> first fire at step 71
  baseline_mode=ema                            -> first fire at step 71    <- NO CHANGE
  ema    + detector warmup=10, min_samples=8   -> first fire at step 60    <- fixed
  frozen + detector warmup=10, min_samples=8   -> first fire at step 60    <- fixed WITHOUT ema
  ema    + detector warmup=5,  min_samples=5   -> first fire at step 60    <- fixed
```

**Changing the baseline mode does nothing. Changing the detector's own warm-up does
everything, in either mode.**

### The correction

v0.1.7 concluded that the "11-step latency" (actually 10) was the **baseline** being unready
at `floor 50 + minimum_history 20 = 70`. That was wrong. The binding gate is the
**gradient detector's own**:

```python
floor = self.config.warmup_steps if self.config.baseline_excludes_warmup else 0
start = max(floor, index - self.config.window)
window_values = values[start:index]
if len(window_values) < self.config.min_samples:
    continue          # cannot fire before step warmup_steps + min_samples
```

`warmup_steps = 50` and `min_samples = 20` give **first possible fire at step 71** — a
number that coincidentally equals the baseline's readiness. **The coincidence is why the
wrong diagnosis looked confirmed.** The baseline is never the bottleneck; reducing its
readiness from 70 to 26 changes no outcome at all.

## 4. Measurement 3 — why the gate must not simply be lowered

The obvious follow-up is "then lower the detector's warm-up". Measured on 6 healthy runs:

```
config                        healthy runs with a gradient fire
warmup=50, min_samples=20     0/6      <- current
warmup=25, min_samples=12     4/6
warmup=10, min_samples=8      6/6      <- every healthy run false-positives
warmup=5,  min_samples=5      6/6
```

**The 70-step blind window is load-bearing.** It is what keeps healthy-run gradient
precision at 0/6 false fires. Shortening it to make C3 constructible destroys that
precision entirely.

Confirmed incidents stayed 0/6 in every configuration — the band gate still blocks
promotion — so the damage is at the *detector* layer, not the incident layer. That is worth
knowing but does not make it acceptable.

## 5. Conclusions

1. **`baseline_mode` stays `"frozen"` by default.** Not because it is better — the two are
   indistinguishable in every configuration measured — but because changing it would churn
   behaviour for no measured benefit.
2. **The EMA implementation is kept**, opt-in, because it is correct, tested and cheap, and
   it removes a real design wart (a freeze that never resumes). It is simply not a win.
3. **v0.1.7's root cause is corrected.** The latency is the detector's warm-up gate, not
   baseline readiness. Both being 70 was a coincidence.
4. **C3's unconstructibility is a consequence of a load-bearing precision gate, not a bug
   to be fixed by tuning.** Making C3 constructible costs 0/6 → 6/6 healthy false fires. The
   honest position is that **C3 is out of reach at the current precision requirement**, and
   should be recorded as such rather than chased.

This is a negative result. It was cheap to obtain and it prevents a change that would have
looked like progress while quietly destroying the precision the project already has.
