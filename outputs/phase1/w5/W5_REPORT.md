# W5 REPORT — Deterministic Detectors

**Wave:** W5
**Status:** **COMPLETE**
**Gate:** A0–A4 corpus with declared detections *and* non-detections — **PASS** (31 tests)

---

## 1. What was built

```
src/training_debugger/detectors/
  base.py        Anomaly, DetectorConfig, DetectorContext, robust statistics
  loss.py        LossSpikeDetector, LossDivergenceDetector
  gradient.py    GradientExplosionDetector, FinitenessDetector
  throughput.py  ThroughputRegressionDetector, StepTimeRegressionDetector
  runner.py      DetectorRunner, multiplicity, suppression, detector-error reporting
tests/scenarios/corpus.py       A0–A4 as declarative data
tests/scenarios/harness.py      one execution path for tests and reports
tests/scenarios/test_scenarios.py  31 tests
```

No LLM, no learned model, no agent logic. Every finding carries its detector id,
version, baseline, threshold, observed value, window and confidence.

---

## 2. Corpus results

| Scenario | Fired | Onset | Expected | Verdict |
|---|---|---|---|---|
| `A0_healthy` | *(none)* | — | nothing | **PASS** |
| `A1_loss_spike` | `grad.robust_zscore`, `loss.robust_zscore` | 210, 220 | both, grad first | **PASS** |
| `A1b_loss_spike_without_gradient_precursor` | `loss.robust_zscore` | 220 | loss only | **PASS** |
| `A2_gradient_explosion` | `grad.robust_zscore` | 150 | gradient | **PASS** |
| `A3_nan` | `numerics.finiteness` | 180 | finiteness | **PASS** |
| `A3b_infinite_gradient` | `numerics.finiteness` | 170 | finiteness | **PASS** |
| `A4_throughput_regression` | `throughput.regression` | 160 | throughput | **PASS** |

**A1 produces a 10-step lead**: the gradient anomaly at 210 is reported before the
loss spike at 220. That gap is the entire product claim, and it is measured rather
than asserted.

Each scenario also declares which detectors must **not** fire. A detector suite
reaches 100 % recall by firing always, so the corpus pins precision too.

---

## 3. Three real detector defects found and fixed

### 3.1 An exponential loss decay made the spike detector blind

The first implementation used a trailing median/MAD z-score. On the A1 corpus it
scored the injected 4.5× jump at **z = 0.93** — no fire at all.

The cause is structural, not a threshold: over a 200-step window the loss decays
from 1.63 to 0.22, so the window's MAD measures *the decay* rather than the noise
around it. **A decaying series cannot be tested with a dispersion estimate that
the trend inflates.**

Fix: detrend the window with a robust local linear fit (anchored on the medians of
the first and last thirds, so it is O(n) and outlier-insensitive), and evaluate the
z-score on residuals. A relative-change criterion (`value >= 2× the window median`)
was added as a second, scale-free test. The same 4.5× jump now scores **z = 8.7**
and fires.

### 3.2 Gradient norms are heavy-tailed, so a z-score alone is the wrong test

With detrending applied, a *real* A1 run still produced false positives at steps 52
and 56. Inspecting the actual series showed why:

```
steps 45..62:  1.40 1.32 1.41 1.39 1.49 1.32 1.85 2.01 1.45 1.19 1.49 2.09 ...
step 80 (injected): 50.21
```

Healthy gradient norms range 1.1–3.3 with unremarkable excursions to 2.8. A
Gaussian-style z-score fires on those. This is precisely why adaptive-clipping
methods exist (Phase 0's survey found ZClip for the same reason).

Fix: the primary criterion is now the one the brief names — **relative change
against the historical baseline** — implemented as `value >= 3× the largest value
in the preceding window`, with the z-score retained as a secondary criterion that
additionally requires 1.5× the window peak, and the absolute guard rail unchanged.
This is also the question a practitioner actually asks: *is this outside anything
we have seen before?*

### 3.3 A detector bug was being swallowed into "no anomaly"

`DetectorRunner._safe_detect` caught detector exceptions and returned an empty
list. A stale variable reference made the loss detector raise on **every** step,
so the corpus reported a clean healthy run **and** a missed spike — the two worst
possible outcomes — with no signal that anything was wrong.

Fix: failures are recorded in `DetectionResult.detector_errors` and asserted on by
the A1 test. Isolation is still correct (one broken detector must not lose the
others' findings); silence was not.

---

## 4. Semantics worth stating

| Property | Implementation |
|---|---|
| The point under test is excluded from its own baseline | window is `values[start:index]`, never including `index` |
| Warm-up is mandatory | no fire before `warmup_steps` and `min_samples` |
| Zero dispersion does not fire | `mad == 0` yields no z-score rather than an infinite one |
| Repeated firings coalesce | per-key cooldown, so a sustained anomaly is one incident |
| A more specific finding supersedes a general one | an OBSERVED finiteness finding suppresses an inferred statistical one at the same step |
| Onset is first-abnormality, not detection time | the sustained-shift detector reports the **first** breaching step, not the step where the run became conclusive |
| Multiplicity is always computed | `affected/population` for layers, ranks and steps |
| Throughput reports the symptom only | `metadata.root_cause = None`, `scope = "symptom_only"` — Phase 1 does not attribute a slowdown to GPU, storage or network |

---

## 5. Not verified in W5

* **Calibration on real incidents.** Thresholds are conservative defaults, not fitted. Phase 1 has no labelled incident corpus; the false-positive claim rests on the A0 scenario and one real A1 run, not on a large sample.
* **A0 is 400 steps.** A longer healthy run (100k steps) would exercise threshold drift far better; the W4 benchmarks run 130–320 steps.
* **Divergence detection is untested on a real diverging run.** A scenario exists in the library but the corpus does not yet include a slow-divergence case with a realistic converged-then-rising curve.
* **Layer-level explosion in A2 is synthetic.** The per-layer attribution path is exercised, but the A2 scenario injects a scaled global factor rather than a genuinely single-layer pathology.
