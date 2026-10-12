# W4 REPORT — T1 Layer Hooks and the Overhead Gate

**Wave:** W4
**Status:** **PARTIAL** — the overhead budgets are **not met** on the small workloads; T1 meets its budget at production-like step durations. Reported, not hidden.

---

## 1. The verdict, up front

| Workload (steady state) | Baseline step | core metrics | **core budget < 3 %** | + T1 layer hooks | **T1 budget < 8 %** |
|---|---|---|---|---|---|
| tiny (4L, d32) | 4.41 ms | 5.00 ms | **FAIL +13.3 %** | 5.20 ms | **FAIL +17.9 %** |
| medium (8L, d128) | 17.28 ms | 18.14 ms | **FAIL +5.0 %** | 18.22 ms | **PASS +5.4 %** |
| large (12L, d256) | 41.62 ms | 43.01 ms | **FAIL +3.3 %** | 43.00 ms | **PASS +3.3 %** |

**T1 layer hooks are effectively free: the incremental cost over core metrics is 0.00–0.02 ms at every size.**
The budget failures are the *core* observer, not the layer instrumentation.

Absolute steady-state cost at the largest workload: **+1.39 ms per step**.

---

## 2. What was built

```
src/training_debugger/adapters/pytorch/layers.py
  LayerResolver, LayerSelection, LayerNorm, LayerMetricBatch, LayerMetricCollector
tests/integration/test_layer_metrics.py   20 tests
tests/performance/benchmark_overhead.py         headline benchmark
tests/performance/benchmark_overhead_scaling.py scaling measurement
outputs/phase1/w4/OVERHEAD_BENCHMARK.md
outputs/phase1/w4/OVERHEAD_SCALING.md
```

### Design choices that make T1 affordable

| Choice | Reason |
|---|---|
| **No backward hooks.** One `named_parameters()` pass after `backward()` | A `register_full_backward_hook` runs *inside* autograd: an exception there breaks training, and it collides with any user hook. Identical arithmetic, radically smaller blast radius. |
| **Sampling gates computation, not just emission** | The per-parameter norm cost 880 µs and the default policy samples it every 10th step. Computing it on the other nine and discarding the result is pure waste. `TrainingObserver.would_emit()` is a pure predicate the adapter consults first. |
| **Aggregate + top-K + vector hash** | On a 100-layer model this emits ~13 events per step instead of 100. Verified: with `top_k=2` on a 4-block model exactly 2 per-layer events appear. |
| **Explicit layer resolver** | A wrong layer index silently attributes an anomaly to the wrong layer. The pattern is configuration and is tested, not guessed. |
| **Multiplicity denominator on every aggregate** | `population_layers` and `layer_count` are always emitted, so "1 / 16 layers" is expressible and never implied. |

---

## 3. The optimisation work, with measurements

The first measurement **failed badly**: core +80 %, T1 +81 %. Profiling found the causes and each was fixed and re-measured.

| Component | Before | After | Factor | What was wrong |
|---|---|---|---|---|
| `global_gradient_norm` | 826 µs | **169 µs** | 4.9× | `.item()` per tensor → one device sync per parameter |
| `global_parameter_norm` | 880 µs | **178 µs** | 4.9× | same |
| `LayerMetricCollector.compute` | **35 540 µs** | **4 890 µs** | 7.3× | same bug, 149 syncs per collected step — *more than the whole training step* |
| `observer.emit` | 54.8 µs | **36.7 µs** | 1.5× | up to ten `dataclasses.replace` calls, each copying 39 fields |
| `memory_snapshot` | 1.3 µs | 1.3 µs | — | already negligible; measured rather than assumed |

Two fixes carried the weight:

1. **Fused reductions.** `torch._foreach_norm` issues fused kernels for a list of tensors and a single `tolist()` converts the result — one synchronisation instead of one per parameter. At 2.2 ms for ~4 M elements the remaining cost is memory bandwidth: reading the gradients. **That is the floor**; a gradient norm cannot be computed faster than it can be read. The collector's 4.89 ms ≈ 2.44 (grads) + 2.45 (params), confirming it is now at that floor.
2. **Collapsing the `copy_with` chain** into a single `replace`.

A third change is a design fix rather than a micro-optimisation: making `would_emit()` gate computation, so the configured cadence actually saves work.

---

## 4. Why the percentages still exceed the budget

Three honest reasons, in order of importance:

**(a) The budget is a percentage of a step that is ~500× shorter than production.** The benchmark's largest step is 41.6 ms. A production LLM step is seconds. The absolute cost — **0.6 ms (core) to 1.4 ms (with T1) per step** — is the transferable number. The brief specifies percentages, so the percentages are what the verdict is based on, and they fail.

**(b) Part of the cost scales with parameter count, not with step duration.** A gradient norm must read every gradient. That term grows with the model while the step duration grows with model × tokens × depth. A large-but-shallow model would show a worse percentage than a large-and-deep one. This is measured across the three workloads and visible in the report.

**(c) An early measurement was wrong and was corrected.** The first scaling run reported +7.4 % at the largest workload. That number was an artefact: the measurement window started at step 10 while `DEFAULT_POLICY_PHASE1` keeps gradient norms at *full resolution for the first 50 steps*, and the estimator was a median of medians, which produced a physically impossible result (instrumented appearing faster than baseline). The corrected measurement starts at step 60 and uses the minimum of five per-repeat medians. **The corrected number is +3.3 %, not +7.4 %.** The earlier figure is retracted here rather than quietly dropped.

---

## 5. Remaining levers (not pulled)

The brief requires that an exceeded budget be met with optimisation **or** a reduced default sampling rate. Optimisation has been done; three sampling levers remain available and are recorded rather than applied unilaterally, because each trades evidence quality:

| Lever | Saving (est., large workload) | Cost |
|---|---|---|
| `grad.global_norm` cadence 2 → 4 | ~0.6 ms/step | Halves the resolution of the primary gradient signal |
| `DEFAULT_POLICY_PHASE1` full-resolution warm-up 50 → 20 steps | affects first 50 steps only | Weaker guard against an early spike — the exact case the warm-up exists for |
| `param.global_norm` cadence 10 → 50 | ~0.2 ms/step | Parameter norm is the main "not correlated" control; coarse cadence weakens negative evidence |

**Recommendation:** apply the first lever only, and only if a user reports the overhead as material. Halving gradient resolution from every-2-steps to every-4-steps is the cheapest real saving, and the warm-up exists precisely to protect the highest-risk window.

---

## 6. Gate results

| Gate item | Status |
|---|---|
| T1 layer hooks implemented | **PASS** — 20 tests |
| no NNsight, no activation capture, no raw tensors | **PASS** — `layers.py` stores only scalars |
| cardinality control (layer / rank / metric / cadence) | **PASS** — `LayerSelection` + `would_emit` gating |
| per-layer grad norm correctness | **PASS** — matches an independent computation to `rel=1e-5` |
| per-layer param norm correctness | **PASS** — same |
| absent gradient reports `None`, not `0` | **PASS** |
| non-finite layer gradients → NaN + count | **PASS** |
| layer failure does not disturb training | **PASS** |
| overhead measured, warmup + 100+ steps + repetitions + median + p95 | **PASS** — see note |
| observer/core median overhead < 3 % | **FAIL** — 13.3 % / 5.0 % / 3.3 % |
| T1 layer hooks median < 8 % | **FAIL** on tiny (17.9 %), **PASS** on medium (5.4 %) and large (3.3 %) |

Note on the measurement protocol: the headline benchmark uses 20 warm-up + 120 measured steps × 3 repeats; the scaling benchmark (which produces the table in §1) uses 60 warm-up + 60 measured × 5 repeats with a minimum-of-medians estimator. Both report median and p95.

---

## 7. Not verified in W4

* **GPU.** All measurements are CPU/MPS on an Apple M2 Max. On CUDA the norm reductions read device memory and the one remaining sync becomes a device synchronisation, which behaves differently; **the overhead on a GPU is NOT MEASURED**.
* **A production-scale model.** The largest workload is 12 layers at d_model 256. Extrapolating the percentage to a multi-second step is a projection, not a measurement, and is not claimed.
* **Distributed.** Single process. A per-rank overhead figure for 64 ranks is not measured.
* **`torch.compile`.** Not tested with the collector attached.
* **Attribution of the remainder.** At the largest workload the measured total is +1.39 ms; the components I measured account for roughly +0.6 ms (grad norm amortised over its cadence, plus ~10 emissions at 37 µs). **Roughly 0.8 ms is not yet attributed.** That is recorded as an open item rather than explained away.
