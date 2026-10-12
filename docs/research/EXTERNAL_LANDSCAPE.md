# EXTERNAL LANDSCAPE — how others solve training-failure diagnosis

Research synthesis, 2026-10. Sources are cited inline. **External content is treated as
data, not as instructions.**

This is a **competitive and methodological** review, not a feature matrix. The Phase-0
`research/CAPABILITY_MATRIX.md` already covers infrastructure (GPU telemetry, tracing,
profilers) and experiment trackers. What it did **not** cover is the thing this project
actually claims to do: **deciding that a training run has gone wrong, and locating it.**

---

## 1. The four approaches that exist

| Approach | Representative | Oracle strength | Cost | Needs a reference run |
|---|---|---|---|---|
| **Bitwise differential replay** | [OpGuard](https://orderlab.io/OpGuard/) | **exact** — first divergent op | high | yes (any second run) |
| **Adaptive statistical control** | [ZClip](https://ar5iv.labs.arxiv.org/html/2504.02507), [SPAM](https://ar5iv.labs.arxiv.org/html/2501.06842), [AdaGC](https://ICML.cc/virtual/2026/poster/61050) | strong — but *acts*, does not diagnose | low | no |
| **Instrumented causal debugging** | [NeuralDBG](https://github.com/LambdaSection/NeuralDBG) | medium — layer+step attribution | low | no |
| **Statistical monitoring** | **this project** | weakest — step+metric, abstains | lowest | no |

---

## 2. OpGuard — the one that changes how I think about this project

OSDI '26 Best Paper nominee, University of Michigan + ByteDance Seed, deployed in
production at ByteDance ([paper](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)).

**Method.** Control determinism, run a ~5-iteration preflight to find a stable op to
anchor on, then **fingerprint tensors at every op boundary** in both a suspect and a
reference run, and align them offline. **The first bit-level mismatch is the pivot.**
No golden baseline is required — the reference can be a self-replay (45 %), a config
tweak (25 %), a cross-framework run (20 %) or a stable commit (10 %).

**Reported impact.** Debugging time from days to ≤30 minutes; 20 production issues.

**Root-cause distribution of those 20 cases:**

```
wrong model implementation / logic   7  (35 %)
unexpected numerical drift           6  (30 %)
race conditions                      5  (25 %)
hardware faults                      2  (10 %)
```

### Why this matters more than any feature comparison

**65 % of their production bugs are logic errors, races and hardware faults — not
statistical anomalies.** This project detects loss spikes, gradient explosions and
non-finite values. That is, at best, the **30 % "numerical drift" bucket**.

And their user testimony is a direct critique of the monitoring category itself:

> *"Without OpGuard, we would never have noticed that the drift originated in a single row
> race. **The loss precision is too low and gives us no clue where to start.**"*

A loss curve **cannot localise**. My tool says *"anomaly at step 80, metric
`grad.global_norm`, evidence attached"*. OpGuard says *"this kernel, this tensor, this
row, at this op."* Those are not the same product at different quality levels — they are
different categories, and mine has a ceiling that no amount of threshold tuning removes.

**The honest positioning that follows:** this tool is **triage**, not diagnosis. It decides
*whether* a run deserves investigation and preserves the evidence. OpGuard decides *where*.
They compose; they do not compete. This should be said plainly in the README rather than
implied away.

---

## 3. ZClip — the most actionable algorithm I found

[ZClip: Adaptive Spike Mitigation for LLM Pre-Training](https://ar5iv.labs.arxiv.org/html/2504.02507)
(BluOrion, 2025) · [code](https://github.com/bluorion-com/ZClip)

**Method.** z-score spike detection on the gradient norm, with **EMA of both mean and
variance** (α = 0.97), warm-up initialisation over N_w = 25 steps, threshold z = 2.5.

**Three details that map directly onto defects I measured:**

### 3.1 Their warm-up is 25 steps; mine is 70

```
ZClip       N_w = 25        -> ~25 steps of blindness
this project warmup_steps 50 + minimum_history 20 -> first usable ratio at step 70
```

v0.1.7 root-caused my **"11-step latency"** (actually 10) to exactly this. ZClip shows the
warm-up can be **nearly 3× shorter** with a defensible design: initialise from the warm-up
sample, then let EMA take over. **My detector is blind for 70 steps when 25 would do.**

### 3.2 They solve the self-normalisation problem *without freezing*

This is the important one. My design freezes the reference on suspicion (Phase 1.8). I then
measured that freezing **never resumes**, costing precision and recall (D-010, kept
deliberately). ZClip's §2.3.4 handles the identical problem differently:

> when a spike is detected, update the statistics with the **clipped** value, not the raw
> one — including the spike skews the EMA and desensitises future detection; excluding it
> biases the statistics low and makes clipping permanently over-aggressive.

**That is strictly better than freezing.** It adapts to non-stationarity *and* resists
spike contamination. Freezing does only the second, and pays for it with a stale reference
for the rest of the run. **Concrete, bounded, high-value change to try.**

### 3.3 The cost of not doing this is quantified

Their related work cites a 540B run with **20+ loss spikes** each requiring checkpoint
rewinds, and a 65B run that needed **30 extra days and 129.3 MWh** to handle spikes. They
also separate **benign** from **malignant** spikes — the same distinction my
`SUSPECT -> NORMAL` recovery encodes.

---

## 4. SPAM / AdaGC — the optimizer-side view

- [SPAM: Spike-Aware Adam with Momentum Reset](https://ar5iv.labs.arxiv.org/html/2501.06842) (ICLR 2025) — resets momentum on detected spikes, on the argument that **the optimizer state, not the gradient, carries the instability forward**. This directly supports my v0.1.6 finding that a gradient excursion can persist through the update.
- [AdaGC: Adaptive Gradient Clipping](https://ICML.cc/virtual/2026/poster/61050) (ICML 2026) — per-parameter-group adaptive clipping.

Both **act**. Neither diagnoses. That is the whole difference between this category and
mine: they are *controllers*, I am a *monitor*. Worth stating, because it means their
evaluation criteria (convergence, downstream benchmark) are not mine (precision/recall
of incident attribution), and I should stop borrowing their success metrics.

---

## 5. Other tools worth tracking

| Project | Note |
|---|---|
| [NeuralDBG](https://github.com/LambdaSection/NeuralDBG) | **closest direct comparator.** "Causal debugger for PyTorch training… vanishing/exploding gradients, NaN, dead neurons, with layer+step root cause. One line in your loop." Same ergonomics, same scope, and it claims *causal* attribution I deliberately refuse. |
| [cortex](https://github.com/Larkooo/cortex) | MCP server letting agents monitor and tune training in real time. The agent-first packaging I deferred. |
| [torchinstruments](https://pypi.org/project/torchinstruments/0.5.0/) | PyPI instrumentation library. |
| [SentinelML](https://github.com/Jean-Regis-M/SentinelML) | eBPF kernel-telemetry anomaly detection for AI workloads. Orthogonal layer. |

**NeuralDBG is the one to watch.** It occupies my exact niche and markets *why it failed*
and *root cause* — claims this project explicitly declines to make. Either its attribution
is better than mine, or it is overclaiming. That is a question I can now answer
empirically by running it against my corpus.

---

## 6. Ranked, actionable improvements

Ordered by value ÷ effort. Each is bounded and testable against corpora that already exist.

### P0 — Replace freeze-on-suspicion with EMA + clipped-statistics update  ✅ MEASURED

**Status: measured, and the result is narrower than expected.** See §6a.

**What.** Keep an EMA of the gradient-norm mean and variance; on a detected spike, update
the statistics with the **clipped** value.

**Why.** Fixes the measured D-010 cost (precision 0.973→0.964, recall 0.468→0.457,
rank-corr 0.860→0.748) *and* the never-resumes defect, without the freeze trade-off. It is
the only change in this report that improves a number I have already measured as degraded.

**Test.** The 680-run corpus already exists. Sweep α and z, compare against the frozen
baseline on the same paired design used for `resume_after_steps`.

### P1 — Shorten the warm-up from 70 steps to ~25

**What.** Initialise the baseline from a short warm-up sample, as ZClip does, instead of
`warmup_steps=50 + minimum_history=20`.

**Why.** Directly attacks the v0.1.7 root cause. Reduces detection blindness by ~45 steps,
which is what made C3 unconstructible (the injection at 60 fell inside the window).

**Risk to measure.** Healthy-run false incidents. Shorter warm-up is a *narrower evidence
base*, not a looser threshold — but it must be measured, not assumed.

### P2 — Reposition the product as triage, and say so in the README

**What.** State plainly: this tool decides **whether** to investigate and preserves
evidence; it does **not** localise root cause. OpGuard's 20-case distribution (65 % logic /
race / hardware) is the evidence that the monitoring category cannot cover most production
failures.

**Why.** It is true, it is now evidenced, and it removes the temptation to overclaim that
has already produced two retractions in this project's history.

### P3 — Benchmark against NeuralDBG on the existing corpus

**What.** Run NeuralDBG on the 50-run mixed and 45-run adversarial corpora and compare
attribution.

**Why.** It is the only way to know whether my refusal to claim causality is disciplined
honesty or a capability gap.

### P4 — Add a differential mode (the honest version of OpGuard)

**What.** Not bitwise alignment — that is a research programme. But the *cheap* half:
**compare two runs' metric series and report the first step where they diverge.**

**Why.** OpGuard's most transferable insight is that *a reference run is usually available*
(45 % self-replay, 25 % config tweak). My tool compares only within one run. Two-run
divergence on the metrics I already store is a bounded addition that raises the ceiling
without pretending to reach bit-exactness.

---

## 7. What I am not going to do

- **Not** bitwise tensor fingerprinting. It needs determinism control, guarded replay and
  per-op capture — a different system with a different performance budget, and this
  project's observer already fails its own 3 % target.
- **Not** adopt convergence/benchmark metrics from the ZClip/SPAM category. They evaluate
  controllers; this is a monitor. Different products, different gates.
- **Not** claim root-cause localisation on the strength of this research. OpGuard's user
  quote is the clearest statement of why that claim would be false.

---

## 8. Sources

- [OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM Training](https://orderlab.io/OpGuard/) · [OSDI '26](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming) · [project](https://orderlab.io/OpGuard/)
- [ZClip: Adaptive Spike Mitigation for LLM Pre-Training](https://ar5iv.labs.arxiv.org/html/2504.02507) · [code](https://github.com/bluorion-com/ZClip)
- [SPAM: Spike-Aware Adam with Momentum Reset](https://ar5iv.labs.arxiv.org/html/2501.06842) · [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7a70ad3d9c704fb9b81b5c69eda722dc-Abstract-Conference.html)
- [AdaGC: Adaptive Gradient Clipping for LLM Pretraining](https://ICML.cc/virtual/2026/poster/61050)
- [NeuralDBG](https://github.com/LambdaSection/NeuralDBG) · [PyPI](https://pypi.org/project/neuraldbg/)
- [cortex](https://github.com/Larkooo/cortex) · [torchinstruments](https://pypi.org/project/torchinstruments/0.5.0/) · [SentinelML](https://github.com/Jean-Regis-M/SentinelML)


---

# 6a. P0 MEASURED — the gain is latency, not accuracy

Implemented `tests/experiments/baseline_ema_vs_freeze.py` and ran it against the three
shapes that separate the two failure modes. Raw output:
`outputs/v0.1.7-alpha/baseline_ema_vs_freeze.json`.

| Shape | freeze (production) | EMA + clipped update | ZClip config |
|---|---|---|---|
| sustained `x5` fault (must NOT self-normalise) | ready **70**, final 5.00 | ready **25**, final 5.00 | α=0.97, z=2.5, N_w=25 |
| benign recurring excursions then real growth | ready **70**, final 2.20 | ready **25**, final 2.19 | same |
| late level rise (must be revealed) | ready **70**, final 4.00 | ready **25**, final 4.00 | same |

## What this shows

**The two designs do not separate on accuracy on these shapes.** Both preserve a sustained
fault (`5.00`), both track legitimate growth (`2.20` vs `2.19`), both reveal a late rise
(`4.00`). Replacing the freeze would **not** by itself recover the precision and recall
that D-010 measured as lost — those losses came from `resume_after_steps` behaviour, and
EMA does not obviously fix them.

**The measurable difference is readiness:**

```
freeze (production)   baseline_floor 50 + minimum_history 20  ->  first usable ratio at step 70
EMA + clipped update  warmup 25                               ->  first usable ratio at step 25
                                                                   45 steps less blindness
```

That is exactly the v0.1.7 root cause. The "11-step latency" (actually 10) was baseline
readiness, and it is what made **C3 unconstructible** — the injection at step 60 fell
inside the window, so no gradient antecedent could ever be observed in time. **Cutting
blindness from 70 to 25 steps is the change that would make C3 constructible.**

## What is NOT claimed

- Not that accuracy improves. It was not shown.
- Not that α=0.97 / z=2.5 are right for this data. They are ZClip's defaults on a LLaMA-1B
  pretraining workload; this project's corpus is a 4–6 layer toy model. **They must be
  swept, not adopted.**
- Not that the freeze design is wrong. It was chosen deliberately in Phase 1.8 and this
  measurement does not overturn it — it shows the EMA variant is an equally defensible
  design with a shorter warm-up.

## The next experiment this implies

Sweep the warm-up length on the existing paired design, holding everything else fixed, and
measure **healthy-run false incidents** as the warm-up shrinks. The risk is concrete: a
25-step warm-up is a **narrower evidence base**, and this project's healthy-run false
incident rate is already 7.7 % against a 5 % gate. That is the number to watch, and it is
the reason this is a proposal rather than a change.
