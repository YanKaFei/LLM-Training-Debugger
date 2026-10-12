# AGENT READINESS

# **NOT_READY**

`NARRATION_AGENT_READY = NO`.

Phase 1.1 set five conditions. Three are now met, two fail by a wide margin, and one
of the failures got **worse** after Phase 1.2's repairs — because the repairs
revealed how much of the previous result was measurement error.

---

## Decision

| Criterion | Target | Measured (holdout) | Verdict |
|---|---|---|---|
| System precision | >= 0.90 | **0.121** | **FAIL** |
| System recall | >= 0.85 | 0.933 | pass |
| Major detector precision | >= 0.80 | `loss.robust_zscore` **0.700**; `grad` 0.063; `trend_slope` 0.000; `throughput` 0.100 | **FAIL** |
| `numerics.finiteness` precision | >= 0.95 | **0.478** | **FAIL** |
| FP / 1k steps | <= 0.5 | **17.0** | **FAIL** |
| `EARLIEST_EVIDENCE_PRECISION` | >= 0.90 | **0.338** | **FAIL** |
| Pre-injection incidents | 0 | **56 of 117** | **FAIL** |
| Incident persistence + provenance | required | append-only, revisions, cross-process reconstruction | pass |
| Negative evidence | required | mandatory, validated, persisted | pass |
| Correlation separated from causation | required | closed phrasing set, guarded by test | pass |

---

## Why this is a worse answer than Phase 1.1, and why that is correct

Phase 1.1 reported system precision **0.632**. Phase 1.2 reports **0.121**.

The detectors did not get three times worse. Three measurement defects were removed,
each of which had been inflating the number:

1. **A crashed run was scored as a detector miss** (D0/D4). Six F10 runs never executed; a false negative is *cheaper* for precision than a false positive.
2. **Detection delay was computed from the raw first fire**, so pre-injection false positives were reported as early detections instead of as errors (D1).
3. **The finiteness detector raised one finding per step** of a persistent non-finite condition — 810 post-fault findings, 94 % of the corpus's false positives (D8).

Only the third is a detector defect. The first two were the harness reporting
favourable numbers. **The honest precision is 0.121, and the previous 0.632 was
partly an artefact.** Fixing the finiteness cooldown alone moved precision from
0.037 to 0.121 while recall held at 0.933.

---

## What blocks an agent, specifically

**The system alarms on healthy runs.** Per-class holdout results:

| Healthy class | Runs that opened an incident |
|---|---|
| `H2_noisy_healthy_loss` | **5 / 5** |
| `H3_scheduler_transition` | 4 / 5 |
| `H1_heavy_tail_gradients` | 3 / 5 |
| `A1_single_huge_harmless_batch` | 3 / 5 |
| `H4_gradient_clipping` | 2 / 5 |
| `F7_lr_jump_no_failure` | 4 / 5 |
| `H0_stable_healthy` | 1 / 5 |

A narration Agent handed this would write a fluent, well-cited diagnosis of an
optimisation failure in runs that had none — and because the incident objects carry
real evidence (real windows, real baselines, real negative evidence), the output
would be *more* persuasive than a bare alarm. That is the exact failure mode the
freeze exists to prevent.

**`EARLIEST_EVIDENCE_PRECISION = 0.338` is disqualifying on its own.** The product's
central claim is *"the earliest defensible abnormal event"*. In two of every three
runs the system names a step that is not fault-related. `KNOWN_FAILURE_MODES.md`
documents three detectors that are outright broken
(`loss.trend_slope` recall 0.000, `throughput.regression` recall 0.250 with 9
pre-injection false positives, `grad.robust_zscore` precision 0.063).

**Abstention does not exist.** There is no `INSUFFICIENT_EVIDENCE` output, so the
system cannot decline to diagnose. An agent needs that path before it can be trusted
to stay silent.

---

## What is ready

Unchanged from Phase 1.1, and still the hardest parts to retrofit:

* **Persistence with provenance** — incidents survive a process restart and render identically; every evidence row names its detector, version, rule version and source events.
* **Negative evidence as a mandatory bucket** — the agent would have something true to say about what did *not* change.
* **Causation fenced off** — a closed phrasing set and a tested guard; an agent inheriting these objects cannot upgrade a correlation into a cause.
* **`KNOWN_FAILURE_MODES.md`** — the constraint set an agent must be given, listing each detector's blind spots including the three broken ones.

The architecture is right. The numbers are not.

---

## The correct next use of an LLM, when it comes

Not diagnosis. **Narration of an already-computed incident**: render a structured
incident, have the agent explain it, and forbid any fact not present in the object.
That is bounded and testable.

It becomes permissible when system precision >= 0.90, every major detector >= 0.80,
`EARLIEST_EVIDENCE_PRECISION` >= 0.90, pre-injection incidents are 0, and
abstention exists.
