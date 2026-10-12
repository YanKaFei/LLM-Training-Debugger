# AGENT READINESS

# **NOT_READY**

No LLM debug agent may be built on this pipeline yet. The Phase 1.1 freeze is
upheld, and this document is the evidence for why.

---

## Decision

| Criterion | State | Verdict |
|---|---|---|
| Detector precision | **0.632 system; 0.107 for the gradient detector** | **FAIL** |
| Detector recall | 0.857 system; `F10` at **0/6** | **FAIL** |
| Incident evidence completeness | four buckets, `not_correlated` mandatory and validated | pass |
| Negative evidence | mandatory, enforced by `Incident.validate`, persisted | pass |
| Persistence | append-only, revisions, provenance, cross-process reconstruction verified | pass |
| Correlation quality | lag distributions, recurrence, BH valid; small-n only | partial |
| Known unknowns | enumerated below and in every report | pass |

**Two hard failures, one of them severe.** The infrastructure is ready; the
detectors are not.

---

## Why precision 0.632 blocks an agent

An agent's output is prose, and prose reads as confident regardless of the
underlying number. Handing a language model a detector set with **0.107 precision
on its primary signal** would convert a 90 %-wrong alarm rate into a fluent
paragraph naming a cause. That is the single failure this project exists to
prevent, and it is exactly what Phase 0's top risk describes.

The specific blocker is not the aggregate: it is that **`grad.robust_zscore` fires
on 6 of 6 clean healthy runs**. An agent built now would explain a gradient
explosion on runs that had none, and would do so in well-formed sentences with
citations to real evidence.

## Why recall 0.857 blocks an agent

`F10_mixed_anomaly` is 0/6 — a NaN loss *and* a gradient explosion produces no
detector fire. An agent asked to diagnose such a run would be diagnosing silence.
Combined with `loss.trend_slope` at 0.167 recall, the system has whole fault
classes it cannot see.

## What would change the answer

In order, all of them measurable:

1. **Fix `grad.robust_zscore` precision.** It must stop firing on `H0_healthy`. The Phase 1 `A0` scenario passed because its gradient noise is milder than a real run's; the calibration corpus is the correct test and the detector fails it.
2. **Explain `F10` 0/6.** A real interaction bug, and the highest-value single investigation in the project.
3. **Re-derive the negative detection delay.** Onset before injection means the reported onset is the first *false* fire, which corrupts the earliest-abnormal-event claim that is the product's whole point.
4. **Re-run calibration after each change on the calibration split only**, then confirm on the holdout. The split exists for this.
5. **Target: system precision ≥ 0.9 and per-detector precision ≥ 0.8** before an agent is considered.

## What is already sufficient for an agent

Three things, and they are the three hardest to retrofit:

* **Persistence with provenance.** Incidents survive a process restart and render identically, and every evidence row names its detector, version, rule version and source events. An agent can cite.
* **Negative evidence as a first-class, mandatory bucket.** The agent has something true to say about what did *not* change, which is what separates a diagnosis from a guess.
* **Correlation statistics separated from causation.** The vocabulary is closed and guarded; an agent inheriting these objects cannot accidentally upgrade a correlation into a cause.
* **Deterministic rendering already exists.** The incident renderer is a pure function, so an agent would be explaining a fixed artifact rather than generating the finding. That containment is the right architecture, and it is already built.

## The correct next use of an LLM

Not diagnosis. The defensible use, once precision is fixed, is **explanation of an
already-computed incident**: render structured incident → agent narrates it,
forbidden from introducing a fact not present in the object. That is a bounded,
testable task. Diagnosis of raw telemetry is not, at this precision.
