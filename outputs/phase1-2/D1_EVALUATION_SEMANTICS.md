# D1 — EVALUATION SEMANTICS

**Status: COMPLETE.** Implemented in `src/training_debugger/evaluation/`.

---

## 1. The defect

Phase 1.1 computed detection delay as `first_fire − expected_effect_start`. A
detector that fired 32 steps *before* the fault existed therefore reported a delay
of **−32**, which reads as *faster than possible* rather than as *wrong*. Four
distinct concepts were collapsed into one number.

## 2. The four moments, separated

| Moment | Known by | Role |
|---|---|---|
| fault injected | harness (ground truth) | what was done, and when |
| expected effect start | harness (ground truth) | when the symptom *could* begin |
| acceptable detection window | harness (ground truth) | the grading region |
| detector fires | detector output | a step, with no notion of truth |

`DetectionWindow` carries `pre_fault_window`, `injection_step`,
`expected_effect_start`, `acceptable_detection_window` and `recovery_window`.
`expected_effect_start` is **never earlier than the injection**, so no fire before
the injection can be graded as a detection.

`effect_lag` exists because a symptom may legitimately follow its cause — the
throughput fault is graded with `effect_lag=1`.

## 3. Classification

`classify_detection(step, window)` returns one of:

| Kind | Condition |
|---|---|
| `true_positive` | inside the acceptable window; **delay = step − expected_effect_start** |
| `pre_injection_false_positive` | before `expected_effect_start` |
| `late_detection` | inside the recovery window (counts for recall, **carries no delay**) |
| `post_fault_false_positive` | after both windows |
| `post_fault_duplicate` | a further fire after the first true positive |

A late detection deliberately reports `delay = None` rather than a large number:
letting it contribute to the delay statistic would let a very late detector flatter
the median.

## 4. Separated metrics (D1.2)

`aggregate()` reports, side by side and never merged:

`pre_injection_false_positives` · `post_fault_false_positives` ·
`healthy_false_positives` · `post_fault_duplicates` · `late_detections` ·
`detection_delay` (**first true positive only**) · `precision` · `recall` ·
`f1` · `false_positives_per_1k_steps` · `runs_with_a_pre_injection_fire` ·
`pre_injection_incident_rate` · `duplicate_rate`.

There is deliberately **no headline single number**. Collapsing these is what
produced the negative delay.

## 5. Earliest-abnormal claims (D1.3)

The production system does not know which fire was a true positive; the evaluation
system does. The calibration report therefore computes
`EARLIEST_EVIDENCE_PRECISION` — of the incidents the system opened, how many were
graded true positives — and `PRE_INJECTION_INCIDENT_RATE`, whose target is **0**.
Any fault run whose first incident-opening fire precedes the injection is counted
as a failure even if the fault is subsequently detected, because the
earliest-abnormal claim would name the wrong step.

## 6. Verified

* A fire at step 43 for a fault injected at 80 is classified
  `pre_injection_false_positive` with **`delay = None`**, not −37.
* Delay is taken from the first true positive only; later duplicates never
  influence it.
* A late detection contributes to recall but not to delay.
* A healthy control classifies every fire as a false positive.
* A crashed run is excluded from scoring entirely rather than counted as a miss —
  the defect that made F10 look like a detector failure.
