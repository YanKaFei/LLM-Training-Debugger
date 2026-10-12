# E2 — STATE MACHINE REPORT

**Status: COMPLETE.** Implemented in `src/training_debugger/episodes/engine.py`.

---

## 1. States and transitions

```
NORMAL ──first abnormal observation──> SUSPECT
SUSPECT ──confirmation rule fires────> CONFIRMED
SUSPECT ──confirm_window elapses─────> NORMAL          <-- the required transition
CONFIRMED ──recovery_steps elapse────> RECOVERING
RECOVERING ─────────────────────────> CLOSED
```

Terminal outcomes: `CONFIRMED_ANOMALY`, `NOT_ANOMALY`, `INSUFFICIENT_EVIDENCE`.

## 2. Verified transitions

| Behaviour | Test | Result |
|---|---|---|
| persistent `Inf` at 111 consecutive steps | one observation per step | **1 episode**, `CONFIRMED_ANOMALY`, onset 80, confirmed 80, `signal_count` 111 |
| single WEAK gradient excursion, never corroborated | one observation | **`NOT_ANOMALY`** — `SUSPECT → NORMAL` |
| multi-layer persistent gradient anomaly | aggregate + ≥40 % layers | `CONFIRMED` |
| recovery | observations stop, `recovery_steps` elapse | `RECOVERING` → `CLOSED` |

## 3. Onset vs confirmation

The two moments are separate fields and are **not** the same number in general:

```
candidate_onset_step = 80     (first abnormal observation)
confirmed_at_step    = 82     (persistence satisfied: 2 observations within 5 steps)
```

A `CERTAIN` (non-finite) observation confirms at onset, because nothing can make a
measurement of `NaN` false and delaying it would be theatre. A statistical
observation confirms only once corroborated. This is reported as
`confirmation_delay_steps`, and the evaluation layer reports it separately from
onset error because the two answer different questions.

## 4. Why `SUSPECT → NORMAL` is the important transition

It is the mechanism by which the system can say **nothing happened**. Measured on
healthy runs from the final holdout:

```
raw signal fires:  71
episodes opened:   many, all unconfirmed
incidents:         0
```

Under the Phase 1.2 model those 71 fires would have been 71 findings and a
`precision = 0.000` score. The transition turned them into silence, which is the
correct answer, and the four-layer scoring model is what makes that visible instead
of penalising it.

## 5. Rules are versioned

`RULE_VERSION = "phase1.3-1.0"` is stored on every episode, and
`fusion_rule_registry.json` records every promotion rule with its id and version.
An episode produced under an older rule set can therefore still be explained.

## 6. Not verified

* **`gap_tolerance` (8 for gradient, 15 for divergence) was not swept.** It was chosen from the observation cadence, not measured.
* **`confirm_window` and `recovery_steps` were not swept.**
* **Multi-rank episodes are untested** — every run is single-rank, so the `rank` dimension of episode identity is unexercised.
