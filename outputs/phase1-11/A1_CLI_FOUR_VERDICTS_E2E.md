# A1 — CLI FOUR-VERDICT E2E

**Status: 3 of 4 verdicts demonstrated. GATE NOT MET.**

Phase 1.10 could only show `NO_ANOMALY` from persisted evidence. This phase produced
**three** of the four through the production pipeline with independent-process CLI
readback, and diagnosed precisely why the fourth is unreachable.

---

## 1. Results

| Scenario | Method | Expected | Observed | CLI exit | Matched |
|---|---|---|---|---|---|
| **A** | REAL PYTORCH TRAINING | `CONFIRMED_INCIDENT` | `CONFIRMED_INCIDENT` | 0 | **YES** |
| **B** | EVENT-DRIVEN PIPELINE INTEGRATION | `CANDIDATE_ONLY` | `NO_ANOMALY` | 0 | **NO** |
| **C** | EVENT-DRIVEN PIPELINE INTEGRATION | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | 0 | **YES** |
| **D** | REAL PYTORCH TRAINING | `NO_ANOMALY` | `NO_ANOMALY` | 0 | **YES** |

```
3 / 4 matched          GATE NOT MET (requires 4)
CLI exit code 0 for all four
0 fabricated verdicts
0 missing evidence references
```

Every verdict is whatever `fuse()` returned from episodes the real `EpisodeEngine`
built from anomalies the real `DetectorRunner` produced. **No verdict object was
constructed and saved to manufacture a pass.**

## 2. Per-scenario evidence

**A — `CONFIRMED_INCIDENT` (real training).** A NaN is injected into the loss at step
110. `numerics.finiteness` classifies it `CERTAIN`; `promote.numerics.certain` accepts
it alone.

```
episodes 1   verdict NAN_DETECTED CONFIRMED_INCIDENT onset=110
             rule=promote.numerics.certain
CLI stdout   "# source: persisted episode store" / "# run_id: a1-confirmed" /
             "CONFIRMED INCIDENT"
```

**C — `INSUFFICIENT_EVIDENCE`.** Per-layer ratios driven so the joint aggregate lands
at 10.0, inside the `UNCERTAIN` band (8.5 – 11.5). The band refuses to confirm by
design, so the episode closes with the `INSUFFICIENT_EVIDENCE` outcome and fusion
returns the matching verdict. This is the abstention the band exists to express,
reaching a user through a separate CLI process.

**D — `NO_ANOMALY` (real training).** A healthy run: the gradient anomaly is
dismissed, and the renderer states explicitly that this is a *positive statement that
nothing was found, not an abstention*.

## 3. Why B failed, and the defect behind it

B was meant to demonstrate `CANDIDATE_ONLY` via `LOSS_SPIKE`, whose promotion is
deliberately disabled — so a confirmed spike must be recorded as a candidate and must
not open an incident. **The spike never confirms at all:**

```
loss.robust_zscore  ->  ONE observation
LOSS_SPIKE rule     ->  persistence_required = 2 within 6 steps
                        result: not confirmed -> NOT_ANOMALY -> verdict NO_ANOMALY
```

**A real design inconsistency:** the loss detector applies `cooldown_steps = 50`, so a
sustained spike yields a single observation, while the `LOSS_SPIKE` episode rule
demands two observations within six steps. **The confirmation rule requires persistence
that the detector's own cooldown makes impossible** — so `LOSS_SPIKE` can essentially
never confirm, and `CANDIDATE_ONLY` is unreachable through that route.

This is the same class of defect Phase 1.6 found with the gradient band: two components
independently specified, never checked against each other. It is reported rather than
worked around, because reaching B by tuning the injection would hide it.

## 4. Honest labelling

Scenarios A and D are **REAL PYTORCH TRAINING E2E**. Scenarios B and C are
**EVENT-DRIVEN PIPELINE INTEGRATION**, which the brief permits provided the distinction
is not blurred. C is a constructed event stream through the production analysis path —
not a training run — and is labelled as such in `cli_e2e_results.json` per scenario.
B is likewise.

Real training was attempted for both first, and the attempts are recorded: B's ×18
loss spike produced `GRADIENT_EXPLOSION CONFIRMED` and `LOSS_SPIKE NO_ANOMALY`; C's ×10
injection landed in `NORMAL` at step 110 and the model destabilised into a genuine
explosion that confirmed at step 160. Neither reached the intended verdict, which is
why the event-driven path was used and is disclosed.

## 5. Gate

```
4 distinct verdict scenarios   FAIL  (3 of 4)
0 fabricated verdicts           PASS
0 missing evidence references   PASS
0 silent CLI fallthrough        PASS
```

**A1 FAIL.** Engineering Closure therefore cannot be PASS in this phase.
