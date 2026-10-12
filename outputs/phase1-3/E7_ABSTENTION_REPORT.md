# E7 — ABSTENTION REPORT

**Status: COMPLETE.** Implemented in `src/training_debugger/fusion/rules.py`.

Phase 1.2 had no way to decline. It could only alarm or stay silent, and it chose
"alarm" far too readily: 17.0 false positives per 1,000 steps and 56 pre-injection
incidents out of 117.

---

## 1. The four verdicts

```
CONFIRMED_INCIDENT
CANDIDATE_ONLY
INSUFFICIENT_EVIDENCE
NO_ANOMALY
```

Nothing else may be returned. `CANDIDATE_ONLY` and `INSUFFICIENT_EVIDENCE` are
**abstentions**: the system observed something and declines to call it an incident.

## 2. Where abstention comes from

| Source | Meaning |
|---|---|
| `INSUFFICIENT_EVIDENCE` | episodes remain unconfirmed with no corroboration — the evidence is genuinely ambiguous |
| `CANDIDATE_ONLY` | episodes **were** confirmed but no promotion rule accepts them (e.g. a single `WEAK` episode) |

Both are counted as abstentions in the abstention rate. `NO_ANOMALY` is not an
abstention: it is a positive statement that observations were dismissed as not
anomalous.

## 3. Reporting requirement (brief section 21)

The evaluation layer reports the abstention rate **alongside** precision and recall,
broken out by population:

```
abstention rate overall
abstention rate on healthy runs
abstention rate on fault runs
```

A system that abstains on everything scores high precision and low recall and is
**visibly FAILING** rather than quietly rewarded. Measured on the final holdout:
overall abstention **0.325**.

## 4. The anti-cheat property

The dangerous failure of an abstention mechanism is that it becomes a way to avoid
being wrong rather than a way to be honest. Two guards:

1. **Recall is reported at every layer**, so abstaining on a fault run is counted as a false negative.
2. **`NO_ANOMALY` is distinguished from abstention.** A system that calls everything `NO_ANOMALY` is not abstaining; it is claiming health, and healthy-run false incident rate stays at 0 while fault recall collapses. That is exactly what the first two final-holdout runs measured, and the four-layer model made it obvious rather than hiding it behind a single precision number.

## 5. Not verified

* **The abstention threshold is not tuned.** How many unconfirmed episodes should tip a verdict from `NO_ANOMALY` to `INSUFFICIENT_EVIDENCE` was chosen by construction (unconfirmed and nothing rejected), not measured.
* **No per-anomaly-type abstention policy.** All types share one rule.
* **Abstention is not surfaced in any user-facing rendering.** The CLI renders incidents; there is no "I could not tell" output.
