# E8 — FUSION REPORT

**Status: COMPLETE.** `src/training_debugger/fusion/rules.py`,
`fusion_rule_registry.json`.

Deterministic. No learned component, no language model. Rules are evaluated in order
and the first match promotes.

---

## 1. Promotion rules (version `phase1.3-1.0`)

| Rule id | Condition | Severity |
|---|---|---|
| `promote.numerics.certain` | ≥1 confirmed episode whose class is `CERTAIN` | error |
| `promote.strong.single` | ≥1 confirmed episode whose class is `CERTAIN` or `STRONG` | warn |
| `promote.concordant.weak` | ≥3 concordant confirmed `WEAK` episodes of the same anomaly type | warn |

A **single `WEAK` episode never promotes.** That path is precisely what produced
Phase 1.2's healthy-run false alarms: one layer of seven departing, with the
aggregate silent.

## 2. Confidence classes, derived from measurement

| Class | Rule | Evidence |
|---|---|---|
| `CERTAIN` | a measured non-finite value | nothing can make a measurement of `NaN` false |
| `STRONG` | aggregate series departed, **or** ≥40 % of layers departed at one step | injected explosion: aggregate 15.7×, 6–7 of 7 layers |
| `WEAK` | one component departed, peers did not | healthy: exactly 1 of 7, aggregate silent |

`LAYER_MULTIPLICITY_STRONG = 0.4` sits between the two measured regimes
(1/7 ≈ 0.14 and 6–7/7 ≈ 0.86–1.0) rather than being chosen by preference.

## 3. Provenance — "why did this become an incident?" without an LLM

Every verdict records:

```
promotion_rule_id, promotion_rule_version
episodes_used, episodes_rejected, abstained_episodes
signals_used, signals_rejected
candidate_onset_step, confirmed_at_step
```

persisted to `incident_results.parquet`. The question is answerable from the stored
row alone.

## 4. Defect found and fixed during the final holdout

The first three final-holdout runs produced **zero confirmed incidents** on 85 fault
runs while the episode layer reported **50 correctly confirmed episodes**.

Cause: `fuse()` selected promotable episodes by `state == "CONFIRMED"`, but the
episode engine closes every episode before fusion runs, so a confirmed episode
carries `state == "CLOSED"` with `outcome == "CONFIRMED_ANOMALY"`. The predicate
matched nothing.

Fix: `is_promotable()` accepts either an open confirmed episode **or** a closed one
whose outcome is `CONFIRMED_ANOMALY`. This is an implementation defect, not a rule
change — no threshold or condition was altered.

## 5. Not verified

* **No cross-anomaly-type fusion.** A concurrent loss spike and gradient explosion produce two verdicts; nothing combines them.
* **`promote.concordant.weak` is untested in practice** — no run in the corpus produced three concordant WEAK episodes.
* **`signals_rejected` is populated only for episodes of the same anomaly type**; signals belonging to other types are not recorded as rejected.
