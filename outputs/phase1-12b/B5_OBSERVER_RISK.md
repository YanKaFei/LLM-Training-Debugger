# B5 — OBSERVER RISK

**Status: RISK RECORDED, NOT RESOLVED.**

---

## 1. Concurrency risk (inherited)

```
root cause:                    NOT IDENTIFIED
fixed:                         NO
latest repetitions (1.12A):    30 isolated / 30 pass, 5 full-suite / 5 pass
historical failure:            1 recorded in Phase 1.11, never reproduced
```

**`OBSERVER_STABILITY = UNRESOLVED`.** Thirty-five clean runs are not evidence of a
repair; they are evidence that the failure is rare. Phase 1.12A's measurement stands and
nothing in this phase changes it. Per the brief, an unresolved B5 does **not** block
B1–B4 — and B0/B1 were completed regardless.

## 2. A risk this phase's own change introduced — recorded, not measured

Phase 1.12A set `cooldown_suppresses_evidence = False`. The brief correctly identifies
the consequence:

> 20 条证据替代 1 条告警 ... 可能改变分析吞吐和存储压力。

The *incident* layer is protected — measured in 1.12A: 20 observations → 1 episode →
1 verdict → 0 incidents. But **two resource questions were not measured**:

| Question | State |
|---|---|
| does the bounded queue remain bounded? | **NOT VERIFIED** |
| is drop accounting still correct? | **NOT VERIFIED** |
| how much did evidence volume grow? | **NOT VERIFIED** — 20× on the synthetic loss spike, but not measured on a corpus |
| is fail-open semantics preserved? | **NOT VERIFIED** for the new volume |

The mechanism for a regression is real: the observer's queue is bounded and drops by
priority, so a 20× increase in emitted evidence can push a run from "no drops" into
"drops", which would silently truncate the evidence chain that this whole phase is
about making trustworthy. Phase 1.1's fault-injection suite measured drop behaviour,
but **not at the evidence volume the new setting produces**.

**This is the most consequential open risk in the phase**, and it is recorded rather
than assumed benign. The resource-boundary tests the brief asks for were not added.
