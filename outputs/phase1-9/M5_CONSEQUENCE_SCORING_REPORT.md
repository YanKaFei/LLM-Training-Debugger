# M5 — CONSEQUENCE SCORING REPORT

**Status: EXECUTED.** The scorer ran over the 169-run development corpus. Its first
execution produced invalid output from a naming mismatch, which is reported because
publishing that output as a measurement would have been wrong.

---

## 1. The first execution was invalid, and why

```
primary_precision: 0.0     counts: UNRELATED_FALSE_POSITIVE 225, CORRECT_ABSTENTION 174
```

Ground truth names a **scenario class** (`F3_gradient_explosion`) while an episode
carries an **anomaly type** (`GRADIENT_EXPLOSION`). Nothing matched, so every
observation scored as unrelated. This was a wiring defect in the evaluation, not a
detector result, and it is now fixed with an explicit documented mapping
(`FAULT_CLASS_TO_ANOMALY`) so the two vocabularies cannot silently diverge again.

## 2. Corrected result

| Class | Count |
|---|---|
| `PRIMARY_TRUE_POSITIVE` | **24** |
| `UNRELATED_FALSE_POSITIVE` | 184 |
| `DUPLICATE_TOP_LEVEL_INCIDENT` | **23** |
| `CORRECT_ABSTENTION` | 174 |

| Metric | Value |
|---|---|
| primary precision | **0.104** |
| primary recall | **1.000** |
| secondary manifestation precision | 0.0 |
| secondary manifestation recall | n/a |
| **duplicate top-level incident rate** | **0.057** |
| relation precision | n/a |
| relation recall | 0.0 |

## 3. Reading it

**Primary recall is 1.000**: when the detector fires, it fires on the right family.
**Primary precision is 0.104**: 184 confirmed observations are of the wrong family for
the fault that was injected.

**Duplicate root incidents: 23, rate 0.057.** This is a real, measured error and it is
the failure mode the brief's R4/J1 named: a consequence claimed as an independent root
cause.

**Secondary manifestations: 0, relation recall 0.0.** Not because there were none, but
because `_persist_run` computes relations while the evaluation reads them from the
result dict, and the result dict does not carry them. Relations are written to
`episode_relations` and are **not** consumed by the evaluator. That is a wiring gap,
not a finding of "no consequences".

## 4. Numerics — the 0.314 is retired

```
numerics observation precision:      NOT VERIFIED
numerics observation recall:         NOT VERIFIED
numerics top-level precision:        NOT VERIFIED
numerics top-level recall:           NOT VERIFIED
```

The per-capability breakdown was **not computed**. What is now established is that
**0.314 must no longer be quoted as the current numerics precision** — it mixed true
secondary manifestations with duplicate root incidents and is semantically invalid.
It is retired rather than replaced by an estimate.
