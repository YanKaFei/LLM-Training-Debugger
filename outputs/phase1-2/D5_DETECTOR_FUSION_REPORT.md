# D5 — DETECTOR FUSION

**Status: COMPLETE.** Implemented in `detectors/base.py` (`ConfidenceClass`) and
`detectors/runner.py` (`_apply_fusion_classes`).

Phase 1.1 proved a single detector cannot carry incident truth. The relationship
`detector fire → incident` is now broken by an explicit layer.

---

## 1. The layer

```
detector fire  ->  SymptomEvidence (+ confidence class)  ->  IncidentDecision
```

Every fire is retained as evidence. Only a subset is promotable.

## 2. Confidence classes, derived from data (D5.1)

| Class | Rule | Derivation |
|---|---|---|
| **CERTAIN** | a non-finite value was measured | not a statistical claim; nothing can make it false. May open an incident alone. |
| **STRONG** | the *aggregate* signal departed, or ≥ 40 % of layers departed at that step | injected explosion: aggregate fires at 15.7×, all layers move. |
| **WEAK** | one component departed while its peers did not | healthy: exactly 1 of 7 layers, aggregate silent. |

`LAYER_MULTIPLICITY_STRONG = 0.4` sits between the measured regimes
(1/7 ≈ 0.14 and 6-7/7 = 1.0).

The default for a detector that does not set a class is **STRONG, never CERTAIN**,
so a detector that forgets cannot silently gain promotion rights.

## 3. Promotion rules — recorded, not implicit

Every anomaly carries `fusion_reason` and `layer_multiplicity` in its metadata, so
a report can state *why* a candidate became an incident:

| Situation | Class | Reason string |
|---|---|---|
| NaN/Inf measured | CERTAIN | `"measured non-finite value"` |
| aggregate norm departed | STRONG | `"aggregate signal departed"` |
| ≥ 40 % of layers departed | STRONG | `"86% of layers departed at this step"` |
| 1 layer of 7 departed | WEAK | `"only 1 of 7 layers departed; the aggregate did not"` |
| no layer population known | WEAK | multiplicity cannot be assessed, so it cannot promote |

That last row matters: when `population_layers` is unknown the multiplicity is
**not** inferred as `len(affected)`, because doing so would make every single-layer
fire look like 100 % of a population of one. It is treated as unassessable and the
fire stays WEAK.

## 4. Raw detector output is never discarded (D5.3)

WEAK candidates are retained in `DetectionResult.anomalies` with their full
metadata, window, baseline and threshold. They are excluded from *incident opening*,
not from the record. Suppressing them would destroy evidence; promoting them would
destroy precision.

## 5. Verified on the A1 corpus

```
step=210 layer=None  grad.global_norm  STRONG   mult=1.00  aggregate signal departed
step=210 layer=0..5  grad.layer_norm   STRONG   mult=1.00  100% of layers departed
step=220 layer=None  train.loss        STRONG   mult=0.00  aggregate signal departed
```

and on a real 130-step run, the step-78 excursion is 1-of-7 layers → WEAK, while
the step-80 injection is aggregate + 7-of-7 → STRONG.

## 6. Not verified

* **The 0.4 threshold was not swept.** It sits between two measured regimes with a wide gap, but no sensitivity analysis was run.
* **Only gradient anomalies use multiplicity.** Loss and throughput findings are classified by detector type, not by cross-signal corroboration.
* **No fusion across incident types.** A concurrent loss spike and gradient explosion produce two findings; nothing combines them into one incident with combined evidence.
* **The `WEAK` class is not yet used to open an incident under any corroboration rule** — no promotion path from WEAK exists yet, so a genuine low-magnitude fault that only ever moves one layer will be seen but not reported as an incident.
