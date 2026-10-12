# R1 — CAPABILITY STATUS

**Status: COMPLETE.** The two-state `SUPPORTED` / `UNSUPPORTED` model is replaced.

---

## 1. Why the old labels were wrong

Phase 1.3 labelled `GRADIENT_EXPLOSION`, `NAN_DETECTED` and `INF_DETECTED` as
**SUPPORTED** while their measured episode precision was **0.181** and **0.314**.
`SUPPORTED` had come to mean "code exists", which is the definition of
`IMPLEMENTED`, not of being trustworthy.

## 2. The four states

| State | Definition |
|---|---|
| `IMPLEMENTED` | code exists |
| `EXPERIMENTAL` | works in some scenarios but fails the reliability gate |
| `VALIDATED` | passes the frozen holdout quality gate |
| `UNSUPPORTED` | unavailable, or deliberately disabled |

Validation gate per capability: **episode precision ≥ 0.80 and episode recall ≥ 0.80**
on a holdout that has never been examined.

## 3. Current labels

| Anomaly type | Capability | Measured evidence |
|---|---|---|
| `GRADIENT_EXPLOSION` | **EXPERIMENTAL** | episode precision 0.181 |
| `NAN_DETECTED` | **EXPERIMENTAL** | episode precision 0.314 |
| `INF_DETECTED` | **EXPERIMENTAL** | episode precision 0.314 (shared detector) |
| `LOSS_SPIKE` | **EXPERIMENTAL** | H2 noisy healthy loss alarmed 8/14 |
| `LOSS_DIVERGENCE` | **UNSUPPORTED** | detector DISABLED, precision/recall 0.000 |
| `THROUGHPUT_REGRESSION` | **UNSUPPORTED** | detector DISABLED, recall 0.250 |

```
validated_capabilities: []
```

**Nothing in this build is VALIDATED.** That is the honest state: no capability has
passed a holdout that has never been looked at.

## 4. Detector status vs capability state

These are different axes and both are recorded:

* **detector status** (`ENABLED` / `EXPERIMENTAL` / `DISABLED`) — whether the detector runs in the default pipeline;
* **capability state** — whether the anomaly type it produces can be trusted.

A detector may be `ENABLED` while its capability is `EXPERIMENTAL`; that is the
current state of three detectors. Nothing is `ENABLED` with a capability of
`UNSUPPORTED`, because a disabled detector cannot run at all.

## 5. Machine-readable

`outputs/phase1-4/capability_registry.json` — the registry, with `capability_states`,
`anomaly_type_capability`, `validation_gate` and `validated_capabilities`.

## 6. Not done

* **README and CLI were not updated.** Brief R1 requires README, registry and CLI to agree; only the registry carries the corrected labels.
* **No promotion path is implemented.** A capability cannot programmatically advance to `VALIDATED`; the label is currently set by hand from measured evidence.
