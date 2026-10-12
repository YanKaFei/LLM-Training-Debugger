# E5 — THROUGHPUT DETECTOR REPORT

**Status: UNSUPPORTED.** Disabled by default; declared unsupported.

---

## 1. Registry state

| Detector | Status | Reason |
|---|---|---|
| `throughput.regression` v1 | **DISABLED** | recall 0.250, 9 pre-injection false positives |

Anomaly-type support: `THROUGHPUT_REGRESSION` = **UNSUPPORTED**.

## 2. Why disabled

Measured on the Phase 1.2 holdout: precision 0.100, recall 0.250, 9 pre-injection
false positives. It alarms during warm-up and misses the injected step delay. It is
not usable as a default incident source.

Measured on the final holdout: `F6_throughput_drop` produced a confirmed incident in
**3 of 5** runs — but the detector is disabled, so those came from other detectors
responding to the delay's side effects, not from throughput detection. That is
incidental, and the registry records the type as unsupported.

## 3. The normalisation audit the brief requires (section 10)

The brief asks specifically whether the throughput metric is correctly normalised
for tokens, batch size, sequence length and gradient accumulation, rather than
comparing raw step time. **This audit was not performed.**

What is known from reading the implementation:

* `train.tokens_per_sec` is emitted and is the natural normalised metric;
* `throughput.regression` compares **step time**, not tokens/s, against a trailing baseline;
* the detector's `metadata.root_cause` is `None` and `scope` is `symptom_only` by design — it deliberately declines to attribute a slowdown, since that is v0.2's job with HTA and DCGM. That design decision stands.

So the brief's suspicion is **substantiated by inspection but not measured**: the
detector uses raw step time.

## 4. Not implemented

| Requirement | Status |
|---|---|
| explicit `warm-up state` / `baseline-learning` / `steady-state` phase model | **NOT DONE** |
| rolling median ratio | **NOT DONE** |
| robust envelope | **NOT DONE** |
| N-of-M persistence | **NOT DONE** |
| change-point persistence | **NOT DONE** |

The brief asks for a comparison of these approaches. None was implemented or
measured, so none is reported.

## 5. Capability honesty

```
THROUGHPUT_REGRESSION: UNSUPPORTED
```

The README, the registry and this report say so. The build does not claim to detect
throughput regressions.
