# M1 — CAPABILITY SCOPE

**Status: COMPLETE.** The gradient capability is described by what was measured, not
by what would sound good.

---

## 1. Capability metadata

```
status:                    EXPERIMENTAL
validation_domain:         DEVELOPMENT_SYNTHETIC
supported_signal:          sustained distributed gradient level shift
development_detection_boundary:
    < x12      NOT RELIABLY DISTINGUISHABLE from healthy variation
    x12        PARTIAL (P(CONFIRMED) = 0.70)
    >= x20     DEVELOPMENT-SUPPORTED (0.95 - 1.00)
known_limitations:
    mild perturbations overlap healthy gradient variation
    healthy confirmed rate 2.4 % (development) remains above the 3 % aspiration
    the boundary is benchmark-specific and does NOT extrapolate to a
    physical "20x" limit on real incidents
```

## 2. The wording this replaces

**Not written:**

> detects gradient explosions >= 20x

**Written instead:**

> On the current synthetic gradient-scaling benchmark, large sustained perturbations
> at the upper tested severity are detected reliably in development data; milder
> perturbations remain non-separable from healthy variation.

The distinction matters because the benchmark injects a uniform multiplicative
scaling with a known onset. Real gradient explosions differ in shape, duration and
locality, and nothing here measures that.

## 3. Two recall figures, both reported

| Figure | Definition | Value |
|---|---|---|
| **full-spectrum recall** | all injected magnitudes ×1.5 … ×40 | 0.440 (incident, M7) |
| **declared-domain recall** | only severities inside the supported domain (≥ ×20) | **0.95 – 1.00** (from the 680-run sweep) |

The brief is explicit that this is **not** a way to obtain a pass by changing the
denominator. It is reported to describe the capability boundary. **The Phase 1.x
gates are unchanged and still FAIL against full-spectrum recall.**

## 4. Healthy false confirmations

```
healthy confirmed rate: 2.4 %   (development, M7)
```

Meets the ≤ 3 % target, and is **not** hidden or excluded. It rose from 1.19 % in
Phase 1.6 to 2.38 % here, because the protected baseline raises the aggregate on some
healthy runs whose gradient level genuinely shifts during a scheduler transition or
clipping onset.

## 5. Status

`GRADIENT_EXPLOSION` stays **EXPERIMENTAL**. It is **not** promoted to
`DEVELOPMENT_READY` or `VALIDATED`: the full-spectrum recall target is not met, and
no blind holdout has been run.
