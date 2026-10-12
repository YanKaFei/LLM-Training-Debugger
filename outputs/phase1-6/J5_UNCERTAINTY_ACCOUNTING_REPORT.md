# J5 — UNCERTAINTY ACCOUNTING REPORT

**Status: COMPLETE.** The band rates are now measured, which is what makes the
anti-cheat check possible.

---

## 1. Why this exists

The brief: precision must not be improved by pushing every difficult case into the
abstention band. So precision, recall **and** abstention must be reported together,
broken out by population.

`uncertainty_rates()` in `evaluation/layers.py` reads `metadata.gradient_band` from
every episode. An episode with no band — from a non-gradient detector — is counted as
`unbanded` rather than being assigned a band it never had.

## 2. Measured (development, 169 runs)

| Population | NORMAL | UNCERTAIN | CONFIRMED | unbanded |
|---|---|---|---|---|
| healthy episodes | **98.6 %** | 0.7 % | 0.7 % | 0 % |
| fault episodes | 41.6 % | 4.3 % | 11.3 % | 42.8 % |

```
HEALTHY_ABSTENTION_RATE = 0.007
FAULT_ABSTENTION_RATE   = 0.043
```

**The anti-cheat check passes decisively.** The healthy `UNCERTAIN` rate is **0.7 %**,
so the precision gain did not come from hiding healthy runs in the abstention band —
the system genuinely classifies 98.6 % of healthy episodes as `NORMAL`.

## 3. By fault magnitude

| Magnitude | NORMAL | UNCERTAIN | CONFIRMED |
|---|---|---|---|
| gradient ×2.5 | most | few | 0 |
| gradient ×4.0 | mixed | some | 1/5 |
| gradient ×7.0 | mixed | some | 0/5 |
| gradient ×12.0 | few | few | 2/5 |
| gradient ×25.0 | none | none | **5/5** |
| gradient ×50.0 | none | none | **5/5** |

## 4. Development-supported boundary

```
development-supported minimum reliably detectable gradient magnitude: ~ x25
```

**Development-supported**, never *validated*: no untouched holdout has been run.

The brief asked for the boundary to be measured rather than preset, and it was — the
answer the measurement gives is far more conservative than the boundary analysis
suggested (≈×4), and the non-monotonicity at ×7.0 (0/5, between ×4 at 1/5 and ×12 at
2/5) shows the per-magnitude sample of 5 runs is too small to place the boundary
precisely.

## 5. Not verified

* **5 runs per magnitude.** The boundary estimate is unstable; the ×7.0 anomaly is direct evidence of that.
* **`unbanded` is 42.8 % of fault episodes**, because the loss and numerics detectors do not produce gradient bands. Those episodes are outside the band accounting entirely.
* **Non-gradient abstention is not measured.** The band is the only abstention source counted.
