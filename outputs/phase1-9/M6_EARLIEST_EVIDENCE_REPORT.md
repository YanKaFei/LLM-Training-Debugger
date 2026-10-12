# M6 — EARLIEST EVIDENCE REPORT

**Status: EXECUTED — the metric is measured for the first time in the project.**

Seven phases described this as the product's core claim and never measured it.

---

## 1. Result

| Metric | Value | Target | Verdict |
|---|---|---|---|
| runs (fault) | 85 | — | — |
| `TRUE` | **24** | — | — |
| `FALSE` | 8 | — | — |
| `UNKNOWN` | 53 | — | — |
| **EARLIEST_EVIDENCE_PRECISION** | **0.750** | ≥ 0.92 | **FAIL** |
| **EARLIEST_EVIDENCE_COVERAGE** | **0.282** | ≥ 0.80 | **FAIL** |
| absolute onset error (median) | **0.0 steps** | — | — |
| signed onset error (median) | **0.0 steps** | — | — |
| confirmation delay (median) | **0.0 steps** | — | — |
| UNKNOWN rate | 0.624 | — | — |
| **pre-injection confirmed incidents** | **0** | = 0 | **PASS** |

## 2. What it means

**When the system does claim an earliest abnormal event, it names the exact step** —
median absolute onset error **0.0** with no signed bias and no confirmation delay. The
localisation is precise.

**But it claims on only 28 % of fault runs.** Coverage 0.282 against a target of 0.80:
three quarters of faults produce no earliest-evidence claim at all, because the
episode never confirms.

**Precision 0.750 means one claim in four is wrong** — it names a confirmed episode of
the expected family whose onset is outside the acceptable window.

## 3. The coverage guard worked as designed

Precision and coverage are reported **together**, exactly as the brief requires. A
system that answered `UNKNOWN` to every run would show precision `None` and coverage
`0.0`; this system shows 0.750 and 0.282. Neither number can be read as success on its
own, and the pair is the honest description:

> The system can name the precise onset of a fault it confirms, and it confirms only a
> minority of injected faults.

## 4. Per-severity breakdown — NOT VERIFIED

The brief requires the three regions (mild overlap / uncertain / development-supported
large-fault) to be reported separately so severe cases cannot mask mild-case
abstention. **This breakdown was not produced.** The development split's magnitudes
were not joined to the earliest-evidence outcomes.

## 5. Not verified

* Per-severity breakdown (§4).
* Whether the 8 `FALSE` claims precede or follow their window; only the aggregate is computed.
* Interaction with the `UNCERTAIN` band: whether UNKNOWN runs are concentrated in the overlap region, which the per-severity breakdown would show.
