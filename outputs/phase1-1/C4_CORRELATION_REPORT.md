# C4 — CORRELATION ENGINE REPORT

**Status: COMPLETE.** 26 tests shared with C3.

---

## 1. Correlation is not causation, enforced

`ALLOWED_PHRASINGS` is a closed set containing no causal verb.
`CAUSAL_VERBS` lists the forbidden words. `assert_causal_language_absent()` raises
on any output containing one, and is applied to the rendered report by test.

Verified: *"gradient anomaly caused the loss spike"*, *"was due to a slow
dataloader"* and *"a bad shard led to divergence"* are all **rejected**; the three
permitted phrasings are accepted.

## 2. Lag distributions, not proximity

For each occurrence of A, the lag in steps to the nearest **following** occurrence
of B. Only `0 <= lag <= max_lag` is kept — a target that came first is not a
negative lag, it is not a lag at all.

Verified: `lag_distribution([10,50,90], [18,61,99,200], max_lag=50) == [8, 9, 11]`;
targets preceding the source yield `[]`; a target outside the window yields `[]`.

Output per pair: `lags`, `median_lag`, `lag_mad`, `lag_min`, `lag_max`,
`effect_size`, `raw_p_value`, `adjusted_p_value`, `recurrence_count`,
`recurrence_population`, `direction`, `confidence`, `limitations`.

## 3. Recurrence, not a boolean

`MIN_OCCURRENCES = 3`; nothing is reported below it. Expected output shape is
`3 / 3 incidents, median lead 9 steps`, and `recurrence_ratio` is exposed rather
than collapsed into a flag. Per-run p-values are pooled with **Fisher's method**;
`chi2_sf` is validated against the closed form (`sf(1,2) = e^-0.5 = 0.60653`).

## 4. Multiple testing

Every ordered pair is tested, so the hypothesis count is quadratic.
`benjamini_hochberg()` returns **adjusted p-values and rejections**, and
`hypotheses_tested` is recorded because an adjusted p-value without its denominator
cannot be re-checked.

**Validated against the canonical BH 1995 example** (m=15, α=0.05): exactly **4
rejections**, `adjusted[0] = 0.0015`. Also verified: monotone, bounded in [0,1],
`adjusted >= raw`, and empty input.

## 5. Negative evidence with equal prominence

The report has four buckets, all rendered: `correlated`, **`not_correlated`**,
`insufficient`, `untestable`.

**A defect was found and fixed here.** The first implementation routed pairs with
too few occurrences into `untestable`, which made `insufficient()` structurally
always-empty and hid every pair the engine had *looked at and could not assess*.
That is precisely the "lying by omission" failure Phase 0 forbids for negative
evidence. Now every tested pair is classified, and `untestable` holds only pairs
whose signal was absent from every run.

## 6. Occurrence detection is independent of the detectors

The engine derives its own occurrences from a robust local baseline rather than
consuming detector fires. If it consumed detector output, correlation recall would
inherit detector recall and would report a clean bill of health whenever the
detectors were blind — the failure mode the calibration in C5 shows is real.

## 7. Not verified

* **Cross-run correlation is validated on 3–4 runs only.** With `MIN_OCCURRENCES = 3` and per-run pooling, the recurrence denominator is small; a ratio of `3/4` carries a wide interval.
* **No lag-distribution test against a known ground-truth causal structure.** The engine was run on runs with an injected gradient→loss lead and found it, which is necessary but not sufficient.
* **No seasonality or confounder control.** Two signals that both track step count will correlate; nothing in the engine corrects for a shared trend beyond using occurrence-based lags rather than raw correlation.
* **`direction` is always `precedes`.** No reverse or bidirectional analysis is implemented.
