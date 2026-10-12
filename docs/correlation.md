# Correlation

## Correlation is not causation

The three are different evidence levels and must never be merged:

| Statement | Level |
|---|---|
| `gradient norm anomalous` | **OBSERVED ANOMALY** |
| `gradient anomaly precedes loss spikes in 8 / 9 comparable incidents, median lead 9 steps` | **CORRELATION / RECURRENCE** |
| `gradient anomaly caused loss divergence` | **NOT ESTABLISHED** |

The third requires replay or intervention. Phase 1.1 has neither. The engine's
vocabulary therefore has no causal verb: `ALLOWED_PHRASINGS` is a closed set and
`assert_causal_language_absent()` rejects output containing one. The guard is
tested.

## Lag *distributions*, not proximity

"A and B were near each other" is not evidence: in a stream where something changes
every step, everything is near everything. What is measured is, for each occurrence
of A, the lag in steps of the nearest **following** occurrence of B. Lags are kept
only when `0 <= lag <= max_lag`; a target that came first is not a negative lag, it
is not a lag at all.

The output records `median_lag`, `lag_mad`, `lag_min`, `lag_max` and the raw lag
list. A median lag of 9 steps with a MAD of 1 is a different finding from a median
of 9 with a MAD of 20.

## Recurrence

A single occurrence is an anecdote. No association is reported below
`MIN_OCCURRENCES = 3`, and the report says `k / n` — "3 / 4 incidents" — rather
than a boolean. Across runs, per-run p-values are pooled with Fisher's method and
the recurrence ratio is reported alongside.

## Multiple testing

Every ordered pair of signals is tested, so the hypothesis count is quadratic.
Benjamini–Hochberg controls the false discovery rate; both the **raw** and the
**adjusted** p-value survive into the output, along with `hypotheses_tested`,
because an adjusted p-value without its denominator cannot be re-checked.

`benjamini_hochberg` is validated against the canonical BH 1995 example (m=15,
α=0.05 → four rejections).

## Negative evidence

The report has three buckets, all rendered:

* `correlated`
* **`not_correlated`** — tested, enough occurrences, no association
* `insufficient` — tested, too few occurrences
* `untestable` — a signal was absent from every run

An earlier implementation dropped the `insufficient` pairs into `untestable`,
which made `insufficient()` structurally always-empty and hid the pairs the engine
had looked at and could not assess. That is the "lying by omission" failure Phase 0
forbids for negative evidence, and it is fixed.

## Occurrence detection is independent of the detectors

The correlation engine computes its own occurrences from a robust local baseline
rather than consuming detector output. If it consumed detector fires, correlation
recall would inherit detector recall and would report a clean bill of health
whenever the detectors were blind.
