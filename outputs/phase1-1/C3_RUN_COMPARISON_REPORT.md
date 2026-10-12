# C3 — CROSS-RUN COMPARISON REPORT

**Status: COMPLETE.** 26 tests shared with C4.

---

## 1. Output is structured, not prose

`compare_runs(root, baseline, candidate) -> RunComparison` with:

| Field | Contents |
|---|---|
| `config_changes` | key, baseline value, candidate value |
| `metric_regressions` | both medians, relative change, **direction**, counts, caveats |
| `incident_changes` | counts, `new_types`, `resolved_types`, earliest step both sides |
| `performance_changes` | throughput, step time, ordering-confidence distribution |
| `new_anomalies` / `missing_anomalies` | incident-level diffs |
| `comparability` | `comparable` flag plus explicit caveats |

`direction` is `better`/`worse` only where polarity is known; otherwise `up`/`down`.
Inventing a polarity for `grad.global_norm` would be an assumption, not a
measurement.

## 2. Position normalisation

`alignment="auto"` selects `relative` (`step / total_steps`) when the runs differ
in length, `step` when they do not, and **records the choice** with the
interpolation method.

**No resampling is performed.** Interpolating two curves onto a shared grid would
manufacture points that were never measured. Both curves are expressed on the axis
and compared by order statistics (medians over the overlapping region).

Verified: a 100-step vs 300-step comparison chooses `relative` and emits the caveat
*"runs differ in length (100 vs 300 steps); compared on relative progress"*.

## 3. Incomparability is stated

A difference in `model_id`, `world_size`, `precision` or `optimizer` sets
`comparable: false` and names the field. Two runs of different models can be
*diffed* but must not be *ranked*.

## 4. Verified

Config-change detection (`lr` change found), metric regressions with both medians,
alignment declaration, relative alignment on unequal lengths, incident diffs,
deterministic rendering (two calls produce identical text), and self-comparison
producing neutral results.

## 5. Not verified

* **Cross-model comparison.** Only same-architecture, different-hyperparameter runs were compared.
* **Statistical significance of a metric regression.** A `relative_change` is reported; whether it is distinguishable from run-to-run noise is **not** tested, and with one run per side it could not be.
* **Baseline selection.** The caller supplies both run ids; there is no notion of a default or rolling baseline.
* **`new_anomalies` semantics across alignment.** Anomalies are matched by *type*, not by aligned position, so a spike at a different relative progress is still matched as "the same type". This is a real limitation: an earlier spike is not reported as new.
