# Run comparison

## The alignment problem

Two runs with different step counts cannot be compared at absolute steps: step
5 000 of a 10 000-step run and step 5 000 of a 100 000-step run are at completely
different points in training.

`compare_runs(..., alignment="auto")` therefore chooses:

| Condition | Axis |
|---|---|
| same total steps | `step` |
| different total steps | `relative` (`step / total_steps`) |

The chosen axis is recorded on the result, along with the interpolation method.
An unstated alignment is an unstated assumption, and the whole comparison rests on
it.

**No interpolation is performed.** Resampling two curves onto a shared grid would
manufacture points that were never measured. Instead both curves are expressed on
the axis and compared by *order statistics* — medians over the overlapping region —
which is honest about what was observed.

## Output

`RunComparison` is structured, never prose. The CLI renders it, but the structure
is primary:

| Field | Contents |
|---|---|
| `config_changes` | key, baseline value, candidate value |
| `metric_regressions` | medians both sides, relative change, **direction**, caveats |
| `incident_changes` | counts, new types, resolved types, earliest step both sides |
| `performance_changes` | throughput, step time, ordering-confidence distribution |
| `new_anomalies` / `missing_anomalies` | incident-level diffs |
| `comparability` | `comparable` flag plus explicit caveats |

`direction` is `better`/`worse` only for metrics whose polarity is known
(`train.loss` and `train.step_time` lower is better; throughput higher is better).
For the rest it is `up`/`down`, because inventing a polarity would be an
assumption.

## Incomparability is stated, not hidden

A difference in `model_id`, `world_size`, `precision` or `optimizer` sets
`comparable: false` with a caveat naming the field. Two runs of different models
can be *diffed* but should not be *ranked*.
