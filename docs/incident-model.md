# The incident model

> The product is not the graph. The product is the incident.

## Four evidence buckets

| Bucket | Meaning |
|---|---|
| `supporting` | the anomaly and the signals that moved with it |
| `contradicting` | evidence against a hypothesis |
| **`not_correlated`** | **signals tested and found stable — mandatory** |
| `unknown` | signals that could not be tested, with the reason |

`Incident.validate()` rejects an incident that has no negative evidence **and** no
stated missing evidence. "We tested nothing" must not render as "nothing was
wrong".

## Evidence levels

`OBSERVED` (measured) · `INFERRED` (a named statistical procedure; the detector is
required) · `HYPOTHESIS` (a causal claim awaiting a probe) · `UNKNOWN`.

Data causation is capped at `HYPOTHESIS` in Phase 1: no replay exists to confirm
that a shard caused a spike, and Phase 0 forbids concluding it from temporal
proximity.

## Earliest abnormal evidence

The earliest evidence is the **set of minimal elements** of the partial order. A
singular `first_abnormal_event_id` is returned only when that set has exactly one
member *and* no comparison was left unsettled.

| Situation | `first_abnormal_event_id` | `ordering_confidence` |
|---|---|---|
| exactly one minimum, no gaps | set | `PROVEN` |
| one minimum, some comparisons unsettled | set | `LOW` |
| several minima, all pairs settled | `None` | `PROVEN` (proved concurrent) |
| several minima, some pairs unsettled | `None` | `UNKNOWN` |

## Ranking

Templates are data. Scores are arithmetic over facts. Every factor is stored in
`score_decomposition`, a multiplicity penalty prevents vagueness from winning, and
`max_level` caps what each template may claim.
