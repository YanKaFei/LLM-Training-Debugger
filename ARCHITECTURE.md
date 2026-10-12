# ARCHITECTURE

```
Adapter -> Observer -> TrainingEvent -> Parquet -> Detector -> AnomalyEpisode
                                                            -> EpisodeStore
                                                            -> Fusion -> verdict
                                                            -> IncidentStore
                                                            -> IncidentRelation
                                                            -> CLI / Scorer
```

Dependencies point one way. `tests/unit/test_layering.py` parses imports and fails the
build on an upward edge; it has caught real violations (silent `except: pass` handlers).

| Layer | Package | Responsibility | Must not |
|---|---|---|---|
| Adapter | `adapters/` | observe a loop the user owns | detect, store, render |
| Observer | `observer/` | sample, validate, queue, write; never block | analyse |
| Schema | `schema/` | the data contract and the partial order | depend on anything above it |
| Storage | `storage/` | Parquet in, DuckDB out | interpret |
| Detectors | `detectors/` | statistical anomaly detection | know about incidents |
| Episodes | `episodes/` | group observations, own the state machine | call a model |
| Fusion | `fusion/` | verdicts, promotion rules, evidence completeness | invent a cause |
| Incidents | `incidents/` | evidence, relations, reconciliation, rendering | assert causation |
| Persistence | `persistence/` | append-only episode / incident / analysis state | interpret |
| CLI | `cli/` | load, call, format | contain analysis |

## Five commitments

1. **The observer never blocks training.** Fail-open, bounded queue, priority eviction,
   its own error accounting, a circuit breaker for a dead writer.
2. **Parquet is the truth; DuckDB queries it.** No database. Trackio is presentation
   only and never in the telemetry path.
3. **Detection is deterministic.** No learned model, no LLM.
4. **The LLM never decides what is true.** `diagnose` is a deterministic renderer.
5. **Ordering is partial and explicit.** A `BEFORE` claim requires a witness
   (`producer_sequence`, `causal_edge`, `barrier`). `event_id` and wall clock never order.

## The observation model

```
raw signal -> candidate observation -> AnomalyEpisode -> fusion -> verdict
```

An episode is a **contiguous run of abnormal observations** for one detector and signal
key. Identity is contiguity, **not** a cooldown: a cooldown may reduce log volume but may
not decide what counts as one anomaly.

```
NORMAL -> SUSPECT -> CONFIRMED -> RECOVERING -> CLOSED
             |
             +-------> NORMAL          an excursion that turned out to be nothing
```

`candidate_onset_step` and `confirmed_at_step` are separate fields. The runtime knows
the second only later, and must never claim to have known at onset what it could only
confirm afterwards.

## Four verdicts

```
CONFIRMED_INCIDENT     a promotion rule matched
CANDIDATE_ONLY         confirmed, but not promotable (capability disabled / no rule)
INSUFFICIENT_EVIDENCE  observed, unresolved — this is a legitimate answer
NO_ANOMALY             observed and dismissed
```

Two orthogonal dimensions keep these honest: `evidence_state`
(`SUSPECT`/`CONFIRMED`/`CLOSED`) and `promotion_disposition`
(`ELIGIBLE`/`DISABLED`/`WITHHELD`), plus a machine-readable `reason_code`. A disabled
capability is never described as "not enough evidence".

Every verdict also carries `data_quality`. Without demonstrated completeness the
renderer will not state that nothing was found.

## The causal fence

`IncidentRelation` supports `POSSIBLE_CONSEQUENCE_OF`, `PRECEDES`, `CO_OCCURS_WITH` and
`UNKNOWN_RELATION`. **`CAUSED_BY` and `ROOT_CAUSE_OF` do not exist**, asserted by test.
An `IncidentGroup` is a presentation unit with `has_proven_root_cause` hard-coded
`False`. Establishing causation requires replay, deferred to a later phase.
