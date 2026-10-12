# Architecture

```
Adapter  ->  Observer  ->  TrainingEvent  ->  Storage  ->  Detector  ->  Incident  ->  CLI
```

Dependencies point one way only. `tests/unit/test_layering.py` parses the imports
and fails the build on an upward edge, so the rule is enforced rather than
documented. It has already caught real violations: silent `except: pass` handlers
in the observer and the storage reader.

| Layer | Package | Responsibility | Must not |
|---|---|---|---|
| Adapter | `adapters/` | observe a training loop the user owns | detect, store, or render |
| Observer | `observer/` | sample, validate, queue, write; never block | analyse |
| Schema | `schema/` | the data contract and the partial order | depend on anything above it |
| Storage | `storage/` | Parquet in, DuckDB out | interpret |
| Detectors | `detectors/` | statistical anomaly detection | know about incidents or agents |
| Incidents | `incidents/` | evidence, ordering, ranking, rendering | call a model |
| CLI | `cli/` | load, call, format | contain analysis logic |

Design commitments carried from Phase 0:

1. **The observer never blocks training.** Fail-open, bounded queue, priority-aware
   eviction, its own error accounting, and a circuit breaker for a dead writer.
2. **Parquet is the truth; DuckDB is a query engine.** Trackio is *not* in the
   telemetry path.
3. **Detection is deterministic.** No learned model, no LLM.
4. **The LLM never decides what is true.** `diagnose` is a deterministic renderer.
5. **Ordering is partial and explicit.** A `BEFORE` claim requires a witness.
