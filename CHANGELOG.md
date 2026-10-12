# CHANGELOG

## v0.1.0-alpha — unreleased

First public experimental release. **Alpha asserts installability and engineering
correctness. It does not assert detector reliability.**

### Capability — what works
- `TrainingEvent` v0.1 schema with a frozen vocabulary and partial order
- Non-blocking, fail-open observer with a bounded priority queue and exact accounting
- PyTorch adapter: loss, lr, gradient/parameter norms, phase timing, throughput
- Per-layer (T1) gradient instrumentation
- Parquet + DuckDB storage, benchmarked at 1M events (10 files, 9.6 MB)
- Deterministic detectors: loss spike, gradient explosion, numeric finiteness
- `AnomalyEpisode` with an explicit state machine and `SUSPECT -> NORMAL`
- Contamination-resistant, state-aware gradient baseline (freeze on suspicion)
- Three-way gradient band: `NORMAL` / `UNCERTAIN` / `CONFIRMED`
- Deterministic fusion with versioned promotion rules and provenance
- Four verdicts including abstention, with `reason_code`, `evidence_state` and
  `promotion_disposition`
- **Evidence completeness** on every verdict: an incomplete stream can never be rendered
  as reassurance
- Append-only episode, incident and analysis-state persistence with cross-process
  recovery of episodes **and** the protected baseline
- `IncidentRelation` with `POSSIBLE_CONSEQUENCE_OF` / `PRECEDES` / `CO_OCCURS_WITH` /
  `UNKNOWN_RELATION` — no causal type exists
- Incident reconciliation separating independent anomalies from linked manifestations
  from duplicate emission
- CLI: `runs`, `incidents`, `incident`, `metrics`, `verdicts`, `diagnose`
- Persisted scoring that reads episodes and relations from disk

### Known defects
- A healthy-run false positive, root-caused but unfixed (frozen baseline never resumes)
- A pre-injection onset in the quickstart example (reported 75, injected 80)
- 23 historical duplicate incidents not reproduced or explained
- Observer accounting race `UNRESOLVED`
- `LOSS_DIVERGENCE` and `THROUGHPUT_REGRESSION` declared `UNSUPPORTED`

### Not included
- No replay, no multi-rank, no GPU telemetry (DCGM/HTA/Flight Recorder/NNsight), no
  learned detector, no LLM agent, no web dashboard
