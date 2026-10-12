# LLM-Training-Debugger v0.1.7-alpha

**Experimental Pre-release.** Source version `v0.1.7-alpha`, Python package version
`0.1.7a1` (PEP 440).

> **This is experimental software.** It is not suitable for making automated decisions to
> stop, roll back or repair a training run.

---

## What this is

An experimental **PyTorch training evidence monitor**. It records what a training run did,
detects some abnormal signs, persists the evidence, and lets you query it later — from a
separate process — with explicit abstention when the evidence is insufficient.

It is a **monitoring and evidence tool**, not a validated root-cause diagnoser.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade "setuptools>=61" wheel
pip install -e ".[torch,dev]"
```

## Try it

```bash
python examples/quickstart.py
```

Real 4-layer transformer, 130 steps, one injected gradient explosion, one incident.

## What works

| | |
|---|---|
| Observer | non-blocking, fail-open, bounded priority queue, exact accounting |
| Storage | Parquet + DuckDB; 1,000,000 events → 10 files, 9.6 MB |
| Detectors | loss spike, gradient explosion, numeric finiteness — deterministic |
| Episodes | explicit state machine, `SUSPECT -> NORMAL` recovery, band-gated confirmation |
| Persistence | append-only episodes, incidents, relations, analysis state, with revisions |
| Recovery | episodes **and** the protected baseline survive a process restart |
| CLI | `runs`, `verdicts`, `incidents`, `incident`, `metrics`, `diagnose` |
| Abstention | `INSUFFICIENT_EVIDENCE` is a first-class verdict |
| Completeness | every verdict states whether evidence completeness was demonstrated |

## Read this before trusting the tool

```
Research-stage diagnostic accuracy.
No blind holdout validation.
No validated detector capabilities.
Observer performance target not fully met.
Automatic root-cause inference is not validated.
```

* **Detection boundary (development-supported):** gradient scaling ×2 → 0/6, ×5 → 0/6,
  ×20 → 6/6, ×40 → 6/6. Mild faults are **not distinguishable** from healthy variation
  by any signal collected. The system abstains rather than guessing.
* **Healthy-run false incidents ≈ 7.7 %** on the smoke configuration (gate: ≤ 5 %).
* **Run-level recall 0.500** on both development corpora.
* **No blind holdout has ever been run.** Every figure is development data.
* **`validated_capabilities = []`.**

## Relation presentation — read this carefully

`POSSIBLE_CONSEQUENCE_OF` is **disabled by default**. It was measured to be
undiscriminable from an independently injected fault in every case that could be
constructed (C3/C12 produced 16 consequence claims against `INDEPENDENT` ground truth).

The default records **order only**:

```
PRECEDES           one anomaly was observed before another
CO_OCCURS_WITH     both were observed together
UNKNOWN_RELATION   nothing could be established
```

The mechanism linker is an explicit opt-in research configuration. **No evidence is
deleted by this** — observations, episodes, order and revisions are all retained; only the
claim changes.

**Mechanism relations are not causal proof.** `CAUSED_BY` and `ROOT_CAUSE_OF` do not
exist in the type system, and a grouping is not a proven root cause.

## Known limitations

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md). The short version:

* mild gradient faults are missed (measured boundary, not a bug)
* healthy runs can produce false incidents
* observer overhead exceeds the 3 % core target at the largest workload
* an observer accounting race is `UNRESOLVED` — mechanism identified, never reproduced
* no GPU benchmark exists; all figures are CPU/MPS on Apple Silicon
* the historical "23 duplicate incidents" is `HISTORICAL_23_NOT_FULLY_REPRODUCED`

## Reproduce every claim

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) gives exact commands. Evidence is kept in three
strictly separated kinds and never mixed:

```
REAL_TRAINING_E2E
EVENT_DRIVEN_PIPELINE_INTEGRATION
SYNTHETIC_DEVELOPMENT_BENCHMARK
```

## Not included

No LLM/agent, no replay, no multi-rank, no GPU telemetry (DCGM/HTA/Flight Recorder/NNsight),
no learned detector, no web dashboard.

## Licence

Apache-2.0.
