# AUTONOMOUS BACKLOG

## v0.1.7-alpha (delivered)

- [x] **A** 11-step latency root-caused: baseline readiness (floor 50 + minimum_history 20)
- [x] **A** Early Evidence Lane NOT built — nothing to surface; recorded INCONCLUSIVE
- [x] **B** injections moved past readiness; C4 and C11 valid; **C3 still blocked** (threshold-to-stability)
- [x] **C** three strategies byte-identical on every corpus; 16 false associations, all from invalid C3/C12
- [x] **D** 531 passing, smoke 8/8, quickstart, installation all PASS
- [ ] **next** lower the gradient confirmation threshold without losing healthy precision
- [ ] **next** find observable midstream evidence for the C4 finite-parameter overflow

## v0.1.6-alpha (delivered)

- [x] **W1** witness contract audit PASS
- [x] **W2** paired OFF/ON on 54 byte-identical runs — identical, benefit exactly zero
- [x] **W3** C1-C9 challenge corpus; C3 not constructible, C4 a coverage gap
- [x] **W4** identifiability INSUFFICIENT — parameter mutation is never observed
- [x] **W6** overhead +0.30% total / +0.97% p95; no GPU benchmark; 3% gate unresolved
- [x] **W8** decision KEEP_RESEARCH_ONLY
- [ ] **W-next** reduce gradient-detector latency below the propagation window
- [ ] **W-next** make C3 constructible so the witness can be tested at all

## v0.1.5-alpha (delivered)

- [x] **W0** norm-reduction artifact audit — v0.1.4's headline was measuring it
- [x] **W1/W2** OptimizerWitness, opt-in, default OFF; ATTEMPTED/INVOKED/SKIPPED/UNKNOWN kept distinct
- [x] **W3** SGD, Adam, accumulation, skip, multi-optimizer, unknown — 14 tests
- [x] **W5** overhead +0.30% total / +0.97% p95, paired and interleaved (7 repeats)
- [ ] **W-next** adversarial corpus with the witness ENABLED as midstream evidence

## v0.1.4-alpha (delivered)

- [x] **W1** signature-dependence audit — CONFIRMED; finite tensor vs infinite norm not distinguished
- [x] **W2** 45-run adversarial corpus; falsified v0.1.3 (10 fabricated associations)
- [x] **W3** relation-opportunity scoring — correct rejection separated from abstention
- [x] **W4/W5** midstream witness required at every stage; optimizer RFC written
- [x] **W7** regression: 516 passing, smoke 8/8, install PASS
- [ ] **W-next** operator decision on the optimizer-boundary witness

## v0.1.3-alpha (delivered)

- [x] **W1** M1/M3 trace — root causes found before any code change
- [x] **W2** metric-scoped finiteness episodes (evidence availability)
- [x] **W3** mechanism-constrained association with E1-E4 hierarchy
- [x] **W4** observability audit — existing telemetry sufficient
- [x] **W5** mixed corpus re-evaluated: relation precision 1.0, recall 1.0
- [x] **W7** incident-level FN bug fixed; corpora scored separately
- [ ] **W-next** O(1) optimizer-boundary witness (needs authorisation)

## v0.1.2-alpha (delivered)

- [x] **WS1** level-separated scoring, fail-closed validation, recall denominator fixed
- [x] **WS2** corpus integrity — loader bug, 42/42 readable
- [x] **WS3** 50-run mixed-fault corpus, 10 relations persisted
- [x] **WS4** relation evaluation — **precision 0.0, recall 0.0, INVERTED**
- [ ] **WS4b** mechanism-aware relation association, measured against the corpus
- [ ] **WS5** observer race

## v0.1.1-alpha

- [x] **W1** Earliest-evidence temporal integrity — onset re-anchored
- [x] **W2** sweep complete: parameter inert, no candidate qualifies (negative result)
- [x] **W3** 42 runs under stable ids, all artefacts persisted, scorer executed from disk
- [ ] **W3b** mixed-fault corpus so relations exist and relation scoring becomes estimable
- [ ] **W3c** pin down the `primary_recall` denominator
- [x] **W4** Observer flaky mechanism identified and test repaired (race UNRESOLVED)

## v0.1.0-alpha (delivered)


Priority order. Work top-down; skip an item only if it is blocked, and say so.

## P0 — Correctness

- [x] **A1** `evidence_completeness`: `data_quality_status`, `drop_count`, `completeness_warning` on the verdict; an unqualified `NO_ANOMALY` is blocked when completeness is unknown or degraded
- [x] **A2** (diagnosed) Diagnose the S1 healthy-run false positive on its named run; keep the failing case as a regression
- [x] **A3** Prove the same source evidence cannot form a duplicate Episode or Incident (idempotency regression at the pipeline level, not only the engine)

## P1 — Persisted scoring

- [x] **B1** Scorer reads `episode_relations` from disk
- [ ] **B2** Execute over the development corpus; report TP/FP/FN and denominators
- [ ] **B3** Split the historical 23 into product duplicates vs scorer misclassifications, per-record
- [ ] **B4** Recompute numerics / relation / earliest-evidence quality

## P2 — Integration

- [x] **C1** Convert the eight smoke scenarios into formal regression tests
- [x] **C2** Verify independent-NaN protection through the persisted path
- [x] **C3** Verify protected baseline and relations survive a restart (already true in experiments; needs a test)

## P3 — Stability

- [ ] **D1** Observer accounting race: deterministic overflow reproduction
- [x] **D2** Measure downstream anomaly volume, memory and disk growth at scale
- [x] **D3** Confirm fail-open cannot mask a completeness problem

## P4 — Alpha packaging

- [x] **E1** LICENSE (Apache-2.0 per Phase 0)
- [x] **E2** README / QUICKSTART / ARCHITECTURE / CAPABILITIES / KNOWN_LIMITATIONS / CONTRIBUTING
- [x] **E3** CHANGELOG / RELEASE_NOTES
- [x] **E4** `examples/` runnable demo
- [x] **E5** Clean-environment install verification (real command, real exit code)
