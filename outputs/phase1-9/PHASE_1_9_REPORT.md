# PHASE 1.9 REPORT

```
PHASE 1.9 STATUS:              PARTIAL
READY_FOR_BLIND_VALIDATION:    NO
DECLARED_DOMAIN_READY:         NO
PHASE 1 ORIGINAL CLOSURE:      STILL PARTIAL
AGENT READINESS:               NO
```

**No final holdout files were created**, per the brief.

`PHASE_1_REPORT.md` and `outputs/phase1-1..1-8/` are **unmodified**.

---

## 1. What was delivered

### M0 — the monotonicity claim was wrong, and the reporter was at fault

Phase 1.8 reported `0 violations` beside a decreasing curve. The reporter only flagged
drops larger than an **arbitrary 0.25**, so every real decrease in that curve fell
below its own threshold. Corrected:

```
inversions: 4   largest drop: 0.17   rank correlation: 0.860   monotone: False
```

Accurate description: **strongly increasing trend with four local inversions confined
to the low-magnitude region**. The tolerance now defaults to **0.0**, and a non-zero
tolerance must be passed explicitly and is echoed into the output. Five regression
tests pin it, including the Phase 1.8 curve itself.

### M1 — capability scope stated honestly

```
status: EXPERIMENTAL
domain: DEVELOPMENT_SYNTHETIC
boundary: <x12 not distinguishable | x12 partial (0.70) | >=x20 supported (0.95-1.00)
```

The claim is *"large sustained perturbations at the upper tested severity are detected
reliably in development data; milder perturbations remain non-separable"* — **not**
"detects gradient explosions ≥ 20×", because the benchmark injects a uniform
multiplicative scaling and nothing here measures real incident shapes.

### M2 — the pipeline now writes its own evidence

`EpisodeStore` and `IncidentStore` are called by `run_one()` on the real path.
Verified on 169 runs: **157 Parquet files, 0 persistence errors**, covering episodes,
revisions and relations.

* **Ordering enforced and tested**: every episode revision is written before any incident that consumes it, so a crash cannot leave an incident with a missing episode.
* **Idempotency tested**: deterministic episode ids mean a re-run adds a revision, never a duplicate episode.
* **Fail-open but surfaced**: errors are returned and carried on the run result, never swallowed.

### M5 / M6 — the two evaluators finally executed

**M5 consequence scoring.** Its *first* execution produced `primary_precision = 0.0`
from a **naming mismatch** — ground truth says `F3_gradient_explosion`, an episode says
`GRADIENT_EXPLOSION` — so nothing matched. Fixed with an explicit documented mapping
and re-run:

| Class | Count |
|---|---|
| `PRIMARY_TRUE_POSITIVE` | 24 |
| `UNRELATED_FALSE_POSITIVE` | 184 |
| `DUPLICATE_TOP_LEVEL_INCIDENT` | **23** |
| `CORRECT_ABSTENTION` | 174 |

primary recall **1.000**, primary precision **0.104**, duplicate root incident rate **0.057**.
**The 0.314 numerics figure is retired as semantically invalid** and is replaced by
`NOT VERIFIED` rather than by an estimate.

**M6 earliest evidence — measured for the first time in the project.**

| Metric | Value | Target |
|---|---|---|
| precision | **0.750** | ≥ 0.92 |
| coverage | **0.282** | ≥ 0.80 |
| absolute onset error | **0.0 steps** | — |
| signed onset error | 0.0 steps | — |
| confirmation delay | 0.0 steps | — |
| pre-injection incidents | **0** | = 0 |

When the system claims, it names the **exact** step. It claims on 28 % of fault runs.

### M7 — full development quality, 169 runs

| Layer | Phase 1.6 | **Phase 1.9** |
|---|---|---|
| EPISODE precision / recall | 0.411 / 0.353 | 0.358 / 0.341 |
| INCIDENT precision / recall | 0.680 / 0.405 | **0.550 / 0.440** |
| RUN precision / recall | 0.972 / 0.412 | 0.956 / **0.506** |
| **healthy-run false incident rate** | 1.19 % | **2.38 %** |
| **false incidents / 1k steps** | 0.030 | **0.059** |
| pre-injection incidents | 0 | **0** |

**Two gates pass:** healthy-run FP 2.38 % (≤ 3 %) and false incidents 0.059 / 1k
(≤ 0.3). Uncertainty rates: healthy abstention 1.4 %, fault abstention 3.9 % — the
precision result is **not** obtained by hiding healthy runs in the abstention band.

## 2. What was NOT delivered

| Item | State |
|---|---|
| **M3** real restart E2E | **NOT DONE** |
| **M3.1** protected baseline survives restart | **NOT VERIFIED** |
| **M4** CLI persisted-data E2E | **NOT DONE** |
| **M8** full-pipeline smoke | **NOT RUN** |
| per-capability numerics quality | **NOT VERIFIED** |
| earliest evidence per-severity breakdown | **NOT VERIFIED** |
| declared-domain end-to-end measurement | **NOT VERIFIED** |

The evidence chain is **written but has never been read back**. Nothing verifies that
a persisted episode can be reconstructed, rendered or scored after the process that
wrote it exited.

## 3. The binding constraint has changed

```
injected fault detected at the incident layer : 44 %
healthy run falsely alarmed                   : 2.4 %
earliest claim made                           : 28 % of fault runs
earliest claim correct, when made             : 75 %
```

**Precision is no longer the problem. Coverage is.** The system is precise about what
it confirms and silent about most of what it does not — a coherent posture, since the
project's governing principle prefers abstention to a confident wrong incident, but
not a detection system with useful coverage.

The residual gap is the ×1.5–×12 region, where a sustained scaling is not separable
from healthy variation by any level statistic. The brief's own remedy — downstream
loss response as **corroborating**, never causal, evidence — was deliberately deferred
and remains the identified next step.

## 4. Tests

```
before:  404
added:     8
total:   412 passing, 1 skipped
```

New coverage: monotonicity reporter correctness (5 tests, including the exact Phase
1.8 curve and a 0.01 drop), pipeline episode/revision writes, pipeline idempotency
across a re-run, relation and incident persistence with ordering.

**One test is flaky under full-suite load:**
`test_accounting_identity_holds_under_queue_overflow` passes 3/3 in isolation and
failed once in a full run — it depends on queue-overflow timing and should be made
deterministic. Recorded rather than retried away.

## 5. Governing principle

> Phase 1.8 fixed the feature defect.
>
> **Phase 1.9 stopped adding detector ideas and connected the chain: the pipeline now
> writes its own episodes, relations and incidents; the consequence scorer and the
> earliest-evidence evaluator finally executed; and the earliest-evidence claim was
> measured for the first time — precise when made (0.0 step error), silent on 72 % of
> faults.**

The chain is written. It has not been read back. That is the next phase.

**No agent. No replay. No new diagnostic source. No causal claim.**
