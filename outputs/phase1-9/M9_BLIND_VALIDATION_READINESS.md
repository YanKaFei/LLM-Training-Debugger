# M9 — BLIND VALIDATION READINESS

# **READY_FOR_BLIND_VALIDATION = NO**

# **DECLARED_DOMAIN_READY = NO**

No final holdout files were created.

`DECLARED_DOMAIN_READY` is also NO: the declared domain (≥ ×20) has development recall
0.95–1.00, but the declared-domain figures were taken from the 680-run synthetic sweep,
not from the full pipeline on the development split, so no end-to-end declared-domain
measurement exists.

---

## 1. Gate

| # | Requirement | State |
|---|---|---|
| 1 | monotonicity contradiction resolved | **PASS** — 4 inversions, tolerance 0.0, tested |
| 2 | capability scope documented honestly | **PASS** |
| 3 | `EpisodeStore` wired to the production pipeline | **PASS** — 157 files written, 0 errors |
| 4 | protected gradient baseline survives restart | **NOT VERIFIED** |
| 5 | real restart E2E | **FAIL** — M3 not executed |
| 6 | CLI persisted-data E2E | **FAIL** — M4 not executed |
| 7 | consequence-aware scorer executed | **PASS** — with a corrected wiring defect |
| 8 | numerics current quality measured correctly | **FAIL** — per-capability breakdown not computed; 0.314 retired |
| 9 | earliest evidence actually measured | **PASS** — first measurement in the project |
| 10 | earliest precision ≥ 0.92 | **FAIL** — 0.750 |
| 11 | earliest coverage ≥ 0.80 | **FAIL** — 0.282 |
| 12 | pre-injection confirmed incidents = 0 | **PASS** |
| 13 | development episode precision ≥ 0.92 | **FAIL** — 0.358 |
| 14 | development episode recall ≥ 0.87 | **FAIL** — 0.341 |
| 15 | development incident precision ≥ 0.92 | **FAIL** — 0.550 |
| 16 | development incident recall ≥ 0.87 | **FAIL** — 0.440 |
| 17 | healthy-run false incident rate ≤ 3 % | **PASS** — 2.38 % |
| 18 | duplicate top-level root incidents = 0 | **FAIL** — 23, rate 0.057 |
| 19 | full-pipeline smoke PASS | **FAIL** — M8 not executed |
| 20 | no final holdout created | **PASS** |
| 21 | no Phase 2 component | **PASS** |
| 22 | no Agent | **PASS** |

**10 PASS / 9 FAIL / 3 NOT VERIFIED.**

## 2. What this phase actually produced

The brief's instruction was explicit:

> 本阶段不再主要优化 detector。把已经存在但彼此断开的 Episode / Relation /
> Incident / Persistence / CLI / Evaluation 真正连接成一条可运行、可恢复、可评分的系统。

**Connected:** the pipeline now writes episodes, revisions, relations and incidents
through the stores (M2, 157 files, 0 errors, ordering and idempotency both tested);
the consequence scorer and the earliest-evidence evaluator now **execute against real
results** (M5, M6).

**Not connected:** restart, CLI, and the smoke corpus. The evidence chain is written
but has never been read back.

**And one long-standing defect was closed:** the monotonicity reporter was found to be
hiding real decreases behind an arbitrary 0.25 tolerance, so a phase-long claim of
"0 violations" was an artefact of the instrument.

## 3. The honest position

```
injected fault detected at the incident layer : 44 % of the time
healthy run falsely alarmed                   : 2.4 % of the time
earliest claim made                           : 28 % of fault runs
earliest claim correct, when made             : 75 %
```

The system is **precise about what it confirms and silent about most of what it does
not**. That is a coherent, defensible operating posture — it prefers abstention to
a confident wrong incident, which is the project's governing principle — but it is not
a detection system with useful coverage, and it cannot be blind-validated as one.

## 4. Blocking reasons, ordered

1. **Coverage, not precision, is now the binding constraint.** Episode recall 0.341 and earliest coverage 0.282. The residual gap is the ×1.5–×12 region where a sustained scaling is not separable from healthy variation by a level statistic. The brief's own suggested remedy — downstream loss response as *corroborating* (never causal) evidence — was deliberately deferred to a later phase and remains the identified next step.
2. **23 duplicate top-level root incidents (rate 0.057).** A consequence is still claimed as an independent cause in ~6 % of observations. The relation model exists; the evaluator does not consume it (M5 §3).
3. **The evidence chain has never been read back.** Restart E2E, CLI E2E and the smoke corpus are all unexecuted, so nothing verifies that a persisted episode can be reconstructed, rendered or scored after the process that wrote it has exited.
4. **Per-capability quality is unmeasured.** Numerics in particular: 0.314 is retired and nothing replaces it.
5. **One test is flaky under load.** `test_accounting_identity_holds_under_queue_overflow` passes 3/3 in isolation and failed once in a full-suite run; it depends on queue-overflow timing and should be made deterministic.

## 5. Recommendation

**Do not spend a holdout.** The correct next phase is the one this phase did not
finish: **read the evidence back**. M3 (restart), M4 (CLI), M8 (smoke), the
per-capability breakdown, and making the consequence relation visible to the scorer.
Those are engineering completions of a chain that now exists, not new detector ideas,
and until they are done a blind holdout would measure a system whose own stored
evidence has never been verified to survive a restart.
