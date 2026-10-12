# PHASE 1 CLOSURE REPORT

```
original status:  PARTIAL
closure status:   STILL PARTIAL
```

`PHASE_1_REPORT.md` is **unmodified**. No original gate was relaxed or redefined.

---

## 1. Release gate, item by item

| # | Gate item | Status | Evidence |
|---|---|---|---|
| 1 | remaining observer overhead attributed | **FAIL** | ledger residual negative at all 3 workloads; dispersion at large (±3.71 ms) exceeds the effect (+2.60 ms) |
| 2 | core observer large workload < 3 % | **FAIL** | +3.3 % (Phase 1, unchanged) |
| 3 | T1 overhead reported honestly on all 3 workloads | **PASS** | 17.9 / 5.4 / 3.3 %, incremental 0.00–0.02 ms, gate not redefined |
| 4 | incident persistence works | **PASS** | 13 tests |
| 5 | restart reconstruction works | **PASS** | separate interpreter, byte-identical render |
| 6 | evidence provenance survives persistence | **PASS** | `event_ids`, `detector`, `detector_version`, `rule_version`, `query_definition` |
| 7 | Trackio integrated | **PASS** | real trackio 0.40.0 on Python 3.12, 208 KB SQLite written |
| 8 | Trackio remains presentation-only | **PASS** | 60 forwarded / 20 filtered; allow-list held in the stored DB |
| 9 | Trackio failure cannot affect training | **PASS** | unavailable / throwing / slow / init failure, all isolated and counted |
| 10 | cross-run comparison works | **PASS** | 26 tests with C4 |
| 11 | lag-distribution correlation works | **PASS** | `[8, 9, 11]` verified |
| 12 | recurrence evidence works | **PASS** | `k / n` reported, `MIN_OCCURRENCES = 3` |
| 13 | BH correction works | **PASS** | canonical BH 1995 example: exactly 4 rejections |
| 14 | negative correlation evidence works | **PASS** | after a defect fix (see §3) |
| 15 | calibration corpus ≥ 50 real training runs | **PASS** | **66 runs**, all real PyTorch |
| 16 | precision / recall measured | **PASS** | P **0.632**, R **0.857**, FP **1.59 / 1k steps** |
| 17 | detector versions persisted | **PASS** | `rule_version` on every row |
| 18 | holdout prevents threshold leakage | **PASS** | seed%3 split, declared; no tuning performed |
| 19 | agent readiness explicitly decided | **PASS** | **NOT_READY** |
| 20 | no Phase 2 diagnostic dependency added | **PASS** | no DCGM, HTA, Flight Recorder, NNsight, LMResiliency, replay, multi-node, agent |

**18 / 20 pass. Two fail, both performance gates.**

## 2. Why STILL PARTIAL

The two failures are the same failure the original report recorded, and C0 did not
close it. Worse, C0's attempt produced a **negative residual**, which is
physically impossible and therefore disqualifies the ledger as a reconciliation
instrument rather than merely failing to attribute the residual. Details and three
identified root causes are in `C0_PERFORMANCE_REPORT.md`.

The **~0.8 ms residual is NOT VERIFIED**, and specifically it is *not* explained by
measurement error: at the largest workload the measurement dispersion exceeds the
quantity being measured.

## 3. Defects found and fixed in Phase 1.1

Every one was invisible to the Phase 1 test suite.

| # | Defect | Consequence if shipped |
|---|---|---|
| 1 | Calibration keyed results by `seed`, collapsing all 11 classes onto one run | Reported **recall 0.000** on a corpus where faults were detected. The harness was lying. |
| 2 | Cost ledger applied one model's component costs to all workloads | Negative residual — arithmetic that cannot be true. |
| 3 | `_chi2_sf` shipped an untested odd-k series expansion | A silently wrong p-value. Now raises instead. |
| 4 | Correlation routed low-occurrence pairs to `untestable` | Made `insufficient()` structurally empty, hiding every pair the engine could not assess — the exact "lying by omission" Phase 0 forbids. |
| 5 | `Hypothesis.from_dict` key mismatch | Persistence reconstruction raised. |
| 6 | `earliest_abnormal.candidates` missing `run_id`/`producer`/`layer_index` | Reconstruction lost the ordering evidence. |
| 7 | `TRACKIO_DIR` resolved at import time | Trackio reported success while persisting **nothing** in a restricted environment. |
| 8 | Phase 0's Trackio API notes wrong for 0.40.0 (`mode`, `log` signature) | Integration would have failed on first contact. |

Items 7 and 8 were only discoverable by **executing the real library**, and item 1
only by **running the corpus**. Neither was findable by reading.

## 4. What Phase 1.1 proved

> Phase 1 proved we can detect an incident.
> **Phase 1.1 proved we cannot yet trust the incident.**

The infrastructure to trust an incident — provenance, revisions, cross-process
durability, correlation with causation fenced off, and calibration with a holdout —
is now built and tested. What it immediately revealed is that the detectors are
much weaker than the test suite suggested: **the primary gradient detector has
precision 0.107 and fires on 6/6 clean runs.**

That is a successful outcome for a calibration phase and a failing outcome for the
detectors. Both are reported.

## 5. Unchanged from Phase 1

* `PHASE_1_REPORT.md` — not modified.
* All original gate targets — not relaxed.
* Out-of-scope list — nothing added: no DCGM, HTA, Flight Recorder, NNsight, LMResiliency, multi-node, replay, LLM agent, automatic actions or web dashboard.

## 6. Test suite

**295 passed, 1 skipped** (up from 256), covering unit, integration, scenario, fault
injection, performance, persistence, restart recovery, cross-run comparison,
correlation, calibration harness and Trackio failure isolation. No original
coverage was removed to make the suite faster.
