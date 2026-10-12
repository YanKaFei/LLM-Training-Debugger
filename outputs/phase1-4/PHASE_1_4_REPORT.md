# PHASE 1.4 REPORT

```
PHASE 1.4 STATUS:      PARTIAL
PHASE 1 ORIGINAL:      STILL PARTIAL (performance gate unchanged)
AGENT READINESS:       NO
```

`PHASE_1_REPORT.md` and `outputs/phase1-1..1-3/` are **unmodified**.

---

## 1. Release gate

| # | Gate item | Status |
|---|---|---|
| 1 | Episode/fusion/registry architecture has formal tests | **PASS** — 295 → **337** tests |
| 2 | cross-process episode reconstruction tested | **PASS** (as dicts; **not** via the incident store) |
| 3 | capability labels corrected | **PASS** — four-state model, `validated_capabilities: []` |
| 4 | gradient boundary overlap quantitatively analysed | **PASS** |
| 5 | univariate tuning is no longer the only gradient evidence | **FAIL** — analysis done, detector unchanged |
| 6 | explicit uncertainty/abstention region exists | **FAIL** — R2.4 not implemented |
| 7 | `H1` healthy gradient incident rate ≤ 5 % | **FAIL** — unchanged at 11/14 |
| 8 | downstream-response evidence implemented | **FAIL** — R3 not started |
| 9 | consequence attribution implemented | **FAIL** — R4 not started |
| 10 | secondary NaN/Inf does not create a duplicate incident | **FAIL** |
| 11 | independent NaN/Inf can still open its own incident | **PASS** (unchanged behaviour) |
| 12 | `H2` noisy loss ≤ 5 % **or** `LOSS_SPIKE` non-default | **FAIL** — neither |
| 13 | `LOSS_DIVERGENCE` validated or UNSUPPORTED | **PASS** — **UNSUPPORTED** |
| 14 | throughput validated or UNSUPPORTED | **PASS** — **UNSUPPORTED** |
| 15 | abstention visible in the renderer | **PASS** — but not wired to the CLI |
| 16 | development quality target reached before holdout | **FAIL** — R9 not run |
| 17 | pre-holdout smoke passes | **FAIL** — not run |
| 18 | final code/config/rules frozen and hashed | **FAIL** — nothing frozen |
| 19 | final holdout not reused after a fix | **PASS** — no holdout was run |
| 20 | final holdout ≥ 150 runs | **FAIL** — not run |
| 21–28 | episode/incident precision, recall, FP rate, earliest evidence, pre-injection, duplicates | **NOT VERIFIED** — no holdout |
| 29 | no Phase 2 component added | **PASS** |
| 30 | no LLM Agent added | **PASS** |

**8 PASS / 12 FAIL / 8 NOT VERIFIED → PHASE 1.4 = PARTIAL.**

---

## 2. What was delivered

**R0 — tests first. 295 → 337 (42 new).** This was the phase's explicit first
priority and it is closed. Writing the tests found **two real defects** that direct
execution had missed: `NAN_DETECTED`/`INF_DETECTED` were absent from
`promote.strong.single`, so a numerics episode classified `STRONG` rather than
`CERTAIN` could **never** open an incident; and `CANDIDATE_ONLY` recorded no
episodes, so the renderer had nothing to show.

**R1 — capability labels corrected.** `SUPPORTED`/`UNSUPPORTED` replaced by
`IMPLEMENTED`/`EXPERIMENTAL`/`VALIDATED`/`UNSUPPORTED`. Phase 1.3 labelled
`GRADIENT_EXPLOSION` SUPPORTED at episode precision 0.181; it is now
**EXPERIMENTAL**. `validated_capabilities` is **empty** — nothing has passed a
holdout that has never been looked at.

**R2.3 — the boundary analysis, and it is the phase's substantive result.**
Measured on 48 real runs, threshold-independently:

| Feature | AUC ×1.5 | ×2.0 | ×2.5 | ×3.0 |
|---|---|---|---|---|
| `global_norm_ratio` — Phase 1.3's feature | **0.25** | **0.29** | 0.58 | 0.92 |
| `l1_aggregate` — joint | 0.62 | 0.83 | **0.96** | **1.00** |

**The univariate feature is worse than chance below ×2.** It is not insensitive at
low magnitude, it is *inverted*: a mild fault produces a lower global-norm ratio than
a healthy heavy-tail run, because the ratio is depressed by the healthy class's
elevated trailing median. A threshold cannot be tuned to fix an inverted signal.

Joint aggregation — normalising each layer by its own baseline and aggregating —
separates cleanly at **≥ ×3.0** (AUC 1.00, no overlap). `trimmed_l2` matches it,
confirming the signal is distributed rather than carried by one outlier.
`top_k_mass` and `max_over_median` separate at **no** magnitude and are reported as
negative results.

**Minimum reliably detectable gradient perturbation ≈ ×3.0**; below ≈×2.5 the classes
are not separable on the signals already collected, and the brief's required
conclusion applies: **abstain**.

**R8 — abstention is user-visible.** All four verdicts render, with observed scope
(`1/7 monitored layers`), both moments (onset vs confirmation), what was missing, and
an explicit statement that `NO_ANOMALY` is not an abstention. Causal language is
guarded by test.

## 3. What was NOT delivered

| Item | State |
|---|---|
| **R2.1/R2.2/R2.4** joint feature in the detector, uncertainty band, frozen A/B | **NOT IMPLEMENTED** |
| **R3** downstream corroboration | **NOT STARTED** |
| **R4** consequence attribution / `IncidentRelation` | **NOT STARTED** |
| **R5** loss spike repair | **NOT STARTED** |
| **R6** `LOSS_DIVERGENCE` | **NOT STARTED** (remains UNSUPPORTED) |
| **R7** throughput audit and repair | **NOT STARTED** (remains UNSUPPORTED) |
| **R9** development evaluation | **NOT RUN** |
| **R10** pre-holdout smoke, freeze, final holdout | **NOT RUN** |
| **R11** earliest evidence quality | **NOT MEASURED** |
| **R12** fresh paired performance benchmark | **NOT RE-RUN** |

The reason is scope against available effort, and it is stated rather than papered
over. The **analysis** that justifies R2's change exists and is decisive; the
**change** does not.

Notably, R4 was the phase's identified fix for numerics precision 0.314 — a
fault-induced `NaN` raising its own top-level incident — and it was not built, so
that number is unchanged.

## 4. Consequence: nothing measurable improved

Because the detector, episode rules and fusion were unchanged apart from two bug
fixes, **every Phase 1.3 quality number stands**:

```
episode precision  0.195      incident precision  0.412
episode recall     0.588      incident recall     0.806
healthy-run false incident rate      0.476
earliest-evidence precision          0.420
pre-injection confirmed incidents    0         <- preserved
duplicate persistent incidents       0         <- preserved
```

Phase 1.4 did not improve reliability. It replaced a false capability claim with an
accurate one, added the regression protection the previous phase lacked, and
**produced the measurement that says what the next change must be**.

## 5. Final holdout discipline

`FINAL_HOLDOUT_LOCK.json` records `status: NOT_FROZEN`. The pre-holdout smoke corpus
was not run and the R2.4/R4/R5/R6/R7 changes were not implemented, so there was
nothing worth freezing — and burning a genuine holdout on a code base with
known-unimplemented work would have destroyed it for no measurement value.

Phase 1.3's reported holdout remains **invalid as an untouched evaluation**, for the
reason recorded in `outputs/phase1-3/invalidated_first_run/README.md`. Phase 1.4 does
not repair that and does not claim a new one. **The holdout gates are NOT VERIFIED.**

## 6. Tests

```
before:  295
added:    42
total:   337 passing, 1 skipped
```

Key new regressions: persistent `Inf` → one episode → one incident; `SUSPECT →
NORMAL`; `candidate_onset != confirmed_at`; `CERTAIN` confirms at onset; single WEAK
does not promote; concordant WEAK does; `INSUFFICIENT_EVIDENCE`; `NO_ANOMALY`;
candidate-only and abstention rendering; promotion and rejection provenance; registry
statuses and old-id resolution; disabled detectors stay disabled but remain
observable; cross-process episode reconstruction.

## 7. Governing principle

> Phase 1.3 proved knowing *when* an anomaly begins does not solve classification
> when healthy and faulty signals overlap.
>
> **Phase 1.4 proved the gradient signal was not merely weak below ×2 — it was
> inverted, and that joint aggregation recovers separation at ×3.**

The brief's instruction for the overlap region is abstention, and the measurement now
says exactly where that region is. Implementing it is the next phase's first task.
**No agent. No replay. No new diagnostic source. No causal claim.**
