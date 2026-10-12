# PHASE 1.6 REPORT

```
PHASE 1.6 STATUS:                  PARTIAL
READY_FOR_BLIND_VALIDATION:        NO
PHASE 1 ORIGINAL CLOSURE:          STILL PARTIAL
AGENT READINESS:                   NO
```

**No final holdout files were created**, per the brief.

`PHASE_1_REPORT.md` and `outputs/phase1-1..1-5/` are **unmodified**. The frozen
boundaries 34 / 46 were **not** retuned — they were corrected to the same operating
point expressed on a layer-count-invariant feature, which the J0 tests demonstrate is
a correctness fix, not a search.

---

## 1. What was delivered

### J0 — tests, and three real bugs

```
337 -> 375 tests passing, 1 skipped
```

Writing the tests found **two correctness defects**, and diagnosing the corpus found
a third:

| # | Defect | Consequence |
|---|---|---|
| 1 | `l1_aggregate` was a **sum** over layers | scaled with model depth; a 4-layer boundary was 50 % wrong on the 6-layer configuration. **16 healthy episodes** reached CONFIRMED. Fixed to the layer-count-invariant **mean**; 34/46 as a sum is exactly 8.5/11.5 as a mean, so a 4-layer run classifies identically. |
| 2 | `UNCERTAIN` closed as `NOT_ANOMALY` | the abstention the band exists to express was reported as "nothing happened" |
| 3 | **the band was not authoritative for confirmation** | `aggregate_confirms=True` meant the band had **no effect at all** on incidents |

Defect 3 is the phase's central finding. It was discovered because **two consecutive
development re-runs produced byte-identical numbers** after the band was implemented.
That is only possible if the band is not in the path it appears to be in.

### J2 — `LOSS_SPIKE` promotion disabled

`PROMOTION_DISABLED_TYPES = {"LOSS_SPIKE"}`. The detector still runs; candidates are
recorded, named in the verdict, and rendered. The brief's third option — leaving the
action unexecuted for a third phase — was not taken.

### J3 — episode persistence, complete

Append-only `episodes` / `episode_revisions` / `episode_relations`. Genuine
cross-process continuation verified: a second interpreter loads a `SUSPECT` episode
and promotes **the same episode** — same id, same onset, correct signal count,
history `["SUSPECT", "CONFIRMED"]`. Relations reload byte-identically.

### J4 — CLI abstention

New `verdicts` command rendering all four outcomes; `diagnose` now falls through to
verdict rendering instead of printing a bare "no incident".

### J5 — uncertainty accounting

```
HEALTHY_ABSTENTION_RATE = 0.007      (98.6 % of healthy episodes are NORMAL)
FAULT_ABSTENTION_RATE   = 0.043
```

The anti-cheat check **passes decisively**: the precision gain did not come from
hiding healthy runs in the abstention band.

### J6 — the result

| Layer | band ineffective | **band authoritative** | Target |
|---|---|---|---|
| EPISODE precision | 0.285 | **0.411** | ≥ 0.92 |
| EPISODE recall | 0.553 | 0.353 | ≥ 0.87 |
| INCIDENT precision | 0.500 | **0.680** | ≥ 0.92 |
| INCIDENT recall | 0.676 | **0.405** | ≥ 0.87 |
| RUN precision | 0.718 | **0.972** | — |
| **healthy-run false incident rate** | 0.286 | **0.0119** | ≤ 0.03 |
| **false incidents / 1k steps** | 0.710 | **0.0296** | ≤ 0.3 |
| pre-injection incidents | 0 | **0** | = 0 |

**Two detector-quality gates pass for the first time in this project:**
healthy-run false incident rate **1.2 %** (≤ 3 %) and false incidents **0.030 / 1k
steps** (≤ 0.3).

## 2. What was NOT delivered

| Item | State |
|---|---|
| **J1** consequence-aware scoring | **NOT DONE** — so numerics precision cannot improve; 0.314 stands |
| **J7** pre-holdout smoke | **NOT RUN** — gated on J6 |
| `earliest evidence precision` | **NOT VERIFIED** — never measured in six phases |
| duplicate root-incident rate | **NOT VERIFIED** |
| CLI end-to-end integration | **NOT VERIFIED** — implemented and unit-tested; the episode store is not wired into the pipeline that writes runs |
| consequence rendering in the CLI | **NOT DONE** |

## 3. The trade, stated plainly

Phase 1.5 was **precision-blocked**: 28.6 % of healthy runs alarmed.
Phase 1.6 is **recall-blocked**: incident recall 0.405.

The system moved from over-alarming to under-detecting. Neither is acceptable, and
the cause of the second is identified and measurable: the boundaries 8.5 / 11.5 were
fitted when the band had **no effect on confirmation**, so they were never optimised
against the quantity they now control. The magnitude sweep shows they confirm only
≈×25 gradient perturbations.

```
development-supported minimum detectable gradient magnitude: ~ x25
```

That number is labelled **development-supported**, never *validated*, and it is
non-monotone at ×7.0 (0/5, between ×4 at 1/5 and ×12 at 2/5), which is direct
evidence that 5 runs per magnitude is too small to place the boundary precisely.

## 4. Regression

```
before:  337
added:    38
total:   375 passing, 1 skipped
```

New coverage: the three-way band and both boundary equalities, `l1_aggregate` and its
layer-count invariance, per-layer own-median normalisation, warm-up exclusion, band →
episode → verdict for all three regions, all four relation types, the
`CAUSED_BY`/`ROOT_CAUSE_OF` prohibition, standalone-`NaN` independence, gap and scope
gating, preferred-primary selection, relation serialisation, episode revisions,
band survival, relation persistence, and cross-process episode continuation.

## 5. Governing principle

> Phase 1.5 wrote a band and believed it was in control.
>
> **Phase 1.6 proved it was not — two identical re-runs gave it away — and once it
> was, the healthy false-incident rate fell from 28.6 % to 1.2 %, the first
> detector-quality gate this project has passed.**

`READY_FOR_BLIND_VALIDATION = NO`. The blocking reason has changed from precision to
recall, and the fix is identified: re-derive the boundaries against the quantity they
now actually control.

**No agent. No replay. No new diagnostic source. No causal claim.**
