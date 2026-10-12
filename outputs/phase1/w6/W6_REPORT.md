# W6 REPORT — Incident Engine, Evidence, CLI

**Wave:** W6
**Status:** **COMPLETE**
**Gate:** A1 on real training passes end to end — **PASS** (19 A1 tests)

---

## 1. What was built

```
src/training_debugger/incidents/
  engine.py            IncidentEngine: detection -> incident
  evidence_builder.py  four evidence buckets + negative evidence
  ranking.py           hypothesis library + decomposable scoring
  render.py            deterministic renderer (not an LLM)
src/training_debugger/cli/main.py   runs | incidents | incident | metrics | diagnose
tests/integration/test_a1_end_to_end.py   19 tests
```

---

## 2. A1 — the release gate, on real training

A real 4-layer transformer trains for 130 steps with `SGD(lr=0.05)`. At step 80
every gradient is multiplied by 40 — a genuine gradient explosion, not a scaled
series. Real autograd then propagates the consequence.

```
INCIDENT inc_2e092777ff6d4f7b

Type:
  GRADIENT_EXPLOSION  (severity error, state OPEN)

Observed symptom:
  grad.global_norm = 50.2139 at step 80 (grad.robust_zscore)
  detector: grad.robust_zscore v1.0 (confidence 0.80)

Abnormal evidence, earliest first:
  [inferred] step 80: gradient norm 13.5371 is 8.0x the largest value in the
             preceding window (1.69198); robust z-score 107.7
  ... five further layers, all at step 80

Affected:
  affected: 5 / 7 layers
  layers: 2, 3, 4, 5, 7

Not correlated (tested and found stable):
  - learning rate remained within baseline at the incident (0.05, +0.0%)
  - parameter norm remained within baseline at the incident (22.4212, +0.7%)
  - throughput (tokens/s) remained within baseline at the incident (7124.94, -0.5%)
  - step time remained within baseline at the incident (4.49127, +0.5%)

Not assessable:
  - allocated memory could not be assessed: this signal was not recorded

Earliest defensible abnormal event:
  Unique earliest abnormal event: grad.global_norm on pytorch-adapter rank 0.
  Every other abnormal event in scope is provably preceded by it.

Probable root cause (ranked, deterministic scores):
  0.525  [INFERRED] gradient.explosion_observed
         score = multiplicity_penalty=0.5, prior=0.35,
                 support:gradient_anomaly_strong=3
```

This text is produced by `render_incident()`, which is a pure function of the
incident's fields. It is deterministic, and `test_rendering_is_deterministic`
asserts it. **No language model is involved.**

---

## 3. The partial-order result

`EventOrdering` returns a single `first_abnormal_event_id` **only** when the
minimal element of the partial order is unique. In the A1 run all findings come
from one producer, so `source_seq` orders them and the answer is unique and
`PROVEN`.

The three required cases are each tested:

| Case | Behaviour | Test |
|---|---|---|
| unique earliest | `first_abnormal_event_id` set, `PROVEN` | `test_case_1_unique_earliest_is_proven` |
| several proven-concurrent | id `None`, candidates listed, `PROVEN` | `test_case_2_concurrent_candidates_yield_no_single_first` |
| ordering not establishable | id `None`, `UNKNOWN`, explanation says so | `test_case_3_missing_evidence_yields_unknown_ordering` |

The rendered refusal reads exactly as the brief requires:

> 2 abnormal events are mutually unordered, so no single event can be named first
> … Their exact ordering cannot be established from the available evidence.

**A producer ordering-violation guard was added.** `source_seq` records *emission*
order, which equals *training* order only if the producer emits in order. When a
producer's sequence and its declared step positions disagree, `compare()` abstains
rather than manufacturing a precedence from emission order.

---

## 4. Evidence: the four buckets

`build_evidence` probes a fixed candidate set for every incident and records each
outcome:

| Signal | Probed | Result in A1 |
|---|---|---|
| learning rate | yes | **not correlated** (+0.0 %) |
| parameter norm | yes | **not correlated** (+0.7 %) |
| throughput (tokens/s) | yes | **not correlated** (−0.5 %) |
| step time | yes | **not correlated** (+0.5 %) |
| allocated memory | yes | **unknown** — not recorded on this backend |
| global gradient norm | yes | supporting (it is the anomaly) |

`Incident.validate()` **rejects** an incident with no negative evidence and no
stated missing evidence, so "we tested nothing" cannot be rendered as "nothing was
wrong".

---

## 5. Ranking

Five hypothesis templates, each a dataclass (not a prompt), scored by arithmetic
over facts derived from the evidence, with **every** factor stored in
`score_decomposition`. A multiplicity penalty divides by `sqrt(applicable
templates)` so a vague hypothesis cannot win by being vague.

Causal claims are capped by `max_level`:

| Template | Cap | Why |
|---|---|---|
| `numerics.nonfinite` | `OBSERVED` | a finiteness finding is measured, not inferred |
| `gradient.explosion_observed` | `INFERRED` | statistical |
| `gradient.explosion_precedes_loss` | `INFERRED` | ordering + statistics |
| `optim.lr_change` | `INFERRED` | statistic |
| `throughput.regression` | `INFERRED` | statistic |
| **`data.outlier_batch`** | **`HYPOTHESIS`** | **no replay exists to confirm it** |

`test_phase1_never_claims_a_confirmed_data_cause` asserts the cap, so a future
change cannot quietly promote a data-causation claim without adding replay.

---

## 6. Defects the CLI exposed

Running `diagnose` over a stored run immediately surfaced three defects that the
unit tests had not:

| Defect | Symptom | Fix |
|---|---|---|
| **Every evidence line read "step 1"** | the renderer used `evidence.window[0]`, which is the start of the *baseline window*, not the anomaly's step | the anomaly's own step is recorded as `metadata.onset_step` and the renderer uses it |
| **"affected: 0 / 1 ranks"** | no finding carried a rank, but the population was reported as 1, reading as "the rank was fine" when nothing was assessed | the rank population is zeroed when no finding names a rank |
| **A GRADIENT_EXPLOSION incident ranked no gradient hypothesis** | `gradient.explosion_precedes_loss` only applied to LOSS_* incidents | added `gradient.explosion_observed`, which raised the top score from 0.144 to **0.525** |

The first is the most instructive: it produced output that *looked* plausible and
was wrong, which is exactly the failure mode Phase 0's top risk names.

---

## 7. Gate results

| Gate item | Status |
|---|---|
| incident lifecycle (OPEN/UPDATED/RESOLVED/CLOSED) | **PASS** |
| evidence levels strictly enforced | **PASS** — `Evidence` rejects INFERRED without a detector |
| negative evidence mandatory | **PASS** — enforced by `Incident.validate` |
| multiplicity implemented | **PASS** — ratio always reported |
| earliest abnormal evidence handles partial ordering | **PASS** — all three cases |
| A1 real training scenario passes | **PASS** — 19 tests |
| deterministic CLI diagnosis | **PASS** — renderer is a pure function |
| no Agent logic inside detectors | **PASS** — `test_layering.py` |
| no storage logic inside adapters | **PASS** — `test_layering.py` |
| no deferred v0.2 component in core | **PASS** — no NNsight, HTA, DCGM, Flight Recorder, replay |

---

## 8. Not verified in W6

* **Multi-rank ordering.** The `CONCURRENT_OR_UNKNOWN` paths are unit-tested, but every end-to-end run is single-process. A real two-rank incident has not been produced.
* **Cross-run comparison.** `compare_runs` was specified in Phase 0 and is **not implemented**.
* **Correlation beyond the fixed candidate set.** `build_evidence` probes six signals; a general lag-distribution correlation engine (Phase 0 section 3) is **not implemented**.
* **Incident persistence.** Incidents serialise to JSON (`test_incident_serialises_for_storage`) but are not written to a store; `diagnose` recomputes them from events each time.
* **Trackio integration.** Not done. See `PHASE_1_REPORT.md`.
