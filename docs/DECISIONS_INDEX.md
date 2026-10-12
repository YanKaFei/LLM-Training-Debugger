# DECISIONS INDEX

A readable index over [`DECISIONS.md`](DECISIONS.md), which is **append-only**: each entry
records what was decided, why, and what it rules out. Nothing in it is rewritten, including
entries that later turned out to be wrong — those are corrected by a later entry and left
standing.

**This index is not a changelog.** It is the reasoning behind the code, grouped by the
question each decision answers. If you want to know why something is the way it is, start
here.

---

## Corrections and retractions at a glance

The most useful page in this repository. A monitor whose own claims are not auditable
cannot be trusted to audit anything else.

| Claim that was made | What was wrong | Disposition |
|---|---|---|
| `numerics` precision = 0.314 | Semantic error in what was being counted. | **Retired.** Never quoted. |
| `primary_recall` = 1.0 | `MISSED_PRIMARY` was never incremented. | Corrected to **0.50** (D-017). |
| Relation precision/recall = 1.0 | The mechanism chain was keyed on a float32 reduction artifact. | Void as a general law (D-025, D-026). |
| “23 duplicate incidents” | Produced by the scorer; never reproduced. | `HISTORICAL_23_NOT_FULLY_REPRODUCED`. |
| “11-step latency” | Arithmetic was inclusive and the prose was wrong. | Corrected to **10**; no timestamp altered (D-035). |
| Latency cause = baseline readiness | The detector's own warm-up gate was the binding constraint. | Corrected (D-038); D-031 superseded. |
| Event volume = largest remaining lever | Micro-cost × count ≠ in-situ cost. | Not established; corrected (D-043). |
| Final holdout = untouched evaluation | Three attempts, each invalidated by a defect. | `NOT_FROZEN`, `NOT DONE — NOT VERIFIED` (D-005). |

---

## What may be claimed, and what may not

| | Decision |
|---|---|
| [`D-003`](DECISIONS.md#d-003--correlation-is-never-causation) | Correlation is never causation: `CAUSED_BY` / `ROOT_CAUSE_OF` do not exist in the type system, and a test asserts they stay absent. |
| [`D-004`](DECISIONS.md#d-004--an-incident-group-is-not-a-proven-root-cause) | An incident group is not a proven root cause; `has_proven_root_cause` is hard-coded `False`. Grouping is presentation. |
| [`D-007`](DECISIONS.md#d-007--abstention-is-a-legitimate-output) | Abstention is a legitimate output. Where healthy and faulty distributions overlap, the system says so. |
| [`D-019`](DECISIONS.md#d-019--relation-association-by-temporal-proximity-is-not-acceptable) | Relation association by temporal proximity is not acceptable: it fired on 10 of 10 deliberately independent fault pairs and 0 of 10 genuine chains. |
| [`D-021`](DECISIONS.md#d-021--mechanism-not-proximity-may-claim-a-consequence) | Only mechanism, not proximity, may claim a consequence. `PRECEDES` asserts order and never consequence. |
| [`D-023`](DECISIONS.md#d-023--a-mechanism-claim-requires-the-propagation-at-every-stage) | A mechanism claim requires the propagation at **every** stage. A suppressed update produces the same timing pattern while nothing propagates. |
| [`D-025`](DECISIONS.md#d-025--the-v013-relation-result-was-generation-signature-dependent) | The v0.1.3 relation result was generation-signature dependent, and is never to be quoted as a general law. |
| [`D-026`](DECISIONS.md#d-026--a-non-finite-norm-is-not-a-non-finite-tensor) | A non-finite norm is not a non-finite tensor. The mapping is definitional; v0.1.3/v0.1.4 keyed a whole chain on a float32 reduction artifact. |
| [`D-027`](DECISIONS.md#d-027--a-witnessed-optimizer-call-is-not-an-applied-parameter-update) | A witnessed optimizer call is not an applied parameter update. `UNKNOWN` is never rendered as `False`. |
| [`D-034`](DECISIONS.md#d-034--possible_consequence_of-is-disabled-by-default) | `POSSIBLE_CONSEQUENCE_OF` is disabled by default. No evidence is deleted — only the claim changes. |

## Evidence architecture

| | Decision |
|---|---|
| [`D-002`](DECISIONS.md#d-002--parquet-is-the-telemetry-truth-duckdb-queries-it) | Parquet is the telemetry truth and DuckDB queries it. No database was introduced, because the trigger for one never became true. |
| [`D-005`](DECISIONS.md#d-005--evidence-observation-is-separated-from-alert-deduplication) | Evidence observation is separated from alert deduplication: a detector cooldown may not suppress an observation. |
| [`D-006`](DECISIONS.md#d-006--idempotency-is-at-step-layer-scoped-per-signal_key) | Idempotency is at `(step, layer)`, scoped per `signal_key`. Not `(step)`, which broke cross-process restart. |
| [`D-020`](DECISIONS.md#d-020--the-signal-key-identifies-the-signal-not-the-detector) | The signal key identifies the signal, not the detector. Otherwise metric identity is lost and the mechanism evidence with it. |
| [`D-024`](DECISIONS.md#d-024--grouping-and-keying-must-agree-on-scope) | Grouping and keying must agree on scope, or the key is decoration. This one silently replaced an upstream observation with its witness. |

## Scoring integrity

| | Decision |
|---|---|
| [`D-015`](DECISIONS.md#d-015--a-scorer-that-reads-00-without-an-error-is-the-most-dangerous-kind) | A scorer that reads 0.0 without an error is the most dangerous kind. Two silent causes, each filing every correct detection as a false positive. |
| [`D-016`](DECISIONS.md#d-016--one-denominator-per-level-and-never-an-episode-count-as-a-run-count) | One denominator per level, and never an episode count as a run count. 42 runs had produced 97 “records”. |
| [`D-017`](DECISIONS.md#d-017--primary_recall-was-always-10-and-is-now-050) | `primary_recall` was always 1.0 because `MISSED_PRIMARY` was never incremented. Corrected to 0.50; earlier quotations are void. |
| [`D-018`](DECISIONS.md#d-018--scoring-fails-closed) | Scoring fails closed. A plausible number is worse than a refusal. |
| [`D-022`](DECISIONS.md#d-022--a-miss-must-never-leave-the-denominator-silently) | A miss must never leave the denominator silently. The identical bug was found twice, a version apart. |

## Detector and baseline behaviour

| | Decision |
|---|---|
| [`D-010`](DECISIONS.md#d-010--keep-the-baseline-resume-fix-despite-a-measured-cost) | Keep the baseline resume fix despite a measured cost, because a documented parameter must not be dead code. |
| [`D-013`](DECISIONS.md#d-013--the-onset-is-the-onset-of-the-abnormality-not-of-the-contiguity-run) | The onset is the onset of the abnormality, not of the contiguity run. |
| [`D-014`](DECISIONS.md#d-014--resume_after_steps-stays-at-40-the-sweep-selected-nothing) | `resume_after_steps` stays at 40: a pre-registered 216-run sweep selected nothing. The parameter is inert and recall is the binding constraint. |
| [`D-031`](DECISIONS.md#d-031--the-11-step-latency-was-baseline-readiness) | The “11-step latency” was baseline readiness. `PROPAGATION_DELAY_STEPS` was deliberately **not** changed to compensate. |
| [`D-032`](DECISIONS.md#d-032--c3-is-blocked-by-the-threshold-to-stability-ratio-not-by-injection-magnitude) | C3 is blocked by the threshold-to-stability ratio, not by injection magnitude. Increasing magnitude was tried and is explicitly not the remedy. |
| [`D-035`](DECISIONS.md#d-035--the-11-step-latency-was-really-10-and-the-report-was-wrong-not-the-data) | The latency was really 10, not 11. The prose was corrected and **no timestamp was altered** to make the wording consistent. |
| [`D-038`](DECISIONS.md#d-038--correction-the-detection-latency-is-the-detectors-warm-up-gate-not-the-baseline) | Correction: the latency is the detector's own warm-up gate, not the baseline. Both were 70, which is why the wrong diagnosis looked confirmed. |
| [`D-039`](DECISIONS.md#d-039--the-70-step-blind-window-is-load-bearing-c3-is-out-of-reach-by-design) | The 70-step blind window is load-bearing. C3 is out of reach by design, because making it constructible costs 0/6 → 6/6 healthy false fires. |
| [`D-040`](DECISIONS.md#d-040--baseline_mode-stays-frozen) | `baseline_mode` stays `“frozen”`: the EMA alternative was measured indistinguishable in every configuration tested. |
| [`D-043`](DECISIONS.md#d-043--t1-layer-observability-cadence-and-a-correction-to-d-042) | T1 layer observability cadence, and a correction to D-042: micro-cost × count does not equal in-situ cost. |

## Measurement discipline

| | Decision |
|---|---|
| [`D-011`](DECISIONS.md#d-011--a-single-scenario-smoke-must-not-assert-a-statistical-property) | A single-scenario smoke must not assert a statistical property. It tests engineering correctness; a separate probe tests detector quality. |
| [`D-030`](DECISIONS.md#d-030--a-case-that-cannot-be-constructed-is-a-finding-not-a-gap-to-paper-over) | A case that cannot be constructed is a finding, not a gap to paper over. No simulation was substituted for it. |
| [`D-041`](DECISIONS.md#d-041--neuraldbgs-causal-layer-is-the-rule-this-project-already-falsified) | A competitor's causal layer is the rule this project already falsified at 10/10 fabricated associations. |
| [`D-042`](DECISIONS.md#d-042--observer-overhead-reduced-30--the-3--gate-failure-is-a-workload-artifact) | Observer overhead reduced ~30 %; the `< 3 %` gate failure is a workload artifact, reported as a failure and not excused. |

## Capability status and release

| | Decision |
|---|---|
| [`D-008`](DECISIONS.md#d-008--capability-states-are-not-promotional) | Capability states are not promotional. Nothing is `VALIDATED` without an untouched blind holdout. |
| [`D-009`](DECISIONS.md#d-009--alpha-is-a-packaging-milestone-not-a-reliability-claim) | Alpha is a packaging milestone, not a reliability claim. |
| [`D-012`](DECISIONS.md#d-012--alpha-asserts-installability-never-reliability) | Alpha asserts installability, never reliability. Passing the first is never reported as progress on the second. |
| [`D-028`](DECISIONS.md#d-028--do-not-integrate-a-witness-whose-evidence-has-not-faced-its-adversarial-cases) | Do not integrate a witness whose evidence has not faced its adversarial cases. |
| [`D-029`](DECISIONS.md#d-029--the-optimizer-witness-is-kept-research-only-measured-benefit-is-exactly-zero) | The optimizer witness is kept research-only: measured benefit is exactly zero, and the reason is structural. |
| [`D-033`](DECISIONS.md#d-033--the-optimizer-witness-is-frozen) | The optimizer witness is frozen. Not rejected — insufficient, not wrong. |
| [`D-036`](DECISIONS.md#d-036--package-version-017a1-maps-to-the-github-tag-v017-alpha) | Package version `0.1.7a1` maps to the GitHub tag `v0.1.7-alpha`. Metadata alignment, not a new development version. |

## Repository hygiene

| | Decision |
|---|---|
| [`D-001`](DECISIONS.md#d-001--no-forks-of-upstream-projects) | No forks of upstream projects. They are integrated behind adapters or deferred. |
| [`D-037`](DECISIONS.md#d-037--generated-experiment-artefacts-must-not-be-published) | Generated experiment artefacts must not be published. String scanning alone was not treated as sufficient: the built sdist was extracted and searched. |

---

## Every decision, in order

| | Title |
|---|---|
| [`D-001`](DECISIONS.md#d-001--no-forks-of-upstream-projects) | No forks of upstream projects |
| [`D-002`](DECISIONS.md#d-002--parquet-is-the-telemetry-truth-duckdb-queries-it) | Parquet is the telemetry truth; DuckDB queries it |
| [`D-003`](DECISIONS.md#d-003--correlation-is-never-causation) | Correlation is never causation |
| [`D-004`](DECISIONS.md#d-004--an-incident-group-is-not-a-proven-root-cause) | An incident group is not a proven root cause |
| [`D-005`](DECISIONS.md#d-005--evidence-observation-is-separated-from-alert-deduplication) | Evidence observation is separated from alert deduplication |
| [`D-006`](DECISIONS.md#d-006--idempotency-is-at-step-layer-scoped-per-signal_key) | Idempotency is at `(step, layer)`, scoped per `signal_key` |
| [`D-007`](DECISIONS.md#d-007--abstention-is-a-legitimate-output) | Abstention is a legitimate output |
| [`D-008`](DECISIONS.md#d-008--capability-states-are-not-promotional) | Capability states are not promotional |
| [`D-009`](DECISIONS.md#d-009--alpha-is-a-packaging-milestone-not-a-reliability-claim) | Alpha is a packaging milestone, not a reliability claim |
| [`D-010`](DECISIONS.md#d-010--keep-the-baseline-resume-fix-despite-a-measured-cost) | Keep the baseline resume fix despite a measured cost |
| [`D-011`](DECISIONS.md#d-011--a-single-scenario-smoke-must-not-assert-a-statistical-property) | A single-scenario smoke must not assert a statistical property |
| [`D-012`](DECISIONS.md#d-012--alpha-asserts-installability-never-reliability) | Alpha asserts installability, never reliability |
| [`D-013`](DECISIONS.md#d-013--the-onset-is-the-onset-of-the-abnormality-not-of-the-contiguity-run) | The onset is the onset of the abnormality, not of the contiguity run |
| [`D-014`](DECISIONS.md#d-014--resume_after_steps-stays-at-40-the-sweep-selected-nothing) | `resume_after_steps` stays at 40; the sweep selected nothing |
| [`D-015`](DECISIONS.md#d-015--a-scorer-that-reads-00-without-an-error-is-the-most-dangerous-kind) | A scorer that reads 0.0 without an error is the most dangerous kind |
| [`D-016`](DECISIONS.md#d-016--one-denominator-per-level-and-never-an-episode-count-as-a-run-count) | One denominator per level, and never an episode count as a run count |
| [`D-017`](DECISIONS.md#d-017--primary_recall-was-always-10-and-is-now-050) | `primary_recall` was always 1.0 and is now 0.50 |
| [`D-018`](DECISIONS.md#d-018--scoring-fails-closed) | Scoring fails closed |
| [`D-019`](DECISIONS.md#d-019--relation-association-by-temporal-proximity-is-not-acceptable) | Relation association by temporal proximity is not acceptable |
| [`D-020`](DECISIONS.md#d-020--the-signal-key-identifies-the-signal-not-the-detector) | The signal key identifies the signal, not the detector |
| [`D-021`](DECISIONS.md#d-021--mechanism-not-proximity-may-claim-a-consequence) | Mechanism, not proximity, may claim a consequence |
| [`D-022`](DECISIONS.md#d-022--a-miss-must-never-leave-the-denominator-silently) | A miss must never leave the denominator silently |
| [`D-023`](DECISIONS.md#d-023--a-mechanism-claim-requires-the-propagation-at-every-stage) | A mechanism claim requires the propagation at EVERY stage |
| [`D-024`](DECISIONS.md#d-024--grouping-and-keying-must-agree-on-scope) | Grouping and keying must agree on scope |
| [`D-025`](DECISIONS.md#d-025--the-v013-relation-result-was-generation-signature-dependent) | The v0.1.3 relation result was generation-signature dependent |
| [`D-026`](DECISIONS.md#d-026--a-non-finite-norm-is-not-a-non-finite-tensor) | A non-finite norm is not a non-finite tensor |
| [`D-027`](DECISIONS.md#d-027--a-witnessed-optimizer-call-is-not-an-applied-parameter-update) | A witnessed optimizer call is not an applied parameter update |
| [`D-028`](DECISIONS.md#d-028--do-not-integrate-a-witness-whose-evidence-has-not-faced-its-adversarial-cases) | Do not integrate a witness whose evidence has not faced its adversarial cases |
| [`D-029`](DECISIONS.md#d-029--the-optimizer-witness-is-kept-research-only-measured-benefit-is-exactly-zero) | The optimizer witness is kept research-only; measured benefit is exactly zero |
| [`D-030`](DECISIONS.md#d-030--a-case-that-cannot-be-constructed-is-a-finding-not-a-gap-to-paper-over) | A case that cannot be constructed is a finding, not a gap to paper over |
| [`D-031`](DECISIONS.md#d-031--the-11-step-latency-was-baseline-readiness) | The "11-step latency" was baseline readiness |
| [`D-032`](DECISIONS.md#d-032--c3-is-blocked-by-the-threshold-to-stability-ratio-not-by-injection-magnitude) | C3 is blocked by the threshold-to-stability ratio, not by injection magnitude |
| [`D-033`](DECISIONS.md#d-033--the-optimizer-witness-is-frozen) | The optimizer witness is frozen |
| [`D-034`](DECISIONS.md#d-034--possible_consequence_of-is-disabled-by-default) | `POSSIBLE_CONSEQUENCE_OF` is disabled by default |
| [`D-035`](DECISIONS.md#d-035--the-11-step-latency-was-really-10-and-the-report-was-wrong-not-the-data) | The "11-step latency" was really 10, and the report was wrong, not the data |
| [`D-036`](DECISIONS.md#d-036--package-version-017a1-maps-to-the-github-tag-v017-alpha) | Package version 0.1.7a1 maps to the GitHub tag v0.1.7-alpha |
| [`D-037`](DECISIONS.md#d-037--generated-experiment-artefacts-must-not-be-published) | Generated experiment artefacts must not be published |
| [`D-038`](DECISIONS.md#d-038--correction-the-detection-latency-is-the-detectors-warm-up-gate-not-the-baseline) | Correction: the detection latency is the detector's warm-up gate, not the baseline |
| [`D-039`](DECISIONS.md#d-039--the-70-step-blind-window-is-load-bearing-c3-is-out-of-reach-by-design) | The 70-step blind window is load-bearing; C3 is out of reach by design |
| [`D-040`](DECISIONS.md#d-040--baseline_mode-stays-frozen) | `baseline_mode` stays `"frozen"` |
| [`D-041`](DECISIONS.md#d-041--neuraldbgs-causal-layer-is-the-rule-this-project-already-falsified) | NeuralDBG's causal layer is the rule this project already falsified |
| [`D-042`](DECISIONS.md#d-042--observer-overhead-reduced-30--the-3--gate-failure-is-a-workload-artifact) | Observer overhead reduced ~30 %; the <3 % gate failure is a workload artifact |
| [`D-043`](DECISIONS.md#d-043--t1-layer-observability-cadence-and-a-correction-to-d-042) | T1 layer observability cadence; and a correction to D-042 |

---

## Reading the numbering

Decision ids are sequential and permanent. A gap or an out-of-order entry is not an error:
`D-038` and `D-039` correct `D-031`, `D-043` corrects `D-042`, and the original entries remain
in place because a record that quietly disappears is not a record.
