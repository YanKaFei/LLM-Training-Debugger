# DECISIONS

Append-only. Each entry: what was decided, why, and what it rules out.

## D-001 — No forks of upstream projects
Trackio/HTA/NNsight/DCGM/Flight Recorder are integrated behind our own adapters or
deferred. Carried from Phase 0.

## D-002 — Parquet is the telemetry truth; DuckDB queries it
No database has been introduced. The trigger for Postgres was concurrent human writers;
still not true.

## D-003 — Correlation is never causation
`CAUSED_BY` / `ROOT_CAUSE_OF` do not exist in the incident-relation type system.
Establishing causation requires replay, deferred to a later phase. A test asserts the
types stay absent.

## D-004 — An incident group is not a proven root cause
`IncidentGroup.has_proven_root_cause` is hard-coded `False`. Grouping is presentation.

## D-005 — Evidence observation is separated from alert deduplication
Phase 1.12A: a detector cooldown may not suppress an observation, because the episode
rule needs it. Deduplication is done by `EpisodeEngine` (contiguity) and `fuse()` (one
verdict per anomaly type). Measured: 20 observations → 1 episode → 1 verdict → 0
incidents.

## D-006 — Idempotency is at `(step, layer)`, scoped per `signal_key`
Not `(step)`: a restarted process supplies new *layers* at an already-seen step, and the
coarser key broke cross-process restart. Limitation: there is no source-event-level
identity, because `Anomaly` carries no `event_id`. The honest claim is
"idempotent at `(step, layer)` granularity", not "fully deduplicated".

## D-007 — Abstention is a legitimate output
`INSUFFICIENT_EVIDENCE` is a first-class verdict. Where healthy and faulty distributions
overlap, the system says so rather than guessing. Phase 1.7 measured that no threshold
separates a mild gradient fault from healthy variation.

## D-008 — Capability states are not promotional
`GRADIENT_EXPLOSION`, `NAN_DETECTED`, `INF_DETECTED`, `LOSS_SPIKE` are EXPERIMENTAL.
Nothing is VALIDATED without an untouched blind holdout. `alpha` does not change this.

## D-009 — Alpha is a packaging milestone, not a reliability claim
`ALPHA_RELEASE_READY` asserts installability and engineering correctness. It does **not**
assert detector reliability, and the original reliability gates remain FAIL.

## D-010 — Keep the baseline resume fix despite a measured cost

`protected_ratio_series` never moved `state` out of `SUSPECT`, so `update`'s resume
branch was unreachable and `resume_after_steps` was dead code: the reference froze
permanently at the first suspicion. Fixed by driving the state back to `NORMAL` when the
value returns below the threshold, restarting the quiet clock on an ongoing suspicion.

**Measured cost (680 runs, identical config):** precision 0.973 → 0.964, recall
0.468 → 0.457, healthy-confirmed 0.042 → 0.056, rank_corr 0.860 → 0.748. Qualifying
operating points remain **0** before and after.

Kept because a documented parameter must not be dead code and the freeze-forever
behaviour was an accident of `suspect()` being the only transition performed. **Not**
tuned to recover the old numbers. A measured sweep of `resume_after_steps` is the
legitimate next step.

## D-011 — A single-scenario smoke must not assert a statistical property

S1 asserted `NO_ANOMALY` for one healthy seed. The detector's healthy false-incident
rate is a *statistical* property (~7.7 % measured over 13 runs), and cannot be asserted
with n = 1 -- nor can its failure be called an engineering fault. S1 now asserts
**well-formedness** (a verdict is produced from persisted evidence, references resolve,
CLI exits 0) and *records* what the verdict was. The rate is measured across 13 runs in
`healthy_false_incident_probe`, reported separately, and **FAILS** the 5 % gate.

This is a corrected category error, not a weakened assertion: the smoke tests
engineering correctness, the probe tests detector quality, and neither hides the other.

## D-012 — Alpha asserts installability, never reliability

`ALPHA_RELEASE_READY` covers installation, engineering correctness and honest
documentation. `DETECTOR_RELIABILITY` remains FAIL independently, and passing the first
is never reported as progress on the second.

## D-013 — The onset is the onset of the abnormality, not of the contiguity run

An episode's `candidate_onset_step` was its first observation whatever the band said, so a
benign NORMAL-band excursion inside `gap_tolerance` inherited the fault next to it and the
quickstart reported onset 75 for a step-80 injection. The onset now anchors to the
earliest `CONFIRMED` **or `UNCERTAIN`** observation; `UNCERTAIN` is retained so genuine
early warning survives. The raw first step stays in `metadata.contiguity_onset_step`.

## D-014 — `resume_after_steps` stays at 40; the sweep selected nothing

Pre-registered sweep, 216 paired runs. `30`/`60`/`never` were byte-identical to `40`;
`20` gained one confirmation (sign-test p = 1.0). No candidate met the gates, so none was
selected. The parameter is essentially inert and **recall is the binding constraint**.

## D-015 — A scorer that reads 0.0 without an error is the most dangerous kind

Persisted scoring reported precision 0.0 on a corpus that plainly detects ×20/×40/NaN,
twice, for two silent reasons: a dropped `ground_truth` key, and a missing
`acceptable_window` that made `in_window` false so every correct detection was filed as
`UNRELATED_FALSE_POSITIVE`. The loader now emits a `scoring_warning` in that state.

## D-016 — One denominator per level, and never an episode count as a run count

42 runs produced 97 "records" because the unit was the episode and 32 runs hold more than
one. `evaluate_levels` now scores SIGNAL / EPISODE / INCIDENT / RUN / RELATION
independently, each with its own named population. Episode-level **recall** is
`NOT ESTIMABLE`: truth names the injected fault, not how many episodes it should yield.

## D-017 — `primary_recall` was always 1.0 and is now 0.50

`MISSED_PRIMARY` was never incremented, so `TP/(TP+0)` was always 1.0. Corrected
run-level recall on the development corpus is **0.50**, not 1.00. Any earlier quotation
of a primary recall of 1.0 is void.

## D-018 — Scoring fails closed

Missing primary fault type, acceptable window or injection step yields `SCORING_INVALID`
with no numbers emitted. Each condition silently produced `precision = 0.0` at some point
in this project; a plausible number is worse than a refusal.

## D-019 — Relation association by temporal proximity is not acceptable

Measured on the 50-run mixed-fault corpus: the linker reported `POSSIBLE_CONSEQUENCE_OF`
on **10 of 10** deliberately independent fault pairs and on **0 of 10** genuine causal
chains. Precision 0.0, recall 0.0. A fabricated relation is an *assertion*, which is
worse than a miss. The rule must become mechanism-aware and be measured against this
corpus; no fix was applied in v0.1.2.

## D-020 — The signal key identifies the signal, not the detector

`numerics.finiteness` multiplexes every metric through one detector id. Keyed on
`detector:rank` alone, a non-finite gradient, loss and counter collapsed into one episode
and the metric identity was lost -- destroying the mechanism evidence. Such detectors are
now metric-scoped. Per-metric detectors are untouched, because aggregate-vs-layer gradient
confirmation *depends* on those sharing an episode.

## D-021 — Mechanism, not proximity, may claim a consequence

`POSSIBLE_CONSEQUENCE_OF` now requires E3: a non-finite value on an **upstream** metric
within `PROPAGATION_DELAY_STEPS = 2` (one optimizer update). E1+E2 alone yields
`PRECEDES`, which asserts order and never consequence. Measured effect: M3 fabricated
associations 10/10 -> 0/10, M1 recovered 0/10 -> 10/10, relation precision and recall
both 0.0 -> 1.0. Only a consequence claim may group incidents; grouping on `PRECEDES`
would repeat the temporal-adjacency error one layer up.

## D-022 — A miss must never leave the denominator silently

v0.1.1's `primary_recall` was always 1.0 because `MISSED_PRIMARY` was never incremented.
v0.1.3 found the identical bug at incident level. Incident-level recall is now
`NOT ESTIMABLE` (no truth for how many incidents a fault should produce) and
`missed_primary_faults` reports the misses as an explicit count equal to RUN-level `fn`.

## D-023 — A mechanism claim requires the propagation at EVERY stage

v0.1.3 accepted an upstream non-finite value plus a two-step delay as a mechanism. The
adversarial corpus falsified that: a **suppressed update** produces exactly that pattern
while nothing propagates, and 10 independent faults were claimed as consequences. The
rule now requires a non-finite observation at each stage (gradient -> parameters ->
loss). `param.global_norm` was already recorded and simply not watched.

## D-024 — Grouping and keying must agree on scope

`EpisodeEngine` keyed episodes by metric but grouped by `(detector, rank)`, so a
multi-metric detector put every metric in one group, took the key from `group[0]`, and
let the last member overwrite `metric_name`. An episode keyed on `grad.global_norm`
reported `param.global_norm` -- the witness replaced the upstream observation instead of
joining it. Both levels are now metric-scoped. **A key and its grouping must use the same
scope, or the key is decoration.**

## D-025 — The v0.1.3 relation result was generation-signature dependent

Holding timing, scope and target fixed, the decision flipped solely on whether the
antecedent observation was non-finite. M1 and M3 were generated so that exactly that
differed. Recorded so the 100% is never quoted as a general law.

## D-026 — A non-finite norm is not a non-finite tensor

`global_gradient_norm` and `global_parameter_norm` return `nan` when any element is
non-finite and leave `inf` when the float32 reduction overflowed with every element
finite. The mapping is definitional. `FinitenessDetector` tested `isfinite(value)` alone,
so v0.1.3/v0.1.4 keyed their entire mechanism chain on a reduction artifact.
`norm_reduction_overflow` is now its own kind and is excluded from mechanism evidence.

## D-027 — A witnessed optimizer call is not an applied parameter update

`OptimizerWitness` keeps ATTEMPTED / INVOKED / SKIPPED / UNKNOWN distinct and always
reports `parameters_changed = None`: proving "invoked but unchanged" needs a parameter
scan. The hooks also fire with no gradients, so invocation alone proves nothing.
`UNKNOWN` is never rendered as `False`, and an optimizer without hook support reports
`UNKNOWN` rather than a confident answer. **Default disabled.**

## D-028 — Do not integrate a witness whose evidence has not faced its adversarial cases

v0.1.5 implements and measures the witness but does **not** wire it into promotion.
Integration needs the adversarial corpus re-run with the witness supplying midstream
evidence; until that number exists, integrating would repeat v0.1.4's mistake of shipping
a rule whose evidence was never validated against the cases designed to break it.

## D-029 — The optimizer witness is kept research-only; measured benefit is exactly zero

Paired on 54 byte-identical runs with the witness given its strongest fair reading:
recovered_true 0, new_false 0, lost_true 0, changed_abstentions 0. Every case identical
with the witness on and off. The reason is structural: a mechanism claim needs an observed
antecedent inside the propagation window, and gradient-detector latency (~11 steps) exceeds
that window, so the witness clause never fires.

Not integrated, and not rejected: the witness works, costs ~0.3 %, never produced a false
association, and is honest about its `UNKNOWN`s. It is **insufficient**, not wrong.

## D-030 — A case that cannot be constructed is a finding, not a gap to paper over

C3 (gradient anomaly observed -> optimizer invoked -> independent NaN one step later) is
not constructible here: the gradient anomaly is detected at step 71 while the injected NaN
is at 62. The witness's discriminating power is therefore `NOT VERIFIED` rather than
demonstrated, and no simulation was substituted for it.

## D-031 — The "11-step latency" was baseline readiness

`baseline_floor = warmup_steps = 50` plus `minimum_history = 20` means no usable ratio
exists until step 70. An injection at 60 therefore cannot be scored until 70; an injection
at 85 fires at 85. The detector is prompt once ready. `PROPAGATION_DELAY_STEPS` was **not**
changed to compensate -- that would have masked the cause. Consequently the Early Evidence
Lane was **not built**: there is no early evidence to surface, only a baseline that is not
yet ready.

## D-032 — C3 is blocked by the threshold-to-stability ratio, not by injection magnitude

Firing the aggregate gradient detector requires an excursion large enough that the run's
parameters corrupt before the case can complete. x100 and x30 both produced non-finite
parameters in 8/8 C3 runs while x30 left C11's parameters finite, so the instability is
triggered by the injected NaN interacting with the excursion. Increasing the magnitude
repeatedly was tried and is explicitly not the remedy; a detector change that lowers the
confirmation threshold without losing healthy-run precision is what the case needs.

## D-033 — The optimizer witness is frozen

Paired identical twice, 54 runs then 32, byte-identical every time. Not rejected (it never
produced a false association and reports `UNKNOWN` honestly), but no further mechanism work
will be added to it. It stays default-disabled.

## D-034 — `POSSIBLE_CONSEQUENCE_OF` is disabled by default

C3/C12 produced 16 consequence claims against `INDEPENDENT` ground truth, and the case
that would have tested the linker's discrimination could not be constructed. Shipping it
by default would present an unvalidated inference as a causal claim. The production
default records `PRECEDES` / `CO_OCCURS_WITH` / `UNKNOWN_RELATION` only; the mechanism
linker is an explicit opt-in research configuration. **No evidence is deleted** — order,
episodes and revisions are all retained; only the claim changes.

## D-035 — The "11-step latency" was really 10, and the report was wrong, not the data

`first usable ratio (70) - injection (60) = 10`, inclusive. v0.1.7 said 11. The prose was
corrected and **no timestamp was altered** to make the wording consistent.

## D-036 — Package version 0.1.7a1 maps to the GitHub tag v0.1.7-alpha

The RC sprint shipped `0.1.0` while the source was developed as `v0.1.7-alpha`. Aligned to
PEP 440 (`0.1.7a1`) with an explicit mapping exposed as `training_debugger.GITHUB_TAG`.
This is metadata alignment, not a new development version.

## D-037 — Generated experiment artefacts must not be published

`tests/experiments/restart_worker.py` and `restart_continue.py` are written at runtime by
`restart_smoke.py` and embedded the local checkout path. Removed and excluded from
`.gitignore` and `MANIFEST.in`. **String scanning alone was not treated as sufficient** --
the final check extracted the built sdist and searched its contents, which is how the leak
was caught.

## D-038 — Correction: the detection latency is the detector's warm-up gate, not the baseline

v0.1.7 attributed the "11-step latency" (actually 10) to the protected baseline being
unready at `floor 50 + minimum_history 20 = 70`. **That was wrong.** The binding gate is
the gradient detector's own `warmup_steps + min_samples` (50 + 20), which also yields step
71. The coincidence is why the wrong diagnosis looked confirmed.

Measured, injection at step 60:

```
baseline_mode=frozen            -> first fire 71
baseline_mode=ema               -> first fire 71     (no change)
+ detector warmup=10,min_samples=8 -> first fire 60  (fixed, in EITHER baseline mode)
```

Reducing baseline readiness from 70 to 26 changes nothing.

## D-039 — The 70-step blind window is load-bearing; C3 is out of reach by design

Lowering the detector's warm-up makes C3 constructible and destroys healthy precision in
the same stroke:

```
warmup=50, min_samples=20   0/6 healthy runs false-fire
warmup=25, min_samples=12   4/6
warmup=10, min_samples=8    6/6
```

C3's unconstructibility is therefore a **consequence of a precision gate**, not a defect.
Recorded as out of reach at the current precision requirement rather than chased.

## D-040 — `baseline_mode` stays `"frozen"`

The EMA baseline (ZClip-style clipped-statistics update) is implemented, tested and
selectable, but is **indistinguishable from frozen in every configuration measured**. It
stays opt-in because changing the default would churn behaviour for no measured benefit.
Kept because it is correct and removes the never-resumes design wart.

## D-041 — NeuralDBG's causal layer is the rule this project already falsified

Ran neuraldbg 1.3.2 on the same scenario. It reported three "cascade propagation" causal
hypotheses at identical confidence 0.72 on a run with one injected gradient excursion, with
its first event at step 0. Reading `enhanced_causality.py::_build_causal_graph`: the causal
edge is added on `current_first <= next_first` -- temporal precedence plus module adjacency
-- with weight `(gap + 1) / 10`, so **closer in time scores as more causal**. That is the
exact rule v0.1.3 measured at 10/10 fabricated associations and removed.

**Not to be copied.** Its *instrumentation* is a different matter: per-module
`Vanishing`/`Exploding`/`Dead`/`Saturated` health states are a real gap in this project,
observational rather than causal, and worth taking.

## D-042 — Observer overhead reduced ~30 %; the <3 % gate failure is a workload artifact

Audit triggered by the activation probe. Two existing performance benchmarks were
**incomparable**: `benchmark_overhead.py` reports a MEDIAN of per-repeat medians,
`benchmark_cost_ledger.py` reports the MINIMUM, and the ledger's residual is NEGATIVE
(accounted 3.86 ms vs observed 0.41 ms) which cannot be a valid attribution. The ledger
is now annotated as not usable for a decision.

Root cause of the overhead: `_accept` called `copy_with` **twice per event** --
`dataclasses.replace` on a 38-field dataclass at 11.3 us each = 22.6 us of a 41.1 us
emit (57 %). Fixed by filling context **in place** when the observer constructed the
event (nothing else can hold a reference), cloning only for caller-supplied events.

```
observer_only  +1.066 -> +0.751 ms    core_layers  +2.414 -> +1.635 ms
emit           41.1 -> 17.8 us/event
```

Validation is a further 38 % of the remaining overhead (+0.314 ms) and is a config flag.

**The <3 % gate verdict is not overturned.** The cost is approximately FIXED per step
(0.2-0.9 ms), so the fraction collapses with workload size: +11.0 % (4L) -> +5.2 % (8L)
-> +0.4 % (12L). The gate measures a 4-layer model with a 3-5 ms step, where no
implementation could pass without going below 0.14 ms. The transferable number is the
absolute per-step cost; the projection to production steps is NOT VERIFIED.

## D-043 — T1 layer observability cadence; and a correction to D-042

`param.layer.norm` was 25 % of event volume with **no consumer** (it appears downstream
only in the presentation sink's FORBIDDEN_PREFIXES). `LayerMetricCollector` now has
`observability_cadence` (default 5) for layer metrics no detector reads, separate from
`cadence` for the detection path. Event volume -20 %; `grad.layer.norm` untouched.

**Measured saving: 0.033 ms/step, about 1 % and within noise.** The audit predicted
0.161 ms from `events_removed x us_per_event`. The prediction was 5x too high.

**Correction to D-042**: that entry lists event volume as the largest remaining lever.
Measured, it is not established. Micro-cost x count does not equal in-situ cost -- the
same error the cost ledger made with its negative residual.

**The experiment's real finding**: resampling the detector's INPUT
(`grad.layer.norm` every 5 steps) leaves detection intact -- still fires at step 90 with
unchanged latency -- but collapses incidents from 9 to 3, because the episode engine
cannot reach CONFIRMED from a sparser series. **Density is required for confirmation, not
for detection.** A detector-level check would not have caught this.
