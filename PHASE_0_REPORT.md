# PHASE_0_REPORT.md

**Project:** LLM-Training-Debugger
**Phase:** 0 — Repository Research & Reuse Decision
**Date:** 2026-10-07
**Status:** **COMPLETE**

---

# PHASE 0 STATUS

## COMPLETE

All 19 gate conditions are satisfied. Evidence for each is given in §Gate below.

**Headline result:** the project is not a dashboard, a profiler, or a metrics collector. Across **31 surveyed repositories** (11 mandated + 3 DSH-ecosystem + 21 additional), **no project fuses loss, gradient, activation, communication, GPU and data-shard evidence into a single incident object with ranked, per-cause-justified root causes.** Every surveyed project detects one thing in one modality. The infrastructure layer is solved and must not be rebuilt; the diagnostic intelligence layer is empty and is what we will build.

**Zero forks. Nine wrap/depend decisions. Two legal hazards caught before any code was read for reuse.**

---

## TOP REUSE DECISIONS

| # | Module | Decision | Target |
|---|---|---|---|
| 1 | Experiment tracking | **DEPEND** | **Trackio** (MIT), pinned, behind our adapter |
| 2 | Telemetry store | **BUILD** on **DEPEND** | Parquet + **DuckDB** (Trackio is *not* used for telemetry) |
| 3 | GPU telemetry | **DEPEND + thin WRAP** | **DCGM exporter** behind a `TelemetrySource` Protocol |
| 4 | Profiler trace analysis | **WRAP**, v0.2 | **HTA** behind `dbg_analyze_profiler` |
| 5 | Distributed hang / collective mismatch | **WRAP + narrow read-extend**, v0.2 | **PyTorch Flight Recorder** + our liveness pass |
| 6 | SDC / straggler localisation | **DEPEND**, v0.2 | **LMResiliency `lm-resiliency`** (BSD-3) — replaces an entire workstream |
| 7 | Deep layer/activation/attention capture | **WRAP**, v0.2 | **NNsight** behind a `DeepInspector` Protocol |
| 8 | DeepSpeed metrics | **DEPEND** (read-only adapter) | DeepSpeed's own profiler + `comm.log_summary` |
| 9 | Agent runtime + tool plumbing | **DEPEND** | **DeepSeek Harness**, as one bundle (host plugin + optional client half) |
| 10 | Skills | **DEPEND/port** | HF `skills` (6 as-is, 5 rewrite, 15 skip) + NDIF `skills` (8 core, 9 advanced, 7 no) |
| 11 | **AVOID** | — | **Arize Phoenix** (Elastic License 2.0), **OpGuard** (no licence) |
| 12 | **FORK** | **NONE** | Both candidates rejected decisively; see Q6 |

---

## TOP ARCHITECTURE DECISIONS

1. **One flat, typed, versioned `TrainingEvent` envelope** with a closed event vocabulary and a closed unit vocabulary; the `metadata` JSON map is an escape hatch and never carries a correlation key. It closes five gaps TorchTitan's record schema has: no layer, no device, no dataset provenance, no unit/severity/confidence, and no incident lifecycle. (`TRAINING_EVENT_SCHEMA_PROPOSAL.md`)
2. **Ordering by `(run_id, rank, step, microstep, seq)`, never by wall clock.** Wall time is for cross-node correlation only. TorchTitan already carries `seq_id` and separates `time`/`time_ms`/`time_us`; we inherit that discipline and add `microstep`, which no framework currently exposes.
3. **`first_abnormal_event_id` is the product.** The engine returns the earliest *anomaly* in the causal subgraph, not the earliest event and not the earliest *report*. In the brief's scenario the reported symptom is 19 steps later than the cause, and that gap is the entire value proposition.
4. **A three-tier cardinality budget with T2 never always-on.** T0 scalars always-on; T1 per-layer as `{max,mean,p99,argmax_layer}` plus top-8 layers plus a vector hash; T2 full capture armed only, step-bounded, auto-expiring. Per-layer × per-step × per-rank is the dominant scaling risk, and it is answered by design rather than by hope.
5. **The LLM never decides what is true.** L1 detection and L2 correlation/ranking are deterministic statistics; L3 (the agent) chooses the next probe and explains proven evidence. Only probes returning `confidence = 1.0` (Observed) evidence may promote a hypothesis. Prose cannot promote a score.
6. **`not_correlated` is mandatory and multiplicity is printed.** The incident report must list what was tested and rejected, and must state how many series were examined. A system that hides the denominator is lying by omission — and the brief's own example report contains that negative list.
7. **Data-causation is detected but not adjudicated in v0.1.** `DATA_SHARD_ANOMALY` ships with detection, correlation and a conditional-recurrence test, capped at `INFERRED` with `missing_evidence: ["replay confirmation"]`. Replay is the only arbiter and it is v0.2. Shipping a confident unverified data-causation verdict would be worse than shipping none.
8. **Parquet-first, DuckDB for queries, Trackio for tracking, no servers.** Driven by measurement: Trackio's metrics table is an EAV with an unindexed JSON blob, its read path `fetchall()`s whole runs with no LIMIT and decodes JSON twice, and all writers serialise on one file lock that drops batches after 10 s. Postgres becomes necessary only when several humans write to the incident store concurrently; ClickHouse only above ~10⁵ events/s sustained. Both triggers are named and measurable.
9. **Observability failure must never degrade to less training.** The observer's hot path is a bounded queue push plus a sequential append. A stopped ingester, a full disk, an absent DCGM or a failed agent all degrade observability only. `OBSERVER_OVERHEAD` is itself a first-class incident type — a debugger that hides its own cost is untrustworthy.
10. **The DSH plugin is a log-emitter with its own error containment.** DSH has **no metrics and no traces** (OTel sdk-logs only), and cordis's generic `emit` dispatch has **zero containment** and is stop-on-throw. Every listener must wrap its own body.

---

## TOP RISKS

| # | Risk | Severity | Why it is real | Mitigation |
|---|---|---|---|---|
| R1 | **`first_abnormal_event_id` is wrong in the multi-rank case** | **HIGH** | Ranks disagree about "step"; the global earliest anomaly must be computed carefully, and TorchTitan's own `rank` field has a documented collision defect across meshes | Global ordering on `(step, microstep, seq)` with rank as an attribute, not a sort key; `mesh_name`/`mesh_rank` mandated in the schema; explicit skewed-rank test at W6; A1 acceptance criterion |
| R2 | **The agent produces plausible prose not grounded in tool output** | **HIGH** | This is how a system like this destroys trust; an LLM can always write a convincing narrative | Every claim must trace to a tool result; tools return data + `evidence_refs`, never conclusions; `missing_evidence` required by schema validation; A5/A6 acceptance criteria; grounding reviewed in transcripts |
| R3 | **Layer hooks measurably slow training** | **HIGH** | 100+ modules × per-step reduction is not free, and a debugger that changes throughput is unusable | T1 emits aggregates + top-K only; full vectors go to the artifact store; auto-expiring T2; `OBSERVER_OVERHEAD` detector; hard A7 gate at <1 % of step time |
| R4 | **NNsight silently corrupts gradients under gradient checkpointing** | **HIGH** *(avoided, not mitigated)* | Standard in LLM training; the failure is silent by construction | **Structural avoidance:** NNsight is excluded from the v0.1 gradient path entirely. T1 uses plain PyTorch hooks. v0.2 wraps NNsight behind a `DeepInspector` that hard-refuses when checkpointing is detected |
| R5 | **Confident wrong data-causation verdicts** | **HIGH** | Shard changes are frequent, so proximity alone is weak evidence; this is the headline story and the easiest to get wrong | Empirical null distribution; conditional recurrence test; `INFERRED` cap in v0.1; replay as the arbiter; anti-pattern #7 in `ROOT_CAUSE_ARCHITECTURE.md` forbids proximity-based conclusions |
| R6 | **Trackio is pre-1.0 and warns its SQLite schema may change; no migration framework** | MEDIUM | An upgrade could break the tracking plane | Pin the exact version; adapter boundary; treat its on-disk schema as private; never store telemetry there |
| R7 | **DCGM is Linux-only and absent on dev boxes** | MEDIUM | Kills the developer experience if the adapter assumes it | Swappable `TelemetrySource`; NVML fallback; `TELEMETRY_UNAVAILABLE` is an honest, first-class state; the DCGM incident types degrade rather than fabricate |
| R8 | **DCGM's default counter set is partial (windowed XID, ECC, throttle, health and P2P are commented out), and its default interval is 30 s** | MEDIUM | The dependency delivers only ~60 % of the needed signal without configuration; `DCGM_FI_DEV_XID_ERRORS` alone is enabled by default | Two named obligations: a custom counters CSV un-commenting `DCGM_EXP_XID_ERRORS_COUNT/TOTAL`, the ECC families, the clock/throttle events, `DCGM_EXP_GPU_HEALTH_STATUS` and `DCGM_EXP_P2P_STATUS`; and a DCGM runtime pin in lockstep with the image tag. Adapter must record both collection cadence and scrape interval |
| R9 | **Four wrapped instruments = four version-compat surfaces** | MEDIUM | DCGM, HTA, Flight Recorder and NNsight all move independently | Each behind a named adapter with a version check and a capability report; wrap boundaries kept thin; `WRAP` never means "call internals" |
| R10 | **Event volume in long runs** | MEDIUM | ~360 MB/day/8 ranks is fine; a 30-day run is ~11 GB | Retention policy defined in W0; Parquet compression; no T2 always-on; per-node pre-aggregation for multi-node |
| R11 | **A dependency goes unmaintained** | LOW–MEDIUM | Precedent: `smdebug` is dead | Port predicates rather than depend on dead packages; keep wrap boundaries thin; `lm-resiliency` is young, so our own heartbeat tripwire is the v0.1 fallback |
| R12 | **DeepSpeed evidence is pinned to a release, not a commit** | LOW | GitHub was unreachable during that window; the git SHA is UNVERIFIED | Content-addressed sdist pin recorded (`0.19.7`, sha256); the brief's four named ZeRO symbols do not exist in that release; re-verify on a networked host |
| R13 | **Elastic-License or unlicensed contamination** | **HIGH** if violated | Phoenix is the single most tempting reuse target and is Elastic License 2.0 | Both hazards recorded as `AVOID`; standing rule L4 forbids GPL/AGPL/SSPL/Elastic/custom code absent a written decision; L5 treats a missing licence as `AVOID` |

---

## PHASE 1 RECOMMENDATION

### Start with W0–W6 in strict serial order, and treat W6 as the go/no-go gate.

The ordering principle: **prove one signal flows end-to-end and that we can name its first abnormal event, before building any intelligence on top of it.**

| Wave | Deliverable | Exit criterion |
|---|---|---|
| **W0** | Schema (typed + validator + unit vocabulary + `schema_version` policy), CI, licence files, the five decisions that must be made now (`run_id`/`attempt_id`, `seq` scope, GPU-metrics-in-stream, artifact layout, retention) | A hand-written fixture event stream validates; CI green |
| **W1** | Observer SDK: bounded queue, non-blocking append, drop accounting, per-rank JSONL | Training runs correctly with the ingester stopped; `SIGKILL` leaves correctly-truncated data |
| **W2** | Raw-PyTorch adapter: `RUN_*`, `STEP_*`, `LOSS`, `LR_CHANGE`, global grad metrics, finiteness | A real run produces real JSONL |
| **W3** | Ingester → Parquet → DuckDB hot window + the `dbg_get_metrics` path | A 10 k-step series returns in <100 ms |
| **W4** | T1 layer instrumentation (plain PyTorch hooks) + `dbg_get_layer_metrics` aggregate/top-K | A layer gradient rises before the global norm, in a synthetic injection |
| **W5** | Detectors + warm-up/suppression + multiplicity accounting | No false incident in a 10 k-step healthy run at the stated budget |
| **W6** | Incident engine + `first_abnormal_event_id` | **A1 passes on a real loss spike. This validates the architecture; everything after is additive.** |
| W7 | DCGM adapter + GPU incidents | GPU incidents fire on synthetic triggers; dev-box fallback is honest |
| W8 | Correlation engine: per-incident windows, lag distributions, recurrence test, BH correction, `not_correlated` | A2/A3 pass |
| W9 | Hypothesis library + decomposable ranking + ambiguity handling | A12 passes for the `LOSS_SPIKE` family |
| W10 | HF Trainer and TorchTitan adapters | The schema proves framework-neutral |
| W11 | DSH plugin bundle + 17 read-only tools + containment | A5/A6 pass |
| W12 | CLI + incident timeline UI on Trackio's custom-frontend path | Alert → root cause without reading JSONL |

### The five things to do first, concretely

1. **Freeze the schema and the five W0 decisions in one PR.** Everything downstream depends on `run_id`/`attempt_id` semantics and `seq` scope; changing them later invalidates stored data.
2. **Build the observer with the fault-injection tests, not after them.** The "training still works with the ingester dead" property is cheap now and impossible to retrofit.
3. **Instrument T1 with plain hooks and measure the overhead in the same wave.** If the overhead gate fails, the whole T1 design changes, and W4 is the cheapest place to discover that.
4. **Wire `first_abnormal_event_id` against a real loss spike as early as possible.** It is the acceptance criterion; discovering a flaw in the ordering model at W6 is recoverable, at W11 it is not.
5. **Write `THIRD_PARTY_NOTICES.md` and adopt Apache-2.0 for our own code in W0.** We copy nothing today, so the obligation set is empty — and it stays empty only if the standing rules are in place before the first port.

### What Phase 1 must not do

No multi-node, no replay, no NNsight, no HTA, no Flight Recorder, no action tools, no auth, no server infrastructure, no learned models. If a Phase-1 plan contains any of these, the plan is wrong and `MVP_SCOPE.md` is the authority.

---

# The 12 mandated questions, answered

## 1. 是否应该基于 Trackio？/ Should we build on Trackio?

**YES — as a pinned dependency, not as a fork, and not as our telemetry store.**

Trackio earns this on three measured properties: its write path is sound (in-memory queue, 0.5 s drain, one `executemany` + one commit per batch, WAL, `synchronous=NORMAL`); its extension surface is designed for exactly our problem (a dynamic dict registry behind one route where adding an endpoint is one function plus one entry, first-class custom frontends via `show(frontend_dir=...)`, and our own tables can live in the same project `.db` through the public `get_project_db_path()` and are thereby visible to the authorizer-guarded `query_project` SQL surface); and its `Trace(messages, metadata, spans)` model already **is** an LLM agent-trace model, so we do not invent a second trace concept.

It is explicitly **not** our telemetry store, and the reason is measured rather than aesthetic: its `metrics` table is an EAV with an unindexed JSON blob payload, every chart read `fetchall()`s an entire run with no LIMIT and decodes each row's JSON twice, the frontend re-fetches full history every ~1 s with no `since-step` parameter, and all writers serialise on a single project file lock that drops the batch after a 10 s timeout. 64 ranks also exceed upstream's certified 8 concurrent processes.

Two caveats are recorded: its `LICENSE:3` is an unfilled `Copyright (c) 2024 Your Name` template placeholder, and its `_vendor/` modules and `commit_scheduler.py` have no per-file licence headers. Both are moot under DEPEND, and both are recorded in `LICENSE_MATRIX.md` so nobody later copies a file assuming MIT makes it trivially safe.

## 2. Trackio vs SwanLab: 我们最终选择谁，为什么？

**We choose Trackio. Definitively. The decision is not close.**

Three findings settle it:

**(a) On our core axis, SwanLab is the *less* mature option.** SwanLab has literally zero occurrences of `incident`, `anomaly`, `alert`, `root_cause`, `rca`, `outlier`, `spike`, `drift` or `regression` across `swanlab/`, `core/` and `protos/`. It ships notification *transport* (9 channels) but **no alert entity and no rule engine** — its callbacks fire only on run init/finish. Trackio already ships `AlertLevel ∈ {INFO, WARN, ERROR}` persisted to SQLite. Choosing the base with no alert concept because it has a nicer hardware-monitor loop would be choosing on the wrong axis.

**(b) Forking SwanLab cannot produce a working system.** The brief's premise is factually wrong: there is no server, no frontend, no database and no Dockerfile in the repository. What exists is a Python SDK, a Go module that is scaffolding only — no `package main`, and `create_core()` raises `NotImplementedError("The SwanLab Go core runtime is not available yet.")` — and 13 `.proto` files. The actual server is closed, seat-metered and **licence-expiring** (`api/self_hosted.py:43-71`). `pip install swanlab` with `mode="local"` needs zero external services and is genuinely useful; self-hosting a server is not possible from this repo at all.

**(c) SwanLab's data plane is being replaced.** `core_python` is explicitly slated for wholesale replacement by the Go core, and the on-disk format is declared not backward-compatible. Forking would mean forking the dying half of a mid-rewrite.

SwanLab is still useful to us as an **optional secondary exporter** at its published `Callback` seam in `mode="local"`, and its `chart_index` concept is worth stealing for our layer heatmap. It is not our base.

## 3. 哪些功能完全不应该自己开发？/ What must we never build ourselves?

Fifteen things, each with the project that already does it and a clean licence:

1. GPU telemetry collection → **DCGM exporter**
2. Profiler trace analysis → **HTA**
3. NCCL / collective-mismatch forensics → **PyTorch Flight Recorder**
4. Silent-data-corruption and straggler localisation → **LMResiliency SCOUT**
5. A profiler → **`torch.profiler`**
6. A training framework → the user's own
7. An experiment tracker → **Trackio**
8. A columnar metric store and query engine → **Parquet + DuckDB**
9. A deep tensor instrumentation framework → thin own hooks for T1; **NNsight** for T2
10. DeepSpeed metrics → DeepSpeed's own FLOPS profiler and `comm.log_summary`
11. Agent runtime, session/agent/tool lifecycles, tool plumbing → **DeepSeek Harness**
12. Training/eval authoring skills → **HF `skills`**
13. Interpretability diagnostics → **NDIF `skills`**
14. Anomaly rule predicates → port from **`smdebug`**'s 23-rule taxonomy rather than invent
15. Step-time / input-bound verdicts → **`traceml`** (v0.2)

## 4. 哪些项目应该作为 dependency？/ Which projects should be dependencies?

**Direct runtime dependencies:** Trackio (pinned), DuckDB, DCGM exporter (deployed, scraped), PyTorch (given), and — from v0.2 — HTA, NNsight, LMResiliency `lm-resiliency`, ZClip's statistic, and optionally SwanLab as a secondary exporter.

**Platform dependencies:** DeepSeek Harness (agent runtime), and its `@deepseek-ai/cordis@4.0.4` / `schemastery@3.18.4` peer set for our plugin.

**Skill-source dependencies (ported, not linked):** HF `skills` (Apache-2.0, 6 as-is + 5 rewritten) and NDIF `skills` (MIT, 8 core).

Every one of these is permissively licensed (MIT / BSD-3 / Apache-2.0), so **no dependency constrains our own licence choice.**

## 5. 哪些项目只应该 reference？/ Which should be reference only?

Seven, each for a specific design idea rather than code:

1. **TorchTitan** — the event-emission pattern (flat JSONL, `seq_id`, `relative_step`, `step_tags`, span pairing with guaranteed close, format-on-caller/write-on-listener, pluggable handlers, schema-enforcing filter). Its code is welded to Python `logging` and its own `step_state`; we reimplement the pattern.
2. **`OrderLab/TrainCheck`** — invariants as an RCA substrate. A violated invariant names a broken relationship, which is a better explanation primitive than a threshold crossing.
3. **`redai-studio/comminsight`** — the fail-open design contract for NCCL-side instrumentation ("observability must never block training"), built on the NCCL Profiler ABI with no PyTorch patch.
4. **NVIDIA `LogSage`** — the closest prior art to our loop, and log-only, which precisely delimits what we must beat.
5. **`awslabs/sagemaker-debugger`** — its documented 23-rule predicate taxonomy (`exploding_tensor`, `vanishing_gradient`, `dead_relu`, `saturated_activation`, `weight_update_ratio`, `stalled_training_rule`, `unchanged_tensor`, …). The project is dead; port the predicates.
6. **`torchlens`** — exhaustive forward+backward activation/gradient capture including dynamic and recurrent models; a candidate complement to NNsight for sampled layer probes.
7. **`loongsuite/dsh-plugin`, `tma1-ai/dsh-otel`, `alebgl77/dsh-plugin-otel-genai`** — peer DSH observability plugins, read to learn the correct plugin shape and telemetry wiring.

## 6. 是否有必要 fork 某个项目？/ Is forking any project necessary?

**NO. We fork nothing.**

- **Trackio: no.** A fork means owning 31.5 kLOC plus a bundled SPA forever, with no migration framework and an upstream warning that the schema will break. It buys nothing, because every extension we need is already a supported seam: a custom frontend directory, an additive endpoint registry, and our own tables in the same database file.
- **SwanLab: no, and it is impossible.** The server is not in the repository and is closed and licence-expiring, so a fork cannot produce a working backend; the data plane is being replaced; the on-disk format is declared non-backward-compatible.
- **TorchTitan: no.** Its logging subsystem is a *reference*, and it exposes `TITAN_STRUCT_LOGGER_HANDLERS` precisely so third parties can register handlers without patching. We use that surface and patch a small number of callsites for signals it does not emit.
- **NNsight, HTA, DCGM, Flight Recorder, DeepSpeed: no.** Each is wrapped or depended on behind a named adapter. DCGM in particular is configurable entirely through `etc/default-counters.csv` — forking would buy nothing and cost upstream DCGM compatibility.

**The consequence is important:** we copy no third-party code, so at v0.1 our licence obligations reduce to dependency metadata rather than source-tree notices. That is a direct, measurable benefit of the no-fork policy.

## 7. NNsight 是否适合训练期间实时 layer monitoring？/ Is NNsight suitable for real-time layer monitoring during training?

**NO for real-time, always-on layer monitoring. YES as incident-triggered deep inspection, from v0.2.**

Three independent code-level reasons rule out always-on:
1. **You cannot observe a forward you did not open a trace around.** The controller is a plain pass-through unless `interleaving and busy`; an edit applies only to work that goes through the interleaver; and `Invoker` refuses to start mid-run. There is no passive tap and no retroactive capture.
2. **Active cost is structural.** `Interleaver.handle` linearly scans all mediators per visit, and the repo's own micro-benchmark measures ~2.5× bare vLLM at TP=1 and ~4.7× at TP=2.
3. **The backward path mutates the process and is silently wrong under gradient checkpointing.** `src/` contains no `torch.utils.checkpoint` handling, and the documentation states that parameter gradients inside a checkpointed segment are simply wrong "with nothing to signal it". Gradient checkpointing is standard in LLM training.

There is a fourth, decisive reason for *our* project specifically: NNsight is **not needed** for the v0.1 requirement. Per-layer gradient and parameter norms are reductions over tensors autograd has already materialised — plain PyTorch module hooks do this with no tracing graph, no intervention semantics, and no correctness hazard.

So the architecture splits cleanly: **T1 layer metrics = BUILD with plain hooks (v0.1); T2 activation/attention/intervention = WRAP NNsight behind a `DeepInspector` Protocol (v0.2), incident-triggered, with a hard refusal when gradient checkpointing is detected.** The idle wrap is cheap — one Python frame plus one `busy` check per module call — so wrapping once and leaving it wrapped is fine; tracing every step is not.

## 8. TorchTitan 中哪些 observability 设计应该借鉴？/ Which TorchTitan observability designs should we borrow?

Eight, in priority order:

1. **The flat JSONL record schema with `seq_id`, `relative_step`, `step_tags`, `caller` (`file:line:function`), `delta_ms` and separate `time`/`time_ms`/`time_us`.** Adopted essentially verbatim as the envelope's foundation.
2. **Format-on-the-emitting-thread, write-on-a-listener-thread.** With the explicit published loss window and the "do not call `flush()` per record" warning. We keep the split and **diverge deliberately on one point**: their queue is unbounded, ours is bounded with drop accounting, because a debugger that OOMs the trainer is unacceptable.
3. **Automatic `step`/`relative_step` stamping from a hybrid global + ContextVar**, so callers never pass step. This is correlation for free, and `relative_step` is what makes events align across a checkpoint resume.
4. **A schema-enforcing filter plus a root-logger forwarder.** `TraceEventsOnlyFilter` rejects records lacking the discriminator, keeping the stream strict; the forwarder on the root logger lets any third-party library opt in by emitting one attribute. Together they are a clean adapter model.
5. **Pluggable handler factories via an environment variable** (`TITAN_STRUCT_LOGGER_HANDLERS`). This is the supported extension point we will use for the TorchTitan adapter instead of monkey-patching.
6. **Guaranteed span closure** — emit `<base>_error` *before* the closing `<base>_end` on exception, so every start has a matching close and analysis never special-cases exceptional spans.
7. **`SDCReplayer` as detect → localise → attribute → typed-report**, using `all_reduce(MAX)` to detect, `all_gather_object` to name the originating rank, and a signature-mismatch finder to name the first diverging tensor. This is the pattern our replay primitive generalises.
8. **Per-rank files joined offline into a human timeline** (`gantt_generator`), with LIFO pairing on `(source, task_name)`, orphan reconstruction, and graceful degradation of unrecognised types to instants.

**And three things we must explicitly *not* inherit:**
- Its **`rank` field is documented as wrong** — a verbatim TODO says it should be `mesh_rank` and that `(source, rank)` collides when two meshes share a name. Our schema mandates `mesh_name`/`mesh_axis`/`mesh_rank` from day one.
- It has **no layer, device, dataset, unit, severity or confidence dimension** — the five gaps that make correlation impossible.
- It has **no GPU telemetry, no anomaly detection, no incident model and no RCA** (0 hits across all the relevant greps). It enables Flight Recorder by default and **never reads the dumps**. That is exactly the gap we fill.

## 9. GPU / distributed / profiling 分别复用什么？/ What do we reuse for GPU, distributed and profiling?

| Domain | Reuse | Boundary | Phase |
|---|---|---|---|
| **GPU telemetry** | **DCGM exporter** (Apache-2.0) | `DcgmExporterSource.sample()` behind a `TelemetrySource` Protocol — HTTP Prometheus text in, `TelemetrySample` out. Not a fork: the metric set is *data* (`etc/default-counters.csv`), so all customisation is configuration. | v0.1 |
| **GPU telemetry fallback** | **BUILD** a thin NVML source | Same Protocol, so a dev box degrades honestly to `TELEMETRY_UNAVAILABLE` rather than fabricating | v0.1 |
| **Distributed hang / collective mismatch** | **PyTorch Flight Recorder** (BSD-3) + a narrow read-extend | `analyze_flight_recorder_dump(dump_dir, world_size, prefix, allow_incomplete, mismatch_cap) -> FrFindings`. The read-extend adds the "first not-completed collective" liveness pass the stock analyser leaves as an explicit TODO, from `pg_status`/`state`/`retired` — fields the dump contains and the analyser ignores. | v0.2 |
| **SDC / straggler localisation** | **LMResiliency `lm-resiliency`** (BSD-3) | Drop-in `torchrun` rendezvous backend; no training-loop rewrite | v0.2 |
| **Always-on hang tripwire** | **BUILD** a cheap heartbeat/progress check | Decides *when* to collect the expensive dump; Flight Recorder alone cannot do it because it has no heartbeats or clocks | v0.1 |
| **Profiling** | `torch.profiler` (given) + **HTA** (**MIT**) for analysis | `dbg_analyze_profiler`; adapter boundary `HtaAnalysisPort.run(HtaRequest) -> HtaResult` implemented by a spawn-isolated `SubprocessHtaAdapter` (HTA's synchronous whole-trace API and pandas DataFrames never leak past the process boundary). **21 public analyses** (24 capabilities / 26 methods). HTA is pure-CPU and needs **no GPU at analysis time**, but asserts a fully-parsed completed trace with no streaming and no timeout — so a **profiling controller in the training loop is a mandatory companion deliverable** | v0.2 |
| **DeepSpeed metrics** | **DeepSpeed** (Apache-2.0) | Read-only `DeepSpeedAdapter` over the public engine surface + `deepspeed.comm.log_summary(return_dict=True)`. No second profiler. | v0.2 |

Three operational notes that must not be lost. **(1)** DCGM's shipped default collection interval is **30 s, not 1 s**, and its default counter set is **partial for our needs** — `DCGM_FI_DEV_XID_ERRORS` is enabled by default (`etc/default-counters.csv:29`), but the windowed XID counters, the ECC families, the clock/throttle events, `DCGM_EXP_GPU_HEALTH_STATUS` and `DCGM_EXP_P2P_STATUS` are commented out, so a custom counters CSV and an explicit cadence are prerequisites for the hardware incident types. **(2)** Polled telemetry has **two independent clocks** (scrape cadence vs the 30 s field-watch refresh), the Prometheus exposition carries **no per-sample timestamps**, and a 200 from `/health` does not prove freshness — so the schema now requires `observation_window_ns`, `freshness_bound_ns`, `sample_age_ms` and `collect_interval_ms` on every polled metric, and correlation must use the observation window rather than the scrape instant (`TRAINING_EVENT_SCHEMA_PROPOSAL.md` §5.4). Without this, a 30 s-stale GPU reading gets attributed to the wrong training step. **(3)** Flight Recorder dumps are **pickle, not JSON**, with `TORCH_FR_BUFFER_SIZE=0` disabling the feature entirely.

## 10. 我们真正需要自己开发的核心代码是什么？/ What must we genuinely build ourselves?

Nine things, and nothing else:

1. **The `TrainingEvent` schema, validator, unit vocabulary and versioning policy** — the cross-domain contract nothing else provides.
2. **Training framework adapters** (PyTorch, HF Trainer, TorchTitan, FSDP2, DeepSpeed) — including the TorchTitan adapter built on its supported handler surface.
3. **Per-layer, per-microstep instrumentation (T1)** with plain PyTorch hooks, plus the top-K/aggregate/vector-hash emission strategy that makes it affordable.
4. **The detector layer (L1)** — robust z-score, trend, ratio, monotonic-run, finiteness, phase-aware band, peer comparison, sequence comparison, progress stall, distribution shift, and the lag-aware categorical detector — with warm-up, suppression, per-`(run, type)` rate limiting, cross-rank dedup and **explicit multiplicity accounting**.
5. **The incident engine** — lifecycle, severity, dedup, and `first_abnormal_event_id`, which is the product.
6. **The correlation engine (L2)** — per-incident-type windows, lag *distributions* rather than a single lag, the empirical null and conditional-recurrence test for categorical streams, Benjamini–Hochberg correction, and the mandatory `not_correlated` set.
7. **The hypothesis library and its decomposable ranking** — templates as reviewable data, with every score factor stored, a multiplicity penalty, ambiguity as a first-class outcome, and promotion only by Observed evidence.
8. **The replay primitive** (v0.2) — deterministic, targeted, data-swap and instrument-only modes, generalising TorchTitan's `SDCReplayer` pattern. This is the only arbiter for data-causation claims.
9. **The agent layer (L3)** — the DSH plugin bundle, the 23 cost-classed read-only `dbg_*` tools with their `covered_range` honesty contract, the probe-selection policy, and the separate authorised action namespace.

Plus the thin glue that is not a "module" but is unavoidable: the ingester (JSONL→Parquet), the artifact store, the CLI, and the incident-timeline UI built on Trackio's custom-frontend path.

## 11. v0.1 的最小功能边界是什么？/ What is v0.1's minimum functional boundary?

**The thesis:** v0.1 must detect a loss spike at step N, identify the *earlier* first-abnormal signal, correlate it against model/data/GPU/distributed evidence, name the most probable root cause with an explainable score, and have an agent explain it and propose the next probe — on a real single-node multi-GPU PyTorch / HF Trainer run.

**IN:** raw PyTorch + HF Trainer + TorchTitan (via its handler surface) + FSDP2 + DeepSpeed adapters; single-GPU and single-node multi-GPU; T0 scalars (loss, micro-loss, lr, grad/param norms, non-finite counts, tokens/s, step time, data-wait, GPU util/VRAM/temp/power/clocks/XID/ECC/throttle, host CPU/RAM/disk/net, heartbeat, shard identity, checkpoint outcomes, NCCL env snapshot); T1 per-layer grad/param/update norms as aggregates + top-8 + vector hash; **17 incident types** (including the required `LOSS_SPIKE`, `LOSS_DIVERGENCE`, `GRADIENT_EXPLOSION`, `GRADIENT_VANISHING`, `NAN/INF_DETECTED`, `GPU_OOM`, `GPU_THERMAL_THROTTLE`, `GPU_XID_ERROR`, `LOW_GPU_UTILIZATION`, `DATALOADER_STARVATION`, `RANK_STALL`, `THROUGHPUT_REGRESSION`, `MEMORY_LEAK`, `RANK_COUNT_MISMATCH`, `DATA_PIPELINE_REGRESSION`, `OBSERVER_OVERHEAD`, `UNKNOWN`); first-abnormal-event identification; lag-distribution correlation; mandatory `not_correlated`; multiplicity accounting; the `LOSS_SPIKE` hypothesis family with decomposed scores; ambiguity as an outcome; `INSUFFICIENT_EVIDENCE`; `compare_runs` with comparability caveats; the DSH plugin with 17 read-only tools and **no action tools**; CLI; thin incident-timeline UI on Trackio's frontend path; Parquet + DuckDB + Trackio stores.

**OUT (deferred, with the trigger):** tested multi-node (v0.2); Flight Recorder / hang incidents (v0.2); HTA / profiler analysis (v0.2); T2 activation/attention/intervention capture (v0.2, after the gradient-checkpointing guard exists); the replay primitive (v0.2); `ACTIVATION_OUTLIER` and `ATTENTION_ANOMALY` (v0.2); network/storage/CPU bottleneck incidents (v0.2, require HTA to explain); checkpoint tensor comparison (v0.2); action tools (v0.2); MoE/expert-parallel taxonomy (v0.2); Postgres/ClickHouse/Kafka (v1.0+, with measured triggers); auth/RBAC/multi-tenancy (v1.0); learned anomaly models (v0.3, after ≥100 labelled incidents); auto-remediation (**never**); eval-accuracy tracking and HPO (**out of scope entirely**).

**The three scoping decisions that must be recorded:** (1) layer gradient monitoring uses plain hooks, **not** NNsight; (2) `DATA_SHARD_ANOMALY` is detected and correlated but capped at `INFERRED` with the replay proposed; (3) the TorchTitan adapter uses its supported handler extension point, never monkey-patching.

**Acceptance:** twelve criteria, of which four are load-bearing — A1 (correct `first_abnormal_event_id` on a real spike), A5 (every agent claim traceable to a tool result), A6 (the agent refuses to conclude a data-causation verdict), A7 (observer overhead <1 % of step time with no drops).

## 12. Phase 1 应该按什么顺序开始开发？/ In what order should Phase 1 begin?

**Strictly serial W0 → W6, then parallelise W7–W12. W6 is the go/no-go gate.**

| Wave | Deliverable | Why it is in this position |
|---|---|---|
| **W0** | Schema + validator + CI + licence files + the five irreversible decisions | Every later wave depends on the schema and on `run_id`/`attempt_id`/`seq` semantics; changing them later invalidates stored data |
| **W1** | Observer SDK: bounded queue, non-blocking append, drop accounting | All adapters need an emitter, and the fault-injection property cannot be retrofitted |
| **W2** | Raw-PyTorch adapter (step, loss, lr, global grad, finiteness) | Produces the first real data; the cheapest way to falsify the schema |
| **W3** | Ingester → Parquet → DuckDB + the metrics query path | Queryability must exist before anything analyses data |
| **W4** | T1 layer instrumentation + `dbg_get_layer_metrics` | The first genuinely novel signal, and the overhead gate lives here |
| **W5** | Detectors + suppression + multiplicity accounting | Detection is meaningless without the false-alarm budget, so they ship together |
| **W6** | Incident engine + `first_abnormal_event_id` | **The architecture's validation point.** A1 passing means the ordering model, the schema and the causality concept are all correct |
| W7 | DCGM adapter + GPU incidents | Independent of the incident model; adds a second evidence domain |
| W8 | Correlation engine + `not_correlated` + recurrence test | Needs real incidents to correlate |
| W9 | Hypothesis library + ranking + ambiguity | Needs correlations to rank |
| W10 | HF Trainer + TorchTitan adapters | Deliberately after the engine, as the design check that the schema is truly framework-neutral rather than PyTorch-shaped |
| W11 | DSH plugin + 17 tools + containment | Needs everything above to have something to serve |
| W12 | CLI + incident timeline UI | Needs a stable report contract |

**Do this first, concretely:** freeze the schema and the five W0 decisions in a single PR; build the observer with its fault-injection tests from the start; measure T1 overhead in the same wave that adds it; wire `first_abnormal_event_id` against a real loss spike as early as possible; and adopt Apache-2.0 plus `THIRD_PARTY_NOTICES.md` in W0, while the obligation set is still empty.

**Explicitly not in Phase 1:** multi-node, replay, NNsight, HTA, Flight Recorder, action tools, auth, server infrastructure, learned models.

---

# Gate

| # | Gate condition | Status | Evidence |
|---|---|---|---|
| 1 | All specified repositories inspected | **MET** | 11 mandated + 3 DSH-ecosystem acquired and read; `REPO_MANIFEST.md` §1–§2 |
| 2 | Commit SHA recorded | **MET, 1 documented exception** | `REPO_MANIFEST.md` §1; DeepSpeed pinned by content-addressed sdist with the git SHA marked UNVERIFIED (§4) |
| 3 | Licence confirmed | **MET** | `LICENSE_MATRIX.md`; two `AVOID` hazards caught (Elastic-2.0 Phoenix, unlicensed OpGuard) |
| 4 | Capability matrix complete | **MET** | `CAPABILITY_MATRIX.md` — 29 mandated rows + 3 added, 10 project columns, per-cell evidence |
| 5 | Trackio vs SwanLab decided | **MET** | **Trackio**, decided on the incident axis + the impossibility of forking SwanLab (Q2) |
| 6 | TorchTitan observability read | **MET** | `notes/torchtitan.md` (656 lines, 378 citations, 105 files); 54 event names and 91 metric names enumerated by exhaustive parse |
| 7 | NNsight verified | **MET** | `notes/nnsight.md` (930 lines); backward mechanism cited to source and tests; the gradient-checkpointing hole found |
| 8 | HTA verified | **MET** | `notes/hta.md` (1074 lines) — 21 public `TraceAnalysis` analyses (24 capabilities / 26 methods); **MIT licence independently verified** (an earlier draft wrongly recorded BSD-3); the completed-trace constraint drives `WRAP`+`DEFER` |
| 9 | DCGM verified | **MET** | `notes/dcgm-exporter.md` (981 lines); default counters, XID path, two deployment modes, 30 s default interval, Linux-only constraint |
| 10 | PyTorch distributed debug verified | **MET** | `notes/pytorch-flight-recorder.md` (452 lines); 14 diagnostics, dump format `2.10`, pickle format, the analyser's ignored fields |
| 11 | HF Skills compatibility checked | **MET** | `notes/hf-skills.md`; 26 skills, exact frontmatter contract, MCP-coupling blocker, 6/5/15 port plan, Apache-2.0 clean |
| 12 | DSH plugin architecture researched | **MET** | `DEEPSEEK_HARNESS_PLUGIN_NOTES.md` (1723 lines, 67 verified source paths); the "no metrics, no traces" and "zero emit containment" findings |
| 13 | ≥10 additional candidate projects surveyed | **MET (21)** | `ADDITIONAL_REPOS.md` — 21 scored entries, ranked table, gap analysis |
| 14 | Unified event schema proposal complete | **MET** | `TRAINING_EVENT_SCHEMA_PROPOSAL.md` |
| 15 | Incident taxonomy complete | **MET** | `INCIDENT_TAXONOMY.md` — 22 mandated types + 6 added, 7 mandatory sections each, detector coverage matrix |
| 16 | Root-cause architecture complete | **MET** | `ROOT_CAUSE_ARCHITECTURE.md` — evidence graph, L1/L2/L3 separation, 12 confounders, report contract, 14 forbidden anti-patterns |
| 17 | Agent tool spec complete | **MET** | `DEBUG_AGENT_TOOL_SPEC.md` — 23 tools with input/output/errors/cost/side-effects, plus a separate authorised action namespace |
| 18 | MVP scope locked | **MET** | `MVP_SCOPE.md` — IN/OUT/DEFERRED, 12 acceptance criteria, W0–W12 plan, risk register |
| 19 | Phase 1 plan complete | **MET** | `MVP_SCOPE.md` §6; Q12 above |

**All 19 gate conditions met. PHASE 0 = COMPLETE.**

## BLOCKERS

**None.** Phase 0 is complete and Phase 1 may begin.

Four items are recorded as **caveats to carry into Phase 1**, not blockers:

| # | Item | Impact | Action |
|---|---|---|---|
| C1 | **DeepSpeed git SHA is UNVERIFIED** (GitHub was unreachable; pinned by content-addressed PyPI sdist `0.19.7`, sha256 recorded) | DeepSpeed citations are release-scoped, and four symbols named in the brief do not exist in that release | Re-verify the SHA and the four symbols on a networked host before writing the DeepSpeed adapter (W10) |
| C2 | **HTA has no local tree** — the clone was interrupted and the tarball exceeded the timeout twice (≥66 MB, no `Content-Length`); the source was obtained via codeload with the SHA in the gzip PAX header, and 3 files were byte-verified against an independent CDN | Citations are valid (two independent SHA routes), but a local tree makes line-level re-verification cheaper | Re-clone at `8fb6d9509318216ded4247f034c97c812bafae50` on a fast network if needed. **Licence correction already applied: HTA is MIT, not BSD-3-Clause as an earlier draft recorded.** Pin by SHA — the `third_party/param` submodule is absent and its licence is UNVERIFIED |
| C3 | **DSH live Slot keys and theme tokens are UNVERIFIED** — the Inspect queries timed out at 10 s | Affects only the optional client half of the plugin, not the host plugin or tools | Re-run `cordis_inspect_query` with the Harness page connected before building the UI half (W12) |
| C4 | **Arize Phoenix and OpGuard are on the AVOID list** | Legal, not technical | Standing rules L4/L5 in `LICENSE_MATRIX.md`; no code from either may be read for reuse |

---

# Deliverables produced

| File | Purpose |
|---|---|
| `REPO_MANIFEST.md` | Pinned revisions, licences, acquisition method, integrity exceptions |
| `REPOSITORY_RESEARCH_REPORT.md` | Per-repository source-level findings (A–M) |
| `DEPENDENCY_REUSE_REPORT.md` | 30 per-module BUILD/DEPEND/WRAP/FORK/REFERENCE/DEFER verdicts |
| `CAPABILITY_MATRIX.md` | 32 capabilities × 10 projects, with evidence |
| `LICENSE_MATRIX.md` | The legal gate, obligations, standing rules, our own licence recommendation |
| `ADDITIONAL_REPOS.md` | 21 scored additional candidates, gaps, surprises |
| `DEEPSEEK_HARNESS_PLUGIN_NOTES.md` | The DSH plugin contract end to end |
| `TRAINING_EVENT_SCHEMA_PROPOSAL.md` | `TrainingEvent v0` |
| `INCIDENT_TAXONOMY.md` | 28 incident types with symptoms/evidence/causes/false-positives/probes/actions |
| `ROOT_CAUSE_ARCHITECTURE.md` | Evidence graph, detection, correlation, ranking, confounders, report contract |
| `DEBUG_AGENT_TOOL_SPEC.md` | 23 read-only tools + the authorised action namespace |
| `MVP_SCOPE.md` | v0.1 boundary, acceptance criteria, Phase 1 waves |
| `TARGET_ARCHITECTURE.md` | System shape, data-layer decision, deployment, failure behaviour |
| `research/notes/*.md` (9 files, ~7 000 lines) | The underlying evidence |
| `research/evidence/*` | Cloning scripts, per-repo logs, the machine-readable manifest |

**Phase 0 stops here.** No product code has been written; no repository has been forked, modified, or copied; no UI has been created. Phase 1 begins only on explicit instruction.
