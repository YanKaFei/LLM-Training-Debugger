<div align="center">

<img src="assets/banner.svg" alt="LLM Training Debugger" width="100%">

<br>

**English** · [简体中文](docs/i18n/README.zh-CN.md) · [한국어](docs/i18n/README.ko.md) · [日本語](docs/i18n/README.ja.md) · [Français](docs/i18n/README.fr.md)

<br>

[![tests](https://img.shields.io/badge/tests-584_passing-3fb950?style=flat-square)](tests)
[![python](https://img.shields.io/badge/python-3.9%2B-58a6ff?style=flat-square)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-8b949e?style=flat-square)](LICENSE)
[![status](https://img.shields.io/badge/status-alpha-d29922?style=flat-square)](#project-status)
[![deepseek-harness](https://img.shields.io/badge/built_with-DeepSeek_Harness-4d6bfe?style=flat-square)](https://github.com/topics/deepseek-harness)

</div>

---

## What this is

An **evidence-first monitoring library for PyTorch training runs**. It watches a training
loop, turns metric anomalies into incidents, and — this is the part that matters — **refuses
to answer when the evidence does not support an answer**.

<div align="center">
<img src="assets/architecture.svg" alt="Architecture" width="100%">
</div>

Most training tooling answers *"what happened?"*. This asks a narrower and harder question:

> **Do I actually know, and if not, what exactly do I have?**

## What this is not

Stated up front, because the honest boundary is more useful than the feature list:

- **Not a replacement for experiment trackers.** It emits to one; it is not a dashboard.
- **Not root-cause analysis.** It reports *a step and a metric*. It cannot tell you a kernel,
  a tensor, or a line of code.
- **Not a controller.** It observes. It will not clip your gradients, reset your optimizer
  state, or stop your run.
- **Not proven at scale.** Every number in this repository was measured on a 4–6 layer model.
- **Not validated.** There is no blind holdout. `validated_capabilities` is empty by design.

## Why it exists

A training run produces a loss curve. When the curve goes wrong, the curve does not say
where, and rerunning rarely reproduces it.

This library takes the position that **the product is not the graph — the product is the
incident**: a bounded, auditable, timestamped claim about what was observed, carrying the
evidence for it and an explicit statement of what could not be concluded.

That position has consequences throughout the design:

| Principle | How it appears in the code |
|---|---|
| Observation must never break training | `emit()` never raises; a bounded P0–P3 queue drops rather than blocks |
| Evidence must be observed before its mechanism is asserted | `OBSERVED` / `INFERRED` / `HYPOTHESIS` / `UNKNOWN`; only `OBSERVED` is promotable |
| A missing observation is not a negative observation | Every verdict carries `data_quality` and `completeness_warning` |
| Prefer abstention to a fabricated relationship | `INSUFFICIENT_EVIDENCE` is a first-class verdict, not an error |
| Some causal claims should be impossible to make | `CAUSED_BY` and `ROOT_CAUSE_OF` **do not exist in the schema**, and a test asserts they never will |

## Requirements

| | |
|---|---|
| **Python** | 3.9 – 3.13. Every version is exercised in CI. |
| **Core** | `numpy>=1.23`, `pyarrow>=12`, `duckdb>=0.10,<1.5` |
| **PyTorch adapter** | `torch>=2.0` — optional, via the `torch` extra. The observer, storage, detectors, episodes, fusion and CLI import no torch. |
| **Development** | `.[dev]` = `pytest` **and** torch, because the test suite imports it. |

> **The `duckdb` upper bound is a workaround, not a preference.** duckdb 1.5.6 breaks
> `fetch_arrow_table()` with `SystemError: <built-in function __import__> returned a result
> with an exception set`. All 23 failures in CI were at that one call; the other 560-odd
> tests passed and `import pyarrow` worked, so the fault is in duckdb's Arrow bridge rather
> than in pyarrow. It was invisible on Python 3.9 because duckdb 1.5.x requires
> **Python >= 3.10**, so 3.9 resolves to 1.4.5 and never sees it — which means anyone on
> 3.10 or newer with an unpinned install had a broken build.
>
> A CI job named *latest dependencies (informational)* deliberately ignores every upper
> bound and is allowed to fail. **While it is red the bound is still required; when it
> turns green the bound should be removed.** See
> [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md), item 15.

## Quick start

```bash
git clone https://github.com/YanKaFei/LLM-Training-Debugger.git
cd LLM-Training-Debugger
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

A complete, runnable example:

```bash
python examples/quickstart.py
```

Or drop it into an existing loop:

```python
from training_debugger.adapters.pytorch import PyTorchTrainingAdapter

adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="my-model")
adapter.run_start(steps=total_steps)

for step in range(1, total_steps + 1):
    with adapter.step(step) as tracker:
        loss = model(**batch)["loss"]
        tracker.record_loss(loss)
        loss.backward()
        tracker.record_gradients()
        with tracker.phase("optimizer"):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tracker.record_parameters()

adapter.run_end()
```

## What you get

<div align="center">
<img src="assets/verdicts.svg" alt="Verdicts" width="100%">
</div>

Each incident carries a `reason_code`, an `evidence_state`, a `promotion_disposition`, and —
when the stream was incomplete — a `data_quality` warning. **A run whose events were dropped
is never rendered as a healthy run.**

## Commands

<div align="center">
<img src="assets/cli-demo.svg" alt="CLI demo" width="100%">
</div>

```bash
training-debugger --root <store> runs                  # list runs
training-debugger --root <store> verdicts  <run>       # all verdicts, including abstentions
training-debugger --root <store> incidents <run>       # incidents with evidence
training-debugger --root <store> incident  <run> <id>  # one incident, in full
training-debugger --root <store> metrics   <run> [m]   # query metric rows
training-debugger --root <store> diagnose  <run>       # deterministic diagnosis
training-debugger --root <store> compare   <a> <b>     # first step two runs disagree
training-debugger --root <store> scan      <r1> ...    # batch: which run is the outlier
```

### `compare` — localise a divergence

Aligns two runs' metric series, uses the **early region where they agree as the null model**,
and reports the first step that leaves it and stays out. Shared noise cancels in the
difference, which is why this localises where a single-run threshold cannot.

Measured on a paired corpus with known injection steps: **9/9 fault pairs localised to the
exact injection step, 3/3 clean pairs correctly abstained.**

Wall-clock and memory metrics are excluded by default — *a timing difference is not an
algorithmic divergence*.

### `scan` — find the outlier in a batch

Every pair is compared once; a run's score is how many peers it diverged from. **It requires
like-for-like runs**: hold the seed and configuration fixed and vary one factor. On a batch
that mixes seeds it returns `NO_SINGLE_OUTLIER` and explains why, rather than naming a run.

## Capabilities

Every capability carries an explicit evidence level. Nothing here is marked validated.

| Capability | Status | Evidence |
|---|---|---|
| Non-finite observation (NaN / Inf) | `EXPERIMENTAL` | exact; no threshold |
| Gradient anomaly detection | `EXPERIMENTAL` | detection boundary ×20 and ×40 → 6/6; ×2 and ×5 → 0/6 |
| Incident fusion with abstention | `EXPERIMENTAL` | run precision 0.947 / recall 0.500 (42-run corpus) |
| Two-run divergence localization | `EXPERIMENTAL` | 9/9 exact, 3/3 abstained (paired corpus) |
| Batch outlier scan | `EXPERIMENTAL` | 5/5 peer support on a like-for-like batch |
| Per-module activation health | `EXPERIMENTAL` | dead module named and dated at OBSERVED 1.0 |
| Optimizer-call witness | `EXPERIMENTAL` | frozen; off by default |
| Relation inference (ordering only) | `UNSUPPORTED` | disabled by default; a measured attempt fabricated 16 associations |

## Known limitations

Published because a monitor whose failures are undocumented cannot be trusted:

- **Run-level recall is 0.500.** Half of the primary faults in the development corpus are missed.
- **Healthy-run false incidents: 7.7 %**, against a self-imposed gate of 5 %. **This gate fails.**
- **Observer overhead fails its own < 3 % target** on the 4-layer benchmark workload. The cost is
  approximately *fixed* per step (0.2–0.9 ms), so the fraction collapses as the workload grows:
  +11.0 % (4L) → +5.2 % (8L) → +0.4 % (12L). Above roughly a 17 ms step, it is inside budget.
  This is reported as a failure, not excused.
- No blind holdout exists; every number is from the development corpus.
- Validated only on 4–6 layer models.

Full list: [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) · Design record:
[DECISIONS.md](docs/DECISIONS.md) · **Readable index: [DECISIONS_INDEX.md](docs/DECISIONS_INDEX.md)**,
which opens with a table of every claim this project has had to correct or withdraw ·
Current state: [DEVELOPMENT_STATE.md](docs/DEVELOPMENT_STATE.md)

## Reproducing the numbers

```bash
pytest -q                                            # 584 passing, 1 skipped
python tests/experiments/full_pipeline_smoke.py      # 8/8
python examples/quickstart.py
python tests/performance/audit_observer_cost.py      # paired overhead, with confidence intervals
```

Every headline figure is now produced by a committed script, and the scoring is
reproducible again. `tests/experiments/level_scoring.py` re-derives the quality numbers
from a store:

```bash
python tests/experiments/level_scoring.py development   # or mixed, adversarial
```

**Reproduced** on the frozen revision `9e4550c`:

| Figure | Stated | Re-derived | |
|---|---|---|---|
| RUN precision (42-run corpus) | 0.947 | **0.9474** | same |
| RUN recall (42-run corpus) | 0.500 | **0.5000** | same |
| `missed_primary_faults` | 18 | **18** | same |
| INCIDENT precision (42-run corpus) | 0.720 | **0.7200** | same |
| RUN precision / recall (50-run mixed) | 1.000 / 0.500 | **1.0000 / 0.5000** | same |
| INCIDENT precision (50-run mixed) | 0.500 | **0.5000** | same |

**Changed**, because the detector changed between when they were measured and now:

| Figure | Stated | Re-derived | |
|---|---|---|---|
| EPISODE precision (42-run corpus) | 0.326 | **0.3960** | improved |
| EPISODE precision (50-run mixed) | 0.500 | **0.2222** | worse |

The EPISODE level counts how many episodes each injected fault produced, and two changes
in v0.1.8 move it: the T1 layer observability cadence reduces `param.layer.norm` emission
five-fold, and a per-module activation detector was registered. Neither was measured
against the episode-precision figure when it was made. **The stated values are superseded;
the re-derived ones are current.**

Nine numbers were restated from an input that was never committed, and could not be
re-derived at all until `tests/experiments/level_scoring.py` and `tests/evaluation_truth.py`
existed. Both now do, and every artifact records the revision that produced it
(`code_sha`, and `code_dirty` scoped to the source tree), so the next drift is visible
without a manual diff. See
[EVIDENCE_REGENERATION_FINDINGS.md](outputs/v0.1.8-alpha/EVIDENCE_REGENERATION_FINDINGS.md).

## Project status

**Alpha.** The interfaces are not stable. The API may change without notice. It is genuinely
useful for instrumenting a run and reading an auditable verdict; it is not yet something to
build a production pipeline on.

## Feedback and discussion

**GitHub is the only place this project is discussed.** There is no chat server, no mailing
list, and no other channel — so anything raised here is visible to everyone and stays
searchable.

| | |
|---|---|
| **Found a bug?** | [Open an issue](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) |
| **Have a question, or want to discuss a design decision?** | [Start a discussion](https://github.com/YanKaFei/LLM-Training-Debugger/discussions) |
| **Want a feature?** | [Open an issue](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) |
| **Disagree with a claim made here?** | [Open an issue](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) — please do |

**Disagreement is welcome, and specifically it is welcome about the numbers.** If you
reproduce a measurement in this README and get a different result, that is a bug report worth
filing. If a claim here is not supported by the evidence in this repository, the claim is
wrong and should be corrected or withdrawn.

When reporting, please include: the command you ran, the model shape and step count, your
PyTorch version, and the full output.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: changes that make a claim weaker
are as welcome as changes that make it stronger, and **a test must never be weakened to make
it pass**.

## License

[Apache-2.0](LICENSE).

<div align="center">
<br>
<sub>Built with <a href="https://github.com/topics/deepseek-harness">DeepSeek Harness</a>.</sub>
</div>
