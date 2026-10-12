# REPRODUCIBILITY

All commands are run from the **project root** — the directory containing `pyproject.toml`.
Clone or unpack the project anywhere; no command depends on where it lives.

Every claim in this repository can be re-run. Three kinds of evidence are kept strictly
separate and are never mixed in a report:

```
REAL_TRAINING_E2E                 real PyTorch autograd, real observer, real detector
EVENT_DRIVEN_PIPELINE_INTEGRATION constructed events through the production analysis path
SYNTHETIC_DEVELOPMENT_BENCHMARK   synthetic harness scenarios (tests/scenarios, helpers.py)
```

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade "setuptools>=61" wheel
pip install -e ".[torch,dev]"
```

## Test suite

```bash
.venv/bin/python -m pytest tests -q
```

## Core evidence

```bash
# Full pipeline smoke — 8 scenarios, engineering correctness only
.venv/bin/python tests/experiments/full_pipeline_smoke.py

# Real training end to end (REAL_TRAINING_E2E)
.venv/bin/python examples/quickstart.py

# Query persisted evidence from a separate process
PYTHONPATH=src .venv/bin/python -m training_debugger.cli.main --root <root> verdicts <run_id>
```

## Development corpora

```bash
# 50-run mixed-fault corpus
.venv/bin/python tests/experiments/mixed_fault_corpus.py

# 45-run adversarial challenge corpus
.venv/bin/python tests/experiments/adversarial_corpus.py

# C3/C4/C11/C12 timeline corpus
.venv/bin/python tests/experiments/timeline_challenge.py

# 680-run gradient recalibration corpus (~16 min)
.venv/bin/python tests/experiments/gradient_recalibration.py
```

## Paired experiments

```bash
# witness OFF vs ON (the result was: identical)
.venv/bin/python tests/experiments/witness_paired_eval.py

# witness overhead, interleaved A/B
.venv/bin/python tests/experiments/witness_overhead.py
```

## Package build

```bash
python -m build --outdir dist .
pip install --no-deps dist/training_debugger-*.whl
python -c "import training_debugger, os; print(os.path.dirname(training_debugger.__file__))"
```

## Notes

* Corpora are seeded and deterministic; run ids are a pure function of the scenario.
* Ground truth lives in a **separate tree** from the stores and never enters production paths.
* Historical reports under `outputs/` are **append-only**; results are never rewritten.
