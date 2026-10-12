# QUICKSTART

Five minutes from clone to your first incident.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade "setuptools>=61" wheel     # PEP 660 editable installs
pip install -e ".[torch,dev]"
```

Verified on a clean venv with Python 3.9.6: `pip install -e .` exits 0 and the
`training-debugger` console script runs.

> macOS system Python ships a setuptools older than 61, which lacks the `build_editable`
> hook and fails with *"build backend is missing the 'build_editable' hook"*. Upgrading
> setuptools first is the fix.

## Run the example

```bash
python examples/quickstart.py
```

It trains a real 4-layer transformer for 130 steps, injects one gradient explosion at
step 80, and prints the resulting incident. Expected shape:

```
CONFIRMED INCIDENT
  type:       GRADIENT_EXPLOSION
  promotion:  promote.strong.single (vphase1.3-1.0)
Evidence:
  - GRADIENT_EXPLOSION at step 75, affecting 8/8 monitored layers
Promotion disposition: ELIGIBLE (reason: PROMOTED)
```

## Query the incident

```bash
PYTHONPATH=src python -m training_debugger.cli.main --root <dataset-root> runs
PYTHONPATH=src python -m training_debugger.cli.main --root <dataset-root> verdicts quickstart
PYTHONPATH=src python -m training_debugger.cli.main --root <dataset-root> incidents quickstart
```

`verdicts` reads the **persisted** episode store, so it reports what was recorded, not
what the current code would decide if re-run.

## Wrap your own loop

```python
from training_debugger.observer import TrainingObserver, SamplingPolicy, ObserverConfig
from training_debugger.storage import ParquetEventWriter
from training_debugger.adapters.pytorch import PyTorchTrainingAdapter

observer = TrainingObserver(
    "my-run", writer=ParquetEventWriter("data", "my-run"),
    sampling=SamplingPolicy.uniform(10), config=ObserverConfig(),
)
adapter = PyTorchTrainingAdapter(observer, model, optimizer)

for step, batch in enumerate(loader, start=1):
    with adapter.step(step) as tracker:
        loss = model(**batch)["loss"]
        tracker.record_loss(loss)
        loss.backward()
        tracker.record_gradients()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

`emit()` never raises and never blocks. The observer is fail-open by design: it drops
rather than delaying training, and a verdict records whether completeness was
demonstrated.
