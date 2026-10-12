# W2 REPORT — PyTorch Adapter

**Wave:** W2
**Status:** **COMPLETE**
**Gate:** real training produces a correct, complete, duplicate-free event stream — **PASS** (23 integration tests on a real transformer)

---

## 1. What was built

```
src/training_debugger/adapters/pytorch/
  adapter.py   PyTorchTrainingAdapter, StepTracker, COUNT_METRICS
  metrics.py   global_gradient_norm, global_parameter_norm, memory_snapshot
tests/tiny_model.py                        a real small transformer
tests/integration/test_pytorch_adapter.py  23 tests
```

The adapter **observes a loop the user already owns** rather than wrapping it in
a `Trainer`. That is what keeps it framework-agnostic and keeps training
semantics out of the debugger.

```python
adapter = PyTorchTrainingAdapter(observer, model, optimizer)
adapter.run_start()
for step, batch in enumerate(loader, start=1):
    with adapter.step(step) as tracker:
        tracker.set_batch(samples=..., tokens=...)
        with tracker.phase("forward"):
            out = model(**batch); loss = out["loss"]
        tracker.record_loss(loss)
        with tracker.phase("backward"):
            loss.backward()
        tracker.record_gradients()          # BEFORE optimizer.step()
        with tracker.phase("optimizer"):
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        tracker.record_parameters()
        tracker.record_learning_rate()
adapter.run_end()
```

---

## 2. Metric definitions are fixed, not per-adapter

Phase 1 brief §4.2 and §4.3 require the *definitions* to be pinned. `metrics.py`
enforces four properties, each with a reason:

| Property | Why |
|---|---|
| **L2 by default, and `norm_type` recorded on every event** | A baseline computed as one norm and an incident computed as another is not comparable; recording the type makes that visible instead of silent. |
| **Sum-of-squares over all elements** (i.e. what `clip_grad_norm_` returns as `total_norm`) | Matches the reference implementation; verified against it with `rel=1e-5`. |
| **Computed in float32** | Summing 10⁹ bf16 squares in bf16 saturates. |
| **Non-mutating** | Gradient norms are computed *without* calling `clip_grad_norm_`, which would rescale the gradients as a side effect and corrupt the next step's measurement. |

`global_parameter_norm` is documented against the confusion the brief warns
about: it is `sqrt(sum(p²))`, **not** `mean(abs(p))`.

`memory_snapshot` reports only the keys the current backend can actually provide,
so a caller never sees a `0` that means "unknown" rather than "zero". On this
machine that is MPS; on a CUDA box it would be `torch.cuda.*`.

---

## 3. A real bug found by the W2 tests

**Two events shared one metric name at one step.**

`record_gradients()` emitted the non-finite *count* under the metric name
`grad.global_norm` — the same name as the norm it qualified. A consumer reading
`grad.global_norm` for the affected step therefore got `794.0` (a count) instead
of `nan`. Because the count was emitted *first*, a naive reader took the wrong
value entirely.

This is precisely the silent corruption Phase 0 warned about, and it was invisible
until a test ran with an injected anomaly. Two fixes:

1. Non-finite observations now have their own names, allocated from one place:
   ```python
   COUNT_METRICS = {"loss": "train.loss_nonfinite_count",
                    "gradient": "grad.nonfinite_count"}
   ```
2. The boolean `nonfinite` flag now belongs **only** to the count event; the
   measurement carries the descriptive `nonfinite_count` in its metadata. The two
   are no longer confusable.

The regression test is parametrised over four injection combinations, because the
un-injected case alone does not exercise the collision:

```python
@pytest.mark.parametrize("kwargs", [{}, {"inject_nan_loss_at": 2},
                                    {"inject_inf_grad_at": 2},
                                    {"inject_nan_loss_at": 3, "inject_inf_grad_at": 2}])
def test_no_metric_name_is_used_twice_at_the_same_step(kwargs): ...
```

**General lesson recorded for the store:** "one metric name, one value, per step"
is an invariant the ingest path should also enforce. It is currently enforced by
test; making it a validation rule is W6 work.

---

## 4. Real training behaviour the tests document

`test_nan_loss_produces_a_p0_nonfinite_event_and_poisons_the_run` asserts that a
NaN loss at step 3 makes steps 4 and 5 NaN too — because the NaN propagates
through `backward()` into every gradient and `optimizer.step()` writes it into the
parameters. That is what training actually does without a skip/rollback policy.

This is not an adapter defect; it is a fact the incident engine will have to
explain. Recording it as an assertion means the W6 engine is developed against
real behaviour rather than an idealised one.

---

## 5. Gate results

```
$ .venv/bin/python -m pytest tests/integration -q
23 passed        (real transformer, real autograd, real optimizer)
```

| Gate item | Status | Evidence |
|---|---|---|
| all core metrics collected | **PASS** | 10 required metric names asserted present |
| gradient norm definition correct | **PASS** | equals `clip_grad_norm_`'s `total_norm` to `rel=1e-5` |
| parameter norm definition correct | **PASS** | matches an independent manual L2 to `rel=1e-6` |
| loss values match the loop | **PASS** | recorded loss == backpropagated loss |
| step alignment | **PASS** | `STEP_END` steps == `1..N` exactly |
| no duplicate events | **PASS** | parametrised over 4 anomaly-injection cases |
| producer sequence monotonic | **PASS** | strictly increasing, all unique |
| every emitted event validates | **PASS** | re-validated through `validate_event` |
| NaN/Inf detected as P0 | **PASS** | `train.loss_nonfinite_count`, `grad.nonfinite_count` |
| recording continues after a NaN | **PASS** | steps 1..6 all present |
| tracker finishes even on exception | **PASS** | step_time present after a raised step |

---

## 6. Not verified in W2

* **CUDA.** This machine is an Apple M2 Max. `torch.cuda` paths in `memory_snapshot`
  are **untested**; the MPS path is tested. The adapter is backend-agnostic by
  construction, but "untested on CUDA" is the honest statement.
* **Multi-GPU / FSDP.** `_parameters()` is documented as the single place that
  would change for gradient accumulation across replicas. No FSDP run exists yet;
  the brief scopes Phase 1 to single-process and only requires the *architecture*
  to be multi-GPU compatible.
* **Gradient accumulation.** `microstep` is plumbed through but no test drives a
  multi-microstep step.
* **AMP / grad scaler.** `GRAD_SCALER_STATE` is in the vocabulary but the adapter
  does not emit it.
* **`torch.compile`.** Not tested; a compiled model may change module traversal.
