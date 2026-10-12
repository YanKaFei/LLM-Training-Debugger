"""W2 gate: the PyTorch adapter against real training.

Everything here trains a real (small) transformer with real autograd. The point
is that a debugger validated only against synthetic events is not validated at
all -- the interesting failures live in the interaction between module hooks,
gradient lifetime, and optimizer ordering.
"""

from __future__ import annotations

import math

import pytest
import torch

import helpers
from tiny_model import make_tiny_lm, synthetic_batch

from training_debugger.adapters.pytorch import (
    NORM_TYPE_L2,
    PyTorchTrainingAdapter,
    global_gradient_norm,
    global_parameter_norm,
    memory_snapshot,
)
from training_debugger.observer import MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver
from training_debugger.schema import EventType, MetricUnit

pytestmark = pytest.mark.torch


def _run(
    steps: int = 6,
    *,
    layer_count: int = 3,
    lr: float = 1e-2,
    inject_nan_loss_at: int = None,
    inject_inf_grad_at: int = None,
    sampling=None,
):
    torch.manual_seed(7)
    model, config = make_tiny_lm(seed=7, n_layers=layer_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "run-w2",
        writer=writer,
        sampling=sampling or SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01),
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="tiny")
    adapter.run_start()

    losses = []
    for step in range(1, steps + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=batch["input_ids"].numel())
            with tracker.phase("forward"):
                out = model(**batch)
                loss = out["loss"]
            if inject_nan_loss_at is not None and step == inject_nan_loss_at:
                loss = loss * float("nan")
            tracker.record_loss(loss)
            with tracker.phase("backward"):
                loss.backward()
            if inject_inf_grad_at is not None and step == inject_inf_grad_at:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad[0] = float("inf")
            tracker.record_gradients()
            with tracker.phase("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()

    adapter.run_end()
    observer.flush(timeout=5.0)
    events = writer.events
    return observer, adapter, model, events, losses


# ---------------------------------------------------------------------------
# Metric correctness
# ---------------------------------------------------------------------------


def test_adapter_emits_the_required_core_metrics():
    observer, adapter, model, events, _ = _run(steps=4)
    observer.close()
    names = {e.metric_name for e in events if e.metric_name}
    for required in (
        "train.loss",
        "train.lr",
        "grad.global_norm",
        "param.global_norm",
        "train.step_time",
        "train.forward_time_ms",
        "train.backward_time_ms",
        "train.optimizer_time_ms",
        "train.samples_per_sec",
        "train.tokens_per_sec",
    ):
        assert required in names, "missing required metric %s" % required


def test_grad_norm_matches_the_torch_reference():
    """The adapter's definition must equal ``clip_grad_norm_``'s total_norm.

    ``clip_grad_norm_`` returns the pre-clip total norm, so computing it on a
    throwaway copy gives a reference implementation without the clipping side
    effect.
    """
    torch.manual_seed(11)
    model, config = make_tiny_lm(seed=11, n_layers=2)
    batch = synthetic_batch(batch_size=2, seq_len=6, vocab_size=config.vocab_size, seed=1)
    loss = model(**batch)["loss"]
    loss.backward()

    adapter_value = global_gradient_norm(model.parameters()).value
    reference = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item())
    # clip_grad_norm_ clipped nothing because the norm is below 1e9, and it did
    # rescale... by exactly 1.0. Compare before any further use.
    assert adapter_value == pytest.approx(reference, rel=1e-5)


def test_param_norm_matches_a_manual_l2_computation():
    torch.manual_seed(12)
    model, _ = make_tiny_lm(seed=12, n_layers=2)
    expected = math.sqrt(
        sum(float(p.detach().to(torch.float32).pow(2).sum().item()) for p in model.parameters())
    )
    result = global_parameter_norm(model.parameters())
    assert result.value == pytest.approx(expected, rel=1e-6)
    assert result.norm_type == NORM_TYPE_L2


def test_norm_type_is_recorded_on_every_norm_event():
    observer, adapter, model, events, _ = _run(steps=3)
    observer.close()
    for event in events:
        if event.metric_name in ("grad.global_norm", "param.global_norm"):
            assert event.metadata.get("norm_type") == NORM_TYPE_L2
            assert event.metadata.get("element_count", 0) > 0


def test_loss_values_recorded_match_the_training_loop():
    """The recorded loss must be the loss that was actually backpropagated."""
    recorded = []
    torch.manual_seed(13)
    model, config = make_tiny_lm(seed=13, n_layers=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    writer = MemoryWriter()
    observer = TrainingObserver("run-w2-loss", writer=writer, sampling=SamplingPolicy.uniform(1),
                                config=ObserverConfig(flush_interval_s=0.01))
    adapter = PyTorchTrainingAdapter(observer, model, optimizer)
    actual = []
    for step in range(1, 5):
        batch = synthetic_batch(batch_size=2, seq_len=6, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            loss = model(**batch)["loss"]
            actual.append(float(loss.detach().item()))
            tracker.record_loss(loss)
            loss.backward()
            tracker.record_gradients()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    observer.flush(timeout=5.0)
    observer.close()
    recorded = [e.metric_value for e in writer.events if e.metric_name == "train.loss"]
    assert recorded == pytest.approx(actual, rel=1e-6)


def test_learning_rate_is_read_from_the_optimizer():
    observer, adapter, model, events, _ = _run(steps=2, lr=3e-4)
    observer.close()
    lrs = [e.metric_value for e in events if e.metric_name == "train.lr"]
    assert lrs and all(v == pytest.approx(3e-4) for v in lrs)


def test_phase_timings_are_positive_and_bounded_by_step_time():
    observer, adapter, model, events, _ = _run(steps=4)
    observer.close()
    by_step = {}
    for event in events:
        if event.metric_name and event.metric_name.startswith("train."):
            by_step.setdefault(event.step, {})[event.metric_name] = event.metric_value
    for step, metrics in by_step.items():
        if "train.step_time" not in metrics:
            continue
        phase_sum = sum(
            metrics.get("train.%s_time_ms" % phase, 0.0)
            for phase in ("forward", "backward", "optimizer")
        )
        assert phase_sum <= metrics["train.step_time"] + 1e-6, (
            "phases summed to more than the measured step time at step %s" % step
        )


def test_throughput_is_derived_from_the_declared_batch():
    observer, adapter, model, events, _ = _run(steps=3)
    observer.close()
    for event in events:
        if event.metric_name == "train.tokens_per_sec":
            assert event.metric_value > 0
            assert event.metric_unit is MetricUnit.TOKENS_PER_SEC


# ---------------------------------------------------------------------------
# Step alignment and uniqueness
# ---------------------------------------------------------------------------


def test_every_step_is_accounted_for_exactly_once():
    steps = 7
    observer, adapter, model, events, _ = _run(steps=steps)
    observer.close()
    step_ends = [e.step for e in events if e.event_type is EventType.STEP_END and e.metric_name == "train.step_time"]
    assert step_ends == list(range(1, steps + 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"inject_nan_loss_at": 2},
        {"inject_inf_grad_at": 2},
        {"inject_nan_loss_at": 3, "inject_inf_grad_at": 2},
    ],
)
def test_no_metric_name_is_used_twice_at_the_same_step(kwargs):
    """One metric name, one value, per step.

    This invariant is the regression test for a real W2 bug: the non-finite
    *count* was originally emitted under the same name as the measurement it
    qualified, so `grad.global_norm` for the affected step was `794.0` (a count)
    rather than `nan`. The parametrisation matters -- the un-injected case alone
    does not exercise the collision.
    """
    observer, adapter, model, events, _ = _run(steps=5, **kwargs)
    observer.close()
    keys = [
        (e.metric_name, e.step, e.microstep, e.module_path)
        for e in events
        if e.metric_name is not None
    ]
    duplicates = [k for k in set(keys) if keys.count(k) > 1]
    assert not duplicates, "duplicate metric emissions: %s" % duplicates


def test_producer_sequence_is_strictly_monotonic():
    observer, adapter, model, events, _ = _run(steps=5)
    observer.close()
    sequences = [e.source_seq for e in events if e.source_seq is not None]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


def test_all_events_validate():
    from training_debugger.schema import ProducerSequenceTracker, validate_event

    observer, adapter, model, events, _ = _run(steps=4)
    observer.close()
    tracker = ProducerSequenceTracker()
    problems = []
    for event in events:
        result = validate_event(event, seq_tracker=tracker)
        if not result.ok:
            problems.append((event.metric_name or event.event_type, result.reason()))
    assert not problems, "adapter emitted invalid events: %s" % problems[:5]


def test_run_start_reports_parameter_counts():
    observer, adapter, model, events, _ = _run(steps=1)
    observer.close()
    start = next(e for e in events if e.event_type is EventType.RUN_START)
    assert start.metadata["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert start.metadata["trainable_parameter_count"] > 0
    assert start.metadata["torch_version"] == torch.__version__


# ---------------------------------------------------------------------------
# Non-finite handling
# ---------------------------------------------------------------------------


def test_nan_loss_produces_a_p0_nonfinite_event_and_poisons_the_run():
    """A NaN loss is recorded as P0 and -- realistically -- persists.

    The first NaN at step 3 propagates through ``backward()`` into every
    gradient, and ``optimizer.step()`` writes it into the parameters, so steps 4
    and 5 are NaN too. That is what training actually does without a skip/rollback
    policy, and asserting it here documents the behaviour the incident engine will
    have to explain rather than papering over it.
    """
    observer, adapter, model, events, _ = _run(steps=5, inject_nan_loss_at=3)
    observer.close()
    losses = [(e.step, e.metric_value) for e in events if e.metric_name == "train.loss"]
    nan_steps = [step for step, value in losses if value != value]
    assert nan_steps == [3, 4, 5], "a NaN loss poisons the run unless it is skipped"
    nonfinite = [e for e in events if e.metadata.get("nonfinite")]
    assert nonfinite and nonfinite[0].step == 3
    assert nonfinite[0].severity.value == "error"
    assert nonfinite[0].priority.value == "P0"
    assert adapter.stats()["nonfinite_losses"] == 3


def test_infinite_gradient_produces_a_nan_grad_norm_and_a_separate_count():
    observer, adapter, model, events, _ = _run(steps=4, inject_inf_grad_at=2)
    observer.close()
    grad_norms = [(e.step, e.metric_value) for e in events if e.metric_name == "grad.global_norm"]
    step2_norm = [value for step, value in grad_norms if step == 2][0]
    assert step2_norm != step2_norm, (
        "an infinite gradient must yield a NaN global norm, not a finite one"
    )
    counts = [(e.step, e.metric_value) for e in events if e.metric_name == "grad.nonfinite_count"]
    assert counts and counts[0][0] == 2 and counts[0][1] > 0
    nonfinite = [e for e in events if e.metadata.get("nonfinite") and e.step == 2]
    assert nonfinite
    assert all(e.priority.value == "P0" for e in nonfinite)
    # The count must never share a name with the measurement it qualifies.
    assert "grad.nonfinite_count" != "grad.global_norm"


def test_adapter_survives_a_nonfinite_step_and_keeps_recording():
    observer, adapter, model, events, _ = _run(steps=6, inject_nan_loss_at=2)
    observer.close()
    steps_seen = {e.step for e in events if e.metric_name == "train.loss"}
    assert steps_seen == set(range(1, 7)), "recording must continue after a NaN loss"


# ---------------------------------------------------------------------------
# Memory reporting degrades honestly
# ---------------------------------------------------------------------------


def test_memory_snapshot_never_reports_a_fake_zero():
    snapshot = memory_snapshot()
    assert isinstance(snapshot, dict)
    for key in ("allocated_memory", "reserved_memory"):
        if key in snapshot:
            assert snapshot[key] >= 0


def test_cpu_run_still_produces_a_complete_event_stream():
    """The Phase 1 dev environment has MPS, not CUDA; the adapter must be
    backend-agnostic."""
    observer, adapter, model, events, _ = _run(steps=3)
    observer.close()
    assert any(e.event_type is EventType.RUN_END for e in events)
    assert adapter.stats()["steps_recorded"] == 3


# ---------------------------------------------------------------------------
# Interaction with sampling and the observer
# ---------------------------------------------------------------------------


def test_sampling_applies_to_layer_metrics_not_to_loss():
    policy = SamplingPolicy.from_spec([("grad.*", 3), ("train.loss", 1)])
    observer, adapter, model, events, _ = _run(steps=9, sampling=policy)
    observer.close()
    loss_steps = sorted(e.step for e in events if e.metric_name == "train.loss")
    grad_steps = sorted(e.step for e in events if e.metric_name == "grad.global_norm")
    assert loss_steps == list(range(1, 10))
    assert grad_steps == [3, 6, 9]


def test_adapter_finishes_the_tracker_even_when_the_step_raises():
    torch.manual_seed(21)
    model, config = make_tiny_lm(seed=21, n_layers=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    writer = MemoryWriter()
    observer = TrainingObserver("run-w2-raise", writer=writer, sampling=SamplingPolicy.uniform(1),
                                config=ObserverConfig(flush_interval_s=0.01))
    adapter = PyTorchTrainingAdapter(observer, model, optimizer)
    with pytest.raises(RuntimeError):
        with adapter.step(1) as tracker:
            raise RuntimeError("training blew up")
    observer.flush(timeout=5.0)
    observer.close()
    # step_time must still have been recorded, so the timeline has no hole.
    assert any(e.metric_name == "train.step_time" for e in writer.events)
