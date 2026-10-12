"""W4 gate: T1 layer instrumentation correctness and cardinality control."""

from __future__ import annotations

import math

import pytest
import torch

import helpers
from tiny_model import make_tiny_lm, synthetic_batch

from training_debugger.adapters.pytorch import (
    LayerMetricCollector,
    LayerResolver,
    LayerSelection,
)
from training_debugger.observer import MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver

pytestmark = pytest.mark.torch


def _setup(layers=3, top_k=3, cadence=1, selection=None, observer=None):
    torch.manual_seed(5)
    model, config = make_tiny_lm(seed=5, n_layers=layers)
    writer = MemoryWriter()
    obs = observer or TrainingObserver(
        "run-w4", writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01),
    )
    collector = LayerMetricCollector(
        model, obs, selection=selection or LayerSelection(top_k=top_k), cadence=cadence
    )
    return model, config, obs, collector, writer


def _backward(model, config, seed=1):
    batch = synthetic_batch(batch_size=2, seq_len=6, vocab_size=config.vocab_size, seed=seed)
    loss = model(**batch)["loss"]
    loss.backward()
    return loss


# ---------------------------------------------------------------------------
# Layer resolution
# ---------------------------------------------------------------------------


def test_layer_resolver_groups_block_parameters():
    resolver = LayerResolver()
    assert resolver.key_for("blocks.3.mlp.0.weight") == "blocks.3"
    assert resolver.key_for("blocks.11.attn.in_proj_weight") == "blocks.11"
    # A parameter outside a repeated block falls back to its first component.
    assert resolver.key_for("token_embedding.weight") == "token_embedding"


def test_layer_indices_are_dense_and_stable():
    resolver = LayerResolver()
    first = resolver.index_for("blocks.0")
    second = resolver.index_for("blocks.1")
    assert (first, second) == (0, 1)
    assert resolver.index_for("blocks.0") == first, "indices must be stable"
    assert resolver.layer_count == 2


def test_collector_finds_the_models_layers():
    model, config, obs, collector, writer = _setup(layers=3)
    keys = collector.layer_keys()
    assert "blocks.0" in keys and "blocks.2" in keys
    assert collector.layers_measured == len(keys)
    obs.close()


# ---------------------------------------------------------------------------
# Correctness of the reduction
# ---------------------------------------------------------------------------


def test_per_layer_grad_norm_matches_a_manual_computation():
    model, config, obs, collector, writer = _setup(layers=2, top_k=99)
    _backward(model, config)
    batch = collector.compute(1)
    by_key = {layer.layer_key: layer for layer in batch.layers}

    # Independent reference: group the parameters the same way, sum squares.
    expected = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        key = collector.resolver.key_for(name)
        expected[key] = expected.get(key, 0.0) + float(
            parameter.grad.detach().to(torch.float32).pow(2).sum().item()
        )
    for key, total in expected.items():
        assert by_key[key].grad_norm == pytest.approx(math.sqrt(total), rel=1e-5), key
    obs.close()


def test_per_layer_param_norm_matches_a_manual_computation():
    model, config, obs, collector, writer = _setup(layers=2, top_k=99)
    batch = collector.compute(1)
    by_key = {layer.layer_key: layer for layer in batch.layers}
    expected = {}
    for name, parameter in model.named_parameters():
        key = collector.resolver.key_for(name)
        expected[key] = expected.get(key, 0.0) + float(
            parameter.detach().to(torch.float32).pow(2).sum().item()
        )
    for key, total in expected.items():
        assert by_key[key].param_norm == pytest.approx(math.sqrt(total), rel=1e-5), key
    obs.close()


def test_layers_without_gradients_report_none_not_zero():
    """A layer with no gradient has *unknown* norm, which is not the same as 0."""
    model, config, obs, collector, writer = _setup(layers=2, top_k=99)
    batch = collector.compute(1)  # no backward, so no gradients at all
    assert all(layer.grad_norm is None for layer in batch.layers)
    assert all(layer.param_norm is not None for layer in batch.layers)
    obs.close()


def test_nonfinite_gradients_yield_nan_layer_norms():
    model, config, obs, collector, writer = _setup(layers=2, top_k=99)
    _backward(model, config)
    target = dict(model.named_parameters())["blocks.0.mlp.0.weight"]
    target.grad[0, 0] = float("inf")
    batch = collector.compute(1)
    by_key = {layer.layer_key: layer for layer in batch.layers}
    assert by_key["blocks.0"].grad_norm != by_key["blocks.0"].grad_norm  # NaN
    assert by_key["blocks.0"].grad_nonfinite > 0
    obs.close()


def test_vector_hash_changes_when_a_layer_changes():
    model, config, obs, collector, writer = _setup(layers=2, top_k=99)
    _backward(model, config)
    first = collector.compute(1).vector_hash
    target = dict(model.named_parameters())["blocks.1.mlp.0.weight"]
    target.grad.mul_(10.0)
    second = collector.compute(1).vector_hash
    assert first != second, "the vector hash must detect a distribution change"
    obs.close()


# ---------------------------------------------------------------------------
# Cardinality control
# ---------------------------------------------------------------------------


def test_top_k_bounds_the_number_of_per_layer_events():
    model, config, obs, collector, writer = _setup(layers=4, top_k=2)
    _backward(model, config)
    collector.collect(1)
    obs.flush(timeout=5.0)
    emitted = [e for e in writer.events if e.metric_name == "grad.layer_norm"]
    assert len(emitted) == 2, "top_k must bound per-layer emission"
    obs.close()


def test_aggregate_reports_the_population_denominator():
    """Multiplicity: '1 / 16 layers' is a different diagnosis from '16 / 16'."""
    model, config, obs, collector, writer = _setup(layers=4, top_k=3)
    _backward(model, config)
    collector.collect(1)
    obs.flush(timeout=5.0)
    aggregate = next(e for e in writer.events if e.metric_name == "grad.layer_norm_max")
    assert aggregate.metadata["population_layers"] == len(collector.layer_keys())
    assert aggregate.metadata["layer_count"] == len(collector.layer_keys())
    assert aggregate.metadata["vector_hash"]
    obs.close()


def test_max_aggregate_equals_the_largest_finite_layer():
    model, config, obs, collector, writer = _setup(layers=3, top_k=99)
    _backward(model, config)
    batch = collector.compute(1)
    finite = [l.grad_norm for l in batch.layers if l.grad_norm is not None and l.grad_norm == l.grad_norm]
    collector.collect(1)
    obs.flush(timeout=5.0)
    maximum = next(e for e in writer.events if e.metric_name == "grad.layer_norm_max")
    assert maximum.metric_value == pytest.approx(max(finite), rel=1e-6)
    obs.close()


def test_cadence_skips_computation_not_just_emission():
    """The saving must be in the reduction, or the cadence means nothing."""
    model, config, obs, collector, writer = _setup(layers=2, cadence=5)
    _backward(model, config)
    for step in range(1, 11):
        collector.collect(step)
    assert collector.steps_considered == 10
    assert collector.steps_collected == 2, "only steps 5 and 10 should be computed"
    obs.close()


def test_include_filter_restricts_the_measured_layers():
    selection = LayerSelection(include=r"blocks\.[01]$", top_k=99)
    model, config, obs, collector, writer = _setup(layers=3, selection=selection)
    assert set(collector.layer_keys()) == {"blocks.0", "blocks.1"}
    obs.close()


def test_exclude_filter_drops_layers():
    selection = LayerSelection(exclude=r"blocks\.1$", top_k=99)
    model, config, obs, collector, writer = _setup(layers=3, selection=selection)
    assert "blocks.1" not in collector.layer_keys()
    obs.close()


def test_max_layers_caps_cardinality():
    selection = LayerSelection(max_layers=2, top_k=99)
    model, config, obs, collector, writer = _setup(layers=4, selection=selection)
    assert collector.layers_measured == 2
    obs.close()


def test_selection_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        LayerSelection(top_k=-1)
    with pytest.raises(ValueError):
        LayerSelection(max_layers=0)


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_collector_failure_does_not_disturb_training():
    model, config, obs, collector, writer = _setup(layers=2)
    # Break the collector's view of the model and confirm collect() degrades.
    collector._selected = [(k, i, None) for k, i, _ in collector._selected]  # type: ignore
    result = collector.collect(1)  # must not raise
    assert result is None or result.layer_count >= 0
    assert obs.stats().observer_internal_errors_total >= 1
    obs.close()


def test_parse_failure_inside_collect_is_isolated():
    model, config, obs, collector, writer = _setup(layers=2)

    def exploding_compute(step):
        raise RuntimeError("synthetic compute failure")

    collector.compute = exploding_compute  # type: ignore[assignment]
    assert collector.collect(1) is None
    assert obs.stats().observer_internal_errors_total >= 1
    assert obs.emit(event_type=__import__(
        "training_debugger.schema", fromlist=["EventType"]
    ).EventType.WARNING) in (True, False)
    obs.close()


def test_events_emitted_by_the_collector_are_unique_per_step():
    model, config, obs, collector, writer = _setup(layers=3, top_k=3)
    for step in range(1, 4):
        _backward(model, config, seed=step)
        collector.collect(step)
    obs.flush(timeout=5.0)
    keys = [
        (e.metric_name, e.step, e.module_path)
        for e in writer.events
        if e.metric_name is not None
    ]
    duplicates = [k for k in set(keys) if keys.count(k) > 1]
    assert not duplicates, duplicates
    obs.close()


def test_collector_stats_are_reported():
    model, config, obs, collector, writer = _setup(layers=2, cadence=2)
    _backward(model, config)
    for step in range(1, 7):
        collector.collect(step)
    stats = collector.stats()
    assert stats["steps_considered"] == 6
    assert stats["steps_collected"] == 3
    assert stats["cadence"] == 2
    assert stats["layers_in_model"] >= 2
    obs.close()
