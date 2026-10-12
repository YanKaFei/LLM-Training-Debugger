"""Per-module activation health — the capability gap a competitor exposed.

This project observed gradient norms per layer and nothing about activations, while
NeuralDBG classifies `Vanishing`/`Exploding`/`Dead`/`Saturated` per named module.

The design rule is the one this repository has already paid for twice: **do not invent a
threshold.** Only the threshold-free states are produced.

```
act.nonfinite_count > 0     -> nonfinite    exact
act.zero_fraction >= 1.0    -> dead         exact
act.abs_max >= float32 max  -> saturated    exact
vanishing / exploding       -> NOT produced (they need a reference)
```
"""

from __future__ import annotations

import torch

from training_debugger.adapters.pytorch.activations import (
    ACTIVATION_METRICS, ActivationProbe, ActivationStats, HealthState, health_from_stats,
)


def stats(**kwargs):
    base = dict(module_path="m", step=1, elements=100, sampled=False,
                zero_fraction=0.0, abs_max=1.0, abs_mean=0.5, global_norm=5.0,
                nonfinite_count=0)
    base.update(kwargs)
    return ActivationStats(**base)


# ---------------------------------------------------------------------------
# Exact classification
# ---------------------------------------------------------------------------


def test_all_zero_elements_is_dead():
    state, reason = health_from_stats(stats(zero_fraction=1.0, abs_max=0.0))
    assert state == HealthState.DEAD
    assert "exactly zero" in reason


def test_a_non_finite_element_is_reported():
    state, reason = health_from_stats(stats(nonfinite_count=3))
    assert state == HealthState.NONFINITE
    assert "3 of 100" in reason


def test_reaching_the_dtype_maximum_is_saturation():
    state, reason = health_from_stats(stats(abs_max=3.4028234663852886e38),
                                      dtype_max=3.4028234663852886e38)
    assert state == HealthState.SATURATED
    assert "dtype maximum" in reason


def test_healthy_needs_no_threshold_to_establish():
    state, reason = health_from_stats(stats())
    assert state == HealthState.HEALTHY
    assert "no exact failure condition" in reason


def test_nonfinite_outranks_dead():
    """A dead module that is also non-finite is reported as non-finite: that is the
    more specific observation."""
    state, _ = health_from_stats(stats(zero_fraction=1.0, nonfinite_count=1))
    assert state == HealthState.NONFINITE


# ---------------------------------------------------------------------------
# Sampling can only ever weaken a claim, never strengthen it
# ---------------------------------------------------------------------------


def test_a_sampled_all_zero_observation_cannot_claim_dead():
    """A sample of zeros does not prove the whole tensor is zero."""
    state, reason = health_from_stats(stats(zero_fraction=1.0, sampled=True))
    assert state == HealthState.UNKNOWN
    assert "sampled" in reason


def test_sampling_does_not_weaken_a_positive_finding():
    """Finding one non-finite element IS proof, however small the sample."""
    state, _ = health_from_stats(stats(nonfinite_count=1, sampled=True))
    assert state == HealthState.NONFINITE


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def test_the_probe_is_disabled_by_default():
    probe = ActivationProbe()
    assert probe.enabled is False
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert probe.attach(model) == 0
    assert probe.attached is False


def test_a_disabled_probe_records_nothing():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    probe = ActivationProbe(enabled=False)
    probe.attach(model)
    model(torch.randn(2, 4))
    assert probe.collect(1) == []
    assert probe.records == []


def test_attach_and_detach_are_reversible_and_bounded():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    probe = ActivationProbe(enabled=True)
    attached = probe.attach(model)
    assert attached > 0 and probe.attached
    probe.detach()
    assert probe.attached is False
    probe.detach()          # idempotent


def test_an_attached_probe_observes_real_activations():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU())
    probe = ActivationProbe(enabled=True, sample_limit=None)
    probe.attach(model)
    model(torch.randn(3, 4))
    produced = probe.collect(1)
    assert produced, "an attached probe must record something"
    assert all(s.elements > 0 for s in produced)


def test_the_metrics_emitted_are_the_declared_ones():
    assert set(ACTIVATION_METRICS) == {
        "act.zero_fraction", "act.abs_max", "act.abs_mean",
        "act.global_norm", "act.nonfinite_count",
    }


def test_edge_triggered_emission_is_the_default():
    """+24.7 % overhead became +3.3 % by emitting only on a state CHANGE.

    The activation computation itself costs +2.7 %; emitting five metrics per module per
    step cost a further +23.5 % (1,600 events over 80 steps with four modules). A module
    healthy for a thousand steps does not need a thousand events saying so.
    """
    probe = ActivationProbe(enabled=True)
    assert probe.emit_on_change_only is True
    assert probe.summary()["emit_on_change_only"] is True


def test_the_summary_counts_every_state():
    probe = ActivationProbe(enabled=True)
    probe.records = [stats(zero_fraction=1.0, abs_max=0.0), stats(),
                     stats(nonfinite_count=1)]
    counts = probe.summary()["states"]
    assert counts["dead"] == 1
    assert counts["healthy"] == 1
    assert counts["nonfinite"] == 1
