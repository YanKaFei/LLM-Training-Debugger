"""W2/W3: the opt-in Optimizer Boundary Witness.

W0 established that `*.global_norm == inf` is a float32 **reduction artifact**, not a
non-finite tensor. Excluding it from mechanism evidence removed *every* relation claim on
both corpora, including the genuine ones — so the existing telemetry contains no direct
observation of the optimizer boundary at all. This witness supplies one, off by default.

The contract under test:

```
A witnessed optimizer call is not necessarily an applied parameter update.
A non-finite norm is not necessarily a non-finite tensor.
UNKNOWN is never False.
```
"""

from __future__ import annotations

import pytest
import torch

from training_debugger.adapters.pytorch import (
    OptimizerStepStatus, OptimizerWitness, WITNESS_SCHEMA_VERSION,
)


def model_and_optimizer(kind="sgd", **kwargs):
    model = torch.nn.Linear(8, 4)
    optimizer = (torch.optim.SGD(model.parameters(), lr=0.01, **kwargs) if kind == "sgd"
                 else torch.optim.Adam(model.parameters(), lr=0.01, **kwargs))
    return model, optimizer


# ---------------------------------------------------------------------------
# Defaults and lifecycle
# ---------------------------------------------------------------------------


def test_the_witness_is_disabled_by_default():
    witness = OptimizerWitness()
    assert witness.enabled is False
    _, optimizer = model_and_optimizer()
    assert witness.attach(optimizer) is False
    assert witness.attached is False


def test_a_disabled_witness_registers_no_hooks_and_emits_nothing():
    model, optimizer = model_and_optimizer()
    witness = OptimizerWitness(enabled=False)
    witness.attach(optimizer)
    witness.note_attempted(training_step_id=1)
    model(torch.zeros(1, 8)).sum().backward()
    optimizer.step()
    witness.note_invoked()
    assert witness.records == [], "a disabled witness must be observably absent"
    assert witness.summary()["attached"] is False


def test_attach_and_detach_are_reversible():
    _, optimizer = model_and_optimizer()
    witness = OptimizerWitness(enabled=True)
    assert witness.attach(optimizer) is True
    assert witness.attached is True
    witness.detach()
    assert witness.attached is False
    witness.detach()          # idempotent


def test_an_optimizer_without_hook_support_yields_unknown_not_false():
    """Never invent a conclusion for an optimizer that cannot be witnessed."""
    class Opaque:
        pass

    witness = OptimizerWitness(enabled=True)
    assert witness.attach(Opaque()) is False
    witness.note_attempted(training_step_id=1)
    witness.note_invoked()
    record = witness.records[0]
    assert record["optimizer_step_status"] == OptimizerStepStatus.UNKNOWN
    assert record["optimizer_step_invoked"] is False
    assert record["observation_confidence"] == "UNKNOWN"
    assert record["witness_source"] == "none"


# ---------------------------------------------------------------------------
# Hook observation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sgd", "adam"])
def test_a_normal_step_is_observed_as_invoked(kind):
    model, optimizer = model_and_optimizer(kind)
    witness = OptimizerWitness(enabled=True)
    assert witness.attach(optimizer)
    witness.note_attempted(training_step_id=1)
    model(torch.zeros(1, 8)).sum().backward()
    optimizer.step()
    witness.note_invoked()
    record = witness.records[0]
    assert record["optimizer_step_status"] == OptimizerStepStatus.INVOKED
    assert record["optimizer_step_invoked"] is True
    assert record["witness_source"] == "torch.optim.register_step_hook"


def test_an_explicitly_reported_skip_is_distinguished_from_an_invocation():
    """The AMP `GradScaler` shape: the call happens and the update does not apply."""
    model, optimizer = model_and_optimizer()
    witness = OptimizerWitness(enabled=True)
    witness.attach(optimizer)
    witness.note_attempted(training_step_id=1, skip_reason="grad_scaler_skip")
    model(torch.zeros(1, 8)).sum().backward()
    optimizer.step()
    witness.note_invoked()
    record = witness.records[0]
    assert record["optimizer_step_status"] == OptimizerStepStatus.SKIPPED
    assert record["optimizer_step_skipped"] is True
    assert record["skip_reason"] == "grad_scaler_skip"


def test_invoked_is_never_reported_as_parameters_changed():
    """Proving "the parameters did not change" needs a scan, which this may not do."""
    model, optimizer = model_and_optimizer()
    witness = OptimizerWitness(enabled=True)
    witness.attach(optimizer)
    witness.note_attempted(training_step_id=1)
    model(torch.zeros(1, 8)).sum().backward()
    optimizer.step()
    witness.note_invoked()
    assert witness.records[0]["parameters_changed"] is None


def test_the_witness_survives_an_optimizer_step_with_no_gradients():
    """Hooks fire even with no grads, so invocation alone proves nothing applied."""
    _, optimizer = model_and_optimizer()
    witness = OptimizerWitness(enabled=True)
    witness.attach(optimizer)
    witness.note_attempted(training_step_id=1)
    optimizer.step()
    witness.note_invoked()
    record = witness.records[0]
    assert record["optimizer_step_status"] == OptimizerStepStatus.INVOKED
    assert record["parameters_changed"] is None


# ---------------------------------------------------------------------------
# W3 — training step vs optimizer step semantics
# ---------------------------------------------------------------------------


def test_gradient_accumulation_separates_training_step_from_optimizer_step():
    """`training_step_id != optimizer_step_id` must be representable."""
    model, optimizer = model_and_optimizer()
    witness = OptimizerWitness(enabled=True)
    witness.attach(optimizer)
    accumulate_every = 3
    for training_step in range(1, 7):
        micro = (training_step - 1) % accumulate_every
        model(torch.zeros(1, 8)).sum().backward()
        if micro == accumulate_every - 1:
            witness.note_attempted(training_step_id=training_step,
                                   micro_step_id=micro)
            optimizer.step()
            witness.note_invoked()
            optimizer.zero_grad(set_to_none=True)
    records = witness.records
    assert len(records) == 2, "two optimizer steps across six training steps"
    assert [r["training_step_id"] for r in records] == [3, 6]
    assert [r["optimizer_step_id"] for r in records] == [1, 2]
    assert records[0]["training_step_id"] != records[0]["optimizer_step_id"], (
        "under accumulation the two ids are different concepts and must not be conflated"
    )


def test_a_multi_optimizer_loop_records_each_optimizer_separately():
    first = torch.nn.Linear(8, 4)
    second = torch.nn.Linear(4, 2)
    opt_a = torch.optim.SGD(first.parameters(), lr=0.01)
    opt_b = torch.optim.Adam(second.parameters(), lr=0.01)
    witness_a = OptimizerWitness(enabled=True, optimizer_id="optimizer:a")
    witness_b = OptimizerWitness(enabled=True, optimizer_id="optimizer:b")
    assert witness_a.attach(opt_a)
    assert witness_b.attach(opt_b)

    second(first(torch.zeros(1, 8))).sum().backward()
    witness_a.note_attempted(training_step_id=1)
    opt_a.step()
    witness_a.note_invoked()
    witness_b.note_attempted(training_step_id=1, skip_reason="frozen")
    opt_b.step()
    witness_b.note_invoked()

    assert witness_a.records[0]["optimizer_id"] == "optimizer:a"
    assert witness_a.records[0]["optimizer_step_status"] == OptimizerStepStatus.INVOKED
    assert witness_b.records[0]["optimizer_id"] == "optimizer:b"
    assert witness_b.records[0]["optimizer_step_status"] == OptimizerStepStatus.SKIPPED


# ---------------------------------------------------------------------------
# Persistence and provenance
# ---------------------------------------------------------------------------


def test_the_witness_record_is_versioned_and_carries_its_provenance():
    _, optimizer = model_and_optimizer()
    witness = OptimizerWitness(enabled=True)
    witness.attach(optimizer)
    witness.note_attempted(training_step_id=1)
    optimizer.step()
    witness.note_invoked()
    record = witness.records[0]
    assert record["schema_version"] == WITNESS_SCHEMA_VERSION
    for key in ("optimizer_id", "training_step_id", "micro_step_id",
                "optimizer_step_id", "optimizer_step_attempted",
                "optimizer_step_invoked", "optimizer_step_skipped",
                "optimizer_step_status", "witness_source", "observation_confidence"):
        assert key in record, "a witness record must be auditable: {} missing".format(key)


def test_the_summary_counts_every_status():
    _, optimizer = model_and_optimizer()
    witness = OptimizerWitness(enabled=True)
    witness.attach(optimizer)
    witness.note_attempted(training_step_id=1)
    optimizer.step()
    witness.note_invoked()
    witness.note_attempted(training_step_id=2, skip_reason="accumulation")
    optimizer.step()
    witness.note_invoked()
    summary = witness.summary()
    assert summary["optimizer_step_count"] == 2
    assert summary["status_counts"]["INVOKED"] == 1
    assert summary["status_counts"]["SKIPPED"] == 1


def test_hooks_do_not_change_training_behaviour():
    """A witness that perturbs the numbers it measures is worthless."""
    def run(enable_witness):
        torch.manual_seed(1234)
        model = torch.nn.Linear(8, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        witness = OptimizerWitness(enabled=enable_witness)
        if enable_witness:
            witness.attach(optimizer)
        losses = []
        for step in range(1, 21):
            torch.manual_seed(step)
            loss = model(torch.randn(4, 8)).sum()
            losses.append(float(loss.detach()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if enable_witness:
                witness.note_attempted(training_step_id=step)
            optimizer.step()
            if enable_witness:
                witness.note_invoked()
        return losses

    assert run(False) == run(True), (
        "enabling the witness must not change a single loss value"
    )
