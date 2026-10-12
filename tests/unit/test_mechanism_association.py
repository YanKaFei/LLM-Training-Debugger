"""W6: mechanism-constrained association.

The v0.1.2 rule used temporal proximity and scope overlap alone. Measured on the
mixed-fault corpus it produced `POSSIBLE_CONSEQUENCE_OF` on **10 of 10 deliberately
independent** fault pairs and on **0 of 10** genuine causal chains.

These tests pin the evidence hierarchy:

```
E1 temporal order      -> only PRECEDES
E2 scope compatibility
E3 mechanistic propagation -> POSSIBLE_CONSEQUENCE_OF
E4 contradiction / independence
```
"""

from __future__ import annotations

import pytest

from training_debugger.episodes import AnomalyEpisode, EpisodeState
from training_debugger.incidents.relations import (
    PROPAGATION_DELAY_STEPS, RelationType, link_consequences, metric_class,
)


def episode(eid, anomaly_type, metric, onset, *, confirmed=None, layers=(0, 1, 2),
            outcome="CONFIRMED_ANOMALY"):
    return AnomalyEpisode(
        episode_id=eid, run_id="r", detector_id="d", detector_version="1",
        rule_version="v", anomaly_type=anomaly_type, signal_key="k",
        metric_name=metric,
        state=(EpisodeState.CONFIRMED.value if confirmed is not None
               else EpisodeState.CLOSED.value),
        outcome=outcome, candidate_onset_step=onset, confirmed_at_step=confirmed,
        affected_layers=list(layers), confidence_class="CERTAIN", confidence=0.95,
    )


# ---------------------------------------------------------------------------
# metric classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric,expected", [
    ("grad.global_norm", "upstream"),
    ("grad.layer_norm", "upstream"),
    ("param.global_norm", "midstream"),
    ("mem.allocated_memory", "midstream"),
    ("train.loss", "downstream"),
    ("train.loss_nonfinite_count", "downstream"),
    (None, "unknown"),
])
def test_metric_class_follows_the_training_mechanism(metric, expected):
    assert metric_class(metric) == expected


# ---------------------------------------------------------------------------
# E3 — the M1 recovery
# ---------------------------------------------------------------------------


def test_nonfinite_gradient_one_step_before_the_loss_is_a_mechanism_link():
    """M1: the real chain, with the propagation observable at EVERY stage.

    v0.1.4 requires the midstream witness. A non-finite *gradient* followed by a
    non-finite *loss* is not enough: the adversarial corpus showed a suppressed update
    produces exactly that pattern while nothing propagates, and it was claimed as a
    consequence five times. The parameter norm is what shows the update really
    overflowed.
    """
    episodes = [
        episode("grad_inf", "NAN_DETECTED", "grad.global_norm", 80, confirmed=80),
        episode("param_inf", "NAN_DETECTED", "param.global_norm", 80, confirmed=80),
        episode("loss_nan", "NAN_DETECTED", "train.loss_nonfinite_count", 81, confirmed=81),
    ]
    relations, linked = link_consequences("r", episodes, mechanism_linker_enabled=True)
    # Any of the three targets may key the relation; the mechanism one is asserted.
    mechanism = [r for r in relations if r.relation_type ==
                 RelationType.POSSIBLE_CONSEQUENCE_OF.value]
    assert mechanism, "a fully observed propagation must be recognised"
    to_loss = [r for r in mechanism if r.target_incident_id == "loss_nan"]
    assert to_loss, "the downstream loss must be linked to the upstream gradient"
    relation = to_loss[0]
    assert relation.relation_type == RelationType.POSSIBLE_CONSEQUENCE_OF.value
    assert relation.evidence["tier"] == "MECHANISM"
    assert relation.gap_steps == 1
    assert relation.evidence["upstream_nonfinite_before_target"] is True
    assert relation.evidence["antecedent_is_nonfinite"] is True
    assert relation.evidence["midstream_nonfinite_witness"] is True
    assert "NOT a causal identification" in relation.evidence["interpretation"]


def test_the_propagation_bound_is_a_mechanism_bound_not_a_tuned_window():
    assert PROPAGATION_DELAY_STEPS == 2


def test_a_nonfinite_gradient_beyond_the_propagation_bound_is_not_a_consequence():
    """Beyond one optimizer update, forward propagation is not mechanically supported."""
    episodes = [
        episode("grad_inf", "NAN_DETECTED", "grad.global_norm", 80, confirmed=80),
        episode("param_inf", "NAN_DETECTED", "param.global_norm", 80, confirmed=80),
        episode("loss_nan", "NAN_DETECTED", "train.loss", 90, confirmed=90),
    ]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    targets = [r for r in relations if r.target_incident_id == "loss_nan"]
    assert targets
    assert targets[0].relation_type == RelationType.PRECEDES.value
    assert targets[0].evidence["tier"] == "TEMPORAL"


def test_a_suppressed_update_is_not_a_consequence_even_one_step_later():
    """W2/A2: the gradient norm is non-finite, but the update never saw it.

    No `param.global_norm` non-finite observation means the propagation was never
    observed, so the downstream failure must not be claimed as its consequence. This
    exact pattern produced 10 fabricated associations in the adversarial corpus.
    """
    episodes = [
        episode("grad_inf", "NAN_DETECTED", "grad.global_norm", 80, confirmed=80),
        episode("loss_nan", "NAN_DETECTED", "train.loss", 81, confirmed=81),
    ]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert relations
    assert all(r.relation_type != RelationType.POSSIBLE_CONSEQUENCE_OF.value
               for r in relations), (
        "without a midstream witness the propagation is unobserved, so no consequence "
        "may be claimed"
    )
    assert all(r.evidence["midstream_nonfinite_witness"] is False for r in relations)


# ---------------------------------------------------------------------------
# E1 only — the M3 fix
# ---------------------------------------------------------------------------


def test_a_finite_gradient_anomaly_40_steps_earlier_is_not_a_consequence():
    """M3: the fabricated association. A finite scale anomaly is not a mechanism."""
    episodes = [
        episode("grad", "GRADIENT_EXPLOSION", "grad.global_norm", 80, confirmed=80),
        episode("nan", "NAN_DETECTED", "train.loss_nonfinite_count", 120, confirmed=120),
    ]
    relations, linked = link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert len(relations) == 1
    assert relations[0].relation_type == RelationType.PRECEDES.value, (
        "a finite gradient anomaly 40 steps earlier must not be claimed as a "
        "consequence; the old rule fabricated exactly this"
    )
    assert relations[0].relation_type != RelationType.POSSIBLE_CONSEQUENCE_OF.value
    assert relations[0].evidence["tier"] == "TEMPORAL"
    assert "asserts precedence, not consequence" in relations[0].evidence["interpretation"]
    assert linked == {"nan"}, "an order-only relation must not absorb the antecedent"


def test_temporal_proximity_alone_never_produces_a_consequence_claim():
    """Even one step apart, a finite scale anomaly is not a mechanism."""
    episodes = [
        episode("grad", "GRADIENT_EXPLOSION", "grad.global_norm", 80, confirmed=80),
        episode("nan", "NAN_DETECTED", "train.loss", 81, confirmed=81),
    ]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert relations
    assert all(r.relation_type != RelationType.POSSIBLE_CONSEQUENCE_OF.value
               for r in relations), (
        "the antecedent is a finite scale anomaly, so no propagation was observed"
    )


# ---------------------------------------------------------------------------
# E4 — independence and abstention
# ---------------------------------------------------------------------------


def test_a_standalone_nonfinite_value_gets_no_relation():
    episodes = [episode("solo", "NAN_DETECTED", "train.loss", 50, confirmed=50)]
    relations, linked = link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert relations == []
    assert linked == set(), "an independent fault must stay free to open its own incident"


def test_no_mechanistic_evidence_means_abstention_not_a_guess():
    """A non-finite value with nothing upstream must not be attached to anything."""
    episodes = [
        episode("param", "NAN_DETECTED", "param.global_norm", 60, confirmed=60),
    ]
    relations, linked = link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert relations == [] and linked == set()


def test_the_consequence_cannot_precede_its_cause():
    episodes = [
        episode("grad_inf", "NAN_DETECTED", "grad.global_norm", 100, confirmed=100),
        episode("loss_nan", "NAN_DETECTED", "train.loss", 90, confirmed=90),
    ]
    relations, linked = link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert relations == [], "a value that came first cannot be a consequence"
    assert linked == set()


def test_disjoint_layer_scope_blocks_association():
    episodes = [
        episode("grad_inf", "NAN_DETECTED", "grad.global_norm", 80, confirmed=80,
                layers=(0,)),
        episode("loss_nan", "NAN_DETECTED", "train.loss", 81, confirmed=81, layers=(9,)),
    ]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert relations == [], "no shared scope means no path"


def test_no_causal_relation_type_is_introduced():
    names = {t.value for t in RelationType}
    assert names == {"POSSIBLE_CONSEQUENCE_OF", "PRECEDES", "CO_OCCURS_WITH",
                     "UNKNOWN_RELATION"}
    assert "CAUSED_BY" not in names and "ROOT_CAUSE_OF" not in names


# ---------------------------------------------------------------------------
# Idempotency and evidence preservation
# ---------------------------------------------------------------------------


def test_linking_is_idempotent():
    episodes = [
        episode("grad_inf", "NAN_DETECTED", "grad.global_norm", 80, confirmed=80),
        episode("param_inf", "NAN_DETECTED", "param.global_norm", 80, confirmed=80),
        episode("loss_nan", "NAN_DETECTED", "train.loss", 81, confirmed=81),
    ]
    first, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    second, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert [r.relation_id for r in first] == [r.relation_id for r in second]
    assert [r.relation_type for r in first] == [r.relation_type for r in second]


def test_linking_preserves_onset_and_confirmation():
    episodes = [
        episode("grad_inf", "NAN_DETECTED", "grad.global_norm", 80, confirmed=80),
        episode("param_inf", "NAN_DETECTED", "param.global_norm", 80, confirmed=80),
        episode("loss_nan", "NAN_DETECTED", "train.loss", 81, confirmed=81),
    ]
    link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert episodes[0].candidate_onset_step == 80
    assert episodes[0].confirmed_at_step == 80
    assert episodes[0].confirmed_at_step == 80, (
        "association must not rewrite when something was confirmed"
    )


def test_the_decision_records_its_evidence():
    episodes = [
        episode("grad_inf", "NAN_DETECTED", "grad.global_norm", 80, confirmed=80),
        episode("param_inf", "NAN_DETECTED", "param.global_norm", 80, confirmed=80),
        episode("loss_nan", "NAN_DETECTED", "train.loss", 81, confirmed=81),
    ]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    evidence = relations[0].evidence
    for key in ("tier", "gap_steps", "propagation_delay_bound", "antecedent_metric",
                "antecedent_metric_class", "target_metric", "target_metric_class",
                "antecedent_is_nonfinite", "scope_compatible", "interpretation",
                "midstream_nonfinite_witness", "midstream_required"):
        assert key in evidence, "a relation must be auditable: {} missing".format(key)


# ---------------------------------------------------------------------------
# R2 — the production default must never assert an unsupported consequence
# ---------------------------------------------------------------------------


def test_the_mechanism_linker_is_disabled_by_default():
    """The safety property the release candidate rests on.

    `POSSIBLE_CONSEQUENCE_OF` was measured to be undiscriminable from an independently
    injected fault in every case that could be constructed (C3/C12: 16 claims against
    `INDEPENDENT` ground truth). Shipping it by default would present an unvalidated
    inference as a causal claim, so the default records ORDER and stops there.
    """
    from training_debugger.incidents.relations import MECHANISM_LINKER_ENABLED_DEFAULT

    assert MECHANISM_LINKER_ENABLED_DEFAULT is False

    episodes = [
        episode("grad_inf", "NAN_DETECTED", "grad.global_norm", 80, confirmed=80),
        episode("param_inf", "NAN_DETECTED", "param.global_norm", 80, confirmed=80),
        episode("loss_nan", "NAN_DETECTED", "train.loss", 81, confirmed=81),
    ]
    relations, _ = link_consequences("r", episodes)
    assert relations, "the order is still recorded"
    assert all(r.relation_type != RelationType.POSSIBLE_CONSEQUENCE_OF.value
               for r in relations), (
        "the production default must not assert a possible consequence"
    )
    assert all(r.evidence["mechanism_linker_enabled"] is False for r in relations)


def test_the_research_configuration_is_opt_in_and_reversible():
    episodes = [
        episode("grad_inf", "NAN_DETECTED", "grad.global_norm", 80, confirmed=80),
        episode("param_inf", "NAN_DETECTED", "param.global_norm", 80, confirmed=80),
        episode("loss_nan", "NAN_DETECTED", "train.loss", 81, confirmed=81),
    ]
    off, _ = link_consequences("r", episodes)
    on, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert {r.relation_type for r in off} == {RelationType.PRECEDES.value}
    assert any(r.relation_type == RelationType.POSSIBLE_CONSEQUENCE_OF.value for r in on)
    # The same evidence is present in both; only the claim differs.
    assert len(off) == len(on)
