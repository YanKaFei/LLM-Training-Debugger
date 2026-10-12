"""W0 gate: partial ordering.

This file is the executable specification of Phase 1 brief sections 2.2-2.4 and
8.4. If a future change makes cross-rank ordering look convenient, these tests
are the thing that should stop it.
"""

from __future__ import annotations

import pytest

import helpers
from training_debugger.schema import (
    EventOrdering,
    EventType,
    EvidenceLevel,
    MetricUnit,
    OrderingConfidence,
    OrderingRelation,
)

P0 = "pytorch-adapter:r0:pid100"
P1 = "pytorch-adapter:r1:pid200"


def _grad(step, value, producer=P0, seq=1, rank=0):
    return helpers.event(
        EventType.GRADIENT_METRIC, producer=producer, rank=rank, source_seq=seq, step=step,
        metric_name="grad.global_norm", metric_value=value, metric_unit=MetricUnit.DIMENSIONLESS,
    )


def _loss(step, value, producer=P0, seq=2, rank=0):
    return helpers.event(
        EventType.LOSS, producer=producer, rank=rank, source_seq=seq, step=step,
        metric_name="train.loss", metric_value=value, metric_unit=MetricUnit.DIMENSIONLESS,
    )


# ---------------------------------------------------------------------------
# Provider-local ordering is total and proven
# ---------------------------------------------------------------------------


def test_same_producer_sequence_proves_before():
    a = _grad(810, 9.2, seq=10)
    b = _loss(812, 4.7, seq=11)
    result = EventOrdering.compare(a, b)
    assert result.relation is OrderingRelation.BEFORE
    assert result.is_proven
    assert result.confidence is OrderingConfidence.PROVEN
    assert result.evidence == "producer_sequence"


def test_same_producer_sequence_reverses_correctly():
    a = _grad(810, 9.2, seq=10)
    b = _loss(812, 4.7, seq=11)
    assert EventOrdering.compare(b, a).relation is OrderingRelation.AFTER


def test_happens_before_predicate_is_conservative():
    a = _grad(810, 9.2, seq=10)
    b = _loss(812, 4.7, seq=11)
    assert EventOrdering.happens_before(a, b)
    assert not EventOrdering.happens_before(b, a)


def test_duplicate_source_seq_within_a_producer_is_flagged_not_ordered():
    a = _grad(810, 1.0, seq=5)
    b = _loss(810, 1.0, seq=5)
    result = EventOrdering.compare(a, b)
    assert result.relation is OrderingRelation.SAME_POSITION
    assert result.hints.get("duplicate_source_seq") is True
    assert not result.is_proven


# ---------------------------------------------------------------------------
# Cross-producer: the case the brief forbids us from guessing
# ---------------------------------------------------------------------------


def test_cross_producer_different_steps_is_not_ordered():
    """The headline rule: a lower step across ranks is a hint, not a happens-before."""
    a = _grad(810, 9.2, producer=P0, seq=1, rank=0)
    b = _loss(812, 4.7, producer=P1, seq=1, rank=1)
    result = EventOrdering.compare(a, b)
    assert result.relation is OrderingRelation.CONCURRENT_OR_UNKNOWN
    assert not result.is_proven
    assert result.evidence == "cross_producer_step_relation_not_provable"
    # Two ranks at different steps are almost certainly ordered in reality; we
    # simply cannot prove which way, so this is a knowledge gap, not a proof of
    # concurrency.
    assert result.is_gap is True
    # The relation we *can* report is the logical-step hint.
    assert result.hints["logical_step_hint"] == "a_lower"


def test_cross_producer_same_step_is_same_position():
    a = _grad(100, 1.0, producer=P0, seq=1, rank=0)
    b = _grad(100, 2.0, producer=P1, seq=1, rank=1)
    result = EventOrdering.compare(a, b)
    assert result.relation is OrderingRelation.SAME_POSITION
    assert result.evidence == "same_logical_position_different_producers"
    assert result.is_gap is False, "proven concurrency is not a knowledge gap"


def test_cross_producer_without_positions_is_a_gap():
    a = helpers.event(EventType.ENV_SNAPSHOT, producer=P0, rank=0, source_seq=1)
    b = helpers.event(EventType.ENV_SNAPSHOT, producer=P1, rank=1, source_seq=1)
    result = EventOrdering.compare(a, b)
    assert result.relation is OrderingRelation.CONCURRENT_OR_UNKNOWN
    assert result.is_gap is True


def test_different_runs_are_never_ordered():
    a = _grad(810, 1.0, producer=P0, seq=1)
    b = _loss(100, 1.0, producer=P0, seq=2)
    b = b.copy_with(run_id="a-different-run")
    result = EventOrdering.compare(a, b)
    assert result.relation is OrderingRelation.CONCURRENT_OR_UNKNOWN
    assert result.evidence == "different_run"
    assert result.confidence is OrderingConfidence.UNKNOWN


def test_barrier_witness_orders_across_producers_but_is_never_invented():
    a = _grad(810, 1.0, producer=P0, seq=1, rank=0)
    b = _loss(812, 1.0, producer=P1, seq=1, rank=1)
    without = EventOrdering.compare(a, b)
    assert without.relation is OrderingRelation.CONCURRENT_OR_UNKNOWN
    with_barrier = EventOrdering.compare(a, b, barrier_witness="step_barrier")
    assert with_barrier.relation is OrderingRelation.BEFORE
    assert with_barrier.is_proven
    # An unrecognised witness must not be trusted.
    bogus = EventOrdering.compare(a, b, barrier_witness="i-say-so")
    assert bogus.relation is OrderingRelation.CONCURRENT_OR_UNKNOWN


# ---------------------------------------------------------------------------
# Explicit causal edges are authoritative
# ---------------------------------------------------------------------------


def test_parent_event_id_proves_happens_before_across_producers():
    parent = _grad(810, 1.0, producer=P0, seq=1, rank=0)
    child = helpers.event(
        EventType.WARNING, producer=P1, rank=1, source_seq=1, step=810,
        parent_event_id=parent.event_id,
    )
    result = EventOrdering.compare(parent, child)
    assert result.relation is OrderingRelation.BEFORE
    assert result.evidence == "causal_edge"
    assert result.is_proven


def test_causal_edge_takes_precedence_over_disagreeing_positions():
    parent = _grad(900, 1.0, producer=P0, seq=1, rank=0)
    child = helpers.event(
        EventType.WARNING, producer=P1, rank=1, source_seq=1, step=100,
        parent_event_id=parent.event_id,
    )
    assert EventOrdering.compare(parent, child).evidence == "causal_edge"


# ---------------------------------------------------------------------------
# The producer ordering-violation guard
# ---------------------------------------------------------------------------


def test_emission_order_contradicting_step_order_is_refused():
    """A producer that flushes out of order must not produce a causal claim.

    ``source_seq`` records emission order, which equals training order only if
    the producer emits each step's observations in order. When the two disagree
    we abstain rather than build a false "X preceded Y" story.
    """
    flushed_late = _grad(810, 9.2, seq=2)   # emitted second, describes step 810
    flushed_early = _loss(812, 4.7, seq=1)  # emitted first, describes step 812
    result = EventOrdering.compare(flushed_early, flushed_late)
    assert result.relation is OrderingRelation.CONCURRENT_OR_UNKNOWN
    assert result.evidence == "producer_sequence_position_conflict"
    assert result.hints["sequence_position_conflict"] is True
    assert result.hints["emission_order_relation"] == "BEFORE"
    assert not result.is_proven


# ---------------------------------------------------------------------------
# earliest(): the three cases of brief section 8.4
# ---------------------------------------------------------------------------


def test_case_1_unique_earliest_is_proven():
    gradient = _grad(810, 9.2, seq=10)
    spike = _loss(812, 4.7, seq=11)
    earliest = EventOrdering.earliest([spike, gradient])
    assert earliest.is_unique
    assert earliest.first_abnormal_event_id == gradient.event_id
    assert earliest.ordering_confidence is OrderingConfidence.PROVEN
    assert "grad.global_norm" in earliest.explanation
    assert earliest.hints["gaps"] == 0


def test_case_2_concurrent_candidates_yield_no_single_first():
    rank0 = _grad(810, 9.2, producer=P0, seq=1, rank=0)
    rank1 = _grad(810, 9.1, producer=P1, seq=1, rank=1)
    earliest = EventOrdering.earliest([rank0, rank1])
    assert not earliest.is_unique
    assert earliest.first_abnormal_event_id is None
    assert set(earliest.candidate_ids()) == {rank0.event_id, rank1.event_id}
    assert earliest.ordering_confidence is OrderingConfidence.PROVEN
    assert "mutually unordered" in earliest.explanation
    assert "provably concurrent" in earliest.explanation
    assert earliest.first_abnormal_event_id is None


def test_case_3_missing_evidence_yields_unknown_ordering():
    a = _grad(810, 9.2, producer=P0, seq=1, rank=0)
    b = _loss(812, 4.7, producer=P1, seq=1, rank=1)
    earliest = EventOrdering.earliest([a, b])
    assert not earliest.is_unique
    assert earliest.first_abnormal_event_id is None
    assert earliest.ordering_confidence is OrderingConfidence.UNKNOWN
    assert "cannot be established" in earliest.explanation


def test_unique_minimum_with_a_gap_is_low_confidence_not_proven():
    """A minimum found despite unsettled comparisons must not be called proven."""
    settled = _grad(810, 9.2, producer=P0, seq=1, rank=0)
    later = _loss(812, 4.7, producer=P0, seq=2, rank=0)
    opaque = helpers.event(EventType.ENV_SNAPSHOT, producer=P1, rank=1)  # no seq, no step
    earliest = EventOrdering.earliest([settled, later, opaque])
    assert not earliest.is_unique
    assert earliest.ordering_confidence is OrderingConfidence.UNKNOWN or (
        earliest.ordering_confidence is OrderingConfidence.LOW
    )


def test_single_candidate_is_trivially_earliest():
    only = _grad(810, 9.2, seq=1)
    earliest = EventOrdering.earliest([only])
    assert earliest.is_unique
    assert earliest.first_abnormal_event_id == only.event_id
    assert "Exactly one" in earliest.explanation


def test_empty_input_yields_no_candidates_and_a_reason():
    earliest = EventOrdering.earliest([])
    assert earliest.candidates == []
    assert earliest.first_abnormal_event_id is None
    assert earliest.ordering_confidence is OrderingConfidence.UNKNOWN
    assert "No abnormal events" in earliest.explanation


def test_earliest_refuses_an_oversized_candidate_set():
    events = [_grad(step, 1.0, seq=step) for step in range(50)]
    with pytest.raises(ValueError):
        EventOrdering.earliest(events, max_pairwise=10)


# ---------------------------------------------------------------------------
# Presentation order is deterministic and explicitly not causal
# ---------------------------------------------------------------------------


def test_storage_sort_is_deterministic_and_stable():
    events = [
        _loss(812, 4.7, seq=2),
        _grad(810, 9.2, seq=1),
        _grad(811, 3.0, seq=3),
    ]
    first = EventOrdering.sort_deterministic(events)
    second = EventOrdering.sort_deterministic(list(reversed(events)))
    assert [e.event_id for e in first] == [e.event_id for e in second]
    assert [e.step for e in first] == [810, 811, 812]


def test_earliest_is_independent_of_input_order():
    a = _grad(810, 9.2, seq=1)
    b = _loss(812, 4.7, seq=2)
    c = _grad(811, 3.0, seq=3)
    one = EventOrdering.earliest([a, b, c])
    two = EventOrdering.earliest([c, b, a])
    assert one.first_abnormal_event_id == two.first_abnormal_event_id
    assert one.candidate_ids() == two.candidate_ids()


def test_describe_origin_includes_rank_only_when_known():
    e = _grad(810, 1.0, producer=P0, seq=1, rank=0)
    assert "rank 0" in EventOrdering.describe_origin(e)
    no_rank = e.copy_with(rank=None)
    assert "rank" not in EventOrdering.describe_origin(no_rank)
