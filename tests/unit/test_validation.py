"""W0 gate: validation, idempotency, and producer sequence tracking."""

from __future__ import annotations

import pytest

import helpers
from training_debugger.schema import (
    EventType,
    IssueCode,
    MetricUnit,
    ProducerSequenceTracker,
    Severity,
    TrainingEvent,
    ValidationContext,
    ValidationError,
    normalize_unit,
    validate_event,
    validate_event_or_raise,
)


def _valid(**kw):
    base = dict(step=1, metric_name="train.loss", metric_value=1.0)
    base.update(kw)
    return helpers.event(EventType.LOSS, **base)


# ---------------------------------------------------------------------------
# Identity and required fields
# ---------------------------------------------------------------------------


def test_valid_event_passes():
    result = validate_event(_valid())
    assert result.ok, result.reason()


def test_missing_run_id_is_fatal():
    event = _valid(run_id="")
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.MISSING_RUN_ID.value in result.codes


def test_missing_producer_and_source_are_fatal():
    event = _valid(producer="", source="")
    codes = validate_event(event).codes
    assert IssueCode.MISSING_PRODUCER.value in codes
    assert IssueCode.MISSING_SOURCE.value in codes


def test_malformed_event_id_is_fatal():
    event = _valid()
    event.event_id = "not-a-ulid"
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.MALFORMED_EVENT_ID.value in result.codes


def test_empty_event_id_is_fatal():
    event = _valid()
    event.event_id = ""
    assert IssueCode.MISSING_EVENT_ID.value in validate_event(event).codes


def test_invalid_event_type_is_fatal():
    event = _valid()
    event.event_type = "NOT_A_REAL_TYPE"  # type: ignore[assignment]
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.INVALID_EVENT_TYPE.value in result.codes


# ---------------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("rank", -1, IssueCode.NEGATIVE_RANK),
        ("step", -5, IssueCode.NEGATIVE_STEP),
        ("microstep", -2, IssueCode.NEGATIVE_MICROSTEP),
        ("layer_index", -3, IssueCode.INVALID_LAYER_INDEX),
        ("duration_ns", -1, IssueCode.NEGATIVE_DURATION),
    ],
)
def test_negative_indices_are_rejected(field, value, code):
    event = _valid()
    setattr(event, field, value)
    result = validate_event(event)
    assert not result.ok
    assert code.value in result.codes


def test_rank_beyond_world_size_is_rejected():
    event = _valid(rank=8, world_size=8)
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.RANK_EXCEEDS_WORLD_SIZE.value in result.codes


def test_rank_within_world_size_is_accepted():
    assert validate_event(_valid(rank=7, world_size=8)).ok


def test_layer_index_beyond_model_depth_is_rejected():
    event = _valid(layer_index=200)
    result = validate_event(event, context=ValidationContext(max_layers=12))
    assert not result.ok
    assert IssueCode.LAYER_INDEX_OUT_OF_RANGE.value in result.codes


def test_module_path_disagreeing_with_layer_index_is_flagged_but_not_fatal():
    event = _valid(layer_index=3, module_path="layers.7.mlp")
    result = validate_event(event)
    assert result.ok, "a provenance mismatch is a warning, not a rejection"
    assert IssueCode.LAYER_MODULE_CONFLICT.value in result.codes


# ---------------------------------------------------------------------------
# Metric payload and units
# ---------------------------------------------------------------------------


def test_metric_name_without_value_is_fatal():
    event = _valid()
    event.metric_value = None
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.METRIC_WITHOUT_VALUE.value in result.codes


def test_metric_value_without_name_is_fatal():
    event = _valid()
    event.metric_name = None
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.METRIC_WITHOUT_NAME.value in result.codes


def test_non_numeric_metric_value_is_fatal():
    event = _valid()
    event.metric_value = "high"  # type: ignore[assignment]
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.INVALID_METRIC_VALUE_TYPE.value in result.codes


def test_nan_on_loss_is_allowed_because_it_is_the_product_signal():
    event = _valid(metric_value=float("nan"))
    assert validate_event(event).ok


def test_nan_on_an_unrelated_event_type_is_rejected():
    event = helpers.event(EventType.STEP_END, step=1, metric_name="train.step_time",
                          metric_value=float("nan"), metric_unit=MetricUnit.MILLISECONDS)
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.NONFINITE_NOT_ALLOWED.value in result.codes


def test_missing_unit_is_a_non_fatal_issue():
    event = _valid()
    event.metric_unit = None
    result = validate_event(event)
    assert result.ok
    assert IssueCode.METRIC_WITHOUT_UNIT.value in result.codes


def test_unknown_unit_is_fatal():
    event = _valid()
    event.metric_unit = "furlongs"  # type: ignore[assignment]
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.INVALID_UNIT.value in result.codes


def test_unit_name_mismatch_is_flagged():
    """A producer that labels milliseconds as seconds is caught, because a
    silent 1000x scale error is exactly the correlation hazard Phase 0 named."""
    event = _valid(metric_name="data.loader_wait_ms", metric_value=12.0)
    event.metric_unit = MetricUnit.SECONDS
    result = validate_event(event)
    assert result.ok
    assert IssueCode.UNIT_NAME_MISMATCH.value in result.codes


def test_consistent_name_and_unit_is_not_flagged():
    event = _valid(metric_name="data.loader_wait_ms", metric_value=12.0)
    event.metric_unit = MetricUnit.MILLISECONDS
    assert IssueCode.UNIT_NAME_MISMATCH.value not in validate_event(event).codes


def test_unit_normalization_converts_to_the_canonical_scale():
    assert normalize_unit(1.5, MetricUnit.SECONDS) == pytest.approx(1500.0)
    assert normalize_unit(1500.0, MetricUnit.MICROSECONDS) == pytest.approx(1.5)
    # Percent is NOT converted to a ratio: that would silently rescale thresholds.
    assert normalize_unit(42.0, MetricUnit.PERCENT) == pytest.approx(42.0)
    assert normalize_unit(3.0, None) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Confidence and metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.5, "high"])
def test_invalid_confidence_is_rejected(bad):
    event = _valid()
    event.confidence = bad  # type: ignore[assignment]
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.INVALID_CONFIDENCE.value in result.codes


def test_metadata_that_is_not_a_dict_is_rejected():
    event = _valid()
    event.metadata = ["nope"]  # type: ignore[assignment]
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.METADATA_NOT_A_DICT.value in result.codes


def test_oversized_metadata_is_rejected():
    event = _valid(metadata={"blob": "x" * 20000})
    result = validate_event(event)
    assert not result.ok
    assert IssueCode.METADATA_TOO_LARGE.value in result.codes


# ---------------------------------------------------------------------------
# Producer sequence
# ---------------------------------------------------------------------------


def test_producer_sequence_tracker_accepts_monotonic_sequences():
    tracker = ProducerSequenceTracker()
    for seq in (1, 2, 3, 10):
        assert validate_event(_valid(source_seq=seq), seq_tracker=tracker).ok


def test_producer_sequence_tracker_rejects_regression():
    tracker = ProducerSequenceTracker()
    assert validate_event(_valid(source_seq=5), seq_tracker=tracker).ok
    result = validate_event(_valid(source_seq=4), seq_tracker=tracker)
    assert not result.ok
    assert IssueCode.NON_MONOTONIC_SOURCE_SEQ.value in result.codes


def test_producer_sequence_tracker_rejects_duplicates():
    tracker = ProducerSequenceTracker()
    validate_event(_valid(source_seq=5), seq_tracker=tracker)
    assert IssueCode.NON_MONOTONIC_SOURCE_SEQ.value in validate_event(
        _valid(source_seq=5), seq_tracker=tracker
    ).codes


def test_producer_sequence_tracker_is_per_producer():
    tracker = ProducerSequenceTracker()
    validate_event(_valid(source_seq=5, producer="a:r0:p1"), seq_tracker=tracker)
    # A different producer starts its own sequence; no cross-talk.
    assert validate_event(_valid(source_seq=1, producer="a:r1:p2"), seq_tracker=tracker).ok


def test_missing_source_seq_is_a_non_fatal_issue():
    event = _valid()
    event.source_seq = None
    result = validate_event(event)
    assert result.ok
    assert IssueCode.MISSING_SOURCE_SEQ.value in result.codes


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotency_key_is_derived_from_run_producer_and_sequence():
    event = _valid(source_seq=7)
    assert event.computed_idempotency_key() == "run-test-0001|{}|7".format(event.producer_id)


def test_idempotency_key_is_stable_across_a_roundtrip():
    event = _valid(source_seq=7)
    restored = TrainingEvent.from_dict(event.to_dict())
    assert restored.computed_idempotency_key() == event.computed_idempotency_key()


def test_duplicate_idempotency_key_is_detected_via_context():
    first = _valid(source_seq=9)
    second = _valid(source_seq=9)
    ctx = ValidationContext(seen_idempotency_keys={first.computed_idempotency_key()})
    result = validate_event(second, context=ctx)
    assert result.ok, "duplicates are reported, not fatal -- the writer deduplicates"
    assert IssueCode.DUPLICATE_IDEMPOTENCY_KEY.value in result.codes


def test_duplicate_event_id_is_detected_via_context():
    first = _valid(source_seq=9)
    ctx = ValidationContext(seen_event_ids={first.event_id})
    result = validate_event(first, context=ctx)
    assert IssueCode.DUPLICATE_EVENT_ID.value in result.codes


def test_event_without_source_seq_still_gets_a_unique_key():
    a = _valid()
    a.source_seq = None
    b = _valid()
    b.source_seq = None
    assert a.computed_idempotency_key() != b.computed_idempotency_key()
    assert a.computed_idempotency_key().startswith("noidem:")


def test_explicit_idempotency_key_overrides_the_derived_one():
    event = _valid(source_seq=1)
    event.idempotency_key = "custom-key"
    assert event.computed_idempotency_key() == "custom-key"


# ---------------------------------------------------------------------------
# The raise-on-failure helper (tests and strict ingest only)
# ---------------------------------------------------------------------------


def test_validate_or_raise_passes_a_valid_event():
    event = _valid()
    assert validate_event_or_raise(event) is event


def test_validate_or_raise_raises_with_a_readable_reason():
    event = _valid(run_id="")
    with pytest.raises(ValidationError) as excinfo:
        validate_event_or_raise(event)
    assert "missing_run_id" in str(excinfo.value)
    assert excinfo.value.result.ok is False


def test_validation_never_raises_for_data_problems():
    """The fail-open contract: ``validate_event`` reports, it does not throw."""
    broken = _valid(run_id="")
    broken.rank = -1
    broken.metric_value = None
    broken.confidence = 9.0
    result = validate_event(broken)  # must not raise
    assert not result.ok
    assert len(result.fatal_issues) >= 3
