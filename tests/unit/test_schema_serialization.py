"""W0 gate: schema serialization and deserialization."""

from __future__ import annotations

import json
import math

import pytest

import helpers
from training_debugger.schema import (
    SCHEMA_VERSION,
    DeviceType,
    EventPriority,
    EventType,
    MetricUnit,
    Severity,
    SpanKind,
    TrainingEvent,
    new_event_id,
)


def test_roundtrip_preserves_every_field():
    original = helpers.event(
        EventType.GRADIENT_METRIC,
        step=42,
        microstep=3,
        rank=2,
        layer_index=47,
        module_path="layers.47.mlp",
        metric_name="grad.layer_norm",
        metric_value=18.75,
        metric_unit=MetricUnit.DIMENSIONLESS,
        severity=Severity.WARN,
        priority=EventPriority.P2,
        confidence=0.9,
        metadata={"k": "v"},
    )
    restored = TrainingEvent.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()
    assert restored.event_id == original.event_id
    assert restored.producer_id == original.producer_id
    assert restored.event_type is EventType.GRADIENT_METRIC
    assert restored.metric_unit is MetricUnit.DIMENSIONLESS
    assert restored.span_kind is SpanKind.EVENT
    assert restored.severity is Severity.WARN
    assert restored.device_type is DeviceType.CPU


def test_json_line_roundtrip_is_lossless():
    original = helpers.event(EventType.LOSS, step=7, metric_name="train.loss", metric_value=2.5)
    line = original.to_json_line()
    assert "\n" not in line
    restored = TrainingEvent.from_json_line(line)
    assert restored.to_dict() == original.to_dict()


def test_json_line_is_key_sorted_for_stable_diffs():
    """Golden-file comparison in ``tests/scenarios`` depends on this."""
    a = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0)
    b = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0)
    assert list(json.loads(a.to_json_line()).keys()) == list(json.loads(b.to_json_line()).keys())


@pytest.mark.parametrize(
    "value,expected_marker",
    [(float("nan"), "NaN"), (float("inf"), "Infinity"), (float("-inf"), "-Infinity")],
)
def test_nonfinite_values_survive_a_roundtrip(value, expected_marker):
    """NaN is a product requirement (NAN_DETECTED), not a defect."""
    original = helpers.event(
        EventType.LOSS, step=9, metric_name="train.loss", metric_value=value,
        severity=Severity.ERROR,
    )
    line = original.to_json_line()
    assert expected_marker in line
    restored = TrainingEvent.from_json_line(line)
    assert not restored.is_finite
    assert math.isnan(restored.metric_value) if math.isnan(value) else restored.metric_value == value


def test_enum_values_serialize_as_strings():
    e = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0)
    payload = e.to_dict()
    for key in ("event_type", "span_kind", "severity", "priority"):
        assert isinstance(payload[key], str), key
    assert payload["metric_unit"] is None or isinstance(payload["metric_unit"], str)
    assert payload["device_type"] is None or isinstance(payload["device_type"], str)


def test_enum_accepts_member_and_string_on_from_dict():
    e = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0)
    payload = e.to_dict()
    payload["severity"] = "warn"
    assert TrainingEvent.from_dict(payload).severity is Severity.WARN
    payload["severity"] = Severity.ERROR
    assert TrainingEvent.from_dict(payload).severity is Severity.ERROR


def test_invalid_enum_value_is_rejected_with_a_helpful_message():
    payload = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0).to_dict()
    payload["severity"] = "catastrophic"
    with pytest.raises(ValueError) as excinfo:
        TrainingEvent.from_dict(payload)
    assert "invalid severity" in str(excinfo.value)
    assert "warn" in str(excinfo.value)


def test_schema_version_is_stamped_and_stable():
    e = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0)
    assert e.schema_version == SCHEMA_VERSION
    assert e.to_dict()["schema_version"] == SCHEMA_VERSION


def test_from_dict_does_not_mutate_its_input():
    payload = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0).to_dict()
    payload["future_field"] = 123
    snapshot = json.loads(json.dumps(payload, allow_nan=True))
    TrainingEvent.from_dict(payload)
    assert payload == snapshot


def test_from_dict_rejects_non_dict():
    with pytest.raises(ValueError):
        TrainingEvent.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_event_ids_are_sortable_and_unique():
    ids = [new_event_id() for _ in range(500)]
    assert len(set(ids)) == 500
    assert ids == sorted(ids), "ids must be non-decreasing in generation order"
    for value in ids:
        assert len(value) == 26


def test_copy_with_does_not_mutate_the_original():
    e = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0)
    changed = e.copy_with(step=2)
    assert e.step == 1
    assert changed.step == 2
    assert changed.event_id == e.event_id


def test_metadata_size_is_measured():
    e = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0,
                      metadata={"blob": "x" * 1000})
    assert e.metadata_size_bytes() > 1000
