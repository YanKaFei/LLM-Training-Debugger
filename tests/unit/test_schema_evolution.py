"""W0 gate: schema evolution, unknown fields, and version compatibility."""

from __future__ import annotations

import pytest

import helpers
from training_debugger.schema import (
    SCHEMA_VERSION,
    EventType,
    IssueCode,
    TrainingEvent,
    validate_event,
)


def _payload() -> dict:
    return helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0).to_dict()


def test_unknown_fields_are_preserved_by_default():
    payload = _payload()
    payload["future_metric"] = 7
    payload["future_nested"] = {"a": [1, 2]}
    restored = TrainingEvent.from_dict(payload)
    assert restored.metadata["_unrecognized"]["future_metric"] == 7
    assert restored.metadata["_unrecognized"]["future_nested"] == {"a": [1, 2]}


def test_preserved_unknown_fields_survive_a_second_roundtrip():
    payload = _payload()
    payload["future_metric"] = 7
    once = TrainingEvent.from_dict(payload)
    twice = TrainingEvent.from_dict(once.to_dict())
    assert twice.metadata["_unrecognized"]["future_metric"] == 7


def test_unknown_fields_can_be_dropped():
    payload = _payload()
    payload["future_metric"] = 7
    restored = TrainingEvent.from_dict(payload, on_unknown="drop")
    assert "_unrecognized" not in restored.metadata


def test_unknown_fields_can_be_an_error():
    payload = _payload()
    payload["future_metric"] = 7
    with pytest.raises(ValueError) as excinfo:
        TrainingEvent.from_dict(payload, on_unknown="error")
    assert "future_metric" in str(excinfo.value)


def test_invalid_on_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        TrainingEvent.from_dict(_payload(), on_unknown="whatever")


def test_unknown_fields_produce_a_non_fatal_validation_issue():
    payload = _payload()
    payload["future_metric"] = 7
    event = TrainingEvent.from_dict(payload)
    result = validate_event(event)
    assert result.ok, "an unknown field must not make the event unusable"
    assert IssueCode.UNKNOWN_FIELDS_PRESENT.value in result.codes
    assert all(not i.fatal for i in result.issues if i.code is IssueCode.UNKNOWN_FIELDS_PRESENT)


def test_old_schema_reader_accepts_a_newer_minor_version():
    payload = _payload()
    payload["schema_version"] = "0.2"
    result = validate_event(TrainingEvent.from_dict(payload))
    assert result.ok
    assert IssueCode.UNKNOWN_SCHEMA_VERSION.value in result.codes


def test_old_schema_reader_rejects_a_newer_major_version():
    payload = _payload()
    payload["schema_version"] = "1.0"
    result = validate_event(TrainingEvent.from_dict(payload))
    assert not result.ok
    assert IssueCode.SCHEMA_MAJOR_MISMATCH.value in result.codes


def test_malformed_schema_version_is_rejected():
    payload = _payload()
    payload["schema_version"] = "banana"
    result = validate_event(TrainingEvent.from_dict(payload))
    assert not result.ok
    assert IssueCode.UNKNOWN_SCHEMA_VERSION.value in result.codes


def test_reading_a_previous_version_payload_works():
    """A hand-written v0.1 payload with no optional fields still loads."""
    minimal = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "01J000000000000000000000AB",
        "run_id": "run-legacy",
        "producer": "pytorch-adapter:r0:pid1",
        "source": "pytorch-adapter",
        "event_type": "LOSS",
        "step": 5,
        "metric_name": "train.loss",
        "metric_value": 1.5,
    }
    event = TrainingEvent.from_dict(minimal)
    assert event.step == 5
    assert event.severity is not None  # defaulted
    assert event.priority is not None
    assert event.metadata == {}


def test_reserved_metadata_keys_are_rejected():
    e = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0,
                      metadata={"_observer_error": "sneaky"})
    result = validate_event(e)
    assert not result.ok
    assert IssueCode.RESERVED_METADATA_KEY.value in result.codes


def test_known_fields_list_matches_dataclass_fields():
    """Guards the easiest schema-change mistake: adding a field to the dataclass
    but forgetting ``KNOWN_FIELDS``, which silently turns it into an unknown
    field on round-trip."""
    import dataclasses

    declared = {f.name for f in dataclasses.fields(TrainingEvent)}
    known = set(TrainingEvent.KNOWN_FIELDS)
    assert declared == known, "dataclass fields and KNOWN_FIELDS diverged: %s" % (
        declared.symmetric_difference(known),
    )


def test_known_fields_matches_to_dict_output():
    e = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0)
    assert set(e.to_dict().keys()) == set(TrainingEvent.KNOWN_FIELDS) | {"idempotency_key"}
