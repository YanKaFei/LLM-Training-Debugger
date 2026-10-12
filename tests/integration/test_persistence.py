"""C1 gate: incident persistence, revisions, provenance, restart reconstruction."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training_debugger.detectors import DetectorConfig, DetectorRunner
from training_debugger.incidents import IncidentEngine, render_incident
from training_debugger.persistence import RULE_VERSION, IncidentStore
from training_debugger.schema import IncidentState

from scenarios.corpus import get_scenario
from scenarios.harness import generate, series_from_events


def _incidents_for(scenario_name: str, run_id: str):
    scenario = get_scenario(scenario_name)
    events, _ = generate(scenario)
    by_metric, layer_series, module_paths = series_from_events(events)
    detection = DetectorRunner(DetectorConfig()).run(
        run_id, by_metric, rank=0, layer_series=layer_series, module_paths=module_paths,
        population_layers=scenario.layer_count, population_ranks=1,
        coverage={"step_count": scenario.steps},
    )
    incidents = IncidentEngine().build(
        run_id, detection, events, by_metric, layer_series=layer_series
    )
    return events, incidents


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    events, incidents = _incidents_for("A1_loss_spike", "run-persist")
    store = IncidentStore(root, detector_config=DetectorConfig().to_dict())
    results = [store.append(i) for i in incidents]
    return root, store, events, incidents, results


def test_every_incident_is_written(stored):
    _, _, _, incidents, results = stored
    assert incidents, "the scenario must produce incidents"
    assert all(r.written for r in results)
    assert all(r.revision == 1 for r in results)


def test_incident_summary_is_queryable(stored):
    root, store, _, incidents, _ = stored
    rows = store.list_incidents("run-persist")
    assert len(rows) == len(incidents)
    for row in rows:
        assert row["run_id"] == "run-persist"
        assert row["incident_type"]
        assert row["state"] == str(IncidentState.OPEN)
        assert row["rule_version"] == RULE_VERSION


def test_reconstruction_renders_identically(stored):
    """The property that makes the store an audit trail rather than a log."""
    _, store, _, incidents, _ = stored
    for incident in incidents:
        loaded = store.load("run-persist", incident.incident_id)
        assert render_incident(loaded) == render_incident(incident), (
            "reconstruction changed the rendered incident for %s" % incident.incident_id
        )


def test_reconstruction_preserves_evidence_and_hypotheses(stored):
    _, store, _, incidents, _ = stored
    for incident in incidents:
        loaded = store.load("run-persist", incident.incident_id)
        assert len(loaded.evidence.all()) == len(incident.evidence.all())
        assert len(loaded.evidence.not_correlated) == len(incident.evidence.not_correlated)
        assert len(loaded.hypotheses) == len(incident.hypotheses)
        assert loaded.first_abnormal_event_id == incident.first_abnormal_event_id
        assert loaded.multiplicity.to_dict() == incident.multiplicity.to_dict()
        assert loaded.ordering_confidence == incident.ordering_confidence


def test_evidence_provenance_survives_persistence(stored):
    """Every evidence row must name the detector, version and rule version."""
    root, store, _, incidents, _ = stored
    table = store._read_table("run-persist", "incident_evidence")
    assert table.num_rows > 0
    for row in table.to_pylist():
        assert row["rule_version"] == RULE_VERSION
        assert row["evidence_id"]
        assert row["bucket"] in ("supporting", "contradicting", "not_correlated", "unknown")
        assert row["level"] in ("OBSERVED", "INFERRED", "HYPOTHESIS", "UNKNOWN")
        # A statistical row must be able to say how it was produced.
        if row["level"] == "INFERRED":
            assert row["query_definition"], "inferred evidence has no query definition"
        assert row["event_ids"] is not None


def test_hypothesis_score_decomposition_survives(stored):
    root, store, _, incidents, _ = stored
    table = store._read_table("run-persist", "incident_hypotheses")
    assert table.num_rows > 0
    for row in table.to_pylist():
        decomposition = json.loads(row["score_decomposition"])
        assert decomposition, "a bare score is unauditable"
        assert "prior" in decomposition


def test_revisions_are_appended_not_overwritten(stored):
    root, store, _, incidents, _ = stored
    incident = incidents[0]
    store.update_state(incident, IncidentState.RESOLVED)
    history = store.history("run-persist", incident.incident_id)
    assert [h["revision"] for h in history] == [1, 2]
    assert history[0]["state"] == "OPEN"
    assert history[1]["state"] == "RESOLVED"


def test_latest_revision_is_returned_by_default(stored):
    root, store, _, incidents, _ = stored
    incident = incidents[0]
    store.update_state(incident, IncidentState.CLOSED)
    latest = [r for r in store.list_incidents("run-persist") if r["incident_id"] == incident.incident_id]
    assert len(latest) == 1
    assert latest[0]["state"] == "CLOSED"
    assert latest[0]["revision"] == 3


def test_historical_revision_can_be_loaded_explicitly(stored):
    _, store, _, incidents, _ = stored
    incident = incidents[0]
    first = store.load("run-persist", incident.incident_id, revision=1)
    assert first.state is IncidentState.OPEN


def test_missing_incident_raises_a_clear_error(stored):
    _, store, _, _, _ = stored
    with pytest.raises(KeyError):
        store.load("run-persist", "inc-does-not-exist")


def test_persistence_failure_is_fail_open(tmp_path):
    """A store failure must not destroy an analysis that already succeeded."""
    import os

    blocking_file = tmp_path / "not-a-dir"
    blocking_file.write_text("x")
    store = IncidentStore(blocking_file, fail_open=True)
    _, incidents = _incidents_for("A3_nan", "run-failopen")
    result = store.append(incidents[0])
    assert result.written is False
    assert result.error
    assert store.errors, "the failure must be recorded, not swallowed"


def test_persistence_can_be_strict(tmp_path):
    import pytest as _pytest

    blocking_file = tmp_path / "not-a-dir"
    blocking_file.write_text("x")
    store = IncidentStore(blocking_file, fail_open=False)
    _, incidents = _incidents_for("A3_nan", "run-strict")
    with _pytest.raises(Exception):
        store.append(incidents[0])


# ---------------------------------------------------------------------------
# Restart reconstruction: a genuinely separate process
# ---------------------------------------------------------------------------


BUILD_AND_PERSIST = """
import json, sys
sys.path.insert(0, {src!r})
sys.path.insert(0, {tests!r})
sys.path.insert(0, {scenarios!r})
from training_debugger.detectors import DetectorConfig, DetectorRunner
from training_debugger.incidents import IncidentEngine, render_incident
from training_debugger.persistence import IncidentStore
from corpus import get_scenario
from harness import generate, series_from_events

scenario = get_scenario("A1_loss_spike")
events, _ = generate(scenario)
by_metric, layer_series, module_paths = series_from_events(events)
detection = DetectorRunner(DetectorConfig()).run(
    "run-restart", by_metric, rank=0, layer_series=layer_series,
    module_paths=module_paths, population_layers=scenario.layer_count,
    population_ranks=1, coverage={{"step_count": scenario.steps}},
)
incidents = IncidentEngine().build(
    "run-restart", detection, events, by_metric, layer_series=layer_series
)
store = IncidentStore({root!r})
for incident in incidents:
    store.append(incident)
json.dump(
    {{i.incident_id: render_incident(i) for i in incidents}},
    open({expected!r}, "w"),
)
print("persisted", len(incidents))
"""

READ_BACK = """
import json, sys
sys.path.insert(0, {src!r})
from training_debugger.incidents import render_incident
from training_debugger.persistence import IncidentStore

store = IncidentStore({root!r})
rows = store.list_incidents("run-restart")
json.dump(
    {{r["incident_id"]: render_incident(store.load("run-restart", r["incident_id"]))
      for r in rows}},
    open({actual!r}, "w"),
)
print("read", len(rows))
"""


def test_restart_reconstruction_in_a_fresh_process(tmp_path):
    """Run, persist, exit the process, read back in a new interpreter.

    An in-process round trip could pass while the store still held a live object
    reference. A subprocess cannot, so this is the test that actually proves
    durability. The scripts are self-contained rather than importing this test
    module, so they exercise only the persisted artefacts.
    """
    project = Path(__file__).resolve().parents[2]
    root = tmp_path / "store"
    expected_path = tmp_path / "expected.json"
    actual_path = tmp_path / "actual.json"
    substitutions = {
        "src": str(project / "src"),
        "tests": str(project / "tests"),
        "scenarios": str(project / "tests" / "scenarios"),
        "root": str(root),
        "expected": str(expected_path),
        "actual": str(actual_path),
    }

    write_script = tmp_path / "write.py"
    write_script.write_text(BUILD_AND_PERSIST.format(**substitutions), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(write_script)], capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert expected_path.exists()

    read_script = tmp_path / "read.py"
    read_script.write_text(READ_BACK.format(**substitutions), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(read_script)], capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, result.stderr[-2000:]

    expected = json.loads(expected_path.read_text())
    actual = json.loads(actual_path.read_text())
    assert actual, "nothing was persisted"
    assert actual == expected, (
        "a restart changed the rendered incident; the store is not a durable audit trail"
    )
