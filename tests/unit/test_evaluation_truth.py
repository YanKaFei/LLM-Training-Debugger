"""Evaluation ground truth must carry the fields the scorer reads, under its names.

The corpora wrote `scenario`; the scorer reads `ground_truth.primary_fault_type` and a
top-level `acceptable_window`. Nothing bridged the two, so this repository's quality
figures could not be re-derived from this repository. These tests pin the bridge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_truth import (  # noqa: E402
    ACCEPTABLE_LATENCY_STEPS, SCENARIO_FAULT_TYPE, acceptable_window, evaluation_truth,
    fault_type_for,
)


def test_the_shape_matches_what_the_scorer_reads():
    """Top-level window, nested fault type. Writing the fault type at the top level
    looks right and scores nothing."""
    truth = evaluation_truth("grad_large", injected=True, injection_step=110)
    assert truth["acceptable_window"] == [110, 110 + ACCEPTABLE_LATENCY_STEPS]
    assert truth["ground_truth"]["primary_fault_type"] == "GRADIENT_EXPLOSION"


def test_a_healthy_run_gets_no_fault_fields():
    """Inventing one would make an unlabelled run creditable as a true positive."""
    assert evaluation_truth("healthy", injected=False) == {}


def test_every_dev_scenario_resolves():
    for scenario in ("loss_spike", "grad_small", "grad_mid", "grad_large",
                     "grad_extreme", "nan_loss"):
        assert fault_type_for(scenario)


def test_every_mixed_scenario_resolves():
    for scenario in ("M1_gradient_then_nonfinite", "M2_standalone_nonfinite",
                     "M3_two_independent_faults", "M4_gradient_plus_loss_noise",
                     "M5_healthy_heavy_tail"):
        assert scenario in SCENARIO_FAULT_TYPE


def test_an_unclassified_label_raises_rather_than_scoring_as_a_miss():
    """A silent None becomes a miss, which is how a scoreboard reads 0.0 with no
    error -- recorded twice in this project (D-015, D-018)."""
    with pytest.raises(KeyError):
        fault_type_for("some_label_nobody_classified")


def test_the_window_is_a_convention_that_can_be_read():
    assert ACCEPTABLE_LATENCY_STEPS == 12
    assert acceptable_window(None) is None
    assert acceptable_window(80) == [80, 92]


def test_the_corpora_emit_the_fields():
    """The bridge is only useful if the corpora cross it."""
    for name in ("corpus_repersist", "mixed_fault_corpus"):
        source = (Path(__file__).resolve().parents[1] / "experiments"
                  / "{}.py".format(name)).read_text(encoding="utf-8")
        assert "evaluation_truth(" in source, "{} does not emit scored fields".format(name)


def test_committed_ground_truth_carries_complete_fields():
    root = Path(__file__).resolve().parents[2]
    for rel in ("outputs/alpha/corpus/ground_truth/development_ground_truth.json",
                "outputs/alpha/mixed/v015/ground_truth/mixed_fault_ground_truth.json"):
        path = root / rel
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        injected = [v for v in data.values() if isinstance(v, dict) and v.get("injected")]
        assert injected, "{} has no injected runs".format(rel)
        for entry in injected:
            assert (entry.get("ground_truth") or {}).get("primary_fault_type"), rel
            assert entry.get("acceptable_window") is not None, rel


def test_ground_truth_is_a_map_and_provenance_is_not_merged_into_it():
    """Merging provenance into a keyed map adds keys that look like run ids."""
    root = Path(__file__).resolve().parents[2]
    path = root / "outputs/alpha/corpus/ground_truth/development_ground_truth.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    assert all(isinstance(v, dict) for v in data.values())
    for key in ("code_sha", "code_dirty", "generated_at"):
        assert key not in data, "provenance leaked into the run-id map"
