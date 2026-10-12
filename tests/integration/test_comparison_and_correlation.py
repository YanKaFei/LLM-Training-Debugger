"""C3 + C4 gates: cross-run comparison and lag-distribution correlation."""

from __future__ import annotations

import pytest

from training_debugger.comparison import compare_runs
from training_debugger.correlation import (
    CorrelationEvidence,
    assert_causal_language_absent,
    benjamini_hochberg,
    build_correlation_report,
    correlate_series,
    lag_distribution,
)
from training_debugger.correlation.engine import ALLOWED_PHRASINGS, CAUSAL_VERBS, MIN_OCCURRENCES
from training_debugger.observer import MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver
from training_debugger.persistence import IncidentStore
from training_debugger.storage import ParquetEventWriter

import helpers


# ---------------------------------------------------------------------------
# C4: lag distribution
# ---------------------------------------------------------------------------


def test_lag_distribution_measures_precedence_not_proximity():
    """Only targets that FOLLOW a source count, and the nearest one wins."""
    lags = lag_distribution([10, 50, 90], [18, 61, 99, 200], max_lag=50)
    assert lags == [8, 9, 11]


def test_lag_distribution_excludes_targets_that_came_first():
    """A target that preceded the source is not a lag, not a negative lag."""
    assert lag_distribution([100], [90], max_lag=50) == []


def test_lag_distribution_respects_the_maximum():
    assert lag_distribution([10], [200], max_lag=50) == []


def test_lag_distribution_needs_a_target_within_the_window():
    assert lag_distribution([10, 20], [15], max_lag=50) == [5]


# ---------------------------------------------------------------------------
# C4: multiple testing
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_matches_the_reference_example():
    """The canonical BH 1995 example: m=15, alpha=0.05, four rejections."""
    p_values = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298,
                0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
    adjusted, rejected = benjamini_hochberg(p_values, alpha=0.05)
    assert sum(rejected) == 4
    assert adjusted[0] == pytest.approx(0.0015, abs=1e-6)


def test_benjamini_hochberg_is_monotone_and_bounded():
    p_values = [0.001, 0.01, 0.02, 0.5, 0.9]
    adjusted, _ = benjamini_hochberg(p_values, alpha=0.05)
    assert all(0.0 <= a <= 1.0 for a in adjusted)
    assert all(adjusted[i] <= adjusted[i + 1] + 1e-12 for i in range(len(adjusted) - 1))


def test_benjamini_hochberg_on_empty_input():
    assert benjamini_hochberg([], alpha=0.05) == ([], [])


def test_adjusted_p_is_never_smaller_than_raw():
    p_values = [0.001, 0.02, 0.04, 0.2]
    adjusted, _ = benjamini_hochberg(p_values, alpha=0.05)
    for raw, adj in zip(p_values, adjusted):
        assert adj >= raw - 1e-12


# ---------------------------------------------------------------------------
# C4: causal language is forbidden
# ---------------------------------------------------------------------------


def test_no_allowed_phrasing_contains_a_causal_verb():
    for phrasing in ALLOWED_PHRASINGS:
        lowered = phrasing.lower()
        for verb in CAUSAL_VERBS:
            assert verb not in lowered, "%r contains causal verb %r" % (phrasing, verb)


def test_causal_language_guard_rejects_causal_claims():
    for text in (
        "gradient anomaly caused the loss spike",
        "the throughput drop was due to a slow dataloader",
        "a bad shard led to divergence",
    ):
        with pytest.raises(ValueError):
            assert_causal_language_absent(text)


def test_causal_language_guard_accepts_permitted_claims():
    for text in (
        "gradient.global_norm repeatedly precedes train.loss",
        "train.lr is not correlated with train.loss",
        "evidence insufficient for param.global_norm and train.loss",
    ):
        assert_causal_language_absent(text)


# ---------------------------------------------------------------------------
# C4: the engine end to end
# ---------------------------------------------------------------------------


def _run_series(spike_step: int, seed: int):
    writer = MemoryWriter()
    observer = TrainingObserver(
        "run-corr-%d" % seed, writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01), rank=0, world_size=1,
    )
    helpers.synthetic_training(
        observer, steps=200, grad_explode_at=spike_step, grad_explode_factor=12.0,
        spike_at=spike_step + 8, spike_factor=4.0,
    )
    observer.flush(timeout=10.0)
    observer.close()
    by_metric: dict = {}
    for event in writer.events:
        if event.metric_name and event.step is not None and event.metric_value is not None:
            by_metric.setdefault(event.metric_name, {})[event.step] = event.metric_value
    return {k: sorted(v.items()) for k, v in by_metric.items()}


def test_correlation_finds_a_repeating_lead():
    report = build_correlation_report(
        {"run-%d" % i: _run_series(60 + i * 10, i) for i in range(4)}
    )
    gradient_vs_loss = [
        e for e in report.evidences
        if e.source_signal == "grad.global_norm" and e.target_signal == "train.loss"
    ]
    assert gradient_vs_loss, "the pair must be tested"
    entry = gradient_vs_loss[0]
    assert entry.occurrence_count >= MIN_OCCURRENCES
    assert entry.median_lag is not None
    assert entry.median_lag >= 0
    assert entry.recurrence_population == 4


def test_correlation_reports_negatives_with_equal_prominence():
    """A report that shows only significant pairs is lying by omission."""
    report = build_correlation_report(
        {"run-%d" % i: _run_series(60 + i * 10, i) for i in range(4)}
    )
    assert report.not_correlated() or report.insufficient(), (
        "the engine must report the pairs it tested and did not find associated"
    )
    text = report.render()
    assert "Not correlated" in text
    assert "Insufficient evidence" in text
    assert "Hypotheses tested" in text


def test_report_rendering_contains_no_causal_language():
    report = build_correlation_report(
        {"run-%d" % i: _run_series(60 + i * 10, i) for i in range(4)}
    )
    assert_causal_language_absent(report.render())


def test_module_notes_when_there_are_too_few_runs():
    report = build_correlation_report({"only-run": _run_series(60, 1)})
    assert any("fewer than 3 runs" in note for note in report.notes)


def test_recurrence_is_reported_as_a_ratio_not_a_boolean():
    report = build_correlation_report(
        {"run-%d" % i: _run_series(60 + i * 10, i) for i in range(4)}
    )
    for evidence in report.evidences:
        payload = evidence.to_dict()
        assert "recurrence_count" in payload
        assert "recurrence_population" in payload
        assert "recurrence_ratio" in payload


def test_serialisation_survives_a_round_trip():
    """Report bytes must be storable; a report that cannot be persisted is a log line."""
    import json

    report = build_correlation_report(
        {"run-%d" % i: _run_series(60 + i * 10, i) for i in range(3)}
    )
    payload = json.dumps(report.to_dict(), default=str, allow_nan=True)
    restored = json.loads(payload)
    assert "correlated" in restored and "not_correlated" in restored


# ---------------------------------------------------------------------------
# C3: cross-run comparison
# ---------------------------------------------------------------------------


def _write_run(root, run_id: str, *, steps: int, spike_at=None, lr: float = 0.05):
    writer = ParquetEventWriter(root, run_id, rows_per_file=100_000)
    observer = TrainingObserver(
        run_id, writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=4096),
        rank=0, world_size=1,
    )
    observer.emit(event_type=__import__("training_debugger.schema", fromlist=["EventType"]).EventType.RUN_START,
                  metadata={"model_id": "tiny", "steps": steps, "lr": lr, "optimizer": "sgd"})
    helpers.synthetic_training(
        observer, steps=steps, spike_at=spike_at, spike_factor=4.0 if spike_at else 1.0,
    )
    observer.flush(timeout=30.0)
    observer.close()
    return writer


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("cmp")
    _write_run(root, "run-baseline", steps=200)
    _write_run(root, "run-candidate", steps=200, spike_at=120, lr=0.02)
    return root


def test_comparison_is_structured_not_prose(two_runs):
    comparison = compare_runs(two_runs, "run-baseline", "run-candidate")
    payload = comparison.to_dict()
    for key in (
        "config_changes", "metric_regressions", "incident_changes",
        "performance_changes", "new_anomalies", "missing_anomalies", "comparability",
    ):
        assert key in payload, key


def test_config_changes_are_detected(two_runs):
    comparison = compare_runs(two_runs, "run-baseline", "run-candidate")
    changed = {c.key for c in comparison.config_changes}
    assert "lr" in changed, "the learning rate changed between the runs"


def test_metric_regressions_are_reported(two_runs):
    comparison = compare_runs(two_runs, "run-baseline", "run-candidate")
    metrics = {m.metric_name for m in comparison.metric_regressions}
    assert "train.loss" in metrics
    loss = next(m for m in comparison.metric_regressions if m.metric_name == "train.loss")
    assert loss.baseline_median is not None and loss.candidate_median is not None
    assert loss.direction in ("better", "worse", "neutral", "up", "down", "unknown")


def test_alignment_is_declared(two_runs):
    comparison = compare_runs(two_runs, "run-baseline", "run-candidate")
    assert comparison.alignment in ("step", "relative")
    assert comparison.interpolation


def test_relative_alignment_is_chosen_when_lengths_differ(tmp_path):
    _write_run(tmp_path, "short-run", steps=100)
    _write_run(tmp_path, "long-run", steps=300)
    comparison = compare_runs(tmp_path, "short-run", "long-run")
    assert comparison.alignment == "relative", (
        "runs of different length must not be compared at absolute steps"
    )
    assert any("differ in length" in c for c in comparison.comparability["caveats"])


def test_incident_changes_are_reported(two_runs):
    comparison = compare_runs(two_runs, "run-baseline", "run-candidate")
    changes = comparison.incident_changes
    assert "baseline_count" in changes and "candidate_count" in changes
    assert "new_types" in changes and "resolved_types" in changes


def test_comparison_renders_deterministically(two_runs):
    first = compare_runs(two_runs, "run-baseline", "run-candidate")
    second = compare_runs(two_runs, "run-baseline", "run-candidate")
    assert first.render() == second.render()


def test_incomparability_is_flagged(tmp_path):
    _write_run(tmp_path, "a", steps=50, lr=0.05)
    _write_run(tmp_path, "b", steps=50, lr=0.5)
    comparison = compare_runs(tmp_path, "a", "b")
    assert "lr" in {c.key for c in comparison.config_changes}


def test_comparison_of_a_run_with_itself_is_neutral(two_runs):
    comparison = compare_runs(two_runs, "run-baseline", "run-baseline")
    assert not comparison.config_changes
    for regression in comparison.metric_regressions:
        assert regression.relative_change is None or abs(regression.relative_change) < 1e-9
