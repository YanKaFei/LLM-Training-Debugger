"""J0: regression coverage for the Phase 1.5 gradient band and consequence relations.

Phase 1.5 implemented a joint gradient feature, a three-way operating band and an
incident-relation model, and the test count did not move. That is the failure this
file closes: the band's boundaries, the aggregate's definition and the
standalone-`NaN` guarantee had no regression protection at all.

Written **before** any further detector or scorer change, per the phase's J0 gate.
"""

from __future__ import annotations

import math

import pytest

from training_debugger.detectors import DetectorConfig, DetectorRunner
from training_debugger.detectors.base import Anomaly, ConfidenceClass
from training_debugger.detectors.runner import joint_gradient_features
from training_debugger.episodes import AnomalyEpisode, EpisodeEngine, EpisodeOutcome, EpisodeState
from training_debugger.fusion import Verdict, fuse
from training_debugger.incidents.relations import (
    DEFAULT_MAX_GAP_STEPS,
    RELATION_RULE_VERSION,
    RelationType,
    link_consequences,
    relation_registry,
)
from training_debugger.schema.enums import EvidenceLevel, Severity

# The development operating point is frozen by the brief and must not move without
# a correctness regression proving the current implementation is wrong.
# The frozen operating point, in the layer-count-invariant (mean) scale.
# The original 34.0 / 46.0 were the same point expressed as a 4-layer sum.
NORMAL_BOUNDARY = 8.5
CONFIRMED_BOUNDARY = 11.5


# ---------------------------------------------------------------------------
# J0: the joint aggregate
# ---------------------------------------------------------------------------


def _layer_series(values_by_layer, steps):
    return {index: list(zip(steps, values)) for index, values in values_by_layer.items()}


def test_l1_aggregate_is_the_mean_of_per_layer_ratios():
    """Each layer normalised by its OWN trailing median, then averaged.

    The mean rather than the sum, so the feature does not scale with model depth.
    """
    steps = list(range(1, 121))
    # layer 0 holds steady at 2.0, layer 1 holds steady at 5.0 -> ratio 1.0 each,
    # so the mean is also 1.0 despite the 2.5x difference in absolute magnitude.
    flat = _layer_series({0: [2.0] * 120, 1: [5.0] * 120}, steps)
    features = joint_gradient_features(flat, 120, baseline_floor=0)
    assert features["l1_aggregate"] == pytest.approx(1.0, abs=1e-6)


def test_each_layer_uses_its_own_trailing_median_not_a_shared_scale():
    """A layer with a large absolute magnitude must not dominate the sum."""
    steps = list(range(1, 121))
    # Both layers are twice their own baseline at step 120, though their absolute
    # magnitudes differ by 100x. A shared scale would report only the larger layer.
    values = {
        0: [1.0] * 100 + [2.0] * 20,
        1: [100.0] * 100 + [200.0] * 20,
    }
    features = joint_gradient_features(_layer_series(values, steps), 120, baseline_floor=0)
    # Both layers are at twice their own baseline, so the mean ratio is 2.0.
    assert features["l1_aggregate"] == pytest.approx(2.0, abs=1e-6)


def test_aggregate_requires_at_least_two_layers():
    steps = list(range(1, 121))
    single = _layer_series({0: [1.0] * 100 + [40.0] * 20}, steps)
    features = joint_gradient_features(single, 120, baseline_floor=0)
    assert features == {}, "one layer is not a joint representation"


def test_warmup_inclusion_does_not_corrupt_the_verdict():
    """Superseded assertion, and the change is an improvement.

    Under the Phase 1.7 trailing-median feature, including the warm-up shifted the
    baseline and therefore changed the aggregate -- that is, the verdict depended on
    an arbitrary window boundary. Under the Phase 1.8 freeze-on-suspicion reference
    the baseline freezes at the first hint of deviation, so both choices converge on
    the same answer for the same fault. The property worth asserting is no longer
    "these differ" but "neither is wrong".
    """
    steps = list(range(1, 121))
    values = {0: [1.0] * 80 + [3.0] * 40, 1: [1.0] * 80 + [3.0] * 40}
    series = _layer_series(values, steps)
    with_warmup = joint_gradient_features(series, 120, baseline_floor=0)
    without_warmup = joint_gradient_features(series, 120, baseline_floor=50)
    assert with_warmup["l1_aggregate"] == pytest.approx(3.0, rel=0.05)
    assert without_warmup["l1_aggregate"] == pytest.approx(3.0, rel=0.05)


# ---------------------------------------------------------------------------
# J0: the three-way band
# ---------------------------------------------------------------------------


def _band_for(aggregate: float, config: DetectorConfig | None = None) -> str:
    """Run the real band function with a synthetic aggregate."""
    from training_debugger.detectors.runner import _apply_gradient_band

    config = config or DetectorConfig()
    # The aggregate is the MEAN ratio, so every layer is set to the target ratio
    # regardless of how many layers there are.
    layers = 4
    ratio = aggregate
    steps = list(range(1, 121))
    series = {
        index: [(s, 1.0 if s < 100 else ratio) for s in steps] for index in range(layers)
    }
    anomaly = Anomaly(
        detector="grad.robust_zscore", detector_version="3.0", metric_name="grad.global_norm",
        first_abnormal_step=120, observed_value=ratio, baseline=1.0, threshold=1.0, score=1.0,
        confidence=0.8, level=EvidenceLevel.INFERRED, severity=Severity.WARN,
        window=(70, 120), rank=0, confidence_class=ConfidenceClass.STRONG, metadata={},
    )
    _apply_gradient_band([anomaly], series, config)
    return anomaly.metadata.get("gradient_band", "?"), anomaly.confidence_class


def test_gradient_normal_region():
    band, cls = _band_for(5.0)
    assert band == "NORMAL"
    assert cls == ConfidenceClass.WEAK, "a NORMAL observation must not be promotable"


def test_gradient_uncertain_region():
    band, cls = _band_for(10.0)
    assert band == "UNCERTAIN"
    assert cls == ConfidenceClass.WEAK, "the abstention band must not be promotable"


def test_gradient_confirmed_region():
    band, cls = _band_for(15.0)
    assert band == "CONFIRMED"
    assert cls == ConfidenceClass.STRONG


def test_normal_boundary_equality_is_uncertain_not_normal():
    """``A <= l1 < B`` is UNCERTAIN, so l1 == A is on the UNCERTAIN side."""
    band, _ = _band_for(NORMAL_BOUNDARY)
    assert band == "UNCERTAIN"


def test_confirmed_boundary_equality_is_confirmed():
    """``l1 >= B`` is CONFIRMED, so l1 == B is confirmed."""
    band, _ = _band_for(CONFIRMED_BOUNDARY)
    assert band == "CONFIRMED"


def test_just_below_the_normal_boundary_is_normal():
    band, _ = _band_for(NORMAL_BOUNDARY - 0.5)
    assert band == "NORMAL"


def test_just_below_the_confirmed_boundary_is_uncertain():
    band, _ = _band_for(CONFIRMED_BOUNDARY - 0.5)
    assert band == "UNCERTAIN"


def test_band_records_its_reason():
    from training_debugger.detectors.runner import _apply_gradient_band

    steps = list(range(1, 121))
    series = {i: [(s, 1.0 if s < 100 else 5.0) for s in steps] for i in range(4)}
    anomaly = Anomaly(
        detector="grad.robust_zscore", detector_version="3.0", metric_name="grad.global_norm",
        first_abnormal_step=120, observed_value=5.0, baseline=1.0, threshold=1.0, score=1.0,
        confidence=0.8, level=EvidenceLevel.INFERRED, severity=Severity.WARN,
        window=(70, 120), rank=0, confidence_class=ConfidenceClass.STRONG, metadata={},
    )
    _apply_gradient_band([anomaly], series, DetectorConfig())
    assert anomaly.metadata["band_reason"]
    assert anomaly.metadata["layers_observed"] == 4
    assert anomaly.metadata["l2_aggregate"] > 0


# ---------------------------------------------------------------------------
# J0: band -> episode -> verdict
# ---------------------------------------------------------------------------


def _gradient_anomaly(step, layer, band):
    cls = ConfidenceClass.STRONG if band == "CONFIRMED" else ConfidenceClass.WEAK
    return Anomaly(
        detector="grad.robust_zscore", detector_version="3.0", metric_name="grad.layer_norm",
        first_abnormal_step=step, observed_value=10.0, baseline=1.0, threshold=1.0, score=1.0,
        confidence=0.8, level=EvidenceLevel.INFERRED, severity=Severity.WARN,
        window=(step - 50, step), rank=0, layer_index=layer, confidence_class=cls,
        metadata={"gradient_band": band, "population_layers": 7},
    )


def test_uncertain_gradient_episode_produces_abstention_not_an_incident():
    engine = EpisodeEngine()
    engine.consume([_gradient_anomaly(78, 2, "UNCERTAIN")], "run")
    engine.finish("run", final_step=300)
    verdicts = fuse("run", engine.all_episodes())
    assert all(v.verdict != Verdict.CONFIRMED_INCIDENT.value for v in verdicts)
    assert verdicts[0].verdict in (
        Verdict.INSUFFICIENT_EVIDENCE.value, Verdict.CANDIDATE_ONLY.value
    )


def test_single_weak_gradient_signal_never_confirms():
    engine = EpisodeEngine()
    engine.consume([_gradient_anomaly(78, 2, "NORMAL")], "run")
    engine.finish("run", final_step=300)
    episodes = engine.all_episodes()
    assert len(episodes) == 1
    assert episodes[0].outcome == EpisodeOutcome.NOT_ANOMALY.value


def test_confirmed_gradient_episode_promotes():
    engine = EpisodeEngine()
    engine.consume(
        [_gradient_anomaly(80, layer, "CONFIRMED") for layer in range(7)], "run"
    )
    engine.finish("run", final_step=300)
    verdicts = fuse("run", engine.all_episodes())
    confirmed = [v for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value]
    assert len(confirmed) == 1


def test_candidate_only_preserves_episode_ids():
    engine = EpisodeEngine()
    engine.consume([_gradient_anomaly(78, 2, "UNCERTAIN")], "run")
    engine.finish("run", final_step=300)
    episodes = engine.all_episodes()
    for episode in episodes:
        episode.confirmed_at_step = 80
        episode.state = EpisodeState.CLOSED.value
        episode.outcome = EpisodeOutcome.CONFIRMED_ANOMALY.value
    verdicts = fuse("run", episodes)
    for verdict in verdicts:
        if verdict.verdict in (Verdict.CANDIDATE_ONLY.value, Verdict.INSUFFICIENT_EVIDENCE.value):
            assert verdict.episodes_used or verdict.abstained_episodes, (
                "an abstention must name what it abstained about"
            )


def test_insufficient_evidence_preserves_episode_ids():
    engine = EpisodeEngine()
    engine.consume([_gradient_anomaly(78, 2, "UNCERTAIN")], "run")
    # Not finished: the episode stays open and SUSPECT.
    verdicts = fuse("run", engine.all_episodes())
    if verdicts[0].verdict == Verdict.INSUFFICIENT_EVIDENCE.value:
        assert verdicts[0].episodes_rejected or verdicts[0].abstained_episodes or \
            verdicts[0].signals_used or verdicts[0].episodes_used


# ---------------------------------------------------------------------------
# J0: relations
# ---------------------------------------------------------------------------


def _episode(eid, anomaly_type, onset, confirmed, layers=(0, 1, 2), outcome="CONFIRMED_ANOMALY"):
    return AnomalyEpisode(
        episode_id=eid, run_id="run", detector_id="d", detector_version="1",
        rule_version="v", anomaly_type=anomaly_type, signal_key="k",
        state=EpisodeState.CLOSED.value, outcome=outcome,
        candidate_onset_step=onset, confirmed_at_step=confirmed,
        affected_layers=list(layers), confidence_class="STRONG", confidence=0.9,
    )


def test_causal_relation_types_are_not_representable():
    """Replay is a Phase 2 capability; the type system must forbid the claim now."""
    names = {t.value for t in RelationType}
    assert "CAUSED_BY" not in names
    assert "ROOT_CAUSE_OF" not in names
    registry = relation_registry()
    assert registry["forbidden_types"] == ["CAUSED_BY", "ROOT_CAUSE_OF"]
    for name in names:
        assert "CAUS" not in name, "no relation type may imply causation"


def test_relation_types_available():
    names = {t.value for t in RelationType}
    assert names == {
        "POSSIBLE_CONSEQUENCE_OF", "PRECEDES", "CO_OCCURS_WITH", "UNKNOWN_RELATION"
    }


def test_gradient_fault_then_nan_is_precedence_not_consequence():
    """Corrected in v0.1.3.

    This test previously asserted `POSSIBLE_CONSEQUENCE_OF` for a *finite* gradient
    anomaly 38 steps before a NaN, on temporal proximity and scope overlap alone. The
    50-run mixed-fault corpus measured that exact rule: it produced a consequence claim
    on **10 of 10 deliberately independent** fault pairs and on **0 of 10** genuine
    causal chains -- precision 0.0 and recall 0.0 simultaneously.

    A finite scale anomaly 38 steps earlier is not mechanistic evidence. The relation
    is still recorded (order is observable and useful), but it asserts precedence only.
    """
    episodes = [
        _episode("e1", "GRADIENT_EXPLOSION", 100, 102),
        _episode("e2", "NAN_DETECTED", 140, 140),
    ]
    relations, linked = link_consequences("run", episodes, mechanism_linker_enabled=True)
    assert len(relations) == 1
    assert relations[0].relation_type == RelationType.PRECEDES.value
    assert relations[0].relation_type != RelationType.POSSIBLE_CONSEQUENCE_OF.value, (
        "temporal proximity alone must never produce a consequence claim"
    )
    assert relations[0].gap_steps == 38
    assert relations[0].evidence["tier"] == "TEMPORAL"
    assert linked == {"e2"}, "an order-only relation must not absorb the antecedent"
    assert relations[0].rule_version == RELATION_RULE_VERSION
    assert relations[0].evidence["antecedent_metric_class"] in ("upstream", "unknown")


def test_standalone_nan_creates_no_relation_and_stays_independent():
    """The guarantee that attribution is not suppression."""
    episodes = [_episode("e3", "NAN_DETECTED", 50, 50, layers=(5,))]
    relations, linked = link_consequences("run", episodes, mechanism_linker_enabled=True)
    assert relations == []
    assert linked == set()


def test_nan_appearing_before_the_primary_is_not_a_consequence():
    episodes = [
        _episode("e1", "GRADIENT_EXPLOSION", 100, 102),
        _episode("e2", "NAN_DETECTED", 60, 60),
    ]
    relations, _ = link_consequences("run", episodes, mechanism_linker_enabled=True)
    assert relations == [], "a consequence cannot precede its cause"


def test_nan_beyond_the_gap_window_is_not_linked():
    episodes = [
        _episode("e1", "GRADIENT_EXPLOSION", 100, 102),
        _episode("e2", "NAN_DETECTED", 102 + DEFAULT_MAX_GAP_STEPS + 5, 200),
    ]
    relations, _ = link_consequences("run", episodes, mechanism_linker_enabled=True)
    assert relations == []


def test_relation_requires_scope_compatibility():
    episodes = [
        _episode("e1", "GRADIENT_EXPLOSION", 100, 102, layers=(0, 1)),
        _episode("e2", "NAN_DETECTED", 110, 110, layers=(9,)),
    ]
    relations, _ = link_consequences("run", episodes, mechanism_linker_enabled=True)
    assert relations == [], "disjoint layer scope must not link"


def test_relation_prefers_the_nearest_primary():
    episodes = [
        _episode("far", "GRADIENT_EXPLOSION", 10, 12),
        _episode("near", "LOSS_DIVERGENCE", 130, 132),
        _episode("nan", "NAN_DETECTED", 140, 140),
    ]
    relations, _ = link_consequences("run", episodes, mechanism_linker_enabled=True)
    assert len(relations) == 1
    assert relations[0].source_incident_id == "near"


def test_relation_serialisation_round_trip():
    episodes = [
        _episode("e1", "GRADIENT_EXPLOSION", 100, 102),
        _episode("e2", "NAN_DETECTED", 140, 140),
    ]
    relations, _ = link_consequences("run", episodes, mechanism_linker_enabled=True)
    restored = type(relations[0]).from_dict(relations[0].to_dict())
    assert restored.to_dict() == relations[0].to_dict()


# ---------------------------------------------------------------------------
# J0/J2 correctness: the aggregate must not scale with the number of layers
# ---------------------------------------------------------------------------


def test_l1_aggregate_is_invariant_to_the_number_of_monitored_layers():
    """A correctness bug found while diagnosing the development corpus.

    ``l1_aggregate`` was a **sum** over layers, so its scale depended on the model's
    depth: a boundary fitted on a 4-layer model was wrong by 1.5x on a 6-layer
    model. The development corpus mixes both shapes and produced 16 healthy episodes
    above the CONFIRMED boundary as a result. The feature is now the **mean** of the
    per-layer ratios, so one boundary applies to any model shape.
    """
    steps = list(range(1, 121))

    def series(layers, ratio):
        return {i: [(s, 1.0 if s < 100 else ratio) for s in steps] for i in range(layers)}

    four = joint_gradient_features(series(4, 9.0), 120, baseline_floor=0)
    six = joint_gradient_features(series(6, 9.0), 120, baseline_floor=0)
    eight = joint_gradient_features(series(8, 9.0), 120, baseline_floor=0)
    assert four["l1_aggregate"] == pytest.approx(six["l1_aggregate"], abs=1e-9)
    assert four["l1_aggregate"] == pytest.approx(eight["l1_aggregate"], abs=1e-9)


def test_boundaries_are_frozen_in_the_invariant_scale():
    """34/46 on a 4-layer sum is exactly 8.5/11.5 on the layer-count-invariant mean."""
    config = DetectorConfig()
    assert config.gradient_agg_normal_below == pytest.approx(34.0 / 4)
    assert config.gradient_agg_confirm_at == pytest.approx(46.0 / 4)


def test_band_classification_is_identical_for_four_and_six_layer_models():
    """The fix must not change how a 4-layer run is classified."""
    from training_debugger.detectors.runner import _apply_gradient_band

    steps = list(range(1, 121))

    def classify(layers, ratio):
        series = {i: [(s, 1.0 if s < 100 else ratio) for s in steps] for i in range(layers)}
        anomaly = Anomaly(
            detector="grad.robust_zscore", detector_version="3.0",
            metric_name="grad.global_norm", first_abnormal_step=120,
            observed_value=ratio, baseline=1.0, threshold=1.0, score=1.0,
            confidence=0.8, level=EvidenceLevel.INFERRED, severity=Severity.WARN,
            window=(70, 120), rank=0, confidence_class=ConfidenceClass.STRONG, metadata={},
        )
        _apply_gradient_band([anomaly], series, DetectorConfig())
        return anomaly.metadata["gradient_band"]

    # One ratio of 3.0 is below the normal boundary in every shape (3.0 < 8.5).
    assert classify(4, 3.0) == "NORMAL"
    assert classify(6, 3.0) == "NORMAL"
    # One ratio of 12.0 is above the confirm boundary in every shape (12.0 >= 11.5).
    assert classify(4, 12.0) == "CONFIRMED"
    assert classify(6, 12.0) == "CONFIRMED"


def test_band_is_authoritative_even_when_the_aggregate_signal_fires():
    """The defect that made two development re-runs byte-identical.

    ``GRADIENT_EXPLOSION`` has ``aggregate_confirms=True``, so an observation of the
    *global* norm confirmed the episode regardless of its band. A healthy run whose
    global norm twitched was therefore confirmed exactly as before the band existed.
    Only a CONFIRMED-band observation may confirm.
    """
    from training_debugger.episodes import EpisodeEngine

    def aggregate_anomaly(step, band):
        return Anomaly(
            detector="grad.robust_zscore", detector_version="3.0",
            metric_name="grad.global_norm", first_abnormal_step=step,
            observed_value=3.0, baseline=1.0, threshold=1.0, score=1.0,
            confidence=0.8, level=EvidenceLevel.INFERRED, severity=Severity.WARN,
            window=(step - 50, step), rank=0, layer_index=None,
            confidence_class=ConfidenceClass.STRONG,
            metadata={"gradient_band": band, "population_layers": 7},
        )

    for blocked in ("NORMAL", "UNCERTAIN"):
        engine = EpisodeEngine()
        engine.consume([aggregate_anomaly(100, blocked)], "run")
        engine.finish("run", final_step=300)
        episodes = engine.all_episodes()
        assert episodes, "the observation must still be recorded"
        assert episodes[0].confirmed_at_step is None, (
            "a %s-band aggregate observation must not confirm" % blocked
        )
        assert episodes[0].metadata.get("band_blocked_confirmation") == [blocked]

    engine = EpisodeEngine()
    engine.consume([aggregate_anomaly(100, "CONFIRMED")], "run")
    engine.finish("run", final_step=300)
    assert engine.all_episodes()[0].confirmed_at_step == 100
