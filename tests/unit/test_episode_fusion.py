"""R0: formal coverage for the episode / fusion / registry architecture.

Phase 1.3 added an entire architectural layer — `AnomalyEpisode`, fusion, the
detector registry, abstention — and the test count did not move. The behaviour was
verified by direct execution during development, which leaves no regression
protection: the next change can silently break episode identity and nothing fails.

This file is that protection. It is written **before** any detector change in
Phase 1.4, per the phase's R0 gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training_debugger.detectors.base import Anomaly, ConfidenceClass
from training_debugger.schema.enums import EvidenceLevel, Severity
from training_debugger.episodes import (
    EPISODE_RULES,
    RULE_VERSION,
    AnomalyEpisode,
    EpisodeEngine,
    EpisodeOutcome,
    EpisodeState,
    episode_id_for,
)
from training_debugger.fusion import (
    FUSION_RULE_VERSION,
    PROMOTION_RULES,
    Verdict,
    fuse,
    fusion_rule_registry,
    is_promotable,
)
from training_debugger.registry import (
    DETECTOR_REGISTRY,
    STATUS_DISABLED,
    STATUS_ENABLED,
    detector_registry,
)

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def anomaly(
    detector: str,
    step: int,
    *,
    layer: int | None = None,
    value: float = 1.0,
    baseline: float = 1.0,
    threshold: float = 1.0,
    confidence_class=None,
    peak_ratio: float = 1.0,
    population: int = 0,
) -> Anomaly:
    metadata = {"peak_ratio": peak_ratio}
    if population:
        metadata["population_layers"] = population
    if confidence_class is None:
        # Mirror production: the runner classifies a measured non-finite value as
        # CERTAIN, everything else by its multiplicity.
        confidence_class = (
            ConfidenceClass.CERTAIN
            if detector == "numerics.finiteness"
            else ConfidenceClass.STRONG
        )
    return Anomaly(
        detector=detector,
        detector_version="test",
        metric_name="grad.global_norm" if layer is None else "grad.layer_norm",
        first_abnormal_step=step,
        observed_value=value,
        baseline=baseline,
        threshold=threshold,
        score=1.0,
        confidence=0.8,
        level=EvidenceLevel.INFERRED,
        severity=Severity.WARN,
        window=(max(1, step - 50), step),
        rank=0,
        layer_index=layer,
        confidence_class=confidence_class,
        metadata=metadata,
    )


def build(detector: str, steps, **kwargs):
    engine = EpisodeEngine()
    engine.consume([anomaly(detector, s, **kwargs) for s in steps], "run")
    engine.finish("run", final_step=(max(steps) + 50 if steps else 1))
    return engine.all_episodes()


# ---------------------------------------------------------------------------
# R0.1 episode lifecycle
# ---------------------------------------------------------------------------


def test_single_weak_signal_goes_suspect_then_normal():
    """The transition the brief insists on: an excursion that becomes nothing."""
    episodes = build(
        "grad.robust_zscore", [50], layer=2,
        confidence_class=ConfidenceClass.WEAK, peak_ratio=1.2,
    )
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.outcome == EpisodeOutcome.NOT_ANOMALY.value
    assert episode.state == EpisodeState.CLOSED.value
    assert episode.confirmed_at_step is None


def test_persistent_gradient_candidate_becomes_one_episode():
    episodes = build(
        "grad.robust_zscore", range(90, 100), layer=3,
        confidence_class=ConfidenceClass.WEAK, peak_ratio=1.2,
    )
    assert len(episodes) == 1, "contiguous observations must not fragment"
    assert episodes[0].signal_count == 10


def test_persistent_inf_is_one_episode():
    episodes = build("numerics.finiteness", range(80, 191))
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.signal_count == 111
    assert episode.candidate_onset_step == 80
    assert episode.last_abnormal_step == 190


def test_persistent_nan_is_one_episode():
    episodes = build("numerics.finiteness", range(100, 140))
    assert len(episodes) == 1
    assert episodes[0].signal_count == 40


def test_111_inf_observations_produce_one_episode_and_one_incident():
    """The brief's headline numerics requirement, end to end through fusion."""
    engine = EpisodeEngine()
    engine.consume([anomaly("numerics.finiteness", s) for s in range(80, 191)], "run")
    engine.finish("run", final_step=250)
    episodes = engine.all_episodes()
    assert len(episodes) == 1

    verdicts = fuse("run", episodes)
    confirmed = [v for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value]
    assert len(confirmed) == 1, "one episode must yield one incident, not one per step"
    assert confirmed[0].episodes_used == [episodes[0].episode_id]


def test_episode_recovery_and_close():
    rule = EPISODE_RULES["NAN_DETECTED"]
    episodes = build("numerics.finiteness", [80, 81])
    episode = episodes[0]
    assert episode.state == EpisodeState.CLOSED.value
    assert episode.outcome == EpisodeOutcome.CONFIRMED_ANOMALY.value
    assert episode.recovered_at_step is not None
    assert episode.closed_wall_time_ns is not None
    _ = rule


def test_candidate_onset_differs_from_confirmed_at_for_statistical_types():
    """Persistence must separate the two moments."""
    episodes = build(
        "grad.robust_zscore", [80, 82], layer=0,
        confidence_class=ConfidenceClass.WEAK, peak_ratio=1.1,
    )
    episode = episodes[0]
    assert episode.candidate_onset_step == 80
    assert episode.confirmed_at_step == 82
    assert episode.confirmation_delay_steps == 2


def test_certain_observation_confirms_at_onset():
    """A measured NaN cannot be made more true by waiting."""
    episodes = build("numerics.finiteness", [80])
    episode = episodes[0]
    assert episode.candidate_onset_step == 80
    assert episode.confirmed_at_step == 80
    assert episode.confirmation_delay_steps == 0
    assert episode.confidence_class == ConfidenceClass.CERTAIN.value


def test_non_contiguous_observations_are_separate_episodes():
    episodes = build("numerics.finiteness", [10, 200])
    assert len(episodes) == 2


# ---------------------------------------------------------------------------
# R0.1 serialisation / identity
# ---------------------------------------------------------------------------


def test_episode_serialisation_round_trip():
    episodes = build("numerics.finiteness", [80, 81])
    episode = episodes[0]
    payload = json.loads(json.dumps(episode.to_dict(), default=str))
    restored = AnomalyEpisode.from_dict(payload)
    assert restored.episode_id == episode.episode_id
    assert restored.candidate_onset_step == episode.candidate_onset_step
    assert restored.confirmed_at_step == episode.confirmed_at_step
    assert restored.outcome == episode.outcome
    assert restored.to_dict() == episode.to_dict()


def test_episode_signature_ignores_observation_count():
    """A re-run over a slightly different number of steps must match."""
    first = build("numerics.finiteness", range(80, 100))[0]
    second = build("numerics.finiteness", range(80, 150))[0]
    assert first.signature() == second.signature()


def test_episode_signature_distinguishes_onset():
    first = build("numerics.finiteness", [80])[0]
    second = build("numerics.finiteness", [120])[0]
    assert first.signature() != second.signature()


def test_episode_id_is_deterministic():
    assert episode_id_for("r", "d", "s", 10) == episode_id_for("r", "d", "s", 10)
    assert episode_id_for("r", "d", "s", 10) != episode_id_for("r", "d", "s", 11)


def test_episode_carries_rule_version():
    episodes = build("numerics.finiteness", [80])
    assert episodes[0].rule_version == RULE_VERSION


# ---------------------------------------------------------------------------
# R0.1 registry
# ---------------------------------------------------------------------------


def test_registry_has_all_three_statuses_available():
    from training_debugger.registry import STATUS_EXPERIMENTAL

    statuses = {entry.status for entry in DETECTOR_REGISTRY}
    assert statuses <= {STATUS_ENABLED, STATUS_EXPERIMENTAL, STATUS_DISABLED}
    assert STATUS_ENABLED in statuses
    assert STATUS_DISABLED in statuses


def test_broken_detectors_are_disabled_with_a_reason():
    registry = detector_registry()
    disabled = {d["detector_id"]: d for d in registry["detectors"]
                if d["status"] == STATUS_DISABLED}
    assert "loss.trend_slope" in disabled
    assert "throughput.regression" in disabled
    for entry in disabled.values():
        assert entry["disabled_reason"], "a disabled detector must say why"


def test_disabled_detectors_are_excluded_from_the_enabled_set():
    registry = detector_registry()
    assert "loss.trend_slope" not in registry["enabled_detector_ids"]
    assert "throughput.regression" not in registry["enabled_detector_ids"]


def test_old_detector_ids_remain_readable():
    """Historical incidents reference these ids; they must still resolve."""
    known = {entry.detector_id for entry in DETECTOR_REGISTRY}
    for old in ("loss.trend_slope", "throughput.regression", "grad.robust_zscore",
                "loss.robust_zscore", "numerics.finiteness"):
        assert old in known, "removing %s would orphan persisted incidents" % old


def test_disabled_detector_produces_no_episode_by_default():
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    engine.consume([anomaly("loss.trend_slope", 100)], "run")
    engine.finish("run", final_step=200)
    assert engine.all_episodes() == []


def test_disabled_detector_can_still_be_observed_when_explicitly_requested():
    """Its failure must remain measurable, not invisible."""
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    engine.consume([anomaly("loss.trend_slope", 100)], "run", include_disabled=True)
    engine.finish("run", final_step=200)
    assert len(engine.all_episodes()) == 1


def test_capability_labels_are_not_overclaimed():
    registry = detector_registry()
    support = registry["anomaly_type_capability"]
    assert support["LOSS_DIVERGENCE"] == "UNSUPPORTED"
    assert support["THROUGHPUT_REGRESSION"] == "UNSUPPORTED"
    # Phase 1.3 over-claimed these as SUPPORTED while measuring episode precision
    # 0.181 and 0.314. Nothing may be VALIDATED without a frozen holdout.
    assert support["GRADIENT_EXPLOSION"] == "EXPERIMENTAL"
    assert support["NAN_DETECTED"] == "EXPERIMENTAL"
    assert registry["validated_capabilities"] == []
    for value in support.values():
        assert value in ("IMPLEMENTED", "EXPERIMENTAL", "VALIDATED", "UNSUPPORTED")


# ---------------------------------------------------------------------------
# R0.1 fusion
# ---------------------------------------------------------------------------


def _episode(anomaly_type, confidence_class, outcome=EpisodeOutcome.CONFIRMED_ANOMALY.value,
             onset=80, confirmed=82, episode_id=None, state=None):
    return AnomalyEpisode(
        episode_id=episode_id or "ep_%s_%s" % (anomaly_type, confidence_class),
        run_id="run",
        detector_id="test",
        detector_version="1",
        rule_version=RULE_VERSION,
        anomaly_type=anomaly_type,
        signal_key="k",
        state=state or EpisodeState.CLOSED.value,
        outcome=outcome,
        candidate_onset_step=onset,
        confirmed_at_step=confirmed,
        confidence_class=confidence_class,
        confidence=0.9,
    )


def test_fusion_certain_promotes_alone():
    verdicts = fuse("run", [_episode("NAN_DETECTED", "CERTAIN")])
    confirmed = [v for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value]
    assert len(confirmed) == 1
    assert confirmed[0].promotion_rule_id == "promote.numerics.certain"


def test_fusion_strong_promotes_alone():
    verdicts = fuse("run", [_episode("GRADIENT_EXPLOSION", "STRONG")])
    confirmed = [v for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value]
    assert len(confirmed) == 1
    assert confirmed[0].promotion_rule_id == "promote.strong.single"


def test_fusion_concordant_weak_promotes():
    episodes = [
        _episode("GRADIENT_EXPLOSION", "WEAK", episode_id="a"),
        _episode("GRADIENT_EXPLOSION", "WEAK", episode_id="b"),
        _episode("GRADIENT_EXPLOSION", "WEAK", episode_id="c"),
    ]
    verdicts = fuse("run", episodes)
    confirmed = [v for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value]
    assert len(confirmed) == 1
    assert confirmed[0].promotion_rule_id == "promote.concordant.weak"


def test_single_weak_does_not_promote():
    """The path that produced Phase 1.2's healthy-run false alarms."""
    verdicts = fuse("run", [_episode("GRADIENT_EXPLOSION", "WEAK")])
    assert all(v.verdict != Verdict.CONFIRMED_INCIDENT.value for v in verdicts)
    assert verdicts[0].verdict == Verdict.CANDIDATE_ONLY.value


def test_candidate_only_is_reported_when_confirmed_but_unpromotable():
    verdicts = fuse("run", [_episode("LOSS_SPIKE", "WEAK")])
    assert verdicts[0].verdict == Verdict.CANDIDATE_ONLY.value
    assert verdicts[0].candidate_onset_step == 80
    assert verdicts[0].confirmed_at_step == 82


def test_insufficient_evidence_when_nothing_resolves():
    episode = _episode("GRADIENT_EXPLOSION", "WEAK", onset=80,
                       state=EpisodeState.SUSPECT.value, outcome=None)
    verdicts = fuse("run", [episode])
    assert verdicts[0].verdict == Verdict.INSUFFICIENT_EVIDENCE.value


def test_no_anomaly_when_every_observation_was_dismissed():
    episode = _episode("GRADIENT_EXPLOSION", "WEAK",
                       outcome=EpisodeOutcome.NOT_ANOMALY.value)
    verdicts = fuse("run", [episode])
    assert verdicts[0].verdict == Verdict.NO_ANOMALY.value


def test_no_anomaly_when_there_are_no_episodes():
    verdicts = fuse("run", [])
    assert verdicts[0].verdict == Verdict.NO_ANOMALY.value


def test_is_promotable_accepts_closed_confirmed_episodes():
    """The defect that made incident recall 0.000 on 85 fault runs."""
    closed_confirmed = _episode("NAN_DETECTED", "CERTAIN")
    assert closed_confirmed.state == EpisodeState.CLOSED.value
    assert is_promotable(closed_confirmed)
    open_confirmed = _episode("NAN_DETECTED", "CERTAIN",
                             state=EpisodeState.CONFIRMED.value, outcome=None)
    assert is_promotable(open_confirmed)
    assert not is_promotable(_episode("NAN_DETECTED", "WEAK",
                                      outcome=EpisodeOutcome.NOT_ANOMALY.value))


# ---------------------------------------------------------------------------
# R0.1 promotion provenance
# ---------------------------------------------------------------------------


def test_promotion_provenance_is_complete():
    verdicts = fuse("run", [_episode("GRADIENT_EXPLOSION", "STRONG")])
    confirmed = [v for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value][0]
    payload = confirmed.to_dict()
    assert payload["promotion_rule_id"]
    assert payload["promotion_rule_version"] == FUSION_RULE_VERSION
    assert payload["episodes_used"]
    assert payload["signals_used"]
    assert payload["candidate_onset_step"] == 80
    assert payload["confirmed_at_step"] == 82
    assert payload["reason"]


def test_rejected_episode_provenance_is_recorded():
    episodes = [
        _episode("GRADIENT_EXPLOSION", "STRONG", episode_id="keep"),
        _episode("GRADIENT_EXPLOSION", "WEAK", episode_id="drop",
                 outcome=EpisodeOutcome.NOT_ANOMALY.value),
    ]
    verdicts = fuse("run", episodes)
    confirmed = [v for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value][0]
    assert "keep" in confirmed.episodes_used
    assert "drop" in confirmed.episodes_rejected


def test_every_promotion_rule_is_versioned_and_registered():
    registry = fusion_rule_registry()
    assert registry["fusion_rule_version"] == FUSION_RULE_VERSION
    assert len(registry["rules"]) == len(PROMOTION_RULES)
    for rule in registry["rules"]:
        assert rule["rule_id"]
        assert rule["version"] == FUSION_RULE_VERSION
        assert rule["description"]


# ---------------------------------------------------------------------------
# R0.1 incident deduplication
# ---------------------------------------------------------------------------


def test_one_episode_yields_one_incident_not_one_per_observation():
    engine = EpisodeEngine()
    engine.consume([anomaly("numerics.finiteness", s) for s in range(80, 191)], "run")
    engine.finish("run", final_step=250)
    verdicts = fuse("run", engine.all_episodes())
    assert sum(1 for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value) == 1


def test_many_episodes_of_one_type_still_yield_one_incident():
    """Fusion decides per anomaly type, so an episode storm cannot multiply incidents."""
    engine = EpisodeEngine()
    observations = []
    for start in (60, 100, 140):
        observations.extend(anomaly("numerics.finiteness", s) for s in range(start, start + 3))
    engine.consume(observations, "run")
    engine.finish("run", final_step=250)
    verdicts = fuse("run", engine.all_episodes())
    confirmed = [v for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value]
    assert len(confirmed) == 1
    assert len(confirmed[0].episodes_used) == 3


# ---------------------------------------------------------------------------
# R0.1 cross-process episode reconstruction
# ---------------------------------------------------------------------------

BUILD = """
import json, sys
sys.path.insert(0, {src!r})
from training_debugger.detectors.base import Anomaly, ConfidenceClass
from training_debugger.schema.enums import EvidenceLevel, Severity
from training_debugger.episodes import EpisodeEngine
from training_debugger.fusion import fuse

def anomaly(step, detector="numerics.finiteness"):
    from training_debugger.schema.enums import EvidenceLevel, Severity
    return Anomaly(detector=detector, detector_version="test",
                   metric_name="grad.global_norm", first_abnormal_step=step,
                   observed_value=1.0, baseline=1.0, threshold=1.0, score=1.0,
                   confidence=0.8, level=EvidenceLevel.INFERRED, severity=Severity.WARN,
                   window=(max(1, step - 50), step),
                   rank=0, confidence_class=ConfidenceClass.STRONG, metadata={{}})

engine = EpisodeEngine()
engine.consume([anomaly(s) for s in range(80, 191)], "run")
engine.finish("run", final_step=250)
episodes = [e.to_dict() for e in engine.all_episodes()]
verdicts = [v.to_dict() for v in fuse("run", engine.all_episodes())]
json.dump({{"episodes": episodes, "verdicts": verdicts}}, open({out!r}, "w"))
print("wrote", len(episodes), "episodes")
"""

READ = """
import json, sys
sys.path.insert(0, {src!r})
from training_debugger.episodes import AnomalyEpisode
from training_debugger.fusion import fuse

payload = json.load(open({out!r}))
episodes = [AnomalyEpisode.from_dict(e) for e in payload["episodes"]]
verdicts = [v.to_dict() for v in fuse("run", episodes)]
json.dump({{"episodes": [e.to_dict() for e in episodes], "verdicts": verdicts}},
          open({read_out!r}, "w"))
print("read", len(episodes), "episodes")
"""


def test_cross_process_episode_reconstruction(tmp_path):
    """Episodes written in one interpreter, fused in a second, must agree exactly."""
    src = str(ROOT / "src")
    written = tmp_path / "written.json"
    read_back = tmp_path / "read.json"

    writer = tmp_path / "write.py"
    writer.write_text(BUILD.format(src=src, out=str(written)), encoding="utf-8")
    result = subprocess.run([sys.executable, str(writer)], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]

    reader = tmp_path / "read.py"
    reader.write_text(
        READ.format(src=src, out=str(written), read_out=str(read_back)), encoding="utf-8"
    )
    result = subprocess.run([sys.executable, str(reader)], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]

    original = json.loads(written.read_text())
    restored = json.loads(read_back.read_text())
    assert restored == original, (
        "a fresh process reconstructed different episodes or verdicts"
    )
