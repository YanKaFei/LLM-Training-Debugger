"""R8: abstention must be visible in the deterministic renderer."""

from __future__ import annotations

import pytest

from training_debugger.correlation import assert_causal_language_absent
from training_debugger.episodes import AnomalyEpisode, EpisodeOutcome, EpisodeState
from training_debugger.fusion import fuse
from training_debugger.incidents.render import render_verdict, render_verdicts


def episode(anomaly_type="GRADIENT_EXPLOSION", outcome=EpisodeOutcome.NOT_ANOMALY.value,
            layers=(2,), population=7, onset=78, confirmed=None, confidence_class="WEAK"):
    return AnomalyEpisode(
        episode_id="ep1", run_id="r", detector_id="grad.robust_zscore",
        detector_version="2.0", rule_version="v", anomaly_type=anomaly_type,
        signal_key="k", state=EpisodeState.CLOSED.value, outcome=outcome,
        candidate_onset_step=onset, confirmed_at_step=confirmed,
        affected_layers=list(layers), confidence_class=confidence_class,
        confidence=0.6, metadata={"population_layers": population},
    )


def test_no_anomaly_renders_as_a_positive_statement_only_when_complete():
    """A1 changed this deliberately.

    "Nothing was found" is only a positive statement when completeness was
    demonstrated. Without it, the same verdict must not read as reassurance --
    otherwise a dropped event is indistinguishable from a healthy step.
    """
    from training_debugger.fusion.quality import DataQuality, DataQualityStatus

    complete = fuse("r", [episode()],
                    data_quality=DataQuality(status=DataQualityStatus.COMPLETE.value))[0]
    text = render_verdict(complete, [episode()])
    assert "NO ANOMALY" in text
    assert "positive statement" in text
    assert "not an abstention" in text

    unreported = fuse("r", [episode()])[0]
    text = render_verdict(unreported, [episode()])
    assert "positive statement" not in text, (
        "an unreported completeness must not be rendered as a clean negative"
    )
    assert "EVIDENCE COMPLETENESS" in text


def test_insufficient_evidence_is_rendered_explicitly():
    suspect = episode(outcome=None, confidence_class="WEAK")
    suspect.state = EpisodeState.SUSPECT.value
    verdict = fuse("r", [suspect])[0]
    text = render_verdict(verdict, [suspect])
    assert "INSUFFICIENT EVIDENCE" in text
    assert "Observed:" in text
    assert "Missing:" in text
    assert "Therefore no incident was opened." in text


def test_candidate_only_is_rendered_with_its_scope():
    weak = episode(outcome=EpisodeOutcome.CONFIRMED_ANOMALY.value, confidence_class="WEAK")
    verdict = fuse("r", [weak])[0]
    assert verdict.verdict == "CANDIDATE_ONLY"
    text = render_verdict(verdict, [weak])
    # The CANDIDATE_ONLY rendering was refined in Phase 1.12B (B0.1) so that a
    # disabled promotion is not described as missing evidence. The scope must still
    # be visible, which is what this test is actually about.
    assert "No incident was opened" in text
    assert "1/7 monitored layers" in text, "scope must be visible to the reader"
    assert "CONFIRMED" in text, "the evidence state must be stated"


def test_confirmed_incident_renders_both_moments():
    strong = episode(outcome=EpisodeOutcome.CONFIRMED_ANOMALY.value,
                     confidence_class="STRONG", confirmed=80)
    verdict = fuse("r", [strong])[0]
    text = render_verdict(verdict, [strong])
    assert "CONFIRMED INCIDENT" in text
    assert "Earliest defensible onset: step 78" in text
    assert "Confirmed at:              step 80" in text
    assert "promotion:" in text


def test_renderer_never_uses_causal_language():
    for build in (
        lambda: [episode()],
        lambda: [episode(outcome=EpisodeOutcome.CONFIRMED_ANOMALY.value, confidence_class="STRONG")],
        lambda: [episode(anomaly_type="NAN_DETECTED", confidence_class="CERTAIN",
                         outcome=EpisodeOutcome.CONFIRMED_ANOMALY.value)],
    ):
        episodes = build()
        text = render_verdicts(fuse("r", episodes), episodes)
        assert_causal_language_absent(text)


def test_rendering_is_deterministic():
    episodes = [episode(outcome=EpisodeOutcome.CONFIRMED_ANOMALY.value, confidence_class="STRONG")]
    first = render_verdicts(fuse("r", episodes), episodes)
    second = render_verdicts(fuse("r", episodes), episodes)
    assert first == second
