"""B0/B1: verdict semantics and incident reconciliation."""

from __future__ import annotations

import pytest

from training_debugger.episodes import AnomalyEpisode
from training_debugger.fusion import PromotionDisposition, ReasonCode, Verdict, fuse
from training_debugger.incidents import reconcile_incidents
from training_debugger.incidents.relations import RelationType, link_consequences
from training_debugger.incidents.render import render_verdict


def episode(eid, anomaly_type, onset, layers=(0, 1, 2), outcome="CONFIRMED_ANOMALY",
            state="CLOSED", confidence_class="STRONG", metric=None):
    return AnomalyEpisode(
        episode_id=eid, run_id="r", detector_id="d", detector_version="1",
        rule_version="v", anomaly_type=anomaly_type, signal_key="k",
        metric_name=metric,
        state=state, outcome=outcome, candidate_onset_step=onset,
        confirmed_at_step=onset + 2 if outcome == "CONFIRMED_ANOMALY" else None,
        affected_layers=list(layers), confidence_class=confidence_class, confidence=0.9,
    )


# ---------------------------------------------------------------------------
# B0.1 verdict semantics
# ---------------------------------------------------------------------------


def test_promotion_disabled_is_not_reported_as_insufficient_evidence():
    """The distinction Phase 1.12A's `CANDIDATE_ONLY` conflated."""
    verdict = fuse("r", [episode("e1", "LOSS_SPIKE", 80)])[0]
    assert verdict.verdict == Verdict.CANDIDATE_ONLY.value
    assert verdict.reason_code == ReasonCode.CONFIRMED_BUT_PROMOTION_DISABLED.value
    assert verdict.evidence_state == "CONFIRMED"
    assert verdict.promotion_disposition == PromotionDisposition.DISABLED.value
    assert "not a statement that the evidence was insufficient" in verdict.reason


def test_genuinely_unresolved_evidence_keeps_its_own_reason_code():
    suspect = episode("e2", "GRADIENT_EXPLOSION", 80, outcome=None, state="SUSPECT",
                      confidence_class="WEAK")
    verdict = fuse("r", [suspect])[0]
    assert verdict.verdict == Verdict.INSUFFICIENT_EVIDENCE.value
    assert verdict.reason_code == ReasonCode.EVIDENCE_INSUFFICIENT.value


def test_no_rule_matched_is_distinct_from_promotion_disabled():
    weak = episode("e3", "GRADIENT_EXPLOSION", 80, confidence_class="WEAK")
    verdict = fuse("r", [weak])[0]
    assert verdict.verdict == Verdict.CANDIDATE_ONLY.value
    assert verdict.reason_code == ReasonCode.CONFIRMED_BUT_NO_RULE_MATCHED.value
    assert verdict.promotion_disposition == PromotionDisposition.WITHHELD.value


def test_promoted_verdict_carries_promoted_reason_code():
    verdict = fuse("r", [episode("e4", "NAN_DETECTED", 80, confidence_class="CERTAIN")])[0]
    assert verdict.verdict == Verdict.CONFIRMED_INCIDENT.value
    assert verdict.reason_code == ReasonCode.PROMOTED.value
    assert verdict.promotion_disposition == PromotionDisposition.ELIGIBLE.value


def test_no_observations_has_its_own_code():
    verdict = fuse("r", [])[0]
    assert verdict.verdict == Verdict.NO_ANOMALY.value
    assert verdict.reason_code == ReasonCode.NO_OBSERVATIONS.value


def test_renderer_does_not_imply_missing_evidence_for_a_disabled_promotion():
    verdict = fuse("r", [episode("e5", "LOSS_SPIKE", 80)])[0]
    text = render_verdict(verdict, [episode("e5", "LOSS_SPIKE", 80)])
    assert "IS CONFIRMED" in text
    assert "NOT a statement that the evidence was insufficient" in text


def test_verdict_api_remains_backward_compatible():
    """Existing consumers read `verdict`, `reason`, `episodes_used` unchanged."""
    verdict = fuse("r", [episode("e6", "NAN_DETECTED", 80, confidence_class="CERTAIN")])[0]
    payload = verdict.to_dict()
    for key in ("verdict", "reason", "episodes_used", "candidate_onset_step",
                "confirmed_at_step", "promotion_rule_id"):
        assert key in payload
    for key in ("reason_code", "evidence_state", "promotion_disposition"):
        assert key in payload


# ---------------------------------------------------------------------------
# B1 reconciliation
# ---------------------------------------------------------------------------


def test_linked_manifestation_forms_a_group_without_proving_a_root_cause():
    """v0.1.3: grouping needs mechanism, not order.

    A non-finite gradient one step before a non-finite loss is a real propagation; a
    finite scale anomaly 40 steps earlier is not, and must stay independent.
    """
    episodes = [
        episode("g1", "NAN_DETECTED", 80, metric="grad.global_norm"),
        episode("p1", "NAN_DETECTED", 80, metric="param.global_norm"),
        episode("p1", "NAN_DETECTED", 80, metric="param.global_norm"),
        episode("n1", "NAN_DETECTED", 81, metric="train.loss"),
    ]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    result = reconcile_incidents("r", episodes, relations)
    assert result.groups, "a mechanism-supported link must group"
    # `n1` must appear as somebody's secondary manifestation.
    secondaries = {sid for group in result.groups for sid in group.secondary_episode_ids}
    assert "n1" in secondaries
    assert all(group.has_proven_root_cause is False for group in result.groups), (
        "a presentation group must never assert a proven root cause"
    )
    assert "does NOT establish" in result.groups[0].presentation_note


def test_standalone_nan_stays_independent():
    episodes = [episode("solo", "NAN_DETECTED", 50, layers=(5,))]
    result = reconcile_incidents("r", episodes, [])
    assert result.groups == []
    assert result.independent_episode_ids == ["solo"]


def test_unrelated_nan_is_not_force_merged():
    """Far outside the window and disjoint in scope: no relation, no group."""
    episodes = [
        episode("g1", "GRADIENT_EXPLOSION", 100, layers=(0, 1)),
        episode("far", "NAN_DETECTED", 900, layers=(9,)),
    ]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    assert relations == []
    result = reconcile_incidents("r", episodes, relations)
    assert result.groups == []
    assert set(result.independent_episode_ids) == {"g1", "far"}, (
        "two independent anomalies must not be merged"
    )


def test_duplicate_episode_emission_is_reported_not_grouped():
    """Situation C is an idempotency defect, not something grouping should hide."""
    episodes = [episode("dup", "NAN_DETECTED", 50), episode("dup", "NAN_DETECTED", 50)]
    result = reconcile_incidents("r", episodes, [])
    assert result.duplicate_episode_ids == ["dup"]
    assert result.raw_episode_count == 1


def test_unknown_relation_does_not_attach():
    relation = {
        "relation_id": "rel_x", "relation_type": RelationType.UNKNOWN_RELATION.value,
        "source_incident_id": "g1", "target_incident_id": "n1",
    }
    episodes = [episode("g1", "GRADIENT_EXPLOSION", 100), episode("n1", "NAN_DETECTED", 81, metric="train.loss")]
    result = reconcile_incidents("r", episodes, [relation])
    assert result.groups == [], "an unclassified association must not be merged"
    assert result.unresolved_relation_ids == ["rel_x"]


def test_reconciliation_preserves_every_episode_id():
    episodes = [
        episode("g1", "GRADIENT_EXPLOSION", 100),
        episode("n1", "NAN_DETECTED", 81, metric="train.loss"),
        episode("n2", "NAN_DETECTED", 900, layers=(9,)),
    ]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    result = reconcile_incidents("r", episodes, relations)
    recorded = set(result.independent_episode_ids)
    for group in result.groups:
        recorded.add(group.primary_episode_id)
        recorded.update(group.secondary_episode_ids)
    assert recorded == {"g1", "n1", "n2"}, "grouping must not delete evidence"


def test_reconciliation_is_idempotent():
    episodes = [episode("g1", "GRADIENT_EXPLOSION", 100), episode("n1", "NAN_DETECTED", 81, metric="train.loss")]
    relations, _ = link_consequences("r", episodes, mechanism_linker_enabled=True)
    first = reconcile_incidents("r", episodes, relations).to_dict()
    second = reconcile_incidents("r", episodes, relations).to_dict()
    assert first == second


def test_no_causal_relation_type_exists():
    names = {t.value for t in RelationType}
    assert "CAUSED_BY" not in names
    assert "ROOT_CAUSE_OF" not in names


# ---------------------------------------------------------------------------
# A1 evidence completeness
# ---------------------------------------------------------------------------


def test_missing_quality_information_is_not_treated_as_complete():
    from training_debugger.fusion.quality import DataQualityStatus

    verdict = fuse("r", [])[0]
    assert verdict.data_quality["status"] == DataQualityStatus.UNKNOWN.value
    assert verdict.completeness_warning, (
        "an unreported completeness must be warned about, not assumed clean"
    )


def test_degraded_evidence_is_surfaced_with_its_counts():
    from training_debugger.fusion.quality import DataQuality, DataQualityStatus

    quality = DataQuality(status=DataQualityStatus.DEGRADED.value, dropped_events=42,
                          overflow_events=7, persistence_failures=1)
    verdict = fuse("r", [], data_quality=quality)[0]
    assert verdict.data_quality["status"] == "DEGRADED"
    assert verdict.data_quality["dropped_events"] == 42
    assert "42 dropped" in verdict.completeness_warning
    assert "NOT that the run was healthy" in verdict.completeness_warning


def test_complete_evidence_carries_no_warning():
    from training_debugger.fusion.quality import DataQuality, DataQualityStatus

    verdict = fuse("r", [], data_quality=DataQuality(status=DataQualityStatus.COMPLETE.value))[0]
    assert verdict.completeness_warning is None


def test_renderer_does_not_claim_normality_from_a_degraded_stream():
    """The brief: missing evidence must never be reported as evidence of normality."""
    from training_debugger.fusion.quality import DataQuality

    verdict = fuse("r", [], data_quality=DataQuality(status="DEGRADED", dropped_events=5))[0]
    text = render_verdict(verdict, [])
    assert "EVIDENCE COMPLETENESS: DEGRADED" in text
    assert "NOT a clean bill of health" in text
    assert "positive statement that nothing was found" not in text, (
        "a degraded stream must not be rendered as reassurance"
    )


def test_observer_health_maps_to_a_quality_status():
    from training_debugger.fusion.quality import DataQuality, DataQualityStatus

    # The snapshot is nested; flat keys are not recognised and must yield UNKNOWN,
    # never COMPLETE (see the test below).
    clean = DataQuality.from_observer_health({
        "observer": {"events_dropped_total": 0, "queue_overflow_total": 0,
                     "writer_failures_total": 0},
        "queue": {"evicted": 0},
    })
    assert clean.status == DataQualityStatus.COMPLETE.value
    dirty = DataQuality.from_observer_health({
        "observer": {"events_dropped_total": 3, "queue_overflow_total": 0,
                     "writer_failures_total": 0},
        "queue": {"evicted": 0},
    })
    assert dirty.status == DataQualityStatus.DEGRADED.value


def test_quality_is_stamped_on_every_verdict_not_only_negative_ones():
    from training_debugger.fusion.quality import DataQuality

    episodes = [episode("e7", "NAN_DETECTED", 80, confidence_class="CERTAIN")]
    verdicts = fuse("r", episodes, data_quality=DataQuality(status="DEGRADED", dropped_events=1))
    confirmed = [v for v in verdicts if v.verdict == Verdict.CONFIRMED_INCIDENT.value]
    assert confirmed, "the positive verdict must still be produced"
    assert confirmed[0].data_quality["status"] == "DEGRADED"


def test_unrecognised_health_snapshot_yields_unknown_not_complete():
    """The bug the quickstart example exposed.

    `from_observer_health` looked for flat keys that do not exist in the nested
    `health()` snapshot, so every value defaulted to zero and completeness was
    reported COMPLETE for a run that had never been read. Absence of a reading is not
    a reading of zero.
    """
    from training_debugger.fusion.quality import DataQuality, DataQualityStatus

    quality = DataQuality.from_observer_health({})
    assert quality.status == DataQualityStatus.UNKNOWN.value
    quality = DataQuality.from_observer_health({"unrelated": 1})
    assert quality.status == DataQualityStatus.UNKNOWN.value


def test_real_nested_health_snapshot_is_read_correctly():
    from training_debugger.fusion.quality import DataQuality, DataQualityStatus

    clean = DataQuality.from_observer_health({
        "observer": {"events_dropped_total": 0, "queue_overflow_total": 0,
                     "writer_failures_total": 0, "written_total": 100,
                     "emitted_total": 100},
        "queue": {"evicted": 0, "rejected": 0},
    })
    assert clean.status == DataQualityStatus.COMPLETE.value
    assert clean.events_persisted == 100

    dirty = DataQuality.from_observer_health({
        "observer": {"events_dropped_total": 5, "queue_overflow_total": 2,
                     "writer_failures_total": 1, "written_total": 90},
        "queue": {"evicted": 2, "rejected": 0},
    })
    assert dirty.status == DataQualityStatus.DEGRADED.value
    assert dirty.dropped_events == 5
    assert dirty.persistence_failures == 1
