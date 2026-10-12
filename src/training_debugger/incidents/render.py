"""Deterministic incident renderer.

This produces the human-readable diagnosis. It is **not** an LLM: every sentence
is assembled from structured fields, so the same incident always renders to the
same text and every claim is traceable to a field.

That property is the point. Phase 0's top risk is an agent writing plausible prose
that outruns its evidence; a deterministic renderer cannot do that. When an LLM is
added in a later phase it will explain *this* output, not replace it.

The output shape follows the brief's worked example, including the explicit
"Not correlated" section -- which is omitted from the text only when the incident
genuinely has nothing to report there, in which case the reason is printed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schema.enums import EvidenceLevel, IncidentType, OrderingConfidence
from ..schema.incident import Incident

__all__ = ["render_incident", "render_incidents_table", "confidence_label"]


def confidence_label(incident: Incident) -> str:
    """A word, derived from numbers, never guessed."""
    value = incident.confidence or 0.0
    ordering = incident.ordering_confidence
    if value >= 0.85 and ordering == str(OrderingConfidence.PROVEN):
        return "HIGH"
    if value >= 0.6:
        return "MEDIUM"
    if value > 0.0:
        return "LOW"
    return "UNKNOWN"


def render_incident(incident: Incident) -> str:
    """Render one incident as text. Pure function of the incident."""
    lines: List[str] = []
    add = lines.append

    add("INCIDENT {}".format(incident.incident_id))
    add("")
    add("Type:")
    add("  {}  (severity {}, state {})".format(
        incident.incident_type, incident.severity, incident.state))

    if incident.symptom is not None:
        add("")
        add("Observed symptom:")
        add("  {}".format(incident.symptom.claim))
        add("  detector: {} v{} (confidence {:.2f})".format(
            incident.symptom.detector, incident.symptom.detector_version or "?",
            incident.symptom.confidence or 0.0))

    # -- earlier abnormal evidence -------------------------------------
    add("")
    add("Abnormal evidence, earliest first:")
    symptom_id = incident.symptom.event_id if hasattr(incident.symptom, "event_id") else None
    def onset_of(evidence):
        """The anomaly's own step, never the start of its baseline window."""
        if evidence.metadata.get("onset_step") is not None:
            return int(evidence.metadata["onset_step"])
        if evidence.window:
            return int(evidence.window[-1])
        return incident.detected_step or 0

    for evidence in sorted(
        incident.evidence.supporting,
        key=lambda e: (onset_of(e), e.claim),
    ):
        marker = "[observed]" if evidence.level is EvidenceLevel.OBSERVED else "[inferred]"
        add("  {marker} step {step}: {claim}".format(
            marker=marker,
            step=onset_of(evidence),
            claim=evidence.claim,
        ))
    _ = symptom_id

    # -- multiplicity ---------------------------------------------------
    add("")
    add("Affected:")
    add("  {}".format(incident.multiplicity.summary()))
    if incident.multiplicity.affected_layers:
        shown = incident.multiplicity.affected_layers[:10]
        more = "" if len(shown) == len(incident.multiplicity.affected_layers) else " (+{} more)".format(
            len(incident.multiplicity.affected_layers) - len(shown))
        add("  layers: {}{}".format(", ".join(str(x) for x in shown), more))

    # -- negative evidence ----------------------------------------------
    add("")
    add("Not correlated (tested and found stable):")
    if incident.evidence.not_correlated:
        for evidence in incident.evidence.not_correlated:
            add("  - {}".format(evidence.claim))
    else:
        add("  (nothing was testable; see Missing evidence)")

    if incident.evidence.contradicting:
        add("")
        add("Contradicting:")
        for evidence in incident.evidence.contradicting:
            add("  - {}".format(evidence.claim))

    if incident.evidence.unknown:
        add("")
        add("Not assessable:")
        for evidence in incident.evidence.unknown:
            add("  - {}".format(evidence.claim))

    # -- earliest defensible abnormal event ------------------------------
    add("")
    add("Earliest defensible abnormal event:")
    if incident.first_abnormal_event_id:
        for line in incident.earliest.explanation.splitlines():
            add("  {}".format(line))
        add("  event_id: {}".format(incident.first_abnormal_event_id))
    else:
        add("  No single earliest event can be named.")
        for line in incident.earliest.explanation.splitlines():
            add("  {}".format(line))
        add("")
        add("  Candidates ({}):".format(len(incident.earliest_abnormal_candidate_ids)))
        for candidate in incident.earliest.candidates:
            add("    - {} = {:.6g} at step {} ({})".format(
                candidate.metric_name, candidate.metric_value or float("nan"),
                candidate.step, candidate.event_id))
        add("  ordering confidence: {}".format(incident.ordering_confidence))

    # -- hypotheses ------------------------------------------------------
    add("")
    add("Probable root cause (ranked, deterministic scores):")
    if not incident.hypotheses:
        add("  no hypothesis template applies")
    for hypothesis in incident.hypotheses[:3]:
        add("  {:.3f}  [{}] {}".format(hypothesis.score, hypothesis.status, hypothesis.template_id))
        add("         {}".format(hypothesis.statement))
        factors = ", ".join(
            "{}={:.3g}".format(k, v) for k, v in sorted(hypothesis.score_decomposition.items())
        )
        add("         score = {}".format(factors))
        if hypothesis.missing:
            for item in hypothesis.missing:
                add("         missing: {}".format(item))

    # -- missing evidence -------------------------------------------------
    add("")
    add("Missing evidence:")
    if incident.missing_evidence:
        for item in incident.missing_evidence:
            add("  - {}".format(item))
    else:
        add("  (none recorded)")

    add("")
    add("Confidence: {}".format(confidence_label(incident)))
    add("  ordering: {}  |  overall: {:.2f}".format(
        incident.ordering_confidence, incident.confidence or 0.0))
    add("  anomalies: {}  |  series examined: {}".format(
        incident.metadata.get("anomaly_count", 0), incident.metadata.get("series_examined", 0)))
    if incident.metadata.get("detector_errors"):
        add("  detector errors: {}".format(incident.metadata["detector_errors"]))
    return "\n".join(lines)


def render_incidents_table(incidents: List[Incident]) -> str:
    if not incidents:
        return "no incidents"
    lines = [
        "{:<20} {:<22} {:>7} {:<8} {:<7} {}".format(
            "incident_id", "type", "step", "severity", "conf", "first abnormal event"),
    ]
    for incident in incidents:
        lines.append("{:<20} {:<22} {:>7} {:<8} {:<7} {}".format(
            incident.incident_id,
            str(incident.incident_type),
            str(incident.detected_step),
            str(incident.severity),
            confidence_label(incident),
            incident.first_abnormal_event_id or "(concurrent; see report)",
        ))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# R8: user-visible abstention
# ---------------------------------------------------------------------------


def render_verdict(verdict: Any, episodes: Optional[List[Any]] = None) -> str:
    """Render a fusion verdict, including the abstentions.

    Phase 1.3 implemented abstention but kept it inside an object, so a user could
    not tell "nothing happened" from "we could not tell". This renders all four
    outcomes, and for the two abstentions it states what was observed, what was
    missing, and therefore why no incident was opened.

    Causal language is forbidden here: this function may say a candidate
    *preceded* something, never that it *caused* it. See
    ``training_debugger.correlation.assert_causal_language_absent``.
    """
    payload = verdict.to_dict() if hasattr(verdict, "to_dict") else dict(verdict)
    kind = payload.get("verdict")
    lines: List[str] = []
    add = lines.append
    episodes = episodes or []
    by_id = {}
    for episode in episodes:
        data = episode.to_dict() if hasattr(episode, "to_dict") else dict(episode)
        by_id[data.get("episode_id")] = data

    def _episode_lines(ids):
        for episode_id in ids or []:
            data = by_id.get(episode_id)
            if not data:
                add("  - episode {} (details unavailable)".format(episode_id))
                continue
            layers = data.get("affected_layers") or []
            population = (data.get("metadata") or {}).get("population_layers")
            scope = "{}/{} monitored layers".format(
                len(layers), population if population else "?"
            ) if layers else "the aggregate signal"
            add("  - {} at step {}, affecting {}".format(
                data.get("anomaly_type"), data.get("candidate_onset_step"), scope))
            delay = data.get("confirmation_delay_steps")
            if delay:
                add("    onset at step {}, confirmed at step {} (+{} steps)".format(
                    data.get("candidate_onset_step"), data.get("confirmed_at_step"), delay))

    if kind == "CONFIRMED_INCIDENT":
        add("CONFIRMED INCIDENT")
        add("")
        add("  type:       {}".format(payload.get("anomaly_type")))
        add("  severity:   {}".format(payload.get("severity")))
        add("  confidence: {:.2f}".format(payload.get("confidence") or 0.0))
        add("  promotion:  {} (v{})".format(
            payload.get("promotion_rule_id"), payload.get("promotion_rule_version")))
        add("")
        add("Evidence:")
        _episode_lines(payload.get("episodes_used"))
        if payload.get("episodes_rejected"):
            add("")
            add("Rejected (observed but not promotion-worthy):")
            _episode_lines(payload.get("episodes_rejected"))
        add("")
        add("Promotion disposition: {} (reason: {})".format(
            payload.get("promotion_disposition"), payload.get("reason_code")))
        add("Earliest defensible onset: step {}".format(payload.get("candidate_onset_step")))
        add("Confirmed at:              step {}".format(payload.get("confirmed_at_step")))
        return "\n".join(lines)

    if kind == "CANDIDATE_ONLY":
        reason_code = payload.get("reason_code")
        if reason_code == "CONFIRMED_BUT_PROMOTION_DISABLED":
            add("No incident was opened. The abnormal episode IS CONFIRMED; "
                "promotion for this experimental capability is switched off in this "
                "build.")
            add("")
            add("This is NOT a statement that the evidence was insufficient.")
        elif reason_code == "CONFIRMED_BUT_NO_RULE_MATCHED":
            add("No incident was opened. The abnormal episode IS CONFIRMED, but no "
                "promotion rule accepted it.")
        else:
            add("No confirmed incident. One or more candidates were observed but did "
                "not meet a promotion rule.")
        add("")
        add("Observed:")
        _episode_lines(payload.get("episodes_used") or payload.get("signals_used"))
        add("")
        add("Therefore no incident was opened.")
        add("Reason: {}".format(payload.get("reason") or "no promotion rule matched"))
        return "\n".join(lines)

    if kind == "INSUFFICIENT_EVIDENCE":
        add("INSUFFICIENT EVIDENCE")
        add("")
        add("Observed:")
        _episode_lines(payload.get("signals_used") or payload.get("abstained_episodes"))
        add("")
        add("Missing:")
        add("  the corroboration needed to confirm the observation -- either a")
        add("  persistent recurrence, an aggregate-series departure, or a")
        add("  cross-layer multiplicity above the validated threshold.")
        add("")
        add("Therefore no incident was opened.")
        add("Reason: {}".format(payload.get("reason") or "evidence did not resolve"))
        if payload.get("candidate_onset_step") is not None:
            add("")
            add("A candidate onset near step {} was observed but is not defensible "
                "as an incident onset.".format(payload["candidate_onset_step"]))
        return "\n".join(lines)

    add("NO ANOMALY")
    add("")
    add("Observations were made and dismissed as not anomalous.")
    if payload.get("reason"):
        add("Reason: {}".format(payload["reason"]))
    add("")
    warning = payload.get("completeness_warning")
    if warning:
        # A1: a negative verdict from an incomplete or unreported stream must NOT be
        # presented as reassurance. Saying "nothing was found" while events may have
        # been dropped is reporting missing evidence as evidence of normality.
        add("EVIDENCE COMPLETENESS: {}".format(
            (payload.get("data_quality") or {}).get("status", "UNKNOWN")))
        add("")
        add(warning)
        add("")
        add("This is a statement about the events that arrived. It is NOT a clean "
            "bill of health, and it does not establish that the run was healthy.")
    else:
        add("This is a positive statement that nothing was found, not an abstention.")
    return "\n".join(lines)


def render_verdicts(verdicts: Sequence[Any], episodes: Optional[List[Any]] = None) -> str:
    return ("\n" + "=" * 72 + "\n").join(
        render_verdict(v, episodes) for v in verdicts
    )
