"""E7/E8: deterministic fusion, promotion rules and abstention.

This is the layer that decides, from episodes, which of four things is true::

    CONFIRMED_INCIDENT
    CANDIDATE_ONLY
    INSUFFICIENT_EVIDENCE
    NO_ANOMALY

Fusion is **explicit and versioned**, never learned and never a model. Every
decision records the rule that produced it and the episodes it accepted, rejected
or abstained on, so that "why did this become an incident?" is answerable from the
stored object alone.

The most important thing this module can do is **decline**. A system that always
answers has no way to be right about what it does not know, and Phase 1.2 measured
what that costs: precision 0.121, earliest-evidence precision 0.338, 56 pre-injection
incidents out of 117.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..episodes.model import AnomalyEpisode, EpisodeOutcome, EpisodeState

__all__ = [
    "Verdict",
    "ReasonCode",
    "PromotionDisposition",
    "IncidentVerdict",
    "PromotionRule",
    "PROMOTION_RULES",
    "FUSION_RULE_VERSION",
    "fuse",
    "fusion_rule_registry",
]

#: Version of the fusion rules. Persisted on every incident.
FUSION_RULE_VERSION = "phase1.3-1.0"


#: Why a verdict came out the way it did. Machine-readable, and the reason
#: ``CANDIDATE_ONLY`` must never be explained to a user as "not enough evidence"
#: when the truth is "the evidence was sufficient but the capability is not cleared
#: to raise incidents".
class ReasonCode(str, Enum):
    #: A promotion rule matched and an incident was opened.
    PROMOTED = "PROMOTED"
    #: The episode **confirmed**, but its capability's promotion is switched off in
    #: this build. The evidence is not in question; the capability is not cleared.
    CONFIRMED_BUT_PROMOTION_DISABLED = "CONFIRMED_BUT_PROMOTION_DISABLED"
    #: The episode confirmed, promotion is enabled, but no rule accepted it --
    #: typically a single WEAK episode.
    CONFIRMED_BUT_NO_RULE_MATCHED = "CONFIRMED_BUT_NO_RULE_MATCHED"
    #: Episodes exist and remain unresolved. This one really is an evidence problem.
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    #: Observations were made and dismissed.
    OBSERVATIONS_DISMISSED = "OBSERVATIONS_DISMISSED"
    #: Nothing was observed at all.
    NO_OBSERVATIONS = "NO_OBSERVATIONS"


class PromotionDisposition(str, Enum):
    """The promotion dimension, kept separate from the evidence dimension."""

    ELIGIBLE = "ELIGIBLE"
    DISABLED = "DISABLED"
    WITHHELD = "WITHHELD"


class Verdict(str, Enum):
    """The four outcomes the brief requires. Nothing else may be returned."""

    CONFIRMED_INCIDENT = "CONFIRMED_INCIDENT"
    CANDIDATE_ONLY = "CANDIDATE_ONLY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NO_ANOMALY = "NO_ANOMALY"


#: Anomaly types whose *promotion* is switched off while their detector remains
#: running. This is the brief's J2 requirement: `LOSS_SPIKE` alarmed on 8 of 14
#: noisy healthy runs, and the brief states that if H2 still systematically alarms,
#: the detector "must not remain default-enabled".
#:
#: Promotion is off, not the detector. Candidates are still recorded, still
#: measurable, and still rendered -- so the failure stays visible instead of being
#: hidden by removal, and the default incident quality is not polluted by a
#: detector that is known not to work yet.
PROMOTION_DISABLED_TYPES = frozenset({"LOSS_SPIKE"})


def promotion_enabled(anomaly_type: str) -> bool:
    return anomaly_type not in PROMOTION_DISABLED_TYPES


#: Which episodes may open an incident on their own.
#:
#: A ``SUSPECT`` or ``NOT_ANOMALY`` episode is by construction not evidence of
#: anything. An episode qualifies when it is **either** still open and confirmed
#: (``state == CONFIRMED``) **or** closed having been confirmed
#: (``outcome == CONFIRMED_ANOMALY``).
#:
#: The second case is not a detail: the engine closes every episode before fusion
#: runs, so a check against ``state`` alone matched nothing and **no incident could
#: ever be promoted** -- measured as incident recall 0.000 on 85 fault runs with 50
#: correctly confirmed episodes sitting unused.
PROMOTABLE_STATES = (EpisodeState.CONFIRMED.value,)


def is_promotable(episode: Any) -> bool:
    """Whether an episode carries confirmed abnormal evidence."""
    return (
        getattr(episode, "state", None) == EpisodeState.CONFIRMED.value
        or getattr(episode, "outcome", None) == EpisodeOutcome.CONFIRMED_ANOMALY.value
    )


@dataclass(frozen=True)
class PromotionRule:
    """One deterministic promotion rule.

    ``rule_id`` and ``version`` are persisted on the incident, because an incident
    that cannot name the rule that produced it cannot be reviewed after the rule
    changes.
    """

    rule_id: str
    version: str
    description: str
    #: Anomaly types this rule may promote.
    anomaly_types: Tuple[str, ...]
    #: Minimum number of concordant confirmed episodes required.
    min_confirmed_episodes: int = 1
    #: Confidence classes accepted as a single-episode promotion.
    accepted_classes: Tuple[str, ...] = ("CERTAIN", "STRONG")
    #: When true, several WEAK episodes that agree may jointly promote.
    allow_concordant_weak: bool = False
    #: Concordant WEAK episodes required when ``allow_concordant_weak``.
    min_weak_episodes: int = 3
    severity: str = "warn"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "version": self.version,
            "description": self.description,
            "anomaly_types": list(self.anomaly_types),
            "min_confirmed_episodes": self.min_confirmed_episodes,
            "accepted_classes": list(self.accepted_classes),
            "allow_concordant_weak": self.allow_concordant_weak,
            "min_weak_episodes": self.min_weak_episodes,
            "severity": self.severity,
        }


#: Evaluated in order; the first rule whose conditions are met promotes.
#:
#: ``CERTAIN`` episodes promote alone because a measured non-finite value cannot be
#: a statistical accident. ``STRONG`` promotes alone because it means the aggregate
#: signal moved or most layers moved together. A single ``WEAK`` episode never
#: promotes -- that path is what produced Phase 1.2's healthy-run false alarms.
PROMOTION_RULES: Tuple[PromotionRule, ...] = (
    PromotionRule(
        rule_id="promote.numerics.certain",
        version=FUSION_RULE_VERSION,
        description=(
            "A confirmed episode whose observations are CERTAIN (a measured "
            "non-finite value) opens an incident alone."
        ),
        anomaly_types=("NAN_DETECTED", "INF_DETECTED"),
        min_confirmed_episodes=1,
        accepted_classes=("CERTAIN",),
        severity="error",
    ),
    PromotionRule(
        rule_id="promote.strong.single",
        version=FUSION_RULE_VERSION,
        description=(
            "One confirmed episode with STRONG evidence (aggregate signal departed, "
            "or cross-layer multiplicity at or above the measured threshold) opens "
            "an incident."
        ),
        anomaly_types=("NAN_DETECTED", "INF_DETECTED", "GRADIENT_EXPLOSION",
                       "LOSS_SPIKE", "LOSS_DIVERGENCE", "THROUGHPUT_REGRESSION"),
        min_confirmed_episodes=1,
        accepted_classes=("CERTAIN", "STRONG"),
        severity="warn",
    ),
    PromotionRule(
        rule_id="promote.concordant.weak",
        version=FUSION_RULE_VERSION,
        description=(
            "Three or more concordant confirmed WEAK episodes of the same anomaly "
            "type promote jointly. A single WEAK episode never promotes."
        ),
        anomaly_types=("GRADIENT_EXPLOSION", "LOSS_SPIKE", "LOSS_DIVERGENCE",
                       "THROUGHPUT_REGRESSION"),
        min_confirmed_episodes=3,
        accepted_classes=(),
        allow_concordant_weak=True,
        min_weak_episodes=3,
        severity="warn",
    ),
)


@dataclass
class IncidentVerdict:
    """The fusion decision, with full provenance."""

    run_id: str
    #: Defaulted so the per-anomaly-type branch can construct a verdict before the
    #: outcome is known. Without a default this raised TypeError on every run that
    #: produced an episode -- 148 of 169 -- which is how it was found.
    verdict: str = Verdict.NO_ANOMALY.value
    anomaly_type: Optional[str] = None
    severity: str = "info"
    confidence: float = 0.0

    promotion_rule_id: Optional[str] = None
    promotion_rule_version: Optional[str] = None

    episodes_used: List[str] = field(default_factory=list)
    episodes_rejected: List[str] = field(default_factory=list)
    abstained_episodes: List[str] = field(default_factory=list)
    signals_used: List[str] = field(default_factory=list)
    signals_rejected: List[str] = field(default_factory=list)

    candidate_onset_step: Optional[int] = None
    confirmed_at_step: Optional[int] = None
    reason: str = ""
    #: B0.1: two independent dimensions, so a user-facing report cannot conflate
    #: "the evidence was insufficient" with "the evidence was sufficient but the
    #: capability is not cleared to raise incidents".
    reason_code: str = ReasonCode.NO_OBSERVATIONS.value
    evidence_state: str = "CLOSED"
    promotion_disposition: str = PromotionDisposition.ELIGIBLE.value
    #: A1: completeness of the evidence this verdict rests on. A negative verdict from
    #: an incomplete stream must never be read as "the run was healthy".
    data_quality: Dict[str, Any] = field(default_factory=dict)
    completeness_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "verdict": self.verdict,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "promotion_rule_id": self.promotion_rule_id,
            "promotion_rule_version": self.promotion_rule_version,
            "episodes_used": list(self.episodes_used),
            "episodes_rejected": list(self.episodes_rejected),
            "abstained_episodes": list(self.abstained_episodes),
            "signals_used": list(self.signals_used),
            "signals_rejected": list(self.signals_rejected),
            "candidate_onset_step": self.candidate_onset_step,
            "confirmed_at_step": self.confirmed_at_step,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "evidence_state": self.evidence_state,
            "promotion_disposition": self.promotion_disposition,
            "data_quality": dict(self.data_quality),
            "completeness_warning": self.completeness_warning,
        }


def fuse(
    run_id: str,
    episodes: Sequence[AnomalyEpisode],
    *,
    rules: Sequence[PromotionRule] = PROMOTION_RULES,
    data_quality: Optional[Any] = None,
) -> List[IncidentVerdict]:
    """Decide what, if anything, the episodes amount to.

    Returns one verdict per anomaly type that has *any* episode, plus an overall
    ``NO_ANOMALY`` verdict when there are none. An anomaly type whose episodes all
    stayed unconfirmed yields ``CANDIDATE_ONLY`` or ``INSUFFICIENT_EVIDENCE`` --
    never an incident.
    """
    from .quality import UNKNOWN_QUALITY, DataQuality

    quality = data_quality if data_quality is not None else UNKNOWN_QUALITY
    if isinstance(quality, dict):
        quality = DataQuality(**{k: v for k, v in quality.items()
                                 if k in DataQuality.__dataclass_fields__})
    quality_payload = quality.to_dict()
    quality_warning = quality.warning_text()

    def _stamp(verdict: IncidentVerdict) -> IncidentVerdict:
        verdict.data_quality = quality_payload
        verdict.completeness_warning = quality_warning
        return verdict

    verdicts: List[IncidentVerdict] = []
    if not episodes:
        return [
            IncidentVerdict(
                run_id=run_id,
                verdict=Verdict.NO_ANOMALY.value,
                confidence=1.0,
                reason="no detector produced an observation",
                reason_code=ReasonCode.NO_OBSERVATIONS.value,
                evidence_state="CLOSED",
                promotion_disposition=PromotionDisposition.WITHHELD.value,
                data_quality=quality_payload,
                completeness_warning=quality_warning,
            )
        ]

    by_type: Dict[str, List[AnomalyEpisode]] = {}
    for episode in episodes:
        by_type.setdefault(episode.anomaly_type, []).append(episode)

    for anomaly_type in sorted(by_type):
        group = by_type[anomaly_type]
        confirmed = [e for e in group if is_promotable(e)]
        unconfirmed = [e for e in group if not is_promotable(e)]
        rejected = [e for e in group if e.outcome == EpisodeOutcome.NOT_ANOMALY.value]
        abstained = [e for e in group if e.outcome == EpisodeOutcome.INSUFFICIENT_EVIDENCE.value]

        verdict = IncidentVerdict(run_id=run_id, anomaly_type=anomaly_type)
        verdict.episodes_rejected = [e.episode_id for e in rejected]
        verdict.abstained_episodes = [e.episode_id for e in abstained]
        verdict.signals_rejected = sorted({e.signal_key for e in rejected + abstained})

        if not confirmed:
            # Nothing was confirmed. If observations existed and were dismissed,
            # the honest answer is that nothing happened; if some are still
            # unresolved, the honest answer is that we cannot tell.
            if unconfirmed and not rejected:
                verdict.verdict = Verdict.INSUFFICIENT_EVIDENCE.value
                verdict.reason_code = ReasonCode.EVIDENCE_INSUFFICIENT.value
                verdict.evidence_state = (
                    unconfirmed[0].state if unconfirmed else "SUSPECT"
                )
                verdict.promotion_disposition = PromotionDisposition.WITHHELD.value
                # Name the episodes. An abstention that does not say what it could
                # not resolve is not actionable, and the renderer has nothing to show.
                verdict.abstained_episodes = [e.episode_id for e in unconfirmed]
                verdict.signals_used = sorted({e.signal_key for e in unconfirmed})
                verdict.candidate_onset_step = _earliest_onset(unconfirmed)
                verdict.reason = (
                    "{} episode(s) remain unconfirmed with no corroboration".format(len(unconfirmed))
                )
            else:
                verdict.verdict = Verdict.NO_ANOMALY.value
                verdict.reason_code = ReasonCode.OBSERVATIONS_DISMISSED.value
                verdict.evidence_state = "CLOSED"
                verdict.promotion_disposition = PromotionDisposition.WITHHELD.value
                verdict.reason = (
                    "{} observation(s) never corroborated; closed as not an anomaly".format(
                        sum(e.signal_count for e in rejected)
                    )
                )
            verdicts.append(_stamp(verdict))
            continue

        if not promotion_enabled(anomaly_type):
            verdict.verdict = Verdict.CANDIDATE_ONLY.value
            verdict.episodes_used = [e.episode_id for e in confirmed]
            verdict.confidence = max((e.confidence for e in confirmed), default=0.0)
            verdict.candidate_onset_step = _earliest_onset(confirmed)
            verdict.confirmed_at_step = min(
                (e.confirmed_at_step for e in confirmed if e.confirmed_at_step is not None),
                default=None,
            )
            verdict.reason_code = ReasonCode.CONFIRMED_BUT_PROMOTION_DISABLED.value
            verdict.evidence_state = "CONFIRMED"
            verdict.promotion_disposition = PromotionDisposition.DISABLED.value
            verdict.reason = (
                "the {} episode is CONFIRMED; promotion for this experimental "
                "capability is switched off in this build, so no incident is opened. "
                "This is not a statement that the evidence was insufficient.".format(
                    anomaly_type
                )
            )
            verdicts.append(_stamp(verdict))
            continue

        matched: Optional[PromotionRule] = None
        for rule in rules:
            if anomaly_type not in rule.anomaly_types:
                continue
            strong = [e for e in confirmed if e.confidence_class in rule.accepted_classes]
            weak = [e for e in confirmed if e.confidence_class not in rule.accepted_classes]
            if len(strong) >= rule.min_confirmed_episodes:
                matched = rule
                verdict.episodes_used = [e.episode_id for e in strong]
                break
            if rule.allow_concordant_weak and len(weak) >= rule.min_weak_episodes:
                matched = rule
                verdict.episodes_used = [e.episode_id for e in weak]
                break

        if matched is None:
            verdict.verdict = Verdict.CANDIDATE_ONLY.value
            # Record the confirmed-but-unpromoted episodes. They are not "used" to
            # promote, but a reader must be able to see what the system observed
            # and declined to escalate.
            verdict.episodes_used = [e.episode_id for e in confirmed]
            verdict.confidence = max((e.confidence for e in confirmed), default=0.0)
            verdict.candidate_onset_step = _earliest_onset(confirmed)
            verdict.confirmed_at_step = min(
                (e.confirmed_at_step for e in confirmed if e.confirmed_at_step is not None),
                default=None,
            )
            verdict.reason_code = ReasonCode.CONFIRMED_BUT_NO_RULE_MATCHED.value
            verdict.evidence_state = "CONFIRMED"
            verdict.promotion_disposition = PromotionDisposition.WITHHELD.value
            verdict.reason = (
                "{} confirmed episode(s) but none met a promotion rule "
                "(best class: {})".format(
                    len(confirmed),
                    max((e.confidence_class for e in confirmed), default="WEAK"),
                )
            )
            verdicts.append(_stamp(verdict))
            continue

        used = [e for e in confirmed if e.episode_id in set(verdict.episodes_used)]
        verdict.verdict = Verdict.CONFIRMED_INCIDENT.value
        verdict.reason_code = ReasonCode.PROMOTED.value
        verdict.evidence_state = "CONFIRMED"
        verdict.promotion_disposition = PromotionDisposition.ELIGIBLE.value
        verdict.promotion_rule_id = matched.rule_id
        verdict.promotion_rule_version = matched.version
        verdict.severity = matched.severity
        verdict.confidence = max((e.confidence for e in used), default=0.0)
        # The two moments the brief requires to be carried forward distinctly.
        verdict.candidate_onset_step = _earliest_onset(used)
        verdict.confirmed_at_step = min(
            (e.confirmed_at_step for e in used if e.confirmed_at_step is not None), default=None
        )
        verdict.signals_used = sorted({e.signal_key for e in used})
        verdict.reason = matched.description
        verdicts.append(_stamp(verdict))

    if not any(v.verdict == Verdict.CONFIRMED_INCIDENT.value for v in verdicts):
        # No incident, but keep every per-type verdict so the abstention rate and
        # the candidate rate remain measurable rather than being flattened into
        # "nothing happened".
        return verdicts
    return verdicts


def _earliest_onset(episodes: Sequence[AnomalyEpisode]) -> Optional[int]:
    steps = [e.candidate_onset_step for e in episodes if e.candidate_onset_step is not None]
    return min(steps) if steps else None


def fusion_rule_registry() -> Dict[str, Any]:
    return {
        "fusion_rule_version": FUSION_RULE_VERSION,
        "verdicts": [v.value for v in Verdict],
        "promotable_episode_states": list(PROMOTABLE_STATES),
        "rules": [rule.to_dict() for rule in PROMOTION_RULES],
        "promotion_disabled_types": sorted(PROMOTION_DISABLED_TYPES),
        "note": (
            "Deterministic. No learned component, no language model. Evaluated in "
            "order; the first matching rule promotes."
        ),
    }
