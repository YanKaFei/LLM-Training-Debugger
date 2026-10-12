"""E1: the AnomalyEpisode model.

Phase 1.2's central defect was architectural, not statistical:

    metric -> detector -> incident

A detector fire became an incident immediately, so a non-finite value that persisted
for 111 steps became 111 findings, and a single healthy gradient excursion became a
gradient explosion. Both are failures of the same missing concept.

This module inserts it::

    raw signal -> candidate anomaly -> ANOMALY EPISODE -> fusion -> incident

An episode is a **contiguous run of abnormal observations for one detector and one
signal key**, with a defined onset, a point at which it became confirmable, a
recovery, and an end.

Three commitments that the phase brief makes explicit, and which this file enforces:

**Cooldown does not define identity.** A cooldown may reduce log volume; it may not
decide what counts as one anomaly. Episode grouping is by *contiguity of
observation*, so a persistent `Inf` from step 80 to 190 is one episode regardless of
any cooldown setting. Detectors therefore emit every observation and the engine
groups them.

**Onset and confirmation are different moments.** ``candidate_onset_step`` is when
the signal was first abnormal. ``confirmed_at_step`` is when enough evidence had
accumulated to confirm it. The runtime knows the first only in hindsight; it must
never claim to have known at onset what it could only confirm later.

**SUSPECT may return to NORMAL.** A single excursion that is never corroborated is
not a small incident -- it is *nothing*, and the machine must be able to say so.
This is what makes abstention possible rather than aspirational.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "EpisodeState",
    "AnomalyEpisode",
    "EpisodeOutcome",
    "EPISODE_STATES",
]


class EpisodeState(str, Enum):
    """The episode state machine.

    ``NORMAL -> SUSPECT -> CONFIRMED -> RECOVERING -> CLOSED``, plus the
    transition the brief insists on: **``SUSPECT -> NORMAL``**, meaning an
    observation that turned out to be nothing.
    """

    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    CONFIRMED = "CONFIRMED"
    RECOVERING = "RECOVERING"
    CLOSED = "CLOSED"


#: Terminal classification of a closed episode. This is where abstention lives.
class EpisodeOutcome(str, Enum):
    #: Confirmed abnormal and then recovered. Became an incident.
    CONFIRMED_ANOMALY = "CONFIRMED_ANOMALY"
    #: One or more observations, never corroborated. **Nothing happened.**
    NOT_ANOMALY = "NOT_ANOMALY"
    #: Observations exist and the evidence is genuinely ambiguous -- too few to
    #: confirm, too many/unusual to dismiss. This is the abstention outcome.
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


EPISODE_STATES = tuple(s.value for s in EpisodeState)


@dataclass
class AnomalyEpisode:
    """One contiguous abnormal episode for one detector and signal key.

    Every field the brief requires is present, plus the provenance needed to answer
    "why did this become an incident?" without a language model.
    """

    episode_id: str
    run_id: str
    detector_id: str
    detector_version: str
    rule_version: str
    anomaly_type: str
    #: Identity of the signal this episode tracks: metric name plus rank/layer.
    signal_key: str
    metric_name: Optional[str] = None
    rank: Optional[int] = None
    layer_index: Optional[int] = None

    state: str = EpisodeState.SUSPECT.value
    outcome: Optional[str] = None

    # -- the two moments the brief requires to be distinct ---------------
    candidate_onset_event_id: Optional[str] = None
    candidate_onset_step: Optional[int] = None
    confirmed_event_id: Optional[str] = None
    confirmed_at_step: Optional[int] = None

    last_abnormal_event_id: Optional[str] = None
    last_abnormal_step: Optional[int] = None
    recovered_event_id: Optional[str] = None
    recovered_at_step: Optional[int] = None

    affected_layers: List[int] = field(default_factory=list)
    affected_ranks: List[int] = field(default_factory=list)
    signal_count: int = 0

    peak_value: Optional[float] = None
    peak_step: Optional[int] = None
    baseline: Optional[float] = None
    threshold: Optional[float] = None
    peak_ratio: Optional[float] = None

    severity: str = "info"
    confidence: float = 0.0
    confidence_class: str = "WEAK"

    supporting_event_ids: List[str] = field(default_factory=list)
    observation_steps: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    opened_wall_time_ns: int = field(default_factory=time.time_ns)
    closed_wall_time_ns: Optional[int] = None

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self.state not in (EpisodeState.CLOSED.value,)

    @property
    def is_confirmed(self) -> bool:
        return self.confirmed_at_step is not None

    @property
    def duration_steps(self) -> int:
        if self.candidate_onset_step is None or self.last_abnormal_step is None:
            return 0
        return self.last_abnormal_step - self.candidate_onset_step + 1

    @property
    def confirmation_delay_steps(self) -> Optional[int]:
        """How long after onset the system could first confirm.

        The brief requires this to be reportable separately from a detection delay,
        because they answer different questions: onset localisation is about *when
        the fault began*, confirmation delay is about *how long the system needed*.
        """
        if self.candidate_onset_step is None or self.confirmed_at_step is None:
            return None
        return self.confirmed_at_step - self.candidate_onset_step

    @property
    def cross_layer_multiplicity(self) -> float:
        population = self.metadata.get("population_layers") or 0
        if not population:
            return 0.0
        return len(self.affected_layers) / float(population)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "detector_id": self.detector_id,
            "detector_version": self.detector_version,
            "rule_version": self.rule_version,
            "anomaly_type": self.anomaly_type,
            "signal_key": self.signal_key,
            "metric_name": self.metric_name,
            "rank": self.rank,
            "layer_index": self.layer_index,
            "state": self.state,
            "outcome": self.outcome,
            "candidate_onset_event_id": self.candidate_onset_event_id,
            "candidate_onset_step": self.candidate_onset_step,
            "confirmed_event_id": self.confirmed_event_id,
            "confirmed_at_step": self.confirmed_at_step,
            "last_abnormal_event_id": self.last_abnormal_event_id,
            "last_abnormal_step": self.last_abnormal_step,
            "recovered_event_id": self.recovered_event_id,
            "recovered_at_step": self.recovered_at_step,
            "affected_layers": list(self.affected_layers),
            "affected_ranks": list(self.affected_ranks),
            "signal_count": self.signal_count,
            "peak_value": self.peak_value,
            "peak_step": self.peak_step,
            "baseline": self.baseline,
            "threshold": self.threshold,
            "peak_ratio": self.peak_ratio,
            "severity": self.severity,
            "confidence": self.confidence,
            "confidence_class": self.confidence_class,
            "supporting_event_ids": list(self.supporting_event_ids),
            "observation_steps": list(self.observation_steps),
            "duration_steps": self.duration_steps,
            "confirmation_delay_steps": self.confirmation_delay_steps,
            "cross_layer_multiplicity": self.cross_layer_multiplicity,
            "opened_wall_time_ns": self.opened_wall_time_ns,
            "closed_wall_time_ns": self.closed_wall_time_ns,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AnomalyEpisode":
        data = dict(payload)
        for derived in (
            "duration_steps",
            "confirmation_delay_steps",
            "cross_layer_multiplicity",
        ):
            data.pop(derived, None)
        return cls(**data)

    def signature(self) -> str:
        """Stable identity for duplicate detection across a restart.

        Deliberately excludes timing and observation *count*, so a re-run that
        observes the same episode over a slightly different number of steps still
        matches. Includes the onset step, which is what actually identifies it.
        """
        payload = "|".join(
            [
                self.run_id,
                self.detector_id,
                self.signal_key,
                str(self.candidate_onset_step),
                str(self.layer_index),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def episode_id_for(run_id: str, detector_id: str, signal_key: str, onset_step: Optional[int]) -> str:
    payload = "|".join([run_id, detector_id, signal_key, str(onset_step)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return "ep_{}".format(digest)
