"""E1/E2: the episode engine and its state machine.

Consumes detector *observations* and produces :class:`AnomalyEpisode` objects. This
is where "one persistent `Inf` is one anomaly, not 111" is enforced, and where the
machine earns the right to say *nothing happened*.

Four design decisions, each fixing a specific Phase 1.2 failure:

**1. Identity is contiguity, not cooldown.** Observations chain into an episode when
they are within ``gap_tolerance`` steps of each other. A detector cooldown would
have defined identity by *how often we bothered to log*, which is why 810 findings
came out of one poisoned optimiser. Detectors now emit every observation; grouping
happens here.

**2. One episode per (detector, rank), covering all layers.** Grouping per layer
would turn a 7-layer explosion into 7 episodes. Layers are recorded as
``affected_layers`` on the single episode, which is also what makes cross-layer
multiplicity available to the fusion layer as evidence rather than as description.

**3. Persistence is required to confirm, and the requirement is per anomaly type.**
A non-finite value is ``CERTAIN`` and confirms on first observation -- delaying it
would be dishonest, since nothing can make a measurement of `NaN` false. A
statistical excursion must be corroborated: by persistence, by cross-layer
multiplicity, or by sheer magnitude.

**4. ``SUSPECT`` may become ``NORMAL``.** An excursion that is never corroborated
within ``confirm_window`` closes as ``NOT_ANOMALY``. It is not a small incident. It
is nothing. Without this transition the system cannot decline to alarm, and
abstention is impossible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..detectors.base import ConfidenceClass
from ..schema.enums import SCHEMA_VERSION
from .model import AnomalyEpisode, EpisodeOutcome, EpisodeState, episode_id_for

__all__ = ["EpisodeEngine", "EpisodeRule", "EPISODE_RULES", "RULE_VERSION"]

#: Version of the episode state machine and its confirmation rules. Persisted on
#: every episode so an old episode can be explained after the rules change.
RULE_VERSION = "phase1.3-1.0"


@dataclass(frozen=True)
class EpisodeRule:
    """Confirmation and recovery policy for one anomaly type."""

    anomaly_type: str
    detector_id: str

    #: Observations required within ``persistence_window`` to confirm.
    persistence_required: int = 2
    persistence_window: int = 5
    #: Cross-layer fraction at or above which one observation suffices.
    multiplicity_confirm: Optional[float] = None
    #: Peak ratio at or above which one observation suffices ("very large single
    #: event override"). ``None`` disables the override.
    magnitude_override: Optional[float] = None
    #: An aggregate-series observation confirms on its own.
    aggregate_confirms: bool = False
    #: ``CERTAIN`` observations confirm immediately regardless of the above.
    certain_confirms: bool = True
    #: Steps without an observation before a SUSPECT episode closes as NOT_ANOMALY.
    confirm_window: int = 10
    #: Consecutive normal steps before a CONFIRMED episode begins recovering.
    recovery_steps: int = 15
    #: Maximum step gap that still chains observations into one episode.
    gap_tolerance: int = 8
    severity: str = "warn"


#: Per-anomaly-type rules. ``multiplicity_confirm`` values come from the measured
#: split between healthy single-layer excursions (1 of 7 = 0.14) and injected
#: explosions (6-7 of 7 = 0.86-1.0).
EPISODE_RULES: Dict[str, EpisodeRule] = {
    "NAN_DETECTED": EpisodeRule(
        anomaly_type="NAN_DETECTED",
        detector_id="numerics.finiteness",
        persistence_required=1,
        confirm_window=10,
        recovery_steps=20,
        severity="error",
    ),
    "INF_DETECTED": EpisodeRule(
        anomaly_type="INF_DETECTED",
        detector_id="numerics.finiteness",
        persistence_required=1,
        confirm_window=10,
        recovery_steps=20,
        severity="error",
    ),
    "GRADIENT_EXPLOSION": EpisodeRule(
        anomaly_type="GRADIENT_EXPLOSION",
        detector_id="grad.robust_zscore",
        persistence_required=2,
        persistence_window=5,
        multiplicity_confirm=0.4,
        magnitude_override=8.0,
        aggregate_confirms=True,
        confirm_window=8,
        recovery_steps=12,
        gap_tolerance=8,
        severity="error",
    ),
    "LOSS_SPIKE": EpisodeRule(
        anomaly_type="LOSS_SPIKE",
        detector_id="loss.robust_zscore",
        persistence_required=2,
        persistence_window=6,
        magnitude_override=6.0,
        confirm_window=8,
        recovery_steps=12,
        severity="warn",
    ),
    "LOSS_DIVERGENCE": EpisodeRule(
        anomaly_type="LOSS_DIVERGENCE",
        detector_id="loss.trend_slope",
        persistence_required=4,
        persistence_window=12,
        confirm_window=20,
        recovery_steps=25,
        gap_tolerance=15,
        severity="error",
    ),
    "THROUGHPUT_REGRESSION": EpisodeRule(
        anomaly_type="THROUGHPUT_REGRESSION",
        detector_id="throughput.regression",
        persistence_required=4,
        persistence_window=12,
        confirm_window=20,
        recovery_steps=25,
        gap_tolerance=15,
        severity="warn",
    ),
}

#: Map raw detector ids onto anomaly types for rule lookup.
_DETECTOR_TO_TYPE = {
    "numerics.finiteness": "NAN_DETECTED",
    "grad.robust_zscore": "GRADIENT_EXPLOSION",
    "loss.robust_zscore": "LOSS_SPIKE",
    "loss.trend_slope": "LOSS_DIVERGENCE",
    "throughput.regression": "THROUGHPUT_REGRESSION",
}


@dataclass
class _OpenEpisode:
    episode: AnomalyEpisode
    observations: List[int] = field(default_factory=list)
    #: ``(step, layer_index)`` pairs already counted, for idempotency.
    observed_pairs: set = field(default_factory=set)
    #: Earliest step the gradient band called CONFIRMED or UNCERTAIN (W1).
    abnormal_onset_step: Optional[int] = None
    abnormal_onset_event_id: Optional[str] = None
    steps_since_observation: int = 0
    confirmed: bool = False
    recovering_since: Optional[int] = None


class EpisodeEngine:
    """Turns detector observations into episodes.

    ``enabled_detectors`` is the mechanism by which a broken detector is kept out
    of the default system without being deleted: its registry entry says
    ``DISABLED``, and the engine declines to open episodes for it. An episode from a
    disabled detector can still be *recorded* when explicitly requested, so its
    failure remains measurable rather than invisible.
    """

    def __init__(
        self,
        rules: Optional[Dict[str, EpisodeRule]] = None,
        *,
        enabled_detectors: Optional[Iterable[str]] = None,
        rule_version: str = RULE_VERSION,
    ) -> None:
        self.rules = dict(rules or EPISODE_RULES)
        self.rule_version = rule_version
        self.enabled_detectors = set(enabled_detectors) if enabled_detectors is not None else None
        self._open: Dict[str, _OpenEpisode] = {}
        self.episodes: List[AnomalyEpisode] = []

    # ------------------------------------------------------------------

    def resume(self, episodes: Sequence[Any]) -> int:
        """P0: re-open persisted, unclosed episodes so processing can continue.

        Without this a restarted process would start from an empty engine and create a
        **new** episode for a fault it had already been tracking -- the same run judged
        twice, with two onsets and two signal counts. Resuming restores the identity and
        the observation history, so ``consume()`` extends the same episode.

        Episode identity, onset and revision are taken from the persisted object; only
        the engine's in-memory view of "what is currently open" is rebuilt.
        """
        restored = 0
        for raw in episodes:
            data = raw.episode.to_dict() if hasattr(raw, "episode") else (
                raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
            )
            if data.get("state") == EpisodeState.CLOSED.value:
                continue
            signal_key = data.get("signal_key") or "{}:rank?".format(data.get("detector_id"))
            episode = AnomalyEpisode.from_dict(data)
            observations = list(episode.observation_steps or [])
            if not observations and episode.candidate_onset_step is not None:
                observations = [episode.candidate_onset_step]
            entry = _OpenEpisode(
                episode=episode,
                observations=observations,
                confirmed=episode.confirmed_at_step is not None,
                observed_pairs=set(),
            )
            self._open[signal_key] = entry
            restored += 1
        return restored

    def consume(
        self, anomalies: Sequence[Any], run_id: str, *, include_disabled: bool = False
    ) -> List[AnomalyEpisode]:
        """Feed observations in step order. Returns episodes closed in this pass."""
        closed: List[AnomalyEpisode] = []
        ordered = sorted(
            anomalies,
            key=lambda a: (a.first_abnormal_step or 0, a.layer_index if a.layer_index is not None else -1),
        )
        by_step: Dict[int, List[Any]] = {}
        for anomaly in ordered:
            if anomaly.first_abnormal_step is None:
                continue
            by_step.setdefault(int(anomaly.first_abnormal_step), []).append(anomaly)

        steps = sorted(by_step)
        if not steps:
            return closed
        previous_step = steps[0]

        for step in steps:
            # Advance every open episode's clock to this step so closures happen
            # in step order rather than at whichever point the data ends.
            closed.extend(self._advance_clock(step, run_id))
            previous_step = step
            groups: Dict[Tuple[str, Optional[int], Optional[str]], List[Any]] = {}
            for anomaly in by_step[step]:
                detector = str(anomaly.detector)
                if not include_disabled and self.enabled_detectors is not None:
                    if detector not in self.enabled_detectors:
                        continue
                # A detector that multiplexes metrics must also be grouped by metric.
                # Keying the episode on the metric while grouping only by
                # (detector, rank) left both metrics in ONE group: the signal key was
                # taken from `group[0]` and the `metric_name` was then overwritten by
                # the last member. The result was an episode keyed on
                # `grad.global_norm` that reported `param.global_norm` -- so the
                # midstream witness replaced the upstream one instead of joining it,
                # and the genuine propagation in A8X became invisible.
                group_metric = (
                    getattr(anomaly, "metric_name", None)
                    if detector in self._multi_metric_detector_ids()
                    else None
                )
                groups.setdefault((detector, anomaly.rank, group_metric), []).append(anomaly)
            for (detector, rank, metric), group in groups.items():
                self._observe(detector, rank, group, step, run_id)
        _ = previous_step
        return closed

    def finish(self, run_id: str, final_step: Optional[int] = None) -> List[AnomalyEpisode]:
        """Close every open episode. A run ending is not an anomaly continuing."""
        step = final_step if final_step is not None else max(
            (e.last_abnormal_step or 0) for e in self._open.values() if self._open
        ) if self._open else 0
        closed = self._advance_clock((step or 0) + 10_000, run_id)
        return closed

    # ------------------------------------------------------------------

    def _advance_clock(self, step: int, run_id: str) -> List[AnomalyEpisode]:
        closed: List[AnomalyEpisode] = []
        for key in list(self._open):
            entry = self._open[key]
            episode = entry.episode
            last = episode.last_abnormal_step or 0
            gap = step - last
            if entry.confirmed:
                if gap >= self._rule(episode).recovery_steps:
                    if episode.state == EpisodeState.CONFIRMED.value:
                        episode.state = EpisodeState.RECOVERING.value
                        episode.recovered_at_step = last + 1
                    episode.state = EpisodeState.CLOSED.value
                    episode.outcome = EpisodeOutcome.CONFIRMED_ANOMALY.value
                    episode.closed_wall_time_ns = time.time_ns()
                    closed.append(episode)
                    self.episodes.append(episode)
                    del self._open[key]
            else:
                if gap > self._rule(episode).confirm_window:
                    # SUSPECT -> NORMAL. The observation was never corroborated.
                    #
                    # Two different things are being closed here and they must not
                    # be conflated. An observation inside the healthy range is
                    # NOT_ANOMALY: a positive statement that nothing happened. An
                    # observation in the UNCERTAIN band is *not* a statement that
                    # nothing happened -- it is a statement that the evidence is
                    # ambiguous, which the brief maps to INSUFFICIENT_EVIDENCE.
                    # Collapsing the second into the first would silently discard
                    # the abstention the gradient band exists to express.
                    uncertain = any(
                        (observation.get("gradient_band") == "UNCERTAIN")
                        for observation in (episode.metadata.get("observations") or [])
                    )
                    episode.state = EpisodeState.CLOSED.value
                    episode.outcome = (
                        EpisodeOutcome.INSUFFICIENT_EVIDENCE.value if uncertain
                        else EpisodeOutcome.NOT_ANOMALY.value
                    )
                    episode.closed_wall_time_ns = time.time_ns()
                    closed.append(episode)
                    self.episodes.append(episode)
                    del self._open[key]
        return closed

    def _multi_metric_detector_ids(self) -> frozenset:
        """Detectors that observe several metrics through one detector id.

        They must be scoped by metric at BOTH the grouping and the key, or the metric
        identity is lost. See `D-020`.
        """
        cached = getattr(self, "_multi_metric_detectors", None)
        if cached is None:
            cached = frozenset({"numerics.finiteness"})
            self._multi_metric_detectors = cached
        return cached

    def _observe(
        self, detector: str, rank: Optional[int], group: Sequence[Any], step: int, run_id: str
    ) -> None:
        anomaly_type = _DETECTOR_TO_TYPE.get(detector, detector)
        rule = self.rules.get(anomaly_type)
        if rule is None:
            rule = EpisodeRule(anomaly_type=anomaly_type, detector_id=detector)
        # W2: the signal key must identify the SIGNAL, not just the detector.
        #
        # `numerics.finiteness` observes every metric in the run through one detector
        # id. With a detector-only key, a non-finite *gradient* at step 80, a non-finite
        # *loss* at step 81 and the non-finite *counter* all landed in ONE episode --
        # and the episode recorded whichever metric name reached it first. The
        # mechanism evidence ("the gradient went non-finite first") was therefore
        # destroyed at the episode layer, which is why M1 produced no relation even
        # though the raw observation existed.
        #
        # Scoped deliberately to detectors that genuinely multiplex metrics. Most
        # detectors are per-metric already, and splitting `grad.robust_zscore` would
        # break aggregate-vs-layer confirmation, which *depends* on those sharing an
        # episode.
        if detector in self._multi_metric_detector_ids():
            metric = str(getattr(group[0], "metric_name", "") or "?")
            signal_key = "{}:rank{}:{}".format(
                detector, rank if rank is not None else "?", metric
            )
        else:
            signal_key = "{}:rank{}".format(detector, rank if rank is not None else "?")

        entry = self._open.get(signal_key)
        if entry is not None and (step - (entry.episode.last_abnormal_step or step)) > rule.gap_tolerance:
            # Too far apart to be the same episode.
            self._open.pop(signal_key, None)
            entry = None
        if entry is None:
            entry = _OpenEpisode(
                episode=AnomalyEpisode(
                    episode_id=episode_id_for(run_id, detector, signal_key, step),
                    run_id=run_id,
                    detector_id=detector,
                    detector_version=str(getattr(group[0], "detector_version", "?")),
                    rule_version=self.rule_version,
                    anomaly_type=anomaly_type,
                    signal_key=signal_key,
                    rank=rank,
                    state=EpisodeState.SUSPECT.value,
                    candidate_onset_step=step,
                    candidate_onset_event_id=_event_id(group[0]),
                    severity=rule.severity,
                )
            )
            self._open[signal_key] = entry

        episode = entry.episode
        # Idempotency is at ``(step, layer)`` granularity, not ``step``.
        #
        # Two observations of the *same* step and layer are the same observation and
        # must not manufacture the persistence a confirmation rule requires -- found
        # by `test_same_observation_cannot_count_twice`. But the same step seen for a
        # *different* layer is a genuinely distinct observation, which is exactly what
        # a restarted process supplies when it resumes mid-step; rejecting those broke
        # `test_cross_process_restart_readback`.
        new_pairs = {
            (step, anomaly.layer_index) for anomaly in group
            if (step, anomaly.layer_index) not in entry.observed_pairs
        }
        if not new_pairs:
            return
        entry.observed_pairs.update(new_pairs)

        # W1: anchor the onset to the earliest observation the band actually called
        # abnormal.
        #
        # The onset used to be the first observation of the contiguity run, whatever
        # its band. That let a *benign* excursion inherit the fault it happened to sit
        # next to: the quickstart injects at step 80, and because a WEAK/NORMAL-band
        # twitch at step 75 fell inside `gap_tolerance = 8`, the incident reported
        # `candidate_onset_step = 75` -- five steps before anything went wrong. The
        # band had already dismissed those steps; a step the band calls NORMAL cannot
        # be the onset of an anomaly.
        #
        # UNCERTAIN counts. Excluding it would throw away genuine early warning from a
        # gradually developing fault, which is the signal this system exists to find.
        # W0: carry the observation KIND onto the episode, so a downstream consumer can
        # tell a real non-finite observation from a float32 reduction artifact.
        for anomaly in group:
            kind = (getattr(anomaly, "metadata", None) or {}).get("kind")
            if kind and episode.metadata.get("anomaly_kind") is None:
                episode.metadata["anomaly_kind"] = kind

        observed_bands = {
            (a.metadata or {}).get("gradient_band")
            for a in group
        } - {None}
        if not observed_bands or observed_bands & {"CONFIRMED", "UNCERTAIN"}:
            if entry.abnormal_onset_step is None:
                entry.abnormal_onset_step = step
                entry.abnormal_onset_event_id = _event_id(group[0])

        if step not in episode.observation_steps:
            # Persistence counts distinct *steps*, so a step is recorded once however
            # many layers report at it.
            entry.observations.append(step)
        episode.last_abnormal_step = step
        episode.last_abnormal_event_id = _event_id(group[-1])
        episode.observation_steps.append(step)
        episode.signal_count += len(group)

        for anomaly in group:
            evidence_id = _event_id(anomaly)
            if evidence_id:
                episode.supporting_event_ids.append(evidence_id)
            if anomaly.layer_index is not None and anomaly.layer_index not in episode.affected_layers:
                episode.affected_layers.append(int(anomaly.layer_index))
            if anomaly.rank is not None and anomaly.rank not in episode.affected_ranks:
                episode.affected_ranks.append(int(anomaly.rank))
            if episode.metric_name is None or anomaly.layer_index is None:
                episode.metric_name = anomaly.metric_name or episode.metric_name
            value = getattr(anomaly, "observed_value", None)
            if value is not None and (episode.peak_value is None or abs(value) > abs(episode.peak_value)):
                episode.peak_value = value
                episode.peak_step = step
                episode.baseline = getattr(anomaly, "baseline", None)
                episode.threshold = getattr(anomaly, "threshold", None)
                episode.peak_ratio = (anomaly.metadata or {}).get("peak_ratio")
            population = (anomaly.metadata or {}).get("population_layers")
            if population:
                episode.metadata["population_layers"] = population
            if anomaly.metadata.get("fusion_reason"):
                episode.metadata.setdefault("fusion_reasons", []).append(
                    anomaly.metadata["fusion_reason"]
                )
            band = anomaly.metadata.get("gradient_band")
            if band:
                episode.metadata.setdefault("observations", []).append(
                    {"step": step, "gradient_band": band}
                )
                episode.metadata["gradient_band"] = band

        _ = sorted(episode.affected_layers)
        self._maybe_confirm(entry, rule, step, group)

    def _maybe_confirm(
        self, entry: _OpenEpisode, rule: EpisodeRule, step: int, group: Sequence[Any]
    ) -> None:
        if entry.confirmed:
            entry.episode.confidence = max(entry.episode.confidence, 0.7)
            return
        episode = entry.episode

        # W1: report the onset of the *abnormality*, not of the contiguity run.
        # The raw first observation is preserved so the grouping remains auditable.
        raw_first = episode.candidate_onset_step
        if entry.abnormal_onset_step is not None and raw_first != entry.abnormal_onset_step:
            episode.metadata["contiguity_onset_step"] = raw_first
            episode.metadata["onset_reanchored"] = (
                "the first observation of this episode was classified NORMAL by the "
                "gradient band, so it is not the onset of an anomaly"
            )
            episode.candidate_onset_step = entry.abnormal_onset_step
            if entry.abnormal_onset_event_id is not None:
                episode.candidate_onset_event_id = entry.abnormal_onset_event_id

        # The gradient band is authoritative for gradient observations.
        #
        # It was NOT, and that is why two development re-runs produced byte-identical
        # numbers after the band was implemented and then corrected. The rule has
        # ``aggregate_confirms=True``, so whenever the *global* norm detector fired
        # the episode confirmed -- regardless of the band, because the band only
        # demoted ``confidence_class``. A healthy run whose global norm twitched was
        # therefore confirmed exactly as before, and 24 of 84 healthy runs opened an
        # incident while the band reported them as NORMAL or UNCERTAIN.
        #
        # An observation's band now decides: only CONFIRMED-band observations may
        # confirm a gradient episode, by any route.
        bands = {
            (a.metadata or {}).get("gradient_band")
            for a in group
            if (a.metadata or {}).get("gradient_band")
        }
        if bands and "CONFIRMED" not in bands:
            # NORMAL or UNCERTAIN only: record why, and do not confirm.
            episode.metadata["band_blocked_confirmation"] = sorted(bands)
            return

        certain = any(
            getattr(a, "confidence_class", None) == ConfidenceClass.CERTAIN for a in group
        )
        aggregate = any(a.layer_index is None for a in group)
        population = episode.metadata.get("population_layers") or 0
        multiplicity = (len(episode.affected_layers) / population) if population else 0.0
        peak_ratio = episode.peak_ratio if episode.peak_ratio is not None else 0.0
        reasons: List[str] = []
        confirmed = False

        if rule.certain_confirms and certain:
            confirmed = True
            reasons.append("measured non-finite value (CERTAIN)")
        if rule.aggregate_confirms and aggregate:
            confirmed = True
            reasons.append("aggregate series departed")
        if rule.multiplicity_confirm is not None and multiplicity >= rule.multiplicity_confirm:
            confirmed = True
            reasons.append(
                "cross-layer multiplicity {:.0f}% >= {:.0f}%".format(
                    multiplicity * 100, rule.multiplicity_confirm * 100
                )
            )
        if rule.magnitude_override is not None and peak_ratio >= rule.magnitude_override:
            confirmed = True
            reasons.append(
                "peak ratio {:.1f} overrides persistence ({:.1f} required)".format(
                    peak_ratio, rule.magnitude_override
                )
            )
        recent = [s for s in entry.observations if step - s < rule.persistence_window]
        if len(recent) >= rule.persistence_required:
            confirmed = True
            reasons.append(
                "{} observations within {} steps (persistence)".format(
                    len(recent), rule.persistence_window
                )
            )

        if not confirmed:
            return
        entry.confirmed = True
        episode.state = EpisodeState.CONFIRMED.value
        episode.confirmed_at_step = step
        episode.confirmed_event_id = _event_id(group[0])
        episode.metadata["confirmation_reasons"] = reasons
        episode.confidence = (
            1.0 if certain else
            min(0.95, 0.6 + 0.35 * min(1.0, multiplicity)) if multiplicity else
            0.7
        )
        episode.confidence_class = (
            ConfidenceClass.CERTAIN.value if certain else
            ConfidenceClass.STRONG.value if (aggregate or multiplicity >= (rule.multiplicity_confirm or 2.0)) else
            ConfidenceClass.WEAK.value
        )

    def _rule(self, episode: AnomalyEpisode) -> EpisodeRule:
        return self.rules.get(episode.anomaly_type) or EpisodeRule(
            anomaly_type=episode.anomaly_type, detector_id=episode.detector_id
        )

    # ------------------------------------------------------------------

    def all_episodes(self) -> List[AnomalyEpisode]:
        return list(self.episodes) + [entry.episode for entry in self._open.values()]

    def summary(self) -> Dict[str, Any]:
        everything = self.all_episodes()
        by_outcome: Dict[str, int] = {}
        for episode in everything:
            key = episode.outcome or episode.state
            by_outcome[key] = by_outcome.get(key, 0) + 1
        return {
            "rule_version": self.rule_version,
            "episodes": len(everything),
            "by_outcome": by_outcome,
            "observations": sum(e.signal_count for e in everything),
        }


def _event_id(anomaly: Any) -> Optional[str]:
    value = getattr(anomaly, "event_id", None)
    if value:
        return str(value)
    return None
