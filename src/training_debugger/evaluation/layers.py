"""E0: the four-layer scoring model.

Phase 1.2's headline number was **event precision 0.121** — a count of raw detector
fires. That is the wrong product metric, and it was actively misleading: one poisoned
optimiser produced 810 fires, so a single fault dominated the score, and fixing the
detector that generated them moved the number without changing whether the system
did anything useful.

Scoring is now separated into four layers, each answering a different question:

===========  ==================================================================
layer        question
===========  ==================================================================
SIGNAL       did the detector see the anomaly at all? (recall/FP per observation)
EPISODE      did it correctly delimit one contiguous anomaly? (onset, confirmation)
INCIDENT     did it correctly decide something happened?
RUN          was the run's overall verdict right? (healthy-run false incident rate)
===========  ==================================================================

A layer is only as meaningful as its ground truth, so each is defined against a
different aspect of the injection:

* a **signal** is graded against the acceptable detection window;
* an **episode** is graded on whether it covers the window *and* where its onset
  landed;
* an **incident** is graded as one decision per (run, anomaly type);
* a **run** is graded on its final verdict, which is what a user experiences.

Detection delay is taken from the **first true-positive confirmation** only. A fire
before the injection is a false positive with no delay, never a negative delay.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..episodes.model import AnomalyEpisode, EpisodeOutcome
from ..fusion.rules import Verdict

__all__ = [
    "LayerScores",
    "evaluate_layers",
    "empty_scores",
]


def _precision(tp: int, fp: int) -> Optional[float]:
    return tp / (tp + fp) if (tp + fp) else None


def _recall(tp: int, fn: int) -> Optional[float]:
    return tp / (tp + fn) if (tp + fn) else None


def _f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    return (2 * p * r / (p + r)) if (p and r) else None


@dataclass
class LayerScores:
    """Counts and derived rates for one evaluation layer."""

    layer: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    #: Fires/episodes/incidents that occurred before the fault could exist.
    pre_injection: int = 0
    #: Extra detections of an already-detected fault.
    duplicates: int = 0
    #: Runs/episodes where the system explicitly declined.
    abstained: int = 0
    delay_samples: List[int] = field(default_factory=list)
    onset_errors: List[int] = field(default_factory=list)
    confirmation_delays: List[int] = field(default_factory=list)
    considered: int = 0

    @property
    def precision(self) -> Optional[float]:
        return _precision(self.true_positives, self.false_positives + self.pre_injection)

    @property
    def recall(self) -> Optional[float]:
        return _recall(self.true_positives, self.false_negatives)

    @property
    def f1(self) -> Optional[float]:
        return _f1(self.precision, self.recall)

    @property
    def median_delay(self) -> Optional[float]:
        return statistics.median(self.delay_samples) if self.delay_samples else None

    @property
    def median_onset_error(self) -> Optional[float]:
        return statistics.median(self.onset_errors) if self.onset_errors else None

    @property
    def median_confirmation_delay(self) -> Optional[float]:
        return statistics.median(self.confirmation_delays) if self.confirmation_delays else None

    @property
    def abstention_rate(self) -> Optional[float]:
        return self.abstained / self.considered if self.considered else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "pre_injection": self.pre_injection,
            "duplicates": self.duplicates,
            "abstained": self.abstained,
            "considered": self.considered,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "median_delay_steps": self.median_delay,
            "median_onset_error_steps": self.median_onset_error,
            "median_confirmation_delay_steps": self.median_confirmation_delay,
            "abstention_rate": self.abstention_rate,
        }


def empty_scores() -> Dict[str, LayerScores]:
    return {name: LayerScores(layer=name) for name in ("signal", "episode", "incident", "run")}


def evaluate_layers(
    scenarios: Sequence[Dict[str, Any]],
    *,
    detector_filter: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Compute all four layers from per-run pipeline output.

    Each ``scenario`` is a mapping with:

    ``run_id``, ``fault_type``, ``injected`` (bool), ``injection_step``,
    ``acceptable_window`` (``(start, end)`` or ``None``), ``fires`` (a list of
    ``(detector_id, step)``), ``episodes`` (a list of dicts), ``verdicts`` (a list
    of :class:`IncidentVerdict`-like dicts), ``run_error``.
    """
    scores = empty_scores()
    detector_filter = set(detector_filter) if detector_filter else None

    healthy_false_incidents = 0
    healthy_runs = 0
    fault_runs = 0
    strict_steps = 0

    for scenario in scenarios:
        if scenario.get("run_error"):
            continue
        injected = bool(scenario.get("injected"))
        window = scenario.get("acceptable_window")
        injection_step = scenario.get("injection_step")
        steps = scenario.get("steps") or 200
        strict_steps += steps

        # ---------------- SIGNAL ----------------------------------------
        fires = [
            (detector, step)
            for detector, step in scenario.get("fires", [])
            if detector_filter is None or detector in detector_filter
        ]
        fires.sort(key=lambda item: item[1])
        if not injected:
            scores["signal"].false_positives += len(fires)
            scores["signal"].true_negatives += 1
        else:
            in_window = [s for _, s in fires if window and window[0] <= s <= window[1]]
            before = [s for _, s in fires if window and s < window[0]]
            after = [s for _, s in fires if window and s > window[1]]
            scores["signal"].pre_injection += len(before)
            scores["signal"].false_positives += len(after)
            scores["signal"].duplicates += max(0, len(in_window) - 1)
            if in_window:
                scores["signal"].true_positives += 1
                scores["signal"].delay_samples.append(min(in_window) - window[0])
            else:
                scores["signal"].false_negatives += 1
            _ = injection_step

        # ---------------- EPISODE ---------------------------------------
        episodes = scenario.get("episodes", [])
        confirmed = [
            e for e in episodes
            if e.get("state") == "CONFIRMED" or e.get("outcome") == EpisodeOutcome.CONFIRMED_ANOMALY.value
        ]
        if not injected:
            scores["episode"].false_positives += len(confirmed)
            scores["episode"].abstained += sum(
                1 for e in episodes if e.get("outcome") == EpisodeOutcome.INSUFFICIENT_EVIDENCE.value
            )
            scores["episode"].considered += 1
            scores["episode"].true_negatives += 1
        else:
            covering = [
                e for e in confirmed
                if window and e.get("candidate_onset_step") is not None
                and window[0] <= e["candidate_onset_step"] <= window[1]
            ]
            early = [
                e for e in confirmed
                if window and e.get("candidate_onset_step") is not None
                and e["candidate_onset_step"] < window[0]
            ]
            scores["episode"].pre_injection += len(early)
            scores["episode"].false_positives += max(0, len(confirmed) - len(covering) - len(early))
            scores["episode"].duplicates += max(0, len(covering) - 1)
            scores["episode"].abstained += sum(
                1 for e in episodes if e.get("outcome") == EpisodeOutcome.INSUFFICIENT_EVIDENCE.value
            )
            scores["episode"].considered += 1
            if covering:
                scores["episode"].true_positives += 1
                best = min(covering, key=lambda e: abs(e["candidate_onset_step"] - window[0]))
                scores["episode"].onset_errors.append(
                    best["candidate_onset_step"] - window[0]
                )
                delay = best.get("confirmed_at_step")
                if delay is not None:
                    scores["episode"].confirmation_delays.append(delay - best["candidate_onset_step"])
                scores["episode"].delay_samples.append(best["candidate_onset_step"] - window[0])
            else:
                scores["episode"].false_negatives += 1

        # ---------------- INCIDENT --------------------------------------
        verdicts = scenario.get("verdicts", [])
        for verdict in verdicts:
            kind = verdict.get("verdict")
            if kind == Verdict.CONFIRMED_INCIDENT.value:
                if not injected:
                    scores["incident"].false_positives += 1
                else:
                    onset = verdict.get("candidate_onset_step")
                    if window and onset is not None and window[0] <= onset <= window[1]:
                        scores["incident"].true_positives += 1
                    elif window and onset is not None and onset < window[0]:
                        scores["incident"].pre_injection += 1
                    else:
                        scores["incident"].false_positives += 1
            elif kind == Verdict.INSUFFICIENT_EVIDENCE.value:
                scores["incident"].abstained += 1
            elif kind == Verdict.CANDIDATE_ONLY.value:
                scores["incident"].abstained += 1
            scores["incident"].considered += 1

        if injected:
            confirmed_types = {
                v.get("anomaly_type") for v in verdicts
                if v.get("verdict") == Verdict.CONFIRMED_INCIDENT.value
            }
            if not confirmed_types:
                scores["incident"].false_negatives += 1
            fault_runs += 1
        else:
            healthy_runs += 1
            if any(v.get("verdict") == Verdict.CONFIRMED_INCIDENT.value for v in verdicts):
                healthy_false_incidents += 1

        # ---------------- RUN -------------------------------------------
        if injected:
            if any(v.get("verdict") == Verdict.CONFIRMED_INCIDENT.value for v in verdicts):
                scores["run"].true_positives += 1
            elif any(v.get("verdict") == Verdict.INSUFFICIENT_EVIDENCE.value for v in verdicts):
                scores["run"].abstained += 1
                scores["run"].false_negatives += 1
            else:
                scores["run"].false_negatives += 1
        else:
            if any(v.get("verdict") == Verdict.CONFIRMED_INCIDENT.value for v in verdicts):
                scores["run"].false_positives += 1
            elif any(v.get("verdict") == Verdict.INSUFFICIENT_EVIDENCE.value for v in verdicts):
                scores["run"].abstained += 1
                scores["run"].true_negatives += 1
            else:
                scores["run"].true_negatives += 1
        scores["run"].considered += 1

    aggregate = {name: layer.to_dict() for name, layer in scores.items()}
    aggregate["run"].update(
        {
            "healthy_runs": healthy_runs,
            "fault_runs": fault_runs,
            "healthy_run_false_incident_rate": (
                healthy_false_incidents / healthy_runs if healthy_runs else None
            ),
            "false_incidents_per_1k_steps": (
                (scores["run"].false_positives / strict_steps * 1000.0) if strict_steps else None
            ),
            "abstention_rate_overall": (
                scores["run"].abstained / scores["run"].considered
                if scores["run"].considered else None
            ),
        }
    )
    aggregate["_meta"] = {
        "scenarios": len(scenarios),
        "steps_observed": strict_steps,
        "layers": ["signal", "episode", "incident", "run"],
    }
    return aggregate


# ---------------------------------------------------------------------------
# J5: uncertainty accounting
# ---------------------------------------------------------------------------


def uncertainty_rates(scenarios: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Band rates by population and by fault magnitude.

    The brief requires this because precision can otherwise be improved by pushing
    every difficult case into the abstention band. Reporting
    ``HEALTHY_ABSTENTION_RATE`` and ``FAULT_ABSTENTION_RATE`` alongside precision
    and recall makes that manoeuvre visible instead of rewarding it.

    Bands are read from episode ``metadata.gradient_band``, which
    ``_apply_gradient_band`` always writes. An observation with no band -- from a
    non-gradient detector -- is counted as ``unbanded`` rather than being assigned
    to a band it never had.
    """
    populations: Dict[str, Dict[str, int]] = {
        "healthy": {"NORMAL": 0, "UNCERTAIN": 0, "CONFIRMED": 0, "unbanded": 0, "runs": 0},
        "fault": {"NORMAL": 0, "UNCERTAIN": 0, "CONFIRMED": 0, "unbanded": 0, "runs": 0},
    }
    by_magnitude: Dict[str, Dict[str, int]] = {}

    for scenario in scenarios:
        if scenario.get("run_error"):
            continue
        population = "fault" if scenario.get("injected") else "healthy"
        bucket = populations[population]
        bucket["runs"] += 1
        seen_bands = set()
        for episode in scenario.get("episodes", []):
            band = (episode.get("metadata") or {}).get("gradient_band")
            key = band if band in ("NORMAL", "UNCERTAIN", "CONFIRMED") else "unbanded"
            bucket[key] += 1
            seen_bands.add(key)
        if population == "fault":
            magnitude = str(scenario.get("magnitude"))
            entry = by_magnitude.setdefault(
                magnitude, {"NORMAL": 0, "UNCERTAIN": 0, "CONFIRMED": 0, "unbanded": 0, "runs": 0}
            )
            entry["runs"] += 1
            for key in seen_bands:
                entry[key] += 1

    def _rates(entry: Dict[str, int]) -> Dict[str, Any]:
        total = entry["NORMAL"] + entry["UNCERTAIN"] + entry["CONFIRMED"] + entry["unbanded"]
        out = dict(entry)
        out["total_observations"] = total
        for key in ("NORMAL", "UNCERTAIN", "CONFIRMED", "unbanded"):
            out["{}_rate".format(key.lower())] = (entry[key] / total) if total else None
        out["abstention_rate"] = (entry["UNCERTAIN"] / total) if total else None
        return out

    return {
        "healthy": _rates(populations["healthy"]),
        "fault": _rates(populations["fault"]),
        "fault_by_magnitude": {k: _rates(v) for k, v in sorted(by_magnitude.items())},
        "healthy_abstention_rate": _rates(populations["healthy"])["abstention_rate"],
        "fault_abstention_rate": _rates(populations["fault"])["abstention_rate"],
        "note": (
            "development-supported boundary; NOT VALIDATED on an untouched holdout"
        ),
    }
