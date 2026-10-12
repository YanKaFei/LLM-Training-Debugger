"""WS1: a counting model with one denominator per level.

The v0.1.1 scoreboard mixed units and nobody could say what the denominator was.
Concretely:

* **42 runs produced 97 classification records.** The classification unit was the
  *episode*, and one run can hold many episodes, so 97 was an episode count being
  read as if it were a run count.
* **`primary_recall` was always 1.0** whenever anything was detected, because its
  false-negative term `MISSED_PRIMARY` was never incremented anywhere. The metric was
  structurally incapable of reporting a miss.

Both are the same mistake: a ratio whose numerator and denominator describe different
populations. This module makes the level explicit and gives every level its own
population, so a number cannot be quoted without saying what it is over.

```
SIGNAL    one observation from one detector
EPISODE   one AnomalyEpisode
INCIDENT  one confirmed incident
RUN       one training run
RELATION  one persisted IncidentRelation
```

**Fail-closed.** If an injected run lacks the ground truth needed to adjudicate it, the
result is ``SCORING_INVALID`` -- never a plausible-looking precision computed over
whatever happened to be present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "LevelScore", "ScoringValidity", "evaluate_levels", "validate_scoring_inputs",
    "SCORING_INVALID",
]

SCORING_INVALID = "SCORING_INVALID"


@dataclass
class LevelScore:
    level: str
    eligible_population: int
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: Optional[int] = None
    abstained: int = 0
    #: What the denominator of `recall` actually is, in words.
    recall_denominator: str = ""
    precision_denominator: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def precision(self) -> Optional[float]:
        base = self.tp + self.fp
        return self.tp / base if base else None

    @property
    def recall(self) -> Optional[float]:
        if self.recall_denominator.startswith("NOT ESTIMABLE"):
            return None
        base = self.tp + self.fn
        return self.tp / base if base else None

    @property
    def abstention_rate(self) -> Optional[float]:
        return self.abstained / self.eligible_population if self.eligible_population else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "eligible_population": self.eligible_population,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "abstained": self.abstained,
            "precision": self.precision,
            "recall": self.recall,
            "abstention_rate": self.abstention_rate,
            "recall_denominator": self.recall_denominator,
            "precision_denominator": self.precision_denominator,
            "notes": list(self.notes),
        }


@dataclass
class ScoringValidity:
    valid: bool = True
    issues: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "VALID" if self.valid else SCORING_INVALID

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "valid": self.valid, "issues": list(self.issues)}


def validate_scoring_inputs(scenarios: Sequence[Dict[str, Any]]) -> ScoringValidity:
    """Fail closed on ground truth that cannot adjudicate a run.

    Every check here corresponds to a defect that silently produced a wrong number in
    this project's history:

    * a dropped ``ground_truth`` key made ``primary_anomaly_type("grad_large")``
      unresolvable, so every primary scored as a miss -> precision 0.0
    * a missing ``acceptable_window`` made ``in_window`` false, so every **correct**
      detection was filed as ``UNRELATED_FALSE_POSITIVE`` -> precision 0.0
    * a missing injection step made `primary_found` unmatched

    None of these raised. All of them looked like a result.
    """
    validity = ScoringValidity()
    for scenario in scenarios:
        run_id = scenario.get("run_id")
        if scenario.get("run_error"):
            validity.valid = False
            validity.issues.append(
                {"run_id": run_id, "issue": "RUN_UNREADABLE",
                 "detail": scenario["run_error"]}
            )
            continue
        if not scenario.get("injected"):
            continue
        truth = scenario.get("ground_truth") or {}
        if not truth.get("primary_fault_type"):
            validity.valid = False
            validity.issues.append(
                {"run_id": run_id, "issue": "MISSING_PRIMARY_FAULT_TYPE",
                 "detail": "an injected run without a resolvable primary fault type "
                           "cannot be matched, and would be scored as a miss"}
            )
        if scenario.get("acceptable_window") is None:
            validity.valid = False
            validity.issues.append(
                {"run_id": run_id, "issue": "MISSING_ACCEPTABLE_WINDOW",
                 "detail": "without a window, a correct detection is classified "
                           "UNRELATED_FALSE_POSITIVE and precision reads 0.0"}
            )
        if scenario.get("injection_step") is None:
            validity.valid = False
            validity.issues.append(
                {"run_id": run_id, "issue": "MISSING_INJECTION_STEP",
                 "detail": "the fault's own step is unknown, so no onset can be judged"}
            )
    return validity


def _opened(verdicts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [v for v in verdicts if v.get("verdict") == "CONFIRMED_INCIDENT"]


def _within(onset: Optional[int], window: Optional[Sequence[int]]) -> bool:
    if onset is None or window is None:
        return False
    try:
        return int(window[0]) <= int(onset) <= int(window[1])
    except (TypeError, ValueError, IndexError):
        return False


def evaluate_levels(scenarios: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Score every level independently. No denominator is shared across levels."""
    validity = validate_scoring_inputs(scenarios)
    if not validity.valid:
        # Fail closed: do not emit numbers that look like results.
        return {
            "status": SCORING_INVALID,
            "validity": validity.to_dict(),
            "levels": {},
            "note": "scoring refused: the ground truth cannot adjudicate every run. "
                    "Fixing the inputs is required before any precision or recall is "
                    "meaningful.",
        }

    signal_ls = LevelScore("SIGNAL", 0, recall_denominator="n/a (no independent signal truth)")
    episode_ls = LevelScore("EPISODE", 0,
                         recall_denominator="NOT ESTIMABLE",
                         precision_denominator="all episodes observed")
    # Episode-level RECALL is not defined: the truth says which fault was injected, not
    # how many episodes it *should* produce. Reporting TP/(TP+FN) here produced a
    # meaningless 1.0, because a miss at episode level shows up as an unwanted episode
    # (a precision error), not as a missing one. Only episode-level PRECISION is
    # meaningful, and the note below says so.
    episode_ls.notes.append(
        "episode-level recall is NOT ESTIMABLE: ground truth names the injected fault, "
        "not the number of episodes it should yield. A missed fault appears here as an "
        "extra episode (a precision error), not as a missing one. Use RUN-level recall "
        "for 'was the fault found'."
    )
    incident_ls = LevelScore("INCIDENT", 0,
                          recall_denominator="NOT ESTIMABLE",
                          precision_denominator="confirmed incidents")
    # W7: incident-level recall was the SAME structural bug as v0.1.1's `primary_recall`.
    #
    # `incident_ls.fn` was never incremented anywhere, so a run whose primary fault was
    # entirely missed contributed {tp: 0, fp: 0, fn: 0} -- the miss silently vanished
    # from the denominator instead of lowering anything. The eligible population here is
    # *confirmed incidents*, and there is no independent truth for how many incidents a
    # fault should produce, so recall over this unit is not defined. The misses are
    # reported as an explicit count (`missed_primary_faults`) that must equal RUN-level
    # `fn`, so nothing can disappear unnoticed.
    incident_ls.notes.append(
        "incident-level recall is NOT ESTIMABLE: there is no independent truth for how "
        "many incidents a fault should produce. `missed_primary_faults` reports the "
        "misses explicitly and always equals RUN-level fn."
    )
    run_ls = LevelScore("RUN", 0,
                     recall_denominator="eligible injected runs",
                     precision_denominator="all runs that opened an incident")
    relation_ls = LevelScore("RELATION", 0,
                          recall_denominator="expected relations (allowed secondary in window)",
                          precision_denominator="all persisted relations")
    signal_ls.recall_denominator = "NOT ESTIMABLE"
    signal_ls.precision_denominator = "NOT ESTIMABLE"

    order_only: List[Dict[str, Any]] = []
    run_episode_counts: List[int] = []
    for scenario in scenarios:
        injected = bool(scenario.get("injected"))
        truth = scenario.get("ground_truth") or {}
        primary = truth.get("primary_fault_type")
        window = scenario.get("acceptable_window")
        episodes = scenario.get("episodes", [])
        verdicts = scenario.get("verdicts", [])
        opened = _opened(verdicts)
        opened_types = {v.get("anomaly_type") for v in opened}
        run_episode_counts.append(len(episodes))

        # -- RUN level ---------------------------------------------------
        run_ls.eligible_population += 1
        if injected:
            # One-to-one matching: a run is matched only if an incident of the
            # primary fault type was opened AND its onset falls in the window.
            matched = any(
                v.get("anomaly_type") == primary and _within(v.get("candidate_onset_step"), window)
                for v in opened
            )
            if matched:
                run_ls.tp += 1
            else:
                run_ls.fn += 1
                if not opened:
                    run_ls.abstained += 1
        else:
            if opened:
                run_ls.fp += 1
            else:
                run_ls.tn = (run_ls.tn or 0) + 1

        # -- INCIDENT level ----------------------------------------------
        for verdict in opened:
            incident_ls.eligible_population += 1
            if injected and verdict.get("anomaly_type") == primary and _within(
                verdict.get("candidate_onset_step"), window
            ):
                incident_ls.tp += 1
            else:
                incident_ls.fp += 1

        # -- EPISODE level -----------------------------------------------
        for episode in episodes:
            episode_ls.eligible_population += 1
            anomaly_type = episode.get("anomaly_type")
            onset = episode.get("candidate_onset_step")
            if not injected:
                # An episode on a healthy run is a false positive by definition, and
                # whether it opened an incident decides which kind.
                if anomaly_type in opened_types:
                    episode_ls.fp += 1
                else:
                    episode_ls.tn = (episode_ls.tn or 0) + 1
                continue
            if anomaly_type == primary and _within(onset, window):
                episode_ls.tp += 1
            elif anomaly_type in (truth.get("allowed_secondary_manifestations") or ()):
                # A real consequence is neither a primary hit nor noise.
                episode_ls.tn = (episode_ls.tn or 0) + 1
                episode_ls.notes.append("secondary manifestation, not a primary match")
            else:
                episode_ls.fp += 1
        if injected and not episodes:
            episode_ls.fn += 1

        # -- RELATION level ----------------------------------------------
        relations = scenario.get("relations", [])
        allowed = set(truth.get("allowed_secondary_manifestations") or ())
        secondary_window = truth.get("secondary_window")
        expected = 0
        for episode in episodes:
            if episode.get("anomaly_type") in allowed:
                if secondary_window is None or _within(episode.get("candidate_onset_step"),
                                                       secondary_window):
                    expected += 1
                    break
        # W3.2: an order-only relation is not a causal claim.
        #
        # `POSSIBLE_CONSEQUENCE_OF` asserts that a manifestation may be downstream of a
        # fault; `PRECEDES` asserts only that one thing came before another, and
        # `CO_OCCURS_WITH` only that two things overlapped. Scoring them identically
        # would punish the system for honestly downgrading a link it could not support
        # -- which is precisely the behaviour this release is trying to reward.
        for relation in relations:
            target = relation.get("target_anomaly_type")
            gap = relation.get("gap_steps")
            kind = relation.get("relation_type")
            if kind == "POSSIBLE_CONSEQUENCE_OF":
                relation_ls.eligible_population += 1
                if target in allowed and gap is not None and 0 <= gap <= 100:
                    relation_ls.tp += 1
                else:
                    relation_ls.fp += 1
                    relation_ls.notes.append(
                        "FABRICATED_ASSOCIATION: {} claimed for a target that ground "
                        "truth does not declare a manifestation of the primary".format(target)
                    )
            else:
                order_only.append({"relation_type": kind, "target": target, "gap": gap,
                                   "run_id": scenario.get("run_id")})
        relation_ls.fn += max(0, expected - sum(
            1 for relation in relations
            if relation.get("relation_type") == "POSSIBLE_CONSEQUENCE_OF"
        ))
        relation_ls.eligible_population += max(0, expected - len([
            relation for relation in relations
            if relation.get("relation_type") == "POSSIBLE_CONSEQUENCE_OF"
        ]))

    # SIGNAL level has no independent ground truth in this corpus.
    signal_ls.notes.append(
        "no independent per-observation truth exists in this corpus, so signal-level "
        "precision/recall are NOT ESTIMABLE rather than assumed"
    )

    one_to_many = sum(1 for c in run_episode_counts if c > 1)
    # Invariant: no missed primary may disappear. The count surfaced at incident level
    # is the run-level false-negative count, by construction.
    assert incident_ls.fn == 0  # never incremented; misses live in `missed_primary_faults`
    return {
        "status": "VALID",
        "missed_primary_faults": run_ls.fn,
        "validity": validity.to_dict(),
        "levels": {
            "SIGNAL": signal_ls.to_dict(),
            "EPISODE": episode_ls.to_dict(),
            "INCIDENT": incident_ls.to_dict(),
            "RUN": run_ls.to_dict(),
            "RELATION": relation_ls.to_dict(),
        },
        "missed_primary_faults": run_ls.fn,
        "order_only_relations": order_only,
        "order_only_count": len(order_only),
        "unit_accounting": {
            "runs": len(scenarios),
            "episodes_classified": sum(run_episode_counts),
            "runs_with_more_than_one_episode": one_to_many,
            "explanation": (
                "The v0.1.1 scoreboard's 97 records were EPISODES, not runs. A run may "
                "hold several episodes, so the same injected fault can appear in more "
                "than one classification record. Episode counts must never be read as "
                "run counts."
            ),
        },
    }
