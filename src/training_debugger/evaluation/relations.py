"""W3: relation-opportunity scoring.

v0.1.3 reported `coverage 1.000` and `abstention 0.800` from the same runs, which is
only possible if "no consequence relation was created" was being counted as both a
success and an abstention. The brief is right that this is incoherent.

The distinction that matters:

```
negative opportunity, no claim   -> CORRECT REJECTION   (a success)
negative opportunity, claim      -> FALSE ASSOCIATION   (an error)
positive opportunity, claim      -> TRUE POSITIVE
positive opportunity, no claim   -> FALSE NEGATIVE      (a miss)
no opportunity at all            -> NOT APPLICABLE      (not a data point)
```

A healthy run with no anomaly anywhere offers no relation opportunity, so it is **not**
a negative sample and must not be counted as one. Reporting it as "abstention" inflates
the denominator with runs that could not have produced a relation either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["RelationOpportunityScore", "score_relation_opportunities"]

CONSEQUENCE = "POSSIBLE_CONSEQUENCE_OF"


@dataclass
class RelationOpportunityScore:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    not_applicable: int = 0
    #: Runs where a relation was recorded but it asserts order only.
    order_only_positive: int = 0
    order_only_negative: int = 0
    failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def positive_cases(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def negative_cases(self) -> int:
        return self.false_positive + self.true_negative

    @property
    def precision(self) -> Optional[float]:
        base = self.true_positive + self.false_positive
        return self.true_positive / base if base else None

    @property
    def recall(self) -> Optional[float]:
        base = self.true_positive + self.false_negative
        return self.true_positive / base if base else None

    @property
    def positive_coverage(self) -> Optional[float]:
        """Of the cases where a consequence exists, how many were claimed."""
        return self.recall

    @property
    def negative_rejection_rate(self) -> Optional[float]:
        """Of the cases where no consequence exists, how many were correctly refused."""
        base = self.negative_cases
        return self.true_negative / base if base else None

    @property
    def false_association_rate(self) -> Optional[float]:
        base = self.negative_cases
        return self.false_positive / base if base else None

    @property
    def true_abstention_rate(self) -> Optional[float]:
        """Order-only relations over negative cases.

        This is the honest abstention number: the system recorded an association but
        explicitly declined to call it a consequence. It is NOT the same as a correct
        rejection (no relation at all), and conflating the two is what produced the
        incoherent v0.1.3 pair.
        """
        base = self.negative_cases
        return self.order_only_negative / base if base else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "not_applicable": self.not_applicable,
            "positive_cases": self.positive_cases,
            "negative_cases": self.negative_cases,
            "precision": self.precision,
            "recall": self.recall,
            "positive_coverage": self.positive_coverage,
            "negative_rejection_rate": self.negative_rejection_rate,
            "false_association_rate": self.false_association_rate,
            "true_abstention_rate": self.true_abstention_rate,
            "order_only_on_positive": self.order_only_positive,
            "order_only_on_negative": self.order_only_negative,
            "failures": list(self.failures),
        }


def score_relation_opportunities(
    scenarios: Sequence[Dict[str, Any]], truth: Dict[str, Dict[str, Any]]
) -> RelationOpportunityScore:
    """Score each run by whether a consequence relation *should* exist.

    ``truth[run_id]["expectation"]`` is ``CONSEQUENCE`` or ``INDEPENDENT``;
    ``truth[run_id]["opportunity"]`` is ``positive``, ``negative`` or absent (which
    means the run offers no relation opportunity and is not scored).
    """
    score = RelationOpportunityScore()
    for scenario in scenarios:
        run_id = scenario.get("run_id")
        entry = truth.get(run_id)
        if entry is None:
            continue
        opportunity = entry.get("opportunity")
        if opportunity is None:
            score.not_applicable += 1
            continue

        relations = scenario.get("relations", [])
        claims = [r for r in relations if r.get("relation_type") == CONSEQUENCE]
        order_only = [r for r in relations if r.get("relation_type") != CONSEQUENCE]
        claimed = bool(claims)

        if opportunity == "positive":
            if claimed:
                score.true_positive += 1
            else:
                score.false_negative += 1
                score.failures.append({
                    "run_id": run_id, "class": "MISSED_CONSEQUENCE",
                    "case": entry.get("case"),
                    "observed": [r.get("relation_type") for r in relations],
                    "expected": entry.get("reason"),
                })
            if order_only:
                score.order_only_positive += 1
        elif opportunity == "negative":
            if claimed:
                score.false_positive += 1
                score.failures.append({
                    "run_id": run_id, "class": "FABRICATED_ASSOCIATION",
                    "case": entry.get("case"),
                    "claimed": [{"target": r.get("target_anomaly_type"),
                                 "gap": r.get("gap_steps"),
                                 "witness": (r.get("evidence") or {}).get(
                                     "midstream_nonfinite_witness")}
                                for r in claims],
                    "expected": entry.get("reason"),
                })
            else:
                score.true_negative += 1
                if order_only:
                    score.order_only_negative += 1
        else:
            score.not_applicable += 1
    return score
