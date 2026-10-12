"""K3: earliest-evidence evaluation.

The brief calls this the product's core claim and notes it was never measured across
six phases. It is measured here.

**The definition matters more than the arithmetic.** The system must be graded on its
*earliest defensible abnormal episode onset*, not on its earliest raw detector fire.
A fire that precedes the fault is not an early detection -- it is a false positive,
and Phase 1.2 showed what treating it as a negative delay costs.

**A claim is only true if it names a fault-associated episode.** Finding a healthy
gradient fluctuation at step 40 when the fault is at step 80 is a **false earliest
claim** even when the anomaly family happens to match. This is the strictest of the
three K3.3 conditions and it is the one that makes the metric meaningful.

**UNKNOWN is a permitted answer** for a fault inside the uncertainty band, and it is
not counted as a false positive. But precision alone would then reward a system that
answers UNKNOWN to everything, so ``EARLIEST_EVIDENCE_COVERAGE`` is reported beside
it and both are compared against their targets together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .consequences import primary_anomaly_type
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["EarliestEvidenceOutcome", "evaluate_earliest_evidence"]


@dataclass
class EarliestEvidenceOutcome:
    run_id: str
    status: str                    # TRUE | FALSE | UNKNOWN
    reason: str = ""
    claimed_step: Optional[int] = None
    true_step: Optional[int] = None
    absolute_error: Optional[int] = None
    signed_error: Optional[int] = None
    confirmation_delay: Optional[int] = None
    pre_injection: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "reason": self.reason,
            "claimed_step": self.claimed_step,
            "true_step": self.true_step,
            "absolute_error": self.absolute_error,
            "signed_error": self.signed_error,
            "confirmation_delay": self.confirmation_delay,
            "pre_injection": self.pre_injection,
        }


def evaluate_earliest_evidence(scenarios: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade the earliest-evidence claim on every fault run."""
    outcomes: List[EarliestEvidenceOutcome] = []

    for scenario in scenarios:
        if scenario.get("run_error") or not scenario.get("injected"):
            continue
        window = scenario.get("acceptable_window")
        injection = scenario.get("injection_step")
        expected_type = (
            scenario.get("primary_fault_type")
            or primary_anomaly_type(scenario.get("fault_type"))
        )
        expected_start = window[0] if window else injection

        confirmed = [
            e for e in scenario.get("episodes", [])
            if (e.get("state") == "CONFIRMED"
                or e.get("outcome") == "CONFIRMED_ANOMALY")
        ]
        # Only episodes of the expected primary family can support a claim about
        # this fault. A confirmed episode of another family is not evidence about it.
        relevant = [e for e in confirmed if e.get("anomaly_type") == expected_type]

        if not relevant:
            abstained = any(
                e.get("outcome") == "INSUFFICIENT_EVIDENCE"
                for e in scenario.get("episodes", [])
            )
            outcomes.append(
                EarliestEvidenceOutcome(
                    run_id=scenario.get("run_id", ""),
                    status="UNKNOWN",
                    reason=(
                        "evidence was judged insufficient" if abstained
                        else "no confirmed episode of the expected family"
                    ),
                    true_step=expected_start,
                )
            )
            continue

        earliest = min(relevant, key=lambda e: e.get("candidate_onset_step") or 10 ** 9)
        onset = earliest.get("candidate_onset_step")
        confirmed_at = earliest.get("confirmed_at_step")
        in_window = (
            onset is not None and window is not None and window[0] <= onset <= window[1]
        )
        outcome = EarliestEvidenceOutcome(
            run_id=scenario.get("run_id", ""),
            status="TRUE" if in_window else "FALSE",
            reason=(
                "onset inside the acceptable window"
                if in_window else
                "onset outside the acceptable window; the earliest confirmed episode "
                "of the expected family is not fault-associated"
            ),
            claimed_step=onset,
            true_step=expected_start,
            absolute_error=(abs(onset - expected_start)
                            if onset is not None and expected_start is not None else None),
            signed_error=(onset - expected_start
                          if onset is not None and expected_start is not None else None),
            confirmation_delay=(
                confirmed_at - onset
                if confirmed_at is not None and onset is not None else None
            ),
            pre_injection=(onset is not None and expected_start is not None
                           and onset < expected_start),
        )
        outcomes.append(outcome)

    total = len(outcomes)
    true = sum(1 for o in outcomes if o.status == "TRUE")
    false = sum(1 for o in outcomes if o.status == "FALSE")
    unknown = sum(1 for o in outcomes if o.status == "UNKNOWN")
    errors = [o.absolute_error for o in outcomes if o.absolute_error is not None]
    signed = [o.signed_error for o in outcomes if o.signed_error is not None]
    delays = [o.confirmation_delay for o in outcomes if o.confirmation_delay is not None]

    def _median(values: Sequence[int]) -> Optional[float]:
        return float(sorted(values)[len(values) // 2]) if values else None

    return {
        "runs": total,
        "true": true,
        "false": false,
        "unknown": unknown,
        "precision": (true / (true + false)) if (true + false) else None,
        "coverage": (true / total) if total else None,
        "unknown_rate": (unknown / total) if total else None,
        "abstention_rate": (unknown / total) if total else None,
        "absolute_onset_error_median": _median(errors),
        "signed_onset_error_median": _median(signed),
        "confirmation_delay_median": _median(delays),
        "pre_injection_confirmed_incidents": sum(1 for o in outcomes if o.pre_injection),
        "outcomes": [o.to_dict() for o in outcomes],
        "note": (
            "Precision is reported only beside coverage. A system that answers "
            "UNKNOWN to every run would score perfect precision, so coverage is the "
            "guard against that."
        ),
    }
