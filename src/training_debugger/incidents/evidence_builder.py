"""Evidence construction, including the mandatory negative evidence.

Phase 0's rule, restated because this module exists to enforce it: *negative
evidence is evidence*. The brief's own worked example lists "learning rate
unchanged / parameter norm within baseline" next to the positive finding, because
that list is what makes the diagnosis credible.

So for every incident this module **tests** a fixed set of candidate signals and
records the outcome in one of four buckets:

* a signal that moved with the anomaly -> ``supporting``
* a signal that would explain it but did not -> ``not_correlated``
* a signal that could not be tested -> ``unknown``, with the reason
* (``contradicting`` is filled by the ranking layer when a hypothesis is refuted)

A signal is never simply absent from the report. "We did not look" and "we looked
and it was normal" are different statements, and only one of them is a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..detectors.base import Anomaly, robust_stats
from ..schema.enums import EvidenceLevel, MetricUnit
from ..schema.evidence import Evidence, EvidenceBuckets

__all__ = [
    "SignalAssessment",
    "CANDIDATE_SIGNALS",
    "assess_signal",
    "build_evidence",
    "anomaly_to_evidence",
]

Series = Sequence[Tuple[int, float]]


#: Signals probed for every incident. Each entry is
#: ``(metric_name, human_label, unit, relative_change_threshold)``.
#:
#: The threshold is the relative change below which a signal is called *stable*.
#: It is deliberately generous: the point of negative evidence is to rule out a
#: large effect, not to detect a 1% drift, and a tight threshold would produce
#: "changed" for ordinary noise and destroy the value of the bucket.
CANDIDATE_SIGNALS: Tuple[Tuple[str, str, MetricUnit, float], ...] = (
    ("train.lr", "learning rate", MetricUnit.DIMENSIONLESS, 0.05),
    ("param.global_norm", "parameter norm", MetricUnit.DIMENSIONLESS, 0.10),
    ("train.tokens_per_sec", "throughput (tokens/s)", MetricUnit.TOKENS_PER_SEC, 0.10),
    ("train.step_time", "step time", MetricUnit.MILLISECONDS, 0.10),
    ("mem.allocated", "allocated memory", MetricUnit.BYTES, 0.15),
    ("grad.global_norm", "global gradient norm", MetricUnit.DIMENSIONLESS, 0.25),
)


@dataclass
class SignalAssessment:
    metric_name: str
    label: str
    unit: MetricUnit
    status: str          # "changed" | "stable" | "unavailable"
    before: Optional[float] = None
    during: Optional[float] = None
    relative_change: Optional[float] = None
    method: str = "window_median_ratio"
    reason: str = ""

    def to_evidence(self, window: Tuple[int, int], event_ids: Optional[List[str]] = None) -> Evidence:
        if self.status == "changed":
            return Evidence(
                claim=(
                    "{label} changed by {change:+.1%} at the incident "
                    "({before:.6g} -> {during:.6g})".format(
                        label=self.label,
                        change=self.relative_change or 0.0,
                        before=self.before or 0.0,
                        during=self.during or 0.0,
                    )
                ),
                level=EvidenceLevel.INFERRED,
                detector="correlation.window_median_ratio",
                metric_name=self.metric_name,
                value=self.during,
                unit=self.unit,
                baseline=self.before,
                window=list(window),
                reason=self.reason or "relative change exceeded the stability threshold",
                event_ids=list(event_ids or []),
                metadata={"relative_change": self.relative_change, "status": self.status},
            )
        if self.status == "stable":
            return Evidence(
                claim=(
                    "{label} remained within baseline at the incident "
                    "({before:.6g}, {change:+.1%})".format(
                        label=self.label,
                        before=self.before if self.before is not None else float("nan"),
                        change=self.relative_change or 0.0,
                    )
                ),
                level=EvidenceLevel.OBSERVED,
                metric_name=self.metric_name,
                value=self.during,
                unit=self.unit,
                baseline=self.before,
                window=list(window),
                reason=self.reason or "relative change stayed below the stability threshold",
                event_ids=list(event_ids or []),
                metadata={"relative_change": self.relative_change, "status": self.status},
            )
        return Evidence(
            claim="{label} could not be assessed: {reason}".format(
                label=self.label, reason=self.reason or "no data"
            ),
            level=EvidenceLevel.UNKNOWN,
            metric_name=self.metric_name,
            unit=self.unit,
            window=list(window),
            reason=self.reason or "no data for this signal in the run",
            metadata={"status": self.status},
        )


def assess_signal(
    metric_name: str,
    label: str,
    unit: MetricUnit,
    series: Optional[Series],
    *,
    onset_step: int,
    baseline_window: int,
    during_window: int,
    threshold: float,
) -> SignalAssessment:
    """Compare a signal's level before and during an incident.

    Uses robust medians on both sides rather than means, for the same reason the
    detectors do: the window being investigated contains the anomaly.
    """
    if not series:
        return SignalAssessment(
            metric_name=metric_name, label=label, unit=unit, status="unavailable",
            reason="this signal was not recorded for this run",
        )
    before = [value for step, value in series if onset_step - baseline_window <= step < onset_step]
    during = [value for step, value in series if onset_step <= step < onset_step + during_window]
    before = [v for v in before if v == v and abs(v) != float("inf")]
    during = [v for v in during if v == v and abs(v) != float("inf")]
    if len(before) < 3 or len(during) < 2:
        return SignalAssessment(
            metric_name=metric_name, label=label, unit=unit, status="unavailable",
            reason="insufficient samples either side of the onset (before=%d, during=%d)"
            % (len(before), len(during)),
        )
    before_median = robust_stats(before)["median"]
    during_median = robust_stats(during)["median"]
    if before_median == 0.0 or before_median != before_median:
        return SignalAssessment(
            metric_name=metric_name, label=label, unit=unit, status="unavailable",
            before=before_median, during=during_median,
            reason="baseline median is zero, so a relative change is undefined",
        )
    change = during_median / before_median - 1.0
    status = "changed" if abs(change) >= threshold else "stable"
    return SignalAssessment(
        metric_name=metric_name,
        label=label,
        unit=unit,
        status=status,
        before=before_median,
        during=during_median,
        relative_change=change,
        reason="relative change {:+.1%} {} the {:.0%} stability threshold".format(
            change,
            "exceeded" if status == "changed" else "stayed below",
            threshold,
        ),
    )


def anomaly_to_evidence(anomaly: Anomaly, event_ids: Optional[List[str]] = None) -> Evidence:
    """Turn a detector finding into an auditable evidence item."""
    return Evidence(
        claim=anomaly.reason or "{} fired at step {}".format(anomaly.detector, anomaly.first_abnormal_step),
        level=anomaly.level,
        detector=str(anomaly.detector),
        detector_version=anomaly.detector_version,
        metric_name=anomaly.metric_name,
        value=anomaly.observed_value,
        baseline=anomaly.baseline,
        threshold=anomaly.threshold,
        window=list(anomaly.window),
        confidence=anomaly.confidence,
        rank=anomaly.rank,
        layer_index=anomaly.layer_index,
        module_path=anomaly.module_path,
        event_ids=list(event_ids or []),
        reason=anomaly.metadata.get("method"),
        metadata=dict(
            anomaly.metadata,
            **{
                "detector_score": anomaly.score,
                # The anomaly's OWN step. ``window`` is
                # ``(baseline_start, onset)``, so rendering ``window[0]`` as the
                # step (an earlier version did) prints the start of the baseline
                # window for every finding -- all of them "step 1" in practice.
                "onset_step": anomaly.first_abnormal_step,
            }
        ),
    )


def build_evidence(
    anomalies: Sequence[Anomaly],
    series_by_metric: Dict[str, Series],
    *,
    onset_step: int,
    event_ids_by_metric: Optional[Dict[str, List[str]]] = None,
    baseline_window: int = 50,
    during_window: int = 30,
) -> Tuple[EvidenceBuckets, List[SignalAssessment]]:
    """Assemble the four evidence buckets for one incident.

    Returns the buckets and the raw assessments, so the report can show what was
    probed even for signals that produced no evidence.
    """
    buckets = EvidenceBuckets()
    ids = event_ids_by_metric or {}

    for anomaly in anomalies:
        buckets.add_supporting(
            anomaly_to_evidence(anomaly, ids.get(anomaly.metric_name))
        )

    assessments: List[SignalAssessment] = []
    anomalous_metrics = {a.metric_name for a in anomalies}
    for metric_name, label, unit, threshold in CANDIDATE_SIGNALS:
        if metric_name in anomalous_metrics:
            # A signal that is itself anomalous belongs in `supporting`, which
            # the loop above already did. Testing it for stability too would be
            # contradictory.
            continue
        assessment = assess_signal(
            metric_name, label, unit, series_by_metric.get(metric_name),
            onset_step=onset_step,
            baseline_window=baseline_window,
            during_window=during_window,
            threshold=threshold,
        )
        assessments.append(assessment)
        evidence = assessment.to_evidence((onset_step - baseline_window, onset_step + during_window))
        if assessment.status == "stable":
            buckets.add_not_correlated(evidence)
        elif assessment.status == "unavailable":
            buckets.add_unknown(evidence)
        else:
            buckets.add_supporting(evidence)

    return buckets, assessments
