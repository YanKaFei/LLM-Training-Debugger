"""Step-level divergence localization between two runs.

The existing `compare_runs` answers *"did this metric get worse on average?"*. It cannot
answer the question that actually shortens a debugging session: **"at which step did these
two runs stop agreeing?"**

That gap matters because of what the external landscape shows. OpGuard (OSDI '26) gets its
precision from comparing two runs at tensor boundaries and taking the **first bit-level
mismatch** as the pivot; its user testimony says plainly that *"the loss precision is too
low and gives us no clue where to start."* This module does the bounded, honest version of
that idea at the **metric** level:

    align two runs' series -> difference d[t] = b[t] - a[t]
    estimate the scale of d from its own EARLY region  (the runs agree there)
    the first step where |d| leaves that scale AND stays out  is the pivot

The two runs' shared noise cancels in `d`, which is why this localises when a single-run
threshold cannot. It is **not** bit-exact and does not claim to be: it reports a step and a
metric, never a kernel or a tensor.

Abstention is a first-class outcome. Too few common steps, an unusable early region, or no
sustained departure all produce an explicit answer rather than a guessed pivot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DivergencePoint", "DivergenceReport", "localize_divergence",
    "DIVERGENCE_RULE_VERSION", "is_deterministic_metric",
    "NON_DETERMINISTIC_PREFIXES",
    "render_divergence_diagnosis",
]

DIVERGENCE_RULE_VERSION = "v0.1.8-1.0"

#: Steps of agreement used to estimate how much two runs normally differ. The pivot is
#: judged against the runs' OWN agreement, not against an absolute tolerance.
DEFAULT_BASELINE_WINDOW = 30
#: How far outside that agreement a difference must sit to count.
DEFAULT_Z_THRESHOLD = 8.0
#: How many consecutive steps must stay outside before it counts. A single-sample blip is
#: not a divergence.
DEFAULT_PERSISTENCE = 3
#: Fewer common steps than this and no pivot can be defended.
DEFAULT_MIN_COMMON_STEPS = 40
#: Agreement is judged relative to the metric's own early level, so metrics of very
#: different magnitudes are treated consistently. Two runs agreeing to within this
#: fraction of the level are "in agreement".
DEFAULT_RELATIVE_FLOOR = 0.01
#: Last-resort guard for a metric whose early level is also zero.
_ABSOLUTE_FLOOR = 1e-12

#: Metric families that are NOT reproducible run to run.
#:
#: Found by running `compare` on a real paired corpus: it reported `train.step_time` at
#: step 58 as the first divergence when the fault was injected at step 90. Wall-clock
#: timings differ from the very first measurement because they depend on machine load,
#: allocator state and thread scheduling — not on the algorithm. Two runs that are
#: algorithmically identical are *expected* to disagree here.
#:
#: This is the same class of error as "a non-finite norm is not a non-finite tensor":
#: **a timing difference is not an algorithmic divergence.** They are excluded by
#: default and can be included explicitly with `include_non_deterministic=True`, in which
#: case the report says so.
NON_DETERMINISTIC_PREFIXES = (
    "train.step_time",
    "train.optimizer_time_ms",
    "train.samples_per_sec",
    "train.tokens_per_sec",
    "train.tokens",
    "mem.",
)


def is_deterministic_metric(metric_name: str) -> bool:
    """Whether two identical runs should produce the same value for this metric."""
    name = str(metric_name or "")
    return not any(name.startswith(prefix) for prefix in NON_DETERMINISTIC_PREFIXES)


@dataclass
class DivergencePoint:
    metric_name: str
    step: int
    value_a: float
    value_b: float
    difference: float
    relative_difference: Optional[float]
    z_score: float
    persisted_steps: int
    direction: str                      # "a_above_b" | "b_above_a"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "step": self.step,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "difference": self.difference,
            "relative_difference": self.relative_difference,
            "z_score": self.z_score,
            "persisted_steps": self.persisted_steps,
            "direction": self.direction,
        }


@dataclass
class DivergenceReport:
    run_a: str
    run_b: str
    rule_version: str = DIVERGENCE_RULE_VERSION
    common_steps: int = 0
    baseline_window: int = DEFAULT_BASELINE_WINDOW
    #: Scale of the two runs' agreement in the early region, per metric.
    agreement_scale: Dict[str, float] = field(default_factory=dict)
    #: The earliest defensible pivot across all metrics, if any.
    first_divergence: Optional[DivergencePoint] = None
    per_metric: List[DivergencePoint] = field(default_factory=list)
    metrics_compared: int = 0
    notes: List[str] = field(default_factory=list)
    non_deterministic_excluded: List[str] = field(default_factory=list)
    verdict: str = "INSUFFICIENT_EVIDENCE"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_a": self.run_a,
            "run_b": self.run_b,
            "rule_version": self.rule_version,
            "common_steps": self.common_steps,
            "baseline_window": self.baseline_window,
            "agreement_scale": dict(self.agreement_scale),
            "first_divergence": (self.first_divergence.to_dict()
                                 if self.first_divergence else None),
            "per_metric": [p.to_dict() for p in self.per_metric],
            "metrics_compared": self.metrics_compared,
            "notes": list(self.notes),
            "non_deterministic_excluded": list(self.non_deterministic_excluded),
            "verdict": self.verdict,
            "reason": self.reason,
        }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _mad(values: Sequence[float]) -> float:
    """Median absolute deviation, scaled to be comparable with a standard deviation."""
    if not values:
        return 0.0
    centre = _median(values)
    return 1.4826 * _median([abs(v - centre) for v in values])


def localize_divergence(
    run_a: str,
    run_b: str,
    series_a: Dict[str, Sequence[Tuple[int, float]]],
    series_b: Dict[str, Sequence[Tuple[int, float]]],
    *,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    persistence: int = DEFAULT_PERSISTENCE,
    min_common_steps: int = DEFAULT_MIN_COMMON_STEPS,
    metrics: Optional[Sequence[str]] = None,
    include_non_deterministic: bool = False,
) -> DivergenceReport:
    """Locate the first step at which two runs stop agreeing.

    Both inputs are ``{metric_name: [(step, value), ...]}``. Only steps present in BOTH
    runs are used; nothing is interpolated, because an interpolated value would be a
    fabricated observation.
    """
    report = DivergenceReport(run_a=run_a, run_b=run_b, baseline_window=baseline_window)
    names = list(metrics) if metrics else sorted(set(series_a) & set(series_b))
    if not include_non_deterministic:
        skipped = [n for n in names if not is_deterministic_metric(n)]
        names = [n for n in names if is_deterministic_metric(n)]
        for name in skipped:
            report.notes.append(
                "{}: excluded — wall-clock/timing metrics are not reproducible between "
                "runs, so a difference there is not an algorithmic divergence".format(name)
            )
        report.non_deterministic_excluded = skipped

    pivot: Optional[DivergencePoint] = None
    compared = 0

    for name in names:
        map_a = {int(s): float(v) for s, v in series_a.get(name, []) if v is not None}
        map_b = {int(s): float(v) for s, v in series_b.get(name, []) if v is not None}
        steps = sorted(set(map_a) & set(map_b))
        if len(steps) < min_common_steps:
            continue
        compared += 1
        if report.common_steps == 0:
            report.common_steps = len(steps)

        differences = [map_b[s] - map_a[s] for s in steps]
        early = differences[:baseline_window]
        scale = _mad(early)

        # How much the two runs normally differ. Two guards, because both failure modes
        # are real:
        #
        #  * `scale` alone is ZERO when the runs agree exactly in the early region --
        #    which is the STRONGEST possible signal (identical up to here, then not), and
        #    an earlier version of this function skipped the metric entirely because of
        #    it. A zero-variance agreement region must not disable the comparison.
        #  * a purely absolute epsilon would be meaningless across metrics of wildly
        #    different magnitudes.
        #
        # So the floor is RELATIVE to the metric's own level: agreement to within 1 % of
        # the metric's early magnitude counts as agreement. Below that, any sustained
        # departure is a pivot.
        level = abs(_median([map_a[s] for s in steps[:baseline_window]]))
        relative_floor = level * DEFAULT_RELATIVE_FLOOR
        effective = max(scale, relative_floor, _ABSOLUTE_FLOOR)
        report.agreement_scale[name] = effective
        if scale == 0.0:
            report.notes.append(
                "{}: the runs were identical over the first {} steps, so the agreement "
                "scale comes from the metric's own level ({:.4g})".format(
                    name, baseline_window, effective)
            )

        index = baseline_window
        while index < len(steps):
            d = differences[index]
            z = d / effective
            if abs(z) <= z_threshold:
                index += 1
                continue
            # require persistence so a single-sample blip is not called a divergence
            run_length = 1
            probe = index + 1
            while probe < len(steps) and abs(differences[probe] / effective) > z_threshold:
                run_length += 1
                probe += 1
            if run_length < persistence:
                index = probe
                continue

            step = steps[index]
            base = abs(map_a[step])
            point = DivergencePoint(
                metric_name=name, step=step,
                value_a=map_a[step], value_b=map_b[step],
                difference=d,
                relative_difference=(d / base) if base > 0 else None,
                z_score=z, persisted_steps=run_length,
                direction="b_above_a" if d > 0 else "a_above_b",
            )
            report.per_metric.append(point)
            if pivot is None or step < pivot.step:
                pivot = point
            break

    report.metrics_compared = compared
    report.first_divergence = pivot

    if compared == 0:
        report.verdict = "INSUFFICIENT_EVIDENCE"
        report.reason = (
            "no metric had at least {} overlapping steps, so no comparison is defensible"
            .format(min_common_steps)
        )
    elif pivot is None:
        report.verdict = "NO_DIVERGENCE"
        report.reason = (
            "on {} metric(s) the two runs stayed within the agreement they themselves "
            "established over the first {} steps; no step is a defensible pivot"
            .format(compared, baseline_window)
        )
    else:
        report.verdict = "DIVERGENCE_LOCALIZED"
        report.reason = (
            "the runs agree for the first {} steps and first depart at step {} on {} "
            "(difference {:+.4g}, {:.1f}x the agreement scale). This is a pivot for "
            "investigation, NOT a root cause: it names a step and a metric, not a kernel "
            "or a tensor.".format(
                baseline_window, pivot.step, pivot.metric_name,
                pivot.difference, abs(pivot.z_score),
            )
        )
    return report


def render_divergence(report: DivergenceReport) -> str:
    lines: List[str] = []
    add = lines.append
    add("RUN DIVERGENCE")
    add("")
    add("  A: {}".format(report.run_a))
    add("  B: {}".format(report.run_b))
    add("  common steps: {}   metrics compared: {}".format(
        report.common_steps, report.metrics_compared))
    add("")
    if report.first_divergence is None:
        add("  VERDICT: {}".format(report.verdict))
        add("")
        for chunk in _wrap(report.reason, 76):
            add("  " + chunk)
        return "\n".join(lines)

    pivot = report.first_divergence
    add("  FIRST DIVERGENCE: step {}  metric {}".format(pivot.step, pivot.metric_name))
    add("")
    add("    A = {:.6g}    B = {:.6g}    difference = {:+.4g}".format(
        pivot.value_a, pivot.value_b, pivot.difference))
    add("    {:.1f}x the agreement scale, sustained for {} step(s)".format(
        abs(pivot.z_score), pivot.persisted_steps))
    add("")
    if report.per_metric:
        add("  per-metric pivots")
        for point in sorted(report.per_metric, key=lambda p: p.step):
            add("    step {:<6} {:<28} {:+.4g}".format(
                point.step, point.metric_name, point.difference))
        add("")
    for chunk in _wrap(report.reason, 76):
        add("  " + chunk)
    return "\n".join(lines)


def render_divergence_diagnosis(
    report: "DivergenceReport", incidents: Sequence[Any]
) -> str:
    """Render a divergence pivot followed by the incidents around it.

    The ordering is the whole content. An incident **at or after** the pivot is a
    *candidate* explanation for what changed; an incident **before** the pivot is a
    separate event that the pivot cannot explain, because it came first. Neither is a
    causal claim, and the renderer says so rather than leaving the reader to infer it.
    """
    lines: List[str] = []
    add = lines.append
    add(render_divergence(report))
    add("")
    add("-" * 72)
    add("")

    pivot_step = report.first_divergence.step if report.first_divergence else None
    if pivot_step is None:
        add("INCIDENTS")
        add("")
        for incident in incidents:
            add("  step {:<6} {}".format(
                _incident_step(incident), _incident_label(incident)))
        add("")
        add("  No pivot was established, so nothing here is dated against one.")
        return "\n".join(lines)

    after = [i for i in incidents if (_incident_step(i) or -1) >= pivot_step]
    before = [i for i in incidents if (_incident_step(i) or -1) < pivot_step]

    add("INCIDENTS AT OR AFTER THE DIVERGENCE  (candidates for what changed)")
    add("")
    for incident in after:
        add("  step {:<6} {}".format(_incident_step(incident), _incident_label(incident)))
    if not after:
        add("  none")
    add("")
    add("INCIDENTS BEFORE THE DIVERGENCE  (these cannot be explained by it)")
    add("")
    for incident in before:
        add("  step {:<6} {}".format(_incident_step(incident), _incident_label(incident)))
    if not before:
        add("  none")
    add("")
    add("  This is an ORDERING, not a cause. The pivot says where the two runs stopped")
    add("  agreeing; it does not say which of them is wrong, nor that an incident after")
    add("  it was caused by whatever changed.")
    return "\n".join(lines)


def _incident_step(incident: Any) -> Optional[int]:
    for attribute in ("detected_step", "confirmed_at_step", "candidate_onset_step"):
        value = getattr(incident, attribute, None)
        if value is not None:
            return int(value)
    payload = incident.to_dict() if hasattr(incident, "to_dict") else dict(incident)
    for key in ("detected_step", "confirmed_at_step", "candidate_onset_step"):
        if payload.get(key) is not None:
            return int(payload[key])
    return None


def _incident_label(incident: Any) -> str:
    payload = incident.to_dict() if hasattr(incident, "to_dict") else dict(incident)
    kind = getattr(getattr(incident, "incident_type", None), "value", None)
    if kind is None:
        kind = payload.get("incident_type")
    return "{}  {}".format(kind, payload.get("incident_id", "")[:20])


def _wrap(text: str, width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines
