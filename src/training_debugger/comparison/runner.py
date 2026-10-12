"""C3: cross-run comparison.

Answering "did this run get worse?" requires three things Phase 1 could not do:
a persisted incident history (C1), a way to read two runs on a common axis, and a
structured answer rather than prose.

**Normalising the training position is the crux.** Two runs with different step
counts cannot be compared at absolute steps: step 5 000 in a 10 000-step run and
step 5 000 in a 100 000-step run are at completely different points in training.
The comparison therefore aligns on **relative progress** by default when the two
runs differ in length, and records which axis and which interpolation were used —
because an unstated alignment is an unstated assumption, and the whole comparison
rests on it.

**Everything is structured.** There is no prose output: a caller gets a
:class:`RunComparison` with config changes, metric regressions, incident changes
and performance changes, each with the numbers behind it. A human-readable
rendering exists for the CLI, but it is derived from the structure, never a
substitute for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["RunComparison", "MetricRegression", "ConfigChange", "compare_runs"]

#: Metrics whose direction of improvement is known. For these, a change is
#: labelled better/worse rather than merely up/down.
LOWER_IS_BETTER = frozenset({"train.loss", "train.step_time", "mem.allocated", "mem.reserved"})
HIGHER_IS_BETTER = frozenset({"train.tokens_per_sec", "train.samples_per_sec"})

#: Metrics compared by default, in report order.
COMPARED_METRICS = (
    "train.loss",
    "grad.global_norm",
    "param.global_norm",
    "train.step_time",
    "train.tokens_per_sec",
    "train.lr",
)


@dataclass
class ConfigChange:
    key: str
    baseline: Any
    candidate: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "baseline": self.baseline, "candidate": self.candidate}


@dataclass
class MetricRegression:
    metric_name: str
    baseline_median: Optional[float]
    candidate_median: Optional[float]
    relative_change: Optional[float]
    direction: str            # "better" | "worse" | "neutral" | "unknown"
    baseline_points: int
    candidate_points: int
    alignment: str
    baseline_at_aligned_end: Optional[float] = None
    candidate_at_aligned_end: Optional[float] = None
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_median": self.baseline_median,
            "candidate_median": self.candidate_median,
            "relative_change": self.relative_change,
            "direction": self.direction,
            "baseline_points": self.baseline_points,
            "candidate_points": self.candidate_points,
            "alignment": self.alignment,
            "baseline_at_aligned_end": self.baseline_at_aligned_end,
            "candidate_at_aligned_end": self.candidate_at_aligned_end,
            "caveats": list(self.caveats),
        }


@dataclass
class RunComparison:
    baseline_run_id: str
    candidate_run_id: str
    alignment: str
    interpolation: str
    baseline_steps: Optional[int] = None
    candidate_steps: Optional[int] = None

    config_changes: List[ConfigChange] = field(default_factory=list)
    metric_regressions: List[MetricRegression] = field(default_factory=list)
    incident_changes: Dict[str, Any] = field(default_factory=dict)
    performance_changes: Dict[str, Any] = field(default_factory=dict)
    new_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    missing_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    comparability: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "alignment": self.alignment,
            "interpolation": self.interpolation,
            "baseline_steps": self.baseline_steps,
            "candidate_steps": self.candidate_steps,
            "config_changes": [c.to_dict() for c in self.config_changes],
            "metric_regressions": [m.to_dict() for m in self.metric_regressions],
            "incident_changes": dict(self.incident_changes),
            "performance_changes": dict(self.performance_changes),
            "new_anomalies": list(self.new_anomalies),
            "missing_anomalies": list(self.missing_anomalies),
            "comparability": dict(self.comparability),
        }

    def render(self) -> str:
        lines: List[str] = []
        add = lines.append
        add("RUN COMPARISON")
        add("  baseline:  {}".format(self.baseline_run_id))
        add("  candidate: {}".format(self.candidate_run_id))
        add("  alignment: {} (interpolation: {})".format(self.alignment, self.interpolation))
        add("")
        comparable = self.comparability.get("comparable")
        add("Comparable: {}".format("yes" if comparable else "NO"))
        for caveat in self.comparability.get("caveats", []):
            add("  caveat: {}".format(caveat))
        add("")
        add("Config changes ({}):".format(len(self.config_changes)))
        for change in self.config_changes:
            add("  {:<24} {} -> {}".format(change.key, change.baseline, change.candidate))
        if not self.config_changes:
            add("  (none)")
        add("")
        add("Metric changes:")
        add("  {:<24} {:>12} {:>12} {:>10}  {}".format(
            "metric", "baseline", "candidate", "change", "direction"))
        for regression in self.metric_regressions:
            change = "n/a" if regression.relative_change is None else "{:+.1%}".format(
                regression.relative_change)
            add("  {:<24} {:>12} {:>12} {:>10}  {}".format(
                regression.metric_name,
                "n/a" if regression.baseline_median is None else "{:.6g}".format(regression.baseline_median),
                "n/a" if regression.candidate_median is None else "{:.6g}".format(regression.candidate_median),
                change, regression.direction))
            for caveat in regression.caveats:
                add("      caveat: {}".format(caveat))
        add("")
        add("Incidents:")
        for key, value in sorted(self.incident_changes.items()):
            add("  {:<28} {}".format(key, value))
        add("")
        add("New anomalies in the candidate ({}):".format(len(self.new_anomalies)))
        for item in self.new_anomalies:
            add("  - {} ({})".format(item.get("incident_type"), item.get("incident_id")))
        if not self.new_anomalies:
            add("  (none)")
        add("")
        add("Anomalies present in the baseline but absent here ({}):".format(
            len(self.missing_anomalies)))
        for item in self.missing_anomalies:
            add("  - {} ({})".format(item.get("incident_type"), item.get("incident_id")))
        if not self.missing_anomalies:
            add("  (none)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def _median(values: Sequence[float]) -> Optional[float]:
    usable = sorted(v for v in values if v == v and abs(v) != float("inf"))
    if not usable:
        return None
    n = len(usable)
    return float(usable[n // 2]) if n % 2 else 0.5 * (usable[n // 2 - 1] + usable[n // 2])


def _align(
    series: Sequence[Tuple[int, float]],
    total_steps: Optional[int],
    baseline_total: Optional[int],
    alignment: str,
) -> List[Tuple[float, float]]:
    """Map a series onto the comparison axis.

    ``step``      -- absolute step, used when the runs are the same length or the
                     caller explicitly asks for it.
    ``relative``  -- ``step / total_steps``, so step 5 000 of a 10 000-step run
                     aligns with step 50 000 of a 100 000-step run.

    No interpolation is performed. Resampling two curves onto a shared grid would
    manufacture points that were never measured; instead both curves are expressed
    on the axis and compared by *order statistics* (medians over the overlapping
    region), which is honest about what was observed.
    """
    if alignment == "step" or not total_steps:
        return [(float(step), value) for step, value in series]
    denominator = float(total_steps)
    _ = baseline_total
    return [(step / denominator, value) for step, value in series]


def _overlapping(
    baseline: Sequence[Tuple[float, float]], candidate: Sequence[Tuple[float, float]]
) -> Tuple[List[float], List[float]]:
    """Values from each series inside the shared axis range."""
    if not baseline or not candidate:
        return [], []
    low = max(baseline[0][0], candidate[0][0])
    high = min(baseline[-1][0], candidate[-1][0])
    if low > high:
        return [], []
    return (
        [v for x, v in baseline if low <= x <= high],
        [v for x, v in candidate if low <= x <= high],
    )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _read_run(root: Path, run_id: str):
    from ..storage import DuckDBReader

    reader = DuckDBReader(root)
    return reader


def _series(reader, run_id: str, metric_name: str) -> List[Tuple[int, float]]:
    try:
        return list(reader.series(run_id, metric_name))
    except Exception:
        return []


def _run_config(reader, run_id: str) -> Dict[str, Any]:
    """Recover the run configuration from its RUN_START event."""
    try:
        table = reader.events(run_id, event_types=["RUN_START"], limit=1)
    except Exception:
        return {}
    rows = table.to_pylist()
    if not rows:
        return {}
    import json

    raw = rows[0].get("metadata")
    try:
        payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (TypeError, ValueError):
        return {}
    flat = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
    return flat


def _incidents(root: Path, run_id: str) -> List[Dict[str, Any]]:
    from ..persistence import IncidentStore

    store = IncidentStore(root)
    try:
        return store.list_incidents(run_id)
    except Exception:
        return []


def compare_runs(
    root: Any,
    baseline_run_id: str,
    candidate_run_id: str,
    *,
    alignment: str = "auto",
    metrics: Optional[Iterable[str]] = None,
    regression_threshold: float = 0.05,
) -> RunComparison:
    """Compare two runs and return a structured result.

    ``alignment="auto"`` chooses ``relative`` when the two runs have different step
    counts and ``step`` when they do not, and records the choice.
    """
    root_path = Path(root)
    reader = _read_run(root_path, candidate_run_id)
    try:
        baseline_coverage = reader.coverage(baseline_run_id)
        candidate_coverage = reader.coverage(candidate_run_id)
        baseline_config = _run_config(reader, baseline_run_id)
        candidate_config = _run_config(reader, candidate_run_id)

        baseline_total = _total_steps(reader, baseline_run_id, baseline_config, baseline_coverage)
        candidate_total = _total_steps(reader, candidate_run_id, candidate_config, candidate_coverage)

        if alignment == "auto":
            chosen = "step" if baseline_total == candidate_total else "relative"
        else:
            chosen = alignment
        if chosen == "relative" and not (baseline_total and candidate_total):
            chosen = "step"
            interpolation = "none (absolute step; relative progress unavailable)"
        else:
            interpolation = "none (order statistics over the shared axis; no resampling)"

        comparison = RunComparison(
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            alignment=chosen,
            interpolation=interpolation,
            baseline_steps=baseline_total,
            candidate_steps=candidate_total,
        )

        # -- config -----------------------------------------------------
        for key in sorted(set(baseline_config) | set(candidate_config)):
            before = baseline_config.get(key)
            after = candidate_config.get(key)
            if before != after:
                comparison.config_changes.append(ConfigChange(key, before, after))

        # -- metrics ----------------------------------------------------
        for metric_name in (metrics or COMPARED_METRICS):
            baseline_series = _series(reader, baseline_run_id, metric_name)
            candidate_series = _series(reader, candidate_run_id, metric_name)
            if not baseline_series and not candidate_series:
                continue
            aligned_baseline = _align(baseline_series, baseline_total, baseline_total, chosen)
            aligned_candidate = _align(candidate_series, candidate_total, baseline_total, chosen)
            values_baseline, values_candidate = _overlapping(aligned_baseline, aligned_candidate)
            baseline_median = _median(values_baseline)
            candidate_median = _median(values_candidate)
            relative = None
            if baseline_median not in (None, 0.0) and candidate_median is not None:
                relative = candidate_median / baseline_median - 1.0
            direction = _direction(metric_name, relative, regression_threshold)
            caveats: List[str] = []
            if not values_baseline or not values_candidate:
                caveats.append("no overlapping region between the two runs")
            if len(values_baseline) < 20 or len(values_candidate) < 20:
                caveats.append("fewer than 20 samples on the shared axis")
            comparison.metric_regressions.append(
                MetricRegression(
                    metric_name=metric_name,
                    baseline_median=baseline_median,
                    candidate_median=candidate_median,
                    relative_change=relative,
                    direction=direction,
                    baseline_points=len(baseline_series),
                    candidate_points=len(candidate_series),
                    alignment=chosen,
                    baseline_at_aligned_end=aligned_baseline[-1][1] if aligned_baseline else None,
                    candidate_at_aligned_end=aligned_candidate[-1][1] if aligned_candidate else None,
                    caveats=caveats,
                )
            )

        # -- incidents ---------------------------------------------------
        baseline_incidents = _incidents(root_path, baseline_run_id)
        candidate_incidents = _incidents(root_path, candidate_run_id)
        baseline_types = {i["incident_type"] for i in baseline_incidents}
        candidate_types = {i["incident_type"] for i in candidate_incidents}
        comparison.incident_changes = {
            "baseline_count": len(baseline_incidents),
            "candidate_count": len(candidate_incidents),
            "count_delta": len(candidate_incidents) - len(baseline_incidents),
            "baseline_types": sorted(baseline_types),
            "candidate_types": sorted(candidate_types),
            "new_types": sorted(candidate_types - baseline_types),
            "resolved_types": sorted(baseline_types - candidate_types),
            "baseline_earliest_step": _earliest_step(baseline_incidents),
            "candidate_earliest_step": _earliest_step(candidate_incidents),
        }
        for incident in candidate_incidents:
            if incident["incident_type"] not in baseline_types:
                comparison.new_anomalies.append(
                    {
                        "incident_id": incident["incident_id"],
                        "incident_type": incident["incident_type"],
                        "step": incident.get("detected_step"),
                    }
                )
        for incident in baseline_incidents:
            if incident["incident_type"] not in candidate_types:
                comparison.missing_anomalies.append(
                    {
                        "incident_id": incident["incident_id"],
                        "incident_type": incident["incident_type"],
                        "step": incident.get("detected_step"),
                    }
                )

        # -- performance -------------------------------------------------
        comparison.performance_changes = {
            "baseline_throughput": _final_value(reader, baseline_run_id, "train.tokens_per_sec"),
            "candidate_throughput": _final_value(reader, candidate_run_id, "train.tokens_per_sec"),
            "baseline_step_time_ms": _median_metric(reader, baseline_run_id, "train.step_time"),
            "candidate_step_time_ms": _median_metric(reader, candidate_run_id, "train.step_time"),
            "baseline_ordering_confidence": _ordering_confidences(baseline_incidents),
            "candidate_ordering_confidence": _ordering_confidences(candidate_incidents),
        }

        # -- comparability ------------------------------------------------
        caveats: List[str] = []
        if comparison.baseline_steps != comparison.candidate_steps:
            caveats.append(
                "runs differ in length ({} vs {} steps); compared on relative progress".format(
                    comparison.baseline_steps, comparison.candidate_steps
                )
            )
        for change in comparison.config_changes:
            if change.key in ("model_id", "world_size", "precision", "optimizer"):
                caveats.append(
                    "{} differs ({} vs {}): the runs may not be comparable at all".format(
                        change.key, change.baseline, change.candidate
                    )
                )
        if baseline_coverage.files_skipped or candidate_coverage.files_skipped:
            caveats.append("some event files were unreadable and were skipped")
        if not (baseline_total and candidate_total):
            caveats.append("total step count unknown for at least one run")
        comparison.comparability = {
            "comparable": not any("may not be comparable" in c for c in caveats),
            "caveats": caveats,
            "baseline_row_count": baseline_coverage.row_count,
            "candidate_row_count": candidate_coverage.row_count,
        }
        return comparison
    finally:
        reader.close()


def _total_steps(reader, run_id: str, config: Dict[str, Any], coverage) -> Optional[int]:
    for key in ("steps", "total_steps", "num_steps"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    if coverage.step_max is not None:
        return int(coverage.step_max)
    return None


def _direction(metric_name: str, relative: Optional[float], threshold: float) -> str:
    if relative is None:
        return "unknown"
    if abs(relative) < threshold:
        return "neutral"
    if metric_name in LOWER_IS_BETTER:
        return "worse" if relative > 0 else "better"
    if metric_name in HIGHER_IS_BETTER:
        return "worse" if relative < 0 else "better"
    return "up" if relative > 0 else "down"


def _earliest_step(incidents: Sequence[Dict[str, Any]]) -> Optional[int]:
    steps = [i.get("detected_step") for i in incidents if i.get("detected_step") is not None]
    return min(steps) if steps else None


def _final_value(reader, run_id: str, metric_name: str) -> Optional[float]:
    series = _series(reader, run_id, metric_name)
    return series[-1][1] if series else None


def _median_metric(reader, run_id: str, metric_name: str) -> Optional[float]:
    return _median([v for _, v in _series(reader, run_id, metric_name)])


def _ordering_confidences(incidents: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for incident in incidents:
        key = str(incident.get("ordering_confidence"))
        counts[key] = counts.get(key, 0) + 1
    return counts
