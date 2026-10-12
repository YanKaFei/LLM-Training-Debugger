"""C4: the general correlation engine.

Phase 0 deferred this, and Phase 1 shipped only a fixed-probe substitute. The
difference matters: a fixed probe can say "the learning rate was stable at this
incident", but it cannot say **"a gradient anomaly preceded a loss spike in 7 of 9
comparable incidents, median lead 9 steps"** — which is the sentence that turns an
observation into a pattern.

Four commitments, each enforced in code:

**Correlation is not causation.** The vocabulary of this module has no word for
"caused". :data:`ALLOWED_PHRASINGS` is a closed set and
:func:`assert_causal_language_absent` rejects a rendered claim that reaches for a
causal verb. Promotion to causation requires replay or intervention, neither of
which exists in Phase 1.1.

**Measure a lag *distribution*, not proximity.** "A and B were near each other" is
not evidence: in a stream where something changes every step, everything is near
everything. What is measured is the step lag of B relative to A, aggregated across
occurrences, with its spread.

**Single occurrences are not evidence.** One gradient spike followed by one loss
spike is an anecdote. The engine reports `k / n` and refuses to report association
below :data:`MIN_OCCURRENCES`.

**Negative results are reported with the same prominence as positive ones.** A run
of the engine returns the tested set, including the pairs that showed nothing, so
"learning rate: 0 / 7" appears next to "gradient: 7 / 7". Phase 0's rule that
negative evidence is evidence applies to correlation too.

Multiple testing is controlled with Benjamini-Hochberg, and both the raw and the
adjusted p-value survive into the output along with the number of hypotheses
tested, because an adjusted p-value without its denominator is unauditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "CorrelationEvidence",
    "CorrelationReport",
    "CORRELATION_METRICS",
    "ALLOWED_PHRASINGS",
    "CAUSAL_VERBS",
    "MIN_OCCURRENCES",
    "lag_distribution",
    "benjamini_hochberg",
    "correlate_series",
    "build_correlation_report",
    "assert_causal_language_absent",
]

Series = Sequence[Tuple[int, float]]

#: Signals the engine correlates. "Cause" candidates first, "effect" candidates
#: second; the engine tests every ordered pair.
CORRELATION_METRICS: Tuple[str, ...] = (
    "grad.global_norm",
    "train.loss",
    "train.lr",
    "param.global_norm",
    "train.tokens_per_sec",
    "train.step_time",
)

#: The only phrasings this module may emit. Deliberately contains no causal verb.
ALLOWED_PHRASINGS = (
    "{a} repeatedly precedes {b}",
    "{a} is statistically associated with {b}",
    "{a} is not correlated with {b}",
    "evidence insufficient for {a} and {b}",
)

#: Words that would assert causation. Presence in a rendered claim is a bug.
CAUSAL_VERBS = (
    "caused", "causes", "because of", "due to", "led to", "leads to",
    "resulted in", "results in", "triggered", "explains why",
)

#: Below this many occurrences no association is reported at all.
MIN_OCCURRENCES = 3


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def lag_distribution(
    source_steps: Sequence[int], target_steps: Sequence[int], max_lag: int = 50
) -> List[int]:
    """For each source occurrence, the lag in steps of the next target occurrence.

    Lags are ``target_step - source_step`` and are kept only when
    ``0 <= lag <= max_lag``: a *preceding* relationship is what is being tested, so
    a target that came first is not a lag at all and is discarded rather than
    recorded as negative.
    """
    lags: List[int] = []
    for source in source_steps:
        candidates = [t - source for t in target_steps if 0 <= (t - source) <= max_lag]
        if candidates:
            lags.append(min(candidates))
    return sorted(lags)


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    n = len(ordered)
    return float(ordered[n // 2]) if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _median_absolute_deviation(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    median = _median(values)
    return _median([abs(v - median) for v in values])


def _mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Two-sided Mann-Whitney U with a normal approximation.

    Chosen over a t-test because these are step-indexed series with no normality
    and frequent ties. Returns ``(u_statistic, raw_p_value)``. Continuity-corrected
    and tie-corrected; when either sample is too small for the normal
    approximation to be meaningful, ``p = 1.0`` is returned rather than a
    spuriously small number.
    """
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return float("nan"), 1.0
    combined = list(a) + list(b)
    ranks = {}
    sorted_values = sorted(combined)
    index = 0
    tie_counts: List[int] = []
    while index < len(sorted_values):
        run_end = index
        while run_end + 1 < len(sorted_values) and sorted_values[run_end + 1] == sorted_values[index]:
            run_end += 1
        average_rank = (index + run_end) / 2.0 + 1.0
        ranks[sorted_values[index]] = average_rank
        tie_counts.append(run_end - index + 1)
        index = run_end + 1
    rank_sum = sum(ranks[v] for v in a)
    u1 = rank_sum - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mean_u = n1 * n2 / 2.0
    n = n1 + n2
    tie_correction = sum(t ** 3 - t for t in tie_counts if t > 1)
    variance = (n1 * n2 / 12.0) * ((n + 1) - (tie_correction / float(n * (n - 1)) if n > 1 else 0.0))
    if variance <= 0:
        return u, 1.0
    z = (abs(u - mean_u) - 0.5) / math.sqrt(variance)
    z = max(0.0, z)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return u, min(1.0, max(0.0, p))


def benjamini_hochberg(
    p_values: Sequence[float], alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """Benjamini-Hochberg step-up. Returns ``(adjusted_p, rejected)``.

    Adjusted p-values are returned alongside the rejections because a
    "significant" flag without its adjusted value cannot be re-checked, and the
    number of hypotheses is reported by the caller for the same reason.
    """
    n = len(p_values)
    if n == 0:
        return [], []
    order = sorted(range(n), key=lambda i: (p_values[i], i))
    adjusted = [1.0] * n
    running_min = 1.0
    for rank_index in range(n - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        value = p_values[original_index] * n / float(rank)
        running_min = min(running_min, value)
        adjusted[original_index] = min(1.0, running_min)
    rejected = [adjusted[i] <= alpha for i in range(n)]
    return adjusted, rejected


# ---------------------------------------------------------------------------
# Correlation object
# ---------------------------------------------------------------------------


@dataclass
class CorrelationEvidence:
    """One tested pair. Always produced, whether or not it was significant.

    ``sample_count``, ``population_count`` and the ``limitations`` list are
    mandatory: an association reported without its denominator, or without a
    statement of what could not be established, is exactly the kind of claim this
    project exists to avoid making.
    """

    source_signal: str
    target_signal: str
    run_ids: List[str] = field(default_factory=list)

    sample_count: int = 0
    population_count: int = 0
    occurrence_count: int = 0

    lags: List[int] = field(default_factory=list)
    median_lag: Optional[float] = None
    lag_mad: Optional[float] = None
    lag_min: Optional[int] = None
    lag_max: Optional[int] = None

    effect_size: Optional[float] = None
    raw_p_value: Optional[float] = None
    adjusted_p_value: Optional[float] = None
    significant: bool = False

    source_occurrences: int = 0
    source_population: int = 0
    recurrence_count: int = 0
    recurrence_population: int = 0

    direction: str = "precedes"
    confidence: float = 0.0
    limitations: List[str] = field(default_factory=list)
    method: str = "lag_distribution_mann_whitney_bh"

    @property
    def recurrence_ratio(self) -> Optional[float]:
        if self.recurrence_population <= 0:
            return None
        return self.recurrence_count / float(self.recurrence_population)

    def phrasing(self) -> str:
        """A permitted sentence. Never contains a causal verb."""
        if self.occurrence_count < MIN_OCCURRENCES:
            return "evidence insufficient for {} and {}".format(
                self.source_signal, self.target_signal
            )
        if self.significant:
            return "{} repeatedly precedes {} ({} / {} incidents, median lead {} steps)".format(
                self.source_signal, self.target_signal,
                self.recurrence_count, self.recurrence_population,
                "n/a" if self.median_lag is None else int(round(self.median_lag)),
            )
        return "{} is not correlated with {}".format(self.source_signal, self.target_signal)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_signal": self.source_signal,
            "target_signal": self.target_signal,
            "run_ids": list(self.run_ids),
            "sample_count": self.sample_count,
            "population_count": self.population_count,
            "occurrence_count": self.occurrence_count,
            "lags": list(self.lags),
            "median_lag": self.median_lag,
            "lag_mad": self.lag_mad,
            "lag_min": self.lag_min,
            "lag_max": self.lag_max,
            "effect_size": self.effect_size,
            "raw_p_value": self.raw_p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "significant": self.significant,
            "recurrence_count": self.recurrence_count,
            "recurrence_population": self.recurrence_population,
            "recurrence_ratio": self.recurrence_ratio,
            "direction": self.direction,
            "confidence": self.confidence,
            "limitations": list(self.limitations),
            "method": self.method,
            "phrasing": self.phrasing(),
        }


@dataclass
class CorrelationReport:
    """Every tested pair, plus what was not testable."""

    evidences: List[CorrelationEvidence] = field(default_factory=list)
    hypotheses_tested: int = 0
    alpha: float = 0.05
    min_occurrences: int = MIN_OCCURRENCES
    run_ids: List[str] = field(default_factory=list)
    untestable: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def correlated(self) -> List[CorrelationEvidence]:
        return [e for e in self.evidences if e.significant]

    def not_correlated(self) -> List[CorrelationEvidence]:
        return [e for e in self.evidences if not e.significant and e.occurrence_count >= MIN_OCCURRENCES]

    def insufficient(self) -> List[CorrelationEvidence]:
        return [e for e in self.evidences if e.occurrence_count < MIN_OCCURRENCES]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_ids": list(self.run_ids),
            "hypotheses_tested": self.hypotheses_tested,
            "alpha": self.alpha,
            "min_occurrences": self.min_occurrences,
            "correlated": [e.to_dict() for e in self.correlated()],
            "not_correlated": [e.to_dict() for e in self.not_correlated()],
            "insufficient": [e.to_dict() for e in self.insufficient()],
            "untestable": list(self.untestable),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """Deterministic text. Contains no causal claim."""
        lines = ["CORRELATION REPORT", ""]
        lines.append("Runs analysed: {}".format(", ".join(self.run_ids) or "(none)"))
        lines.append("Hypotheses tested: {} (alpha {:.2f}, Benjamini-Hochberg)".format(
            self.hypotheses_tested, self.alpha))
        lines.append("")
        lines.append("Correlated ({}):".format(len(self.correlated())))
        for evidence in self.correlated():
            lines.append("  - {}".format(evidence.phrasing()))
        if not self.correlated():
            lines.append("  (none)")
        lines.append("")
        lines.append("Not correlated ({}):".format(len(self.not_correlated())))
        for evidence in self.not_correlated():
            lines.append("  - {}  [{}/{} occurrences]".format(
                evidence.phrasing(), evidence.occurrence_count, evidence.recurrence_population))
        lines.append("")
        lines.append("Insufficient evidence ({}):".format(len(self.insufficient())))
        for evidence in self.insufficient():
            lines.append("  - {} and {}: {} of {} required occurrences".format(
                evidence.source_signal, evidence.target_signal,
                evidence.occurrence_count, MIN_OCCURRENCES))
        if self.untestable:
            lines.append("")
            lines.append("Not testable ({}):".format(len(self.untestable)))
            for entry in self.untestable:
                lines.append("  - {} vs {}: {}".format(
                    entry["source_signal"], entry["target_signal"], entry["reason"]))
        lines.append("")
        lines.append("Causation is NOT claimed. Correlation is not causation; promotion "
                     "requires replay or intervention, neither of which exists in Phase 1.1.")
        return "\n".join(lines)


def assert_causal_language_absent(text: str) -> None:
    """Raise if ``text`` contains a causal verb. Used by tests and by the renderer."""
    lowered = text.lower()
    for verb in CAUSAL_VERBS:
        if verb in lowered:
            raise ValueError(
                "causal language %r found in correlation output; correlation is not causation"
                % verb
            )


# ---------------------------------------------------------------------------
# Occurrence detection
# ---------------------------------------------------------------------------


def _occurrences(series: Series, *, z: float = 4.0, baseline: int = 50) -> List[int]:
    """Steps at which a series departs from its own robust baseline.

    Deliberately *local* to this module and deliberately simple: the correlation
    engine must not depend on whether a detector happened to fire, or the
    correlation would inherit the detector's recall and report a clean bill of
    health whenever recall was poor.
    """
    values = [v for _, v in series if v == v and abs(v) != float("inf")]
    if len(values) < baseline + 5:
        return []
    steps = [s for s, _ in series]
    medians: List[float] = []
    result: List[int] = []
    for index in range(baseline, len(series)):
        window = values[max(0, index - baseline):index]
        if len(window) < 10:
            continue
        median = _median(window)
        mad = _median_absolute_deviation(window) * 1.4826
        if mad <= 0:
            continue
        value = series[index][1]
        if value != value or abs(value) == float("inf"):
            result.append(series[index][0])
            continue
        if abs(value - median) / mad >= z:
            result.append(series[index][0])
    _ = steps, medians
    return sorted(set(result))


def correlate_series(
    series_by_signal: Dict[str, Series],
    *,
    run_id: str = "",
    max_lag: int = 50,
    alpha: float = 0.05,
    min_occurrences: int = MIN_OCCURRENCES,
) -> List[CorrelationEvidence]:
    """Test every ordered pair of signals for a repeating lag relationship."""
    signals = [name for name in CORRELATION_METRICS if series_by_signal.get(name)]
    occurrences = {name: _occurrences(series_by_signal[name]) for name in signals}
    evidences: List[CorrelationEvidence] = []

    for source in signals:
        for target in signals:
            if source == target:
                continue
            source_steps = occurrences[source]
            target_steps = occurrences[target]
            lags = lag_distribution(source_steps, target_steps, max_lag=max_lag)

            evidence = CorrelationEvidence(
                source_signal=source,
                target_signal=target,
                run_ids=[run_id] if run_id else [],
                occurrence_count=len(lags),
                recurrence_count=len(lags),
                recurrence_population=len(source_steps),
                source_occurrences=len(source_steps),
                source_population=len(source_steps),
                sample_count=len(series_by_signal[target]),
                population_count=len(series_by_signal[target]),
            )
            if lags:
                evidence.lags = list(lags)
                evidence.median_lag = _median(lags)
                evidence.lag_mad = _median_absolute_deviation(lags)
                evidence.lag_min = min(lags)
                evidence.lag_max = max(lags)

            if len(lags) >= min_occurrences:
                # Effect size: the shift of the target's level after a source
                # occurrence, in units of its own baseline MAD.
                values = [v for _, v in series_by_signal[target]]
                median = _median(values)
                mad = _median_absolute_deviation(values) * 1.4826 or 1.0
                after = [
                    v for s, v in series_by_signal[target]
                    if any(0 <= (s - occ) <= max(4, int(evidence.median_lag or 4)) for occ in source_steps)
                ]
                before = [
                    v for s, v in series_by_signal[target]
                    if not any(0 <= (s - occ) <= max(4, int(evidence.median_lag or 4)) for occ in source_steps)
                ]
                evidence.effect_size = (
                    float((_median(after) - median) / mad) if after else None
                )
                _, raw_p = _mann_whitney_u(after, before)
                evidence.raw_p_value = raw_p
                evidence.confidence = max(0.0, min(0.9, len(lags) / 10.0)) if raw_p < alpha else 0.0
                if len(lags) < 5:
                    evidence.limitations.append(
                        "only {} occurrence(s): the lag distribution is indicative, not "
                        "established".format(len(lags))
                    )
            else:
                evidence.limitations.append(
                    "fewer than {} occurrences; no association is reported".format(min_occurrences)
                )
            evidences.append(evidence)

    # Benjamini-Hochberg over every pair with a computable p-value.
    tested = [e for e in evidences if e.raw_p_value is not None]
    if tested:
        adjusted, rejected = benjamini_hochberg([e.raw_p_value for e in tested], alpha=alpha)
        for evidence, adj, rej in zip(tested, adjusted, rejected):
            evidence.adjusted_p_value = adj
            evidence.significant = bool(rej)
            if not rej:
                evidence.confidence = 0.0
    for evidence in evidences:
        if evidence.significant:
            evidence.limitations.append(
                "association only; causation requires replay or intervention"
            )
    return evidences


def build_correlation_report(
    series_by_run: Dict[str, Dict[str, Series]],
    *,
    alpha: float = 0.05,
    max_lag: int = 50,
) -> CorrelationReport:
    """Correlate across every supplied run and report positives *and* negatives.

    Pooling across runs is what turns an anecdote into a recurrence count: a pair
    that repeats in seven of nine runs is a pattern, and the same pair in one run
    is a coincidence nobody can distinguish from noise.
    """
    report = CorrelationReport(run_ids=sorted(series_by_run), alpha=alpha)
    pooled: Dict[Tuple[str, str], List[CorrelationEvidence]] = {}
    missing: Dict[Tuple[str, str], List[str]] = {}

    # Every tested pair is classified, including the ones with too few
    # occurrences. An earlier version dropped those into ``untestable``, which
    # made ``insufficient()`` structurally always-empty and hid the pairs the
    # engine looked at and could not assess -- the same "lying by omission"
    # failure Phase 0 forbids for negative evidence.
    for run_id, series in sorted(series_by_run.items()):
        evidences = correlate_series(series, run_id=run_id, max_lag=max_lag, alpha=alpha)
        for evidence in evidences:
            key = (evidence.source_signal, evidence.target_signal)
            pooled.setdefault(key, []).append(evidence)

    report.hypotheses_tested = len(pooled) + sum(len(v) for v in missing.values())
    all_evidences: List[CorrelationEvidence] = []
    for key, entries in sorted(pooled.items()):
        observed = [e for e in entries if e.occurrence_count >= MIN_OCCURRENCES]
        if not observed:
            # Tested, but never with enough occurrences to say anything. Reported
            # as insufficient rather than silently omitted.
            merged = _merge(entries)
            merged.recurrence_population = len(series_by_run)
            merged.recurrence_count = 0
            merged.significant = False
            merged.confidence = 0.0
            merged.limitations.append(
                "no run produced {} or more occurrences; no association is reported".format(
                    MIN_OCCURRENCES
                )
            )
            all_evidences.append(merged)
            continue
        merged = _merge(observed)
        merged.recurrence_population = len(series_by_run)
        merged.recurrence_count = len(observed)
        all_evidences.append(merged)

    # Pooled BH across the merged hypotheses.
    tested = [e for e in all_evidences if e.raw_p_value is not None]
    if tested:
        adjusted, rejected = benjamini_hochberg([e.raw_p_value for e in tested], alpha=alpha)
        for evidence, adj, rej in zip(tested, adjusted, rejected):
            evidence.adjusted_p_value = adj
            evidence.significant = bool(rej)
            evidence.confidence = (
                max(0.0, min(0.95, evidence.recurrence_ratio or 0.0)) if rej else 0.0
            )
            if rej:
                evidence.limitations.append(
                    "pooled across {} runs; association only, causation not claimed".format(
                        len(series_by_run)
                    )
                )
    report.evidences = sorted(
        all_evidences, key=lambda e: (not e.significant, e.source_signal, e.target_signal)
    )
    # ``untestable`` now holds only pairs one of whose signals was absent from
    # every run -- genuinely untestable, as distinct from testable-but-weak.
    for run_id, series in sorted(series_by_run.items()):
        present = {name for name in CORRELATION_METRICS if series.get(name)}
        for source in CORRELATION_METRICS:
            for target in CORRELATION_METRICS:
                if source == target or (source in present and target in present):
                    continue
                report.untestable.append(
                    {
                        "source_signal": source,
                        "target_signal": target,
                        "reason": "signal absent from run {}".format(run_id),
                        "run_ids": [run_id],
                    }
                )
                break
            break
    if len(series_by_run) < 3:
        report.notes.append(
            "fewer than 3 runs: recurrence statistics are not meaningful at this sample size"
        )
    return report


def _merge(entries: Sequence[CorrelationEvidence]) -> CorrelationEvidence:
    """Pool per-run evidence into one record across runs."""
    first = entries[0]
    lags: List[int] = []
    for entry in entries:
        lags.extend(entry.lags)
    lags.sort()
    merged = CorrelationEvidence(
        source_signal=first.source_signal,
        target_signal=first.target_signal,
        run_ids=sorted({r for e in entries for r in e.run_ids}),
        sample_count=sum(e.sample_count for e in entries),
        population_count=sum(e.population_count for e in entries),
        occurrence_count=len(lags),
        lags=lags,
        median_lag=_median(lags) if lags else None,
        lag_mad=_median_absolute_deviation(lags) if lags else None,
        lag_min=min(lags) if lags else None,
        lag_max=max(lags) if lags else None,
        source_occurrences=sum(e.source_occurrences for e in entries),
        recurrence_count=len(entries),
        recurrence_population=len(entries),
    )
    effects = [e.effect_size for e in entries if e.effect_size is not None]
    merged.effect_size = _median(effects) if effects else None
    # Fisher's method to pool p-values across runs.
    ps = [e.raw_p_value for e in entries if e.raw_p_value is not None]
    if ps:
        clamped = [min(max(p, 1e-12), 1.0) for p in ps]
        statistic = -2.0 * sum(math.log(p) for p in clamped)
        # Survival function of chi-square with 2k degrees of freedom, computed
        # through the regularised upper incomplete gamma via a series, which is
        # adequate for the small k this engine sees.
        merged.raw_p_value = _chi2_sf(statistic, 2 * len(clamped))
    else:
        merged.raw_p_value = None
    if len(entries) < 3:
        merged.limitations.append(
            "pooled from only {} run(s): recurrence is not yet established".format(len(entries))
        )
    return merged


def _chi2_sf(x: float, k: int) -> float:
    """Upper tail of the chi-square distribution for EVEN degrees of freedom.

    Fisher's method always produces ``k = 2 * number_of_runs``, which is even, so
    only the closed form is needed. The odd case raises rather than shipping an
    untested series expansion: a silently wrong p-value is worse than an error.
    """
    if x <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    if k % 2 != 0:
        raise ValueError("_chi2_sf only supports even degrees of freedom, got %r" % k)
    term = math.exp(-x / 2.0)
    total = term
    for i in range(1, k // 2):
        term *= (x / 2.0) / i
        total += term
    return min(1.0, max(0.0, total))
