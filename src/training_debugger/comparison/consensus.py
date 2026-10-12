"""Multi-run divergence: find the outlier in a batch.

`localize_divergence` compares two runs and needs one designated as the reference. The
practical question is usually the other one: *"here are forty runs from last night — which
one is wrong?"* Nobody wants to name a reference, and naming a bad one poisons the answer.

This module answers it by comparing **every run against every other** and reporting:

* which run disagrees with the most others (**the outlier**),
* the step at which that disagreement starts (**the consensus pivot**),
* and how many of its peers support that pivot (**the support**).

A single reference can be wrong. **A majority cannot all be wrong in the same direction**,
which is why consensus is more robust than any pairwise comparison — and why the support
count is reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .divergence import (
    DEFAULT_BASELINE_WINDOW, DEFAULT_MIN_COMMON_STEPS, DEFAULT_PERSISTENCE,
    DEFAULT_Z_THRESHOLD, DivergenceReport, localize_divergence,
)

__all__ = ["RunOutlier", "ConsensusReport", "find_outlier", "render_consensus"]

CONSENSUS_RULE_VERSION = "v0.1.8-1.0"

#: How close two pivots must be to count as the same event, in steps. Two runs that
#: diverge at 90 and 92 are describing one departure, not two.
DEFAULT_PIVOT_TOLERANCE = 2


@dataclass
class RunOutlier:
    run_id: str
    #: How many peers this run diverged from.
    divergence_count: int = 0
    peers: int = 0
    #: Median step at which it diverged, across those peers.
    median_pivot_step: Optional[int] = None
    pivot_steps: List[int] = field(default_factory=list)
    #: The metric most often named as the first departure.
    dominant_metric: Optional[str] = None

    @property
    def divergence_rate(self) -> Optional[float]:
        return self.divergence_count / self.peers if self.peers else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "divergence_count": self.divergence_count,
            "peers": self.peers,
            "divergence_rate": self.divergence_rate,
            "median_pivot_step": self.median_pivot_step,
            "pivot_steps": list(self.pivot_steps),
            "dominant_metric": self.dominant_metric,
        }


@dataclass
class ConsensusReport:
    runs: List[str] = field(default_factory=list)
    pairs_compared: int = 0
    #: Every run, ranked by how much it disagrees with its peers.
    ranking: List[RunOutlier] = field(default_factory=list)
    outlier: Optional[RunOutlier] = None
    #: Peers that independently localised a pivot close to the outlier's median.
    supporting_peers: int = 0
    #: ``SINGLE_OUTLIER`` | ``DIFFUSE`` | ``NONE``.
    #:
    #: ``DIFFUSE`` is the important one and it was found on real data: scanning six
    #: **healthy** runs with different seeds reported an outlier. It was not one. Runs
    #: drawn independently differ from each other legitimately, so there is no majority
    #: for anything to be an outlier *from*. Consensus needs a coherent agreeing
    #: majority, and that has to be checked rather than assumed.
    pattern: str = "NONE"
    #: Fraction of pairs among the NON-outlier runs that did not diverge. High means a
    #: coherent majority exists; low means the batch is simply varied.
    consensus_strength: Optional[float] = None
    rule_version: str = CONSENSUS_RULE_VERSION
    verdict: str = "INSUFFICIENT_EVIDENCE"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runs": list(self.runs),
            "pairs_compared": self.pairs_compared,
            "ranking": [r.to_dict() for r in self.ranking],
            "outlier": self.outlier.to_dict() if self.outlier else None,
            "supporting_peers": self.supporting_peers,
            "pattern": self.pattern,
            "consensus_strength": self.consensus_strength,
            "rule_version": self.rule_version,
            "verdict": self.verdict,
            "reason": self.reason,
        }


def _median(values: Sequence[int]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return float(ordered[n // 2]) if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def find_outlier(
    series_by_run: Dict[str, Dict[str, Sequence[Tuple[int, float]]]],
    *,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    persistence: int = DEFAULT_PERSISTENCE,
    min_common_steps: int = DEFAULT_MIN_COMMON_STEPS,
    pivot_tolerance: int = DEFAULT_PIVOT_TOLERANCE,
    include_non_deterministic: bool = False,
) -> ConsensusReport:
    """Rank a batch of runs by how much each disagrees with the others.

    Every unordered pair is compared once. A run's score is the number of peers it
    diverged from, and the pivot it reports with them.

    **Ties are reported as ties.** If two runs disagree with each other and agree with
    everyone else, both are outliers and neither is "the" answer; the ranking shows that
    rather than picking one.
    """
    run_ids = sorted(series_by_run)
    report = ConsensusReport(runs=run_ids)
    if len(run_ids) < 2:
        report.verdict = "INSUFFICIENT_EVIDENCE"
        report.reason = "a consensus needs at least two runs"
        return report

    divergences: Dict[str, List[Tuple[str, Optional[int], Optional[str]]]] = {
        run_id: [] for run_id in run_ids
    }
    for left, right in combinations(run_ids, 2):
        result = localize_divergence(
            left, right, series_by_run[left], series_by_run[right],
            baseline_window=baseline_window, z_threshold=z_threshold,
            persistence=persistence, min_common_steps=min_common_steps,
            include_non_deterministic=include_non_deterministic,
        )
        report.pairs_compared += 1
        if result.first_divergence is None:
            continue
        step = result.first_divergence.step
        metric = result.first_divergence.metric_name
        # A divergence between two runs is evidence about BOTH of them: we do not know
        # which one moved. It is scored symmetrically and the report says so.
        divergences[left].append((right, step, metric))
        divergences[right].append((left, step, metric))

    # Exclude the eventual top run and ask whether the REST agree with each other.
    # A real outlier stands against a coherent majority; if the remainder also disagree
    # among themselves, the batch is varied rather than one-of-these-is-different.
    def _rest_agreement(excluded: str) -> Tuple[int, int]:
        agreeing = total = 0
        for left, right in combinations([r for r in run_ids if r != excluded], 2):
            total += 1
            pair = localize_divergence(
                left, right, series_by_run[left], series_by_run[right],
                baseline_window=baseline_window, z_threshold=z_threshold,
                persistence=persistence, min_common_steps=min_common_steps,
                include_non_deterministic=include_non_deterministic,
            )
            if pair.first_divergence is None:
                agreeing += 1
        return agreeing, total

    ranking: List[RunOutlier] = []
    for run_id in run_ids:
        peers = len(run_ids) - 1
        entries = divergences[run_id]
        steps = [step for _, step, _ in entries if step is not None]
        metrics = [metric for _, _, metric in entries if metric]
        dominant = None
        if metrics:
            dominant = max(sorted(set(metrics)), key=metrics.count)
        ranking.append(RunOutlier(
            run_id=run_id, divergence_count=len(entries), peers=peers,
            median_pivot_step=_median(steps),
            pivot_steps=sorted(steps), dominant_metric=dominant,
        ))
    ranking.sort(key=lambda r: (-r.divergence_count, r.run_id))
    report.ranking = ranking

    if not ranking or ranking[0].divergence_count == 0:
        report.pattern = "NONE"
        report.verdict = "NO_OUTLIER"
        report.reason = (
            "across {} run pairs no pair diverged beyond the agreement it established "
            "for itself; the batch is internally consistent".format(report.pairs_compared)
        )
        return report

    outlier = ranking[0]
    report.outlier = outlier
    agreeing, total = _rest_agreement(outlier.run_id)
    report.consensus_strength = (agreeing / total) if total else None

    centre = outlier.median_pivot_step
    support = 0
    if centre is not None:
        support = sum(
            1 for step in outlier.pivot_steps
            if abs(step - centre) <= pivot_tolerance
        )
    report.supporting_peers = support

    # The discriminator is whether the peers agree on **when** the divergence happened,
    # not merely on whether there was one.
    #
    # A genuine outlier departs at the SAME step from every peer, because there is one
    # event. Independently sampled runs diverge at scattered steps, because the
    # differences are accumulated variance rather than an event. Measured:
    #
    #   one faulty run among five clean   -> 5/5 peers localise within 2 steps
    #   six healthy runs, three seeds     -> 0/4 peers localise within 2 steps
    #
    # An earlier version keyed on pairwise agreement instead and still reported an
    # outlier on six healthy runs.
    if outlier.divergence_count >= 2:
        support_ratio = support / float(outlier.divergence_count)
        if support_ratio < 0.5:
            report.pattern = "DIFFUSE"
            report.verdict = "NO_SINGLE_OUTLIER"
            report.reason = (
                "no single run stands out. {} diverges from {} peer(s), but only {} of "
                "them localise the departure within {} step(s) of each other -- the "
                "peers do not agree on WHEN it happened. Scattered pivots are the "
                "signature of independently sampled runs (different seeds or "
                "configurations) rather than one faulty run, because accumulated "
                "variance has no single onset. Compare like with like: hold the seed "
                "and configuration fixed and vary one factor."
                .format(outlier.run_id, outlier.divergence_count, support,
                        pivot_tolerance)
            )
            return report
    report.pattern = "SINGLE_OUTLIER"
    ties = [r for r in ranking[1:] if r.divergence_count == outlier.divergence_count]
    report.verdict = "OUTLIER_FOUND"
    report.reason = (
        "{} diverges from {} of its {} peer(s), most often at step {} on {}. {} peer(s) "
        "localise a pivot within {} step(s) of that. This is a run to investigate, NOT a "
        "faulty run: the comparison cannot say which side of a divergence moved."
        .format(outlier.run_id, outlier.divergence_count, outlier.peers,
                None if centre is None else int(centre),
                outlier.dominant_metric or "an unnamed metric",
                support, pivot_tolerance)
    )
    if ties:
        report.reason += (
            " NOTE: {} other run(s) diverge from the same number of peers — this is a "
            "tie, not a single answer.".format(len(ties))
        )
    return report


def render_consensus(report: ConsensusReport) -> str:
    lines: List[str] = []
    add = lines.append
    add("BATCH DIVERGENCE SCAN")
    add("")
    add("  runs: {}   pairs compared: {}".format(len(report.runs), report.pairs_compared))
    add("")
    add("  {:<34} {:>8} {:>10} {:>18}".format("run", "diverged", "of peers", "median pivot"))
    for entry in report.ranking[:10]:
        add("  {:<34} {:>8} {:>10} {:>18}".format(
            entry.run_id[:34], entry.divergence_count, entry.peers,
            "-" if entry.median_pivot_step is None else int(entry.median_pivot_step)))
    if len(report.ranking) > 10:
        add("  ... {} more".format(len(report.ranking) - 10))
    add("")
    add("  VERDICT: {}".format(report.verdict))
    add("")
    for chunk in _wrap(report.reason, 74):
        add("  " + chunk)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> List[str]:
    words = text.split()
    out: List[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            out.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        out.append(current)
    return out
