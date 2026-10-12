"""Cross-run comparison: structured, alignment-explicit, no prose.
"""

from __future__ import annotations

from .consensus import (
    CONSENSUS_RULE_VERSION,
    ConsensusReport,
    RunOutlier,
    find_outlier,
    render_consensus,
)
from .divergence import (
    DIVERGENCE_RULE_VERSION,
    DivergencePoint,
    DivergenceReport,
    localize_divergence,
    render_divergence,
    render_divergence_diagnosis,
    is_deterministic_metric,
    NON_DETERMINISTIC_PREFIXES,
)
from .runner import (
    COMPARED_METRICS,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    ConfigChange,
    MetricRegression,
    RunComparison,
    compare_runs,
)

__all__ = [
    "compare_runs",
    "RunComparison",
    "DivergencePoint",
    "DivergenceReport",
    "localize_divergence",
    "render_divergence",
    "render_divergence_diagnosis",
    "DIVERGENCE_RULE_VERSION",
    "ConsensusReport",
    "RunOutlier",
    "find_outlier",
    "render_consensus",
    "CONSENSUS_RULE_VERSION",
    "is_deterministic_metric",
    "NON_DETERMINISTIC_PREFIXES",
    "MetricRegression",
    "ConfigChange",
    "COMPARED_METRICS",
    "LOWER_IS_BETTER",
    "HIGHER_IS_BETTER",
]
