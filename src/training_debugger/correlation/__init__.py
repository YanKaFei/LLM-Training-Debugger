"""General lag-distribution correlation.

Correlation is not causation. This package measures whether one signal
*repeatedly precedes* another, across occurrences and across runs, and reports the
pairs that showed nothing with the same prominence as the ones that did.
"""

from __future__ import annotations

from .engine import (
    ALLOWED_PHRASINGS,
    CAUSAL_VERBS,
    CORRELATION_METRICS,
    MIN_OCCURRENCES,
    CorrelationEvidence,
    CorrelationReport,
    assert_causal_language_absent,
    benjamini_hochberg,
    build_correlation_report,
    correlate_series,
    lag_distribution,
)

__all__ = [
    "CorrelationEvidence",
    "CorrelationReport",
    "correlate_series",
    "build_correlation_report",
    "lag_distribution",
    "benjamini_hochberg",
    "assert_causal_language_absent",
    "CORRELATION_METRICS",
    "ALLOWED_PHRASINGS",
    "CAUSAL_VERBS",
    "MIN_OCCURRENCES",
]
