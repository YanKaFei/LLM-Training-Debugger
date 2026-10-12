"""Presentation layer: a filtered, fail-open window onto the run.

Enforces the Phase 0 rule in code rather than in convention -- Parquet is the
telemetry truth, the incident store is the incident truth, and a presentation sink
sees only an allow-listed subset of high-level metrics.
"""

from __future__ import annotations

from .trackio_sink import (
    FORBIDDEN_PREFIXES,
    PRESENTED_METRICS,
    PRESENTED_SUMMARY_KEYS,
    FaultySink,
    NullSink,
    PresentationBridge,
    PresentationSink,
    RecordingSink,
    SlowSink,
    TrackioSink,
    build_sink,
    is_presentable,
    trackio_available,
)

__all__ = [
    "PresentationSink",
    "PresentationBridge",
    "NullSink",
    "RecordingSink",
    "TrackioSink",
    "FaultySink",
    "SlowSink",
    "PRESENTED_METRICS",
    "PRESENTED_SUMMARY_KEYS",
    "FORBIDDEN_PREFIXES",
    "is_presentable",
    "build_sink",
    "trackio_available",
]
