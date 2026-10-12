"""Deterministic anomaly detectors.

No LLM, no learned model, no agent logic. Every finding carries the detector id,
baseline, threshold, observed value, window and confidence that produced it, so a
reviewer can recompute the decision by hand.
"""

from __future__ import annotations

from .base import (
    Anomaly,
    BaseDetector,
    DetectorConfig,
    DetectorContext,
    Series,
    robust_stats,
    theil_sen_slope,
    trailing_median,
)
from .gradient import (
    GRAD_LAYER_METRIC,
    GRAD_NONFINITE_METRIC,
    GRAD_NORM_METRIC,
    LOSS_NONFINITE_METRIC,
    ActivationHealthDetector,
    FinitenessDetector,
    GradientExplosionDetector,
)
from .loss import LOSS_METRIC, LossDivergenceDetector, LossSpikeDetector
from .baseline import BaselineState, ProtectedBaseline, protected_ratio_series
from .runner import DETECTOR_TO_INCIDENT, DetectionResult, DetectorRunner, default_detectors, suggest_incident_type
from .throughput import (
    SAMPLES_PER_SEC,
    STEP_TIME,
    TOKENS_PER_SEC,
    StepTimeRegressionDetector,
    ThroughputRegressionDetector,
)

__all__ = [
    "ProtectedBaseline",
    "BaselineState",
    "protected_ratio_series",
    "Anomaly",
    "BaseDetector",
    "DetectorConfig",
    "DetectorContext",
    "Series",
    "robust_stats",
    "theil_sen_slope",
    "trailing_median",
    "LossSpikeDetector",
    "LossDivergenceDetector",
    "GradientExplosionDetector",
    "FinitenessDetector",
    "ActivationHealthDetector",
    "ThroughputRegressionDetector",
    "StepTimeRegressionDetector",
    "DetectorRunner",
    "DetectionResult",
    "default_detectors",
    "suggest_incident_type",
    "DETECTOR_TO_INCIDENT",
    "LOSS_METRIC",
    "GRAD_NORM_METRIC",
    "GRAD_LAYER_METRIC",
    "GRAD_NONFINITE_METRIC",
    "LOSS_NONFINITE_METRIC",
    "TOKENS_PER_SEC",
    "SAMPLES_PER_SEC",
    "STEP_TIME",
]
