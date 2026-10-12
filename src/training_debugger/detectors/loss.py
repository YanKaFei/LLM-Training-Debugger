"""Loss detectors: LOSS_SPIKE and LOSS_DIVERGENCE.

The two are deliberately separate, because they answer different questions and
have different false-positive profiles:

``LOSS_SPIKE``
    a point anomaly: one step (or a few) departs sharply from a robust baseline.
    Detected with a trailing median/MAD z-score over the *preceding* window only,
    so the spike cannot inflate the baseline it is compared against.

``LOSS_DIVERGENCE``
    a trend: the loss stops improving and starts rising, with no single step
    being remarkable. Detected with a Theil-Sen slope plus a "no new minimum"
    condition. A single spike does **not** produce this, and a slow drift does
    **not** produce a spike -- which is why both exist.

Both exclude the point under test from its own baseline. A detector that
includes the outlier in the baseline is self-masking, and that is the single most
common way a naive robust-z implementation fails.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..schema.enums import DetectorId, EvidenceLevel, Severity
from .base import (
    Anomaly,
    BaseDetector,
    DetectorContext,
    Series,
    detrended_residuals,
    robust_stats,
    theil_sen_slope,
)

__all__ = ["LossSpikeDetector", "LossDivergenceDetector", "LOSS_METRIC"]


LOSS_METRIC = "train.loss"


class LossSpikeDetector(BaseDetector):
    """Robust z-score on the loss, evaluated against the trailing window only."""

    detector_id = DetectorId.LOSS_ROBUST_ZSCORE
    version = "1.0"
    metric_names = (LOSS_METRIC,)

    def detect(self, series: Series, context: DetectorContext) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        values = [value for _, value in series]
        steps = [step for step, _ in series]

        for index, (step, value) in enumerate(series):
            if not self._enough_history(series, index):
                continue
            if value != value or abs(value) == float("inf"):
                # A non-finite loss is the finiteness detector's business, not a
                # statistical claim. Firing here too would double-report.
                continue
            start = max(0, index - self.config.window)
            window_values = values[start:index]
            window_steps = steps[start:index]
            if len(window_values) < self.config.min_samples:
                continue

            # Detrend first: a decaying loss curve makes the raw window MAD a
            # measure of the decay rather than of the noise, which is exactly how
            # a 4.5x jump scored z=0.93 before this was added.
            if self.config.detrend:
                residuals, slope, intercept = detrended_residuals(window_steps, window_values)
                resid_stats = robust_stats(residuals)
                expected = intercept + slope * step
                spread = resid_stats["mad"]
            else:
                resid_stats = robust_stats(window_values)
                expected = resid_stats["median"]
                spread = resid_stats["mad"]

            statistics_median = robust_stats(window_values)["median"]
            z_score = (value - expected) / spread if spread > 0 else 0.0
            ratio = (value / statistics_median) if statistics_median > 0 else None

            z_fired = spread > 0.0 and z_score > self.config.spike_z
            ratio_fired = (
                self.config.relative_factor is not None
                and ratio is not None
                and ratio >= self.config.relative_factor
            )
            if not (z_fired or ratio_fired):
                continue
            score = z_score if z_fired else float(ratio or 0.0)
            key = "{}:{}".format(context.metric_name or LOSS_METRIC, context.rank)
            suppressed = self._in_cooldown(key, step)
            if suppressed and self.config.cooldown_suppresses_evidence:
                # Legacy behaviour: the observation is discarded entirely and the
                # episode never learns of it. Retained behind a flag so the defect
                # stays reproducible.
                continue
            if not suppressed:
                self._mark_fired(key, step)
            window_start = steps[max(0, index - self.config.window)]
            anomalies.append(
                Anomaly(
                    detector=self.detector_id,
                    detector_version=self.version,
                    metric_name=context.metric_name or LOSS_METRIC,
                    first_abnormal_step=step,
                    observed_value=value,
                    baseline=statistics_median,
                    threshold=(
                        expected + self.config.spike_z * spread
                        if z_fired
                        else statistics_median * float(self.config.relative_factor)
                    ),
                    score=score,
                    # Inferred, not observed: a z-score is a statistical claim.
                    confidence=(
                        min(0.99, 0.5 + z_score / (4.0 * self.config.spike_z))
                        if z_fired
                        else 0.75
                    ),
                    level=EvidenceLevel.INFERRED,
                    severity=Severity.ERROR,
                    window=(window_start, step),
                    rank=context.rank,
                    sample_count=len(window_values),
                    reason=(
                        "loss {:.6g} is {:.1f} robust z-scores above a detrended expectation of "
                        "{:.6g} (residual MAD {:.6g}); ratio to the window median is {:.2f}x".format(
                            value, z_score, expected, spread,
                            ratio if ratio is not None else float("nan"),
                        )
                    ),
                    metadata={
                        "cooldown_suppressed": suppressed,
                        "baseline_median": statistics_median,
                        "detrended_expectation": expected,
                        "residual_mad": spread,
                        "local_slope_per_step": slope if self.config.detrend else 0.0,
                        "z_score": z_score,
                        "ratio_to_median": ratio,
                        "criterion": "z_score" if z_fired else "relative_factor",
                        "baseline_window": self.config.window,
                        "method": "detrended_trailing_median_mad" if self.config.detrend else "trailing_median_mad",
                    },
                )
            )
        return anomalies


class LossDivergenceDetector(BaseDetector):
    """Sustained rise with no new minimum.

    Two conditions must both hold, which is what separates divergence from a
    spike:

    1. the Theil-Sen slope over the trailing window is positive beyond
       ``divergence_slope_min`` as a fraction of the baseline; and
    2. the window contains **no new minimum**, i.e. the run has stopped
       improving. A single spike followed by recovery leaves a new minimum
       behind and therefore does not fire.
    """

    detector_id = DetectorId.LOSS_TREND
    version = "1.0"
    metric_names = (LOSS_METRIC,)

    def detect(self, series: Series, context: DetectorContext) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        if len(series) < self.config.divergence_min_points:
            return anomalies

        window = self.config.divergence_window
        for index in range(self.config.warmup_steps, len(series)):
            if index + 1 < self.config.divergence_min_points:
                continue
            start = max(0, index - window)
            segment = series[start : index + 1]
            if len(segment) < self.config.divergence_min_points:
                continue
            finite = [(s, v) for s, v in segment if v == v and abs(v) != float("inf")]
            if len(finite) < self.config.divergence_min_points:
                continue

            baseline = robust_stats([v for _, v in series[max(0, start - window) : start + 1]])["median"]
            if baseline != baseline or baseline == 0.0:
                continue
            slope = theil_sen_slope(finite)
            relative_slope = slope / abs(baseline)
            if relative_slope <= self.config.divergence_slope_min:
                continue

            values = [v for _, v in finite]
            running_min = min(v for _, v in series[: index + 1] if v == v and abs(v) != float("inf"))
            window_min = min(values)
            # A new minimum anywhere in the window means the run is still
            # capable of improving, so this is not divergence.
            new_minimum_in_window = window_min <= running_min + 1e-12
            if new_minimum_in_window:
                continue

            step = series[index][0]
            key = "{}:{}:divergence".format(context.metric_name or LOSS_METRIC, context.rank)
            suppressed = self._in_cooldown(key, step)
            if suppressed and self.config.cooldown_suppresses_evidence:
                # Legacy behaviour: the observation is discarded entirely and the
                # episode never learns of it. Retained behind a flag so the defect
                # stays reproducible.
                continue
            if not suppressed:
                self._mark_fired(key, step)
            anomalies.append(
                Anomaly(
                    detector=self.detector_id,
                    detector_version=self.version,
                    metric_name=context.metric_name or LOSS_METRIC,
                    # The first step of the window is when the trend began, not
                    # the step at which it became detectable. The incident engine
                    # needs the former.
                    first_abnormal_step=segment[0][0],
                    observed_value=series[index][1],
                    baseline=baseline,
                    threshold=None,
                    score=relative_slope,
                    confidence=0.6,
                    level=EvidenceLevel.INFERRED,
                    severity=Severity.ERROR,
                    window=(segment[0][0], step),
                    rank=context.rank,
                    sample_count=len(finite),
                    reason=(
                        "loss has been rising for {} steps (Theil-Sen slope {:.3g}/step, "
                        "{:.2%} of baseline {:.6g} per step) with no new minimum".format(
                            step - segment[0][0], slope, relative_slope, baseline
                        )
                    ),
                    metadata={
                        "cooldown_suppressed": suppressed,
                        "slope_per_step": slope,
                        "relative_slope": relative_slope,
                        "baseline_median": baseline,
                        "method": "theil_sen_no_new_minimum",
                    },
                )
            )
            break  # one divergence report per series is enough
        return anomalies
