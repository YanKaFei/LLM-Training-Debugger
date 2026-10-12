"""Throughput regression detection.

Phase 1 brief section 7.5 is explicit about the *scope* of this detector:

    Phase 1 does not infer GPU / storage / network root cause. It only creates
    the symptom.

So this module answers exactly one question -- *did throughput drop, and when did
it start?* -- and deliberately stops there. Attributing the drop to a GPU, a
filesystem or a collective is the correlation engine's job (W6) and, beyond that,
v0.2's job with HTA and DCGM. A detector that guessed a cause here would be
producing the confident-but-unfounded verdict Phase 0 forbids.

Two signals are watched because they fail in opposite directions:

``train.tokens_per_sec``
    falls when work per unit time drops.
``train.step_time``
    rises for the same reason, and is available even when the token count is not.

Either alone is enough; when both fire, the incident engine sees two independent
symptoms of one regression.
"""

from __future__ import annotations

from typing import List, Optional

from ..schema.enums import DetectorId, EvidenceLevel, Severity
from .base import Anomaly, BaseDetector, DetectorContext, Series, robust_stats

__all__ = [
    "ThroughputRegressionDetector",
    "StepTimeRegressionDetector",
    "TOKENS_PER_SEC",
    "STEP_TIME",
    "SAMPLES_PER_SEC",
]

TOKENS_PER_SEC = "train.tokens_per_sec"
SAMPLES_PER_SEC = "train.samples_per_sec"
STEP_TIME = "train.step_time"


class _SustainedShiftDetector(BaseDetector):
    """Shared logic: a sustained shift away from a robust baseline.

    "Sustained" is the operative word. A single slow step is noise (a
    checkpoint, a garbage collection, a page fault); ten consecutive slow steps
    is a regression. Requiring consecutive samples is what keeps this detector
    from firing on the ordinary variance of a training loop.
    """

    detector_id = DetectorId.THROUGHPUT_REGRESSION
    version = "1.0"

    #: ``True`` for signals that should go DOWN, ``False`` for signals that
    #: should go UP when the run regresses.
    expect_decrease = True

    def detect(self, series: Series, context: DetectorContext) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        if len(series) < self.config.warmup_steps + self.config.throughput_consecutive:
            return anomalies

        values = [value for _, value in series]
        base_window = self.config.throughput_baseline_window
        needed = self.config.throughput_consecutive

        index = self.config.warmup_steps
        while index < len(series):
            baseline_values = values[max(0, index - base_window) : index]
            if len(baseline_values) < self.config.min_samples:
                index += 1
                continue
            baseline = robust_stats(baseline_values)["median"]
            if baseline != baseline or baseline <= 0.0:
                index += 1
                continue

            # Find a run of consecutive samples that all breach the threshold.
            run_start: Optional[int] = None
            run_length = 0
            cursor = index
            while cursor < len(series):
                value = values[cursor]
                if value != value or abs(value) == float("inf"):
                    break
                ratio = value / baseline
                breached = (
                    ratio < (1.0 - self.config.throughput_drop_ratio)
                    if self.expect_decrease
                    else ratio > (1.0 + self.config.throughput_drop_ratio)
                )
                if breached:
                    if run_start is None:
                        run_start = cursor
                    run_length += 1
                    cursor += 1
                else:
                    break

            if run_start is not None and run_length >= needed:
                step = series[run_start][0]
                key = "{}:{}".format(context.metric_name, context.rank)
                if not self._in_cooldown(key, step):
                    self._mark_fired(key, step)
                    observed = values[run_start + run_length - 1]
                    change = observed / baseline - 1.0
                    anomalies.append(
                        Anomaly(
                            detector=self.detector_id,
                            detector_version=self.version,
                            metric_name=context.metric_name or STEP_TIME,
                            # The FIRST step of the sustained run is the onset,
                            # which is what the incident engine needs. Reporting
                            # the step at which the run became long enough would
                            # systematically bias the onset late by
                            # `throughput_consecutive` steps.
                            first_abnormal_step=step,
                            observed_value=observed,
                            baseline=baseline,
                            threshold=baseline * (
                                1.0 - self.config.throughput_drop_ratio
                                if self.expect_decrease
                                else 1.0 + self.config.throughput_drop_ratio
                            ),
                            score=abs(change),
                            confidence=0.7,
                            level=EvidenceLevel.INFERRED,
                            severity=Severity.WARN,
                            window=(series[max(0, run_start - base_window)][0], series[cursor - 1][0]),
                            rank=context.rank,
                            sample_count=len(baseline_values),
                            reason=(
                                "{} {} for {} consecutive steps: {:.6g} vs a baseline median of "
                                "{:.6g} ({:+.1%})".format(
                                    context.metric_name,
                                    "dropped" if self.expect_decrease else "rose",
                                    run_length, observed, baseline, change,
                                )
                            ),
                            metadata={
                                "baseline_median": baseline,
                                "change_ratio": change,
                                "consecutive_steps": run_length,
                                "baseline_window": base_window,
                                "method": "sustained_shift",
                                # Explicitly recorded so no consumer can mistake
                                # this for a root-cause finding.
                                "root_cause": None,
                                "scope": "symptom_only",
                            },
                        )
                    )
                index = cursor
            else:
                index += 1
        return anomalies


class ThroughputRegressionDetector(_SustainedShiftDetector):
    """Tokens (or samples) per second falling."""

    detector_id = DetectorId.THROUGHPUT_REGRESSION
    metric_names = (TOKENS_PER_SEC, SAMPLES_PER_SEC)
    expect_decrease = True


class StepTimeRegressionDetector(_SustainedShiftDetector):
    """Step time rising. The same regression seen from the other side."""

    detector_id = DetectorId.THROUGHPUT_REGRESSION
    metric_names = (STEP_TIME,)
    expect_decrease = False
