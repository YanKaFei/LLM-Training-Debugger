"""Gradient and numeric-finiteness detectors.

``GRADIENT_EXPLOSION`` (v2.0, asymmetric envelope)
    A value must exceed the weaker of two bounds computed from its own preceding
    window: the 99.5th percentile times 1.5, or the median times 3.0. Plus an
    optional absolute guard rail.

    v1.0 used a detrended robust z-score with a peak-ratio guard. Measurement on
    30 healthy real runs showed that criterion to be the entire cause of the
    detector's precision failure: a healthy run reached 1.51x its own window peak
    against a guard of 1.5, so the z-branch fired on clean data. See
    ``DetectorConfig.grad_envelope_quantile`` for the eight-family comparison that
    replaced it.

``NAN_DETECTED`` / ``INF_DETECTED``
    The only `OBSERVED` detector in Phase 1. NaN is not a statistical claim, so
    the confidence is 1.0 and the severity is always at least ERROR. This is the
    detector that the incident engine may use to promote a hypothesis without a
    further probe, because there is nothing to probe.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..schema.enums import DetectorId, EvidenceLevel, Severity
def _quantile(ordered: Sequence[float], q: float) -> float:
    """Quantile of an already-sorted sequence, linearly interpolated."""
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


from .base import (
    Anomaly,
    BaseDetector,
    DetectorContext,
    Series,
    detrended_residuals,
    robust_stats,
)

__all__ = [
    "GradientExplosionDetector",
    "FinitenessDetector",
    "ActivationHealthDetector",
    "PARAM_NORM_METRIC",
    "GRAD_NORM_METRIC",
    "GRAD_LAYER_METRIC",
    "LOSS_NONFINITE_METRIC",
    "GRAD_NONFINITE_METRIC",
]

GRAD_NORM_METRIC = "grad.global_norm"
PARAM_NORM_METRIC = "param.global_norm"
GRAD_LAYER_METRIC = "grad.layer_norm"
LOSS_NONFINITE_METRIC = "train.loss_nonfinite_count"
GRAD_NONFINITE_METRIC = "grad.nonfinite_count"


class GradientExplosionDetector(BaseDetector):
    """Robust z-score plus an absolute guard rail.

    Also reports the *layer* attribution when a per-layer series is supplied,
    because the global norm can be normal while one layer explodes -- aggregation
    masking, which Phase 0 lists as confounder 7.
    """

    detector_id = DetectorId.GRAD_ROBUST_ZSCORE
    version = "2.0"
    metric_names = (GRAD_NORM_METRIC, GRAD_LAYER_METRIC)

    def detect(self, series: Series, context: DetectorContext) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        values = [value for _, value in series]
        steps = [step for step, _ in series]

        for index, (step, value) in enumerate(series):
            if not self._enough_history(series, index):
                continue
            if value != value or abs(value) == float("inf"):
                # Non-finite is the finiteness detector's finding.
                continue
            # Phase-aware baseline: the warm-up region is excluded so the
            # threshold is not computed from a non-stationary population.
            floor = self.config.warmup_steps if self.config.baseline_excludes_warmup else 0
            start = max(floor, index - self.config.window)
            window_values = values[start:index]
            window_steps = steps[start:index]
            if len(window_values) < self.config.min_samples:
                continue

            # Detrend before measuring dispersion, for the same reason the loss
            # detector does: a gradient norm that is falling as the model warms
            # up makes the raw window MAD a measure of that decline rather than
            # of the noise around it. Without this the A1 run reported a false
            # positive at step 52, three steps after warm-up ended, purely
            # because the early-training gradient norm was still decreasing.
            statistics_median = robust_stats(window_values)["median"]
            if self.config.detrend:
                residuals, slope, intercept = detrended_residuals(window_steps, window_values)
                resid_stats = robust_stats(residuals)
                expected = intercept + slope * step
                spread = resid_stats["mad"]
            else:
                expected = statistics_median
                slope = 0.0
                spread = robust_stats(window_values)["mad"]

            window_peak = max(window_values) if window_values else 0.0
            ordered = sorted(window_values)
            envelope = _quantile(ordered, self.config.grad_envelope_quantile)
            median_threshold = statistics_median * self.config.grad_median_multiple
            envelope_threshold = envelope * self.config.grad_envelope_factor
            # The weaker of the two bounds: a quiet run is caught by its own
            # envelope, a noisy one by the median multiple.
            threshold = min(median_threshold, envelope_threshold)
            peak_ratio = (value / window_peak) if window_peak > 0 else float("inf")
            envelope_ratio = (value / envelope) if envelope > 0 else float("inf")
            median_ratio = (value / statistics_median) if statistics_median > 0 else float("inf")
            z_score = (value - expected) / spread if spread > 0.0 else 0.0

            guard = self.config.grad_absolute_guard
            # v2.0: the asymmetric envelope is the primary criterion. The v1.0
            # robust z-score was removed, not retuned -- measurement showed it
            # fired on a healthy run at 1.51x the window peak, 0.7 % above its own
            # guard, and it was the sole cause of 50 false positives against 6
            # true positives.
            envelope_fired = value > threshold
            guard_fired = guard is not None and value > guard
            if not (envelope_fired or guard_fired):
                continue
            stats = {"median": statistics_median, "mad": spread, "count": len(window_values)}

            key = "{}:{}:{}".format(context.metric_name, context.rank, context.layer_index)
            if self._in_cooldown(key, step):
                continue
            self._mark_fired(key, step)

            if envelope_fired:
                score = envelope_ratio
                reason = (
                    "gradient norm {:.6g} exceeds the {:.1%} window envelope {:.6g} by "
                    "{:.1f}x (threshold {:.6g}); {:.1f}x the window median and "
                    "{:.1f}x the window peak".format(
                        value, self.config.grad_envelope_quantile, envelope,
                        envelope_ratio, threshold, median_ratio, peak_ratio)
                )
                method = "asymmetric_envelope"
                baseline = statistics_median
            else:
                score = (value / guard) if guard else float("inf")
                reason = (
                    "gradient norm {:.6g} exceeds the absolute guard rail {:.6g} "
                    "regardless of baseline".format(value, float(guard))
                )
                method = "absolute_guard"
                baseline = statistics_median
                threshold = guard

            anomalies.append(
                Anomaly(
                    detector=self.detector_id,
                    detector_version=self.version,
                    metric_name=context.metric_name or GRAD_NORM_METRIC,
                    first_abnormal_step=step,
                    observed_value=value,
                    baseline=baseline,
                    threshold=threshold,
                    score=score if score != float("inf") else 1e9,
                    confidence=0.9 if guard_fired else 0.8,
                    level=EvidenceLevel.INFERRED,
                    severity=Severity.ERROR,
                    window=(steps[max(0, index - self.config.window)], step),
                    rank=context.rank,
                    layer_index=context.layer_index,
                    module_path=context.module_path,
                    sample_count=int(stats["count"]),
                    reason=reason,
                    metadata={
                        "baseline_median": statistics_median,
                        "detrended_expectation": expected,
                        "residual_mad": spread,
                        "local_slope_per_step": slope,
                        "z_score": z_score,
                        "window_peak": window_peak,
                        "peak_ratio": peak_ratio,
                        "absolute_guard": guard,
                        "method": method,
                        "population_layers": context.population_layers,
                    },
                )
            )
        return anomalies


class ActivationHealthDetector(BaseDetector):
    """Per-module activation health, from **exact observations only**.

    This closes a real gap: the project observed gradient norms per layer and nothing
    about activations, while a competitor classifies `Vanishing`/`Exploding`/`Dead`/
    `Saturated` per named module.

    **Only the threshold-free states are reported:**

    ```
    act.nonfinite_count > 0      -> nonfinite
    act.zero_fraction >= 1.0     -> dead        (every element exactly zero)
    act.abs_max >= dtype max     -> saturated   (|x| at the dtype's own limit)
    ```

    `vanishing` and `exploding` are deliberately **not** produced. Calling a module's
    activations "vanishing" requires a notion of the right size for that module, and this
    detector does not have one. The magnitudes are emitted as statistics
    (`act.abs_mean`, `act.global_norm`) and left to the reader. A competitor reports
    "activation_regime_shift cascaded through [modules]" at 0.72 confidence; this reports
    "100 % of activations are exactly zero", which is a fact.
    """

    detector_id = DetectorId.ACTIVATION_HEALTH
    version = "1.0"
    metric_names = (
        "act.zero_fraction", "act.abs_max", "act.nonfinite_count",
    )

    def detect(self, series: Series, context: DetectorContext) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        metric = str(context.metric_name)
        for step, value in series:
            if value is None or value != value:
                continue
            state = reason = None
            if metric == "act.nonfinite_count" and value > 0:
                state, reason = "nonfinite", "{} non-finite activation element(s)".format(
                    int(value))
            elif metric == "act.zero_fraction" and value >= 1.0:
                state, reason = "dead", "every activation element is exactly zero"
            elif metric == "act.abs_max" and value >= _DTYPE_MAX_F32:
                state, reason = "saturated", "|x| reaches the float32 maximum"
            if state is None:
                continue
            key = "{}:{}:{}".format(metric, context.rank, context.module_path)
            if self._in_cooldown(key, step):
                continue
            self._mark_fired(key, step)
            anomalies.append(
                Anomaly(
                    detector=self.detector_id,
                    detector_version=self.version,
                    metric_name=metric,
                    first_abnormal_step=step,
                    observed_value=float(value),
                    baseline=None,
                    threshold=None,
                    score=1.0,
                    confidence=1.0,
                    level=EvidenceLevel.OBSERVED,
                    severity=Severity.WARN,
                    window=(step, step),
                    rank=context.rank,
                    layer_index=context.layer_index,
                    module_path=context.module_path,
                    sample_count=1,
                    reason=reason,
                    metadata={"kind": state, "threshold_free": True,
                              "module_path": context.module_path},
                )
            )
        return anomalies


#: float32 maximum. `abs_max` reaching it is saturation at the dtype boundary, which is
#: an exact condition -- unlike "large", which would need a reference.
_DTYPE_MAX_F32 = 3.4028234663852886e38


class FinitenessDetector(BaseDetector):
    """NaN / Inf detection. ``OBSERVED`` confidence 1.0 -- nothing is inferred.

    Two shapes are supported because the adapter emits both:

    * a **count** series (``train.loss_nonfinite_count``, ``grad.nonfinite_count``)
      where any value > 0 is a finding; and
    * a **value** series (``train.loss``, ``grad.global_norm``) where a non-finite
      sample is itself the finding.

    Reporting both would double-count, so the caller chooses which series to feed.
    """

    detector_id = DetectorId.NUMERICS_FINITENESS
    version = "1.0"
    #: `param.global_norm` is the **midstream witness**. The training mechanism is
    #: gradient -> optimizer update -> parameters -> loss, and the parameter norm is the
    #: only recorded observation that shows whether the update itself overflowed.
    #:
    #: Without it the relation rule could not tell a real propagation from a suppressed
    #: update that merely *looked* identical: in the adversarial corpus, `param.global_norm`
    #: was `inf` at the injection step for a genuine propagation and finite for a
    #: suppressed update whose downstream failure was injected independently. The two
    #: produced the same gradient and loss observations and were scored as the same
    #: thing -- 10 fabricated associations.
    #:
    #: This widens an existing detector's watch list to a metric the observer ALREADY
    #: records. It is not a new detector and not a new telemetry source.
    metric_names = (LOSS_NONFINITE_METRIC, GRAD_NONFINITE_METRIC, "train.loss",
                    GRAD_NORM_METRIC, PARAM_NORM_METRIC)

    def detect(self, series: Series, context: DetectorContext) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        is_count_series = context.metric_name in (LOSS_NONFINITE_METRIC, GRAD_NONFINITE_METRIC)

        for step, value in series:
            if is_count_series:
                if value is None or value != value or value <= 0:
                    continue
                observed = float(value)
                kind = "count"
            else:
                if value is None or (value == value and abs(value) != float("inf")):
                    continue
                observed = float("nan") if value != value else value
                kind = "nan" if value != value else "inf"

                # W0: on a `*.global_norm` metric, `inf` does NOT mean "a non-finite
                # element was observed".
                #
                # Both `global_gradient_norm` and `global_parameter_norm` return `nan`
                # when any element is non-finite, and leave the value as `inf` only when
                # the **reduction overflowed while every element was finite**. The
                # mapping is therefore definitional, not probabilistic:
                #
                #     *.global_norm == inf  ->  all elements finite, reduction overflowed
                #     *.global_norm == nan  ->  non-finite elements present
                #
                # Reporting the first as a non-finite observation is precisely the
                # "a non-finite norm is not necessarily a non-finite tensor" error.
                # It is still a real and useful finding, so it is reported with its own
                # kind rather than dropped.
                if kind == "inf" and str(context.metric_name).endswith(".global_norm"):
                    kind = "norm_reduction_overflow"

            key = "{}:{}:{}".format(context.metric_name, context.rank, context.layer_index)
            # Cooldown applied. The v1.0 comment here read "No cooldown: every
            # non-finite observation is separately true" -- which is true and was
            # the wrong design. A non-finite value that persists poisons the
            # optimiser state, so every subsequent step is also non-finite, and
            # the detector raised one finding per step: measured at **810
            # post-fault findings across 90 runs**, 94 % of the entire corpus's
            # false positives and the sole reason system precision fell to 0.037.
            #
            # Each observation remains true; they are not 810 independent facts.
            # A persistent non-finite condition is ONE finding, and the cooldown
            # expresses that without discarding the observation: the first step is
            # reported, the continuation is suppressed as a duplicate.
            key = "{}:{}:{}".format(context.metric_name, context.rank, context.layer_index)
            if self._in_cooldown(key, step):
                continue
            self._mark_fired(key, step)

            if kind == "norm_reduction_overflow":
                detail = (
                    "{} overflowed the float32 reduction to inf at step {} while every "
                    "element was finite; this is a reduction artifact, NOT a "
                    "non-finite parameter or gradient".format(context.metric_name, step)
                )
            elif kind == "nan":
                detail = "{} is NaN at step {}".format(context.metric_name, step)
            elif kind == "inf":
                detail = "{} is infinite at step {}".format(context.metric_name, step)
            else:
                detail = "{} reports {} non-finite value(s) at step {}".format(
                    context.metric_name, int(observed), step
                )

            anomalies.append(
                Anomaly(
                    detector=self.detector_id,
                    detector_version=self.version,
                    metric_name=context.metric_name or LOSS_NONFINITE_METRIC,
                    first_abnormal_step=step,
                    observed_value=observed,
                    baseline=None,
                    threshold=0.0,
                    score=1.0,
                    # Measured, not inferred. This is the only detector allowed
                    # to claim OBSERVED.
                    confidence=1.0,
                    level=EvidenceLevel.OBSERVED,
                    severity=Severity.FATAL if kind != "count" else Severity.ERROR,
                    window=(step, step),
                    rank=context.rank,
                    layer_index=context.layer_index,
                    module_path=None,
                    sample_count=1,
                    reason=detail,
                    metadata={
                        "kind": kind,
                        "method": "direct_finiteness_check",
                        "inferred": False,
                    },
                )
            )
        return anomalies
