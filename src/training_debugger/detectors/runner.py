"""Detector runner: applies every detector to a run and assembles multiplicity.

Two responsibilities beyond simply looping:

**Suppression.** When the finiteness detector reports a NaN loss at step *s*, a
loss-spike anomaly at the same step is not additional evidence -- it is the same
observation described statistically. Reporting both would inflate the incident's
evidence count and make the spike look better supported than it is. The rule is
that a more *specific* finding supersedes a more *general* one at the same step.

**Multiplicity.** Phase 1 brief section 7.6 requires every incident to carry
``affected_ranks``, ``affected_layers``, ``affected_steps``, ``event_count`` and
``population_count``. It is computed here, once, from the anomalies themselves --
so "Layer 12 anomalous" is never reported without its denominator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schema.enums import EvidenceLevel, IncidentType
from ..schema.evidence import Multiplicity
from .base import (
    Anomaly,
    BaseDetector,
    ConfidenceClass,
    DetectorConfig,
    DetectorContext,
    Series,
)
from .gradient import (
    GRAD_NONFINITE_METRIC,
    GRAD_NORM_METRIC,
    LOSS_NONFINITE_METRIC,
    ActivationHealthDetector,
    FinitenessDetector,
    GradientExplosionDetector,
)
from .loss import LOSS_METRIC, LossDivergenceDetector, LossSpikeDetector
from .throughput import SAMPLES_PER_SEC, STEP_TIME, StepTimeRegressionDetector, ThroughputRegressionDetector

__all__ = ["DetectionResult", "DetectorRunner", "default_detectors"]


def default_detectors(config: Optional[DetectorConfig] = None) -> List[BaseDetector]:
    """The Phase 1 detector set. Order matters only for reporting."""
    return [
        LossSpikeDetector(config),
        LossDivergenceDetector(config),
        GradientExplosionDetector(config),
        ThroughputRegressionDetector(config),
        StepTimeRegressionDetector(config),
        FinitenessDetector(config),
        ActivationHealthDetector(config),
    ]


@dataclass
class DetectionResult:
    """Everything the incident engine needs, and nothing it does not."""

    run_id: str
    anomalies: List[Anomaly] = field(default_factory=list)
    multiplicity: Multiplicity = field(default_factory=Multiplicity)
    suppressed: List[Dict[str, Any]] = field(default_factory=list)
    detectors_run: int = 0
    series_examined: List[str] = field(default_factory=list)
    coverage: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    detector_errors: List[Dict[str, Any]] = field(default_factory=list)

    def by_detector(self, detector_id: Any) -> List[Anomaly]:
        return [a for a in self.anomalies if a.detector == detector_id]

    def observed(self) -> List[Anomaly]:
        return [a for a in self.anomalies if a.level is EvidenceLevel.OBSERVED]

    def inferred(self) -> List[Anomaly]:
        return [a for a in self.anomalies if a.level is EvidenceLevel.INFERRED]

    def steps(self) -> List[int]:
        return sorted({a.first_abnormal_step for a in self.anomalies})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "multiplicity": self.multiplicity.to_dict(),
            "suppressed": list(self.suppressed),
            "detectors_run": self.detectors_run,
            "series_examined": list(self.series_examined),
            "coverage": dict(self.coverage),
            "config": dict(self.config),
            "detector_errors": list(self.detector_errors),
        }


#: Fraction of the layer population that must depart at one step for a
#: per-layer fire to count as corroborated (D2.4).
#:
#: Measured: healthy single-layer tail events affect **1 of 7** layers while the
#: aggregate stays silent (3.26 against a 4.48 threshold). A real injected
#: explosion moves every layer and the aggregate by 15.7x. The threshold sits
#: between those regimes rather than being chosen by taste.
LAYER_MULTIPLICITY_STRONG = 0.4


def _layer_ratio_series_protected(layer_series, baseline_floor: int, window: int = 150,
                                  freeze_at_ratio: float = 1.3,
                                  minimum_history: int = 20,
                                  resume_after_steps: int = 40,
                                  mode: str = "frozen",
                                  ema_warmup: int = 25,
                                  ema_alpha: float = 0.97,
                                  ema_z: float = 2.5) -> Dict[Any, Dict[int, float]]:
    """L2: per-layer ratios against a **freeze-on-suspicion** reference.

    This replaces the trailing-median ratio that Phase 1.7 proved defective on 680
    runs: a sustained uniform fault scaling dragged the trailing median along with it,
    so the ratio decayed back toward 1 and the fault erased itself. Here the
    reference freezes at the first hint of deviation, so the sustained level stays
    visible for as long as it lasts.
    """
    from .baseline import adaptive_ratio_series, protected_ratio_series

    ratios: Dict[Any, Dict[int, float]] = {}
    for index, series in (layer_series or {}).items():
        if mode == "ema":
            # The warm-up floor is deliberately NOT applied: removing the readiness
            # blindness is the entire point of this mode. The EMA has its own warm-up.
            points = [(step, value) for step, value in series]
            if len(points) < ema_warmup + 1:
                continue
            computed = adaptive_ratio_series(
                points, warmup=ema_warmup, alpha=ema_alpha, z_threshold=ema_z,
            )
        else:
            points = [(step, value) for step, value in series if step >= baseline_floor]
            if len(points) < minimum_history:
                continue
            computed = protected_ratio_series(
                points, minimum_history=minimum_history, window=window,
                freeze_at_ratio=freeze_at_ratio, resume_after_steps=resume_after_steps,
            )
        kept = {step: value for step, value in computed.items() if value is not None}
        if kept:
            ratios[index] = kept
    return ratios


def _layer_ratio_series(layer_series, baseline_floor: int, window: int = 150):
    """Per-layer values normalised by each layer's own trailing median.

    Each layer is normalised by *its own* baseline rather than by a shared scale,
    because per-layer gradient magnitudes differ by depth by more than 1.5x
    (measured), so an unnormalised sum would simply report which layer is deepest.
    """
    ratios = {}
    for index, series in (layer_series or {}).items():
        # ``series`` is a list of (step, value) pairs.
        lookup = dict(series)
        steps = sorted(lookup)
        normalised = {}
        for step in steps:
            history = [lookup[s] for s in steps if baseline_floor <= s < step][-window:]
            if len(history) < 20:
                continue
            ordered = sorted(history)
            median = ordered[len(ordered) // 2]
            if median <= 0:
                continue
            normalised[step] = lookup[step] / median
        if normalised:
            ratios[index] = normalised
    return ratios


def joint_gradient_features(layer_series, step: int, *, baseline_floor: int = 50,
                            resume_after_steps: int = 40,
                            baseline_mode: str = "frozen",
                            baseline_warmup_steps: int = 25,
                            baseline_ema_alpha: float = 0.97,
                            baseline_ema_z: float = 2.5,
                            window: int = 150, protected: bool = True) -> Dict[str, float]:
    """The joint representation, evaluated at one step.

    ``protected=True`` (the default since Phase 1.8) uses the freeze-on-suspicion
    reference. The unprotected trailing-median variant is retained so the defect it
    causes stays reproducible and comparable.
    """
    if protected:
        ratios = _layer_ratio_series_protected(
            layer_series, baseline_floor, window,
            resume_after_steps=resume_after_steps, mode=baseline_mode,
            ema_warmup=baseline_warmup_steps, ema_alpha=baseline_ema_alpha,
            ema_z=baseline_ema_z,
        )
    else:
        ratios = _layer_ratio_series(layer_series, baseline_floor, window)
    values = [r[step] for r in ratios.values() if step in r]
    if len(values) < 2:
        return {}
    ordered = sorted(values)
    # ``l1_aggregate`` is the MEAN of the per-layer ratios, not the sum.
    #
    # A sum scales with the number of monitored layers, so a boundary fitted on a
    # 4-layer model is wrong by 1.5x on a 6-layer model. That defect was measured:
    # the development corpus (which mixes 4-layer d32 and 6-layer d48 runs) produced
    # 16 healthy episodes above the CONFIRMED boundary, against a fitted healthy
    # maximum of 33.32 -- because 6 layers inflate a sum by 50 %. The feature is now
    # layer-count-invariant, so one boundary applies to any model shape.
    return {
        "l1_aggregate": (sum(values) / len(values)) if values else float("nan"),
        "l2_aggregate": math.sqrt(sum(v * v for v in values)),
        "trimmed_l2": math.sqrt(sum(v * v for v in ordered[:-1])) if len(ordered) > 1 else 0.0,
        "max_layer_ratio": ordered[-1],
        "layers_observed": float(len(values)),
    }


def _apply_gradient_band(
    anomalies: List[Anomaly], layer_series, config: DetectorConfig
) -> None:
    """I0.2: assign each gradient observation a NORMAL / UNCERTAIN / CONFIRMED band.

    This is what turns the Phase 1.4 measurement into behaviour. A marginal
    excursion below ``A`` is *marked* rather than deleted -- it stays in the record
    as evidence -- but it can no longer be confirmed, which is what removes the
    healthy heavy-tail false alarms. The middle band is the abstention region the
    brief requires: the classes overlap there and the system says so.
    """
    if not config.gradient_band_enabled:
        return
    cache: Dict[int, Dict[str, float]] = {}
    for anomaly in anomalies:
        if str(anomaly.detector) != "grad.robust_zscore":
            continue
        step = anomaly.first_abnormal_step
        if step is None:
            continue
        if step not in cache:
            cache[step] = joint_gradient_features(
                layer_series, step,
                baseline_floor=config.warmup_steps if config.baseline_excludes_warmup else 0,
                window=config.window,
                resume_after_steps=config.baseline_resume_after_steps,
                baseline_mode=config.baseline_mode,
                baseline_warmup_steps=config.baseline_warmup_steps,
                baseline_ema_alpha=config.baseline_ema_alpha,
                baseline_ema_z=config.baseline_ema_z,
            )
        features = cache[step]
        aggregate = features.get("l1_aggregate")
        if aggregate is None or features.get("layers_observed", 0) < config.gradient_agg_min_layers:
            # Not enough monitored layers to aggregate. Fall back to the marginal
            # classification already assigned rather than inventing a band.
            anomaly.metadata.setdefault("gradient_band", "UNCERTAIN")
            anomaly.metadata.setdefault("band_reason", "insufficient monitored layers")
            continue
        anomaly.metadata["l1_aggregate"] = aggregate
        anomaly.metadata["l2_aggregate"] = features.get("l2_aggregate")
        anomaly.metadata["trimmed_l2"] = features.get("trimmed_l2")
        anomaly.metadata["layers_observed"] = features.get("layers_observed")
        if aggregate >= config.gradient_agg_confirm_at:
            anomaly.metadata["gradient_band"] = "CONFIRMED"
            anomaly.metadata["band_reason"] = (
                "joint aggregate {:.1f} >= {:.1f}".format(aggregate, config.gradient_agg_confirm_at)
            )
            anomaly.confidence_class = ConfidenceClass.STRONG
        elif aggregate < config.gradient_agg_normal_below:
            anomaly.metadata["gradient_band"] = "NORMAL"
            anomaly.metadata["band_reason"] = (
                "joint aggregate {:.1f} < {:.1f}: within the healthy range".format(
                    aggregate, config.gradient_agg_normal_below)
            )
            anomaly.confidence_class = ConfidenceClass.WEAK
        else:
            anomaly.metadata["gradient_band"] = "UNCERTAIN"
            anomaly.metadata["band_reason"] = (
                "joint aggregate {:.1f} lies between {:.1f} and {:.1f}: healthy and mild "
                "fault distributions overlap here".format(
                    aggregate, config.gradient_agg_normal_below, config.gradient_agg_confirm_at)
            )
            anomaly.confidence_class = ConfidenceClass.WEAK


def _apply_fusion_classes(anomalies: List[Anomaly], population_layers: int = 0) -> None:
    """Assign a D5.1 confidence class and record cross-layer multiplicity.

    Two mechanisms are at work and both are needed:

    * **Finiteness is CERTAIN.** A measured non-finite value is not a statistical
      claim, so no amount of healthy variation can make it false.
    * **A single component's departure is WEAK** unless its peers departed with
      it. This is the classification that separates a healthy heavy tail from a
      real explosion, and it is derived from the measured 1/7 versus 5-7/7 split.
    """
    by_step_layers: Dict[int, set] = {}
    for anomaly in anomalies:
        if anomaly.layer_index is not None and anomaly.first_abnormal_step is not None:
            by_step_layers.setdefault(anomaly.first_abnormal_step, set()).add(anomaly.layer_index)

    for anomaly in anomalies:
        if str(anomaly.detector) == "numerics.finiteness":
            anomaly.confidence_class = ConfidenceClass.CERTAIN
            anomaly.metadata["fusion_reason"] = "measured non-finite value"
            continue

        is_aggregate = anomaly.layer_index is None
        affected = by_step_layers.get(anomaly.first_abnormal_step, set())
        population = population_layers or anomaly.metadata.get("population_layers") or 0
        if not population:
            # No layer population known: multiplicity cannot be assessed, so the
            # fire is not promotable on multiplicity grounds alone. Falling back
            # to len(affected) would make every single-layer fire look like 100 %
            # of a population of one.
            population = 0
        multiplicity = (len(affected) / float(population)) if population else 0.0
        anomaly.metadata["layer_multiplicity"] = multiplicity
        anomaly.metadata["affected_layers_at_step"] = sorted(affected)
        anomaly.metadata["population_layers"] = population

        if is_aggregate:
            # The aggregate moved. Under an injected explosion it moved 15.7x; on
            # healthy runs it did not move at all.
            anomaly.confidence_class = ConfidenceClass.STRONG
            anomaly.metadata["fusion_reason"] = "aggregate signal departed"
        elif population and multiplicity >= LAYER_MULTIPLICITY_STRONG:
            anomaly.confidence_class = ConfidenceClass.STRONG
            anomaly.metadata["fusion_reason"] = (
                "{:.0f}% of layers departed at this step".format(multiplicity * 100.0)
            )
        else:
            anomaly.confidence_class = ConfidenceClass.WEAK
            anomaly.metadata["fusion_reason"] = (
                "only {} of {} layers departed; the aggregate did not".format(
                    len(affected), population or "an unknown number of"
                )
            )


class DetectorRunner:
    """Runs detectors over a set of series and assembles the result."""

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        detectors: Optional[Sequence[BaseDetector]] = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.detectors = list(detectors) if detectors is not None else default_detectors(self.config)
        #: Detector failures, surfaced rather than swallowed.
        self.errors: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def run(
        self,
        run_id: str,
        series_by_metric: Dict[str, Series],
        *,
        rank: Optional[int] = None,
        layer_series: Optional[Dict[int, Series]] = None,
        module_paths: Optional[Dict[int, str]] = None,
        population_layers: int = 0,
        population_ranks: int = 1,
        coverage: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        """Detect over ``series_by_metric``.

        ``layer_series`` maps a layer index to its own series, so a per-layer
        gradient explosion is attributable to a layer rather than to the model.
        """
        self.errors = []
        result = DetectionResult(run_id=run_id, coverage=dict(coverage or {}))
        result.config = self.config.to_dict()
        result.detectors_run = len(self.detectors)

        # -- plain series --------------------------------------------------
        for metric_name, series in series_by_metric.items():
            if not series:
                continue
            result.series_examined.append(metric_name)
            context = DetectorContext(
                run_id=run_id, rank=rank, metric_name=metric_name,
                population_layers=population_layers, population_ranks=population_ranks,
            )
            for detector in self.detectors:
                if not self._applies(detector, metric_name):
                    continue
                result.anomalies.extend(self._safe_detect(detector, series, context))

        # -- per-layer series ----------------------------------------------
        if layer_series:
            for layer_index, series in sorted(layer_series.items()):
                if not series:
                    continue
                metric_name = "grad.layer_norm"
                result.series_examined.append("{}[layer={}]".format(metric_name, layer_index))
                context = DetectorContext(
                    run_id=run_id, rank=rank, metric_name=metric_name,
                    layer_index=layer_index,
                    module_path=(module_paths or {}).get(layer_index),
                    population_layers=population_layers or len(layer_series),
                    population_ranks=population_ranks,
                )
                for detector in self.detectors:
                    if not self._applies(detector, metric_name):
                        continue
                    result.anomalies.extend(self._safe_detect(detector, series, context))

        result.anomalies = self._suppress(result.anomalies, result.suppressed)
        result.anomalies.sort(
            key=lambda a: (a.first_abnormal_step, str(a.detector), a.layer_index or -1)
        )
        result.detector_errors = list(self.errors)
        result.multiplicity = self._multiplicity(
            result.anomalies,
            population_layers=max(population_layers, len(layer_series or {})),
            population_ranks=population_ranks,
            steps_available=coverage.get("step_count") if coverage else None,
            layer_series=layer_series,
            config=self.config,
        )
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _applies(detector: BaseDetector, metric_name: str) -> bool:
        if not detector.metric_names:
            return True
        return metric_name in detector.metric_names

    def _safe_detect(self, detector: BaseDetector, series: Series, context: DetectorContext) -> List[Anomaly]:
        """A detector raising must not lose the other detectors' findings.

        The failure is **recorded**, not merely swallowed. An earlier version
        returned an empty list silently, which hid a real bug: a stale variable
        reference made the loss detector raise on every step, the runner turned
        that into "no anomalies", and the corpus reported a clean healthy run and
        a missed spike. A detector that is silently broken is worse than one that
        crashes, because nothing surfaces it.
        """
        try:
            return list(detector.detect(series, context))
        except Exception as exc:
            self.errors.append(
                {
                    "detector": str(detector.detector_id),
                    "metric_name": context.metric_name,
                    "rank": context.rank,
                    "layer_index": context.layer_index,
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            )
            return []

    @staticmethod
    def _suppress(anomalies: List[Anomaly], log: List[Dict[str, Any]]) -> List[Anomaly]:
        """A more specific finding supersedes a more general one at the same step.

        Currently: an observed finiteness finding suppresses an inferred
        statistical finding on the *same metric* at the *same step*. Statistical
        claims about a NaN are meaningless -- the mean of a set containing NaN is
        NaN, not a large number.
        """
        observed_keys = {
            (a.metric_name, a.first_abnormal_step)
            for a in anomalies
            if a.level is EvidenceLevel.OBSERVED
        }
        kept: List[Anomaly] = []
        for anomaly in anomalies:
            if anomaly.level is EvidenceLevel.INFERRED and (
                anomaly.metric_name, anomaly.first_abnormal_step
            ) in observed_keys:
                log.append(
                    {
                        "suppressed_detector": str(anomaly.detector),
                        "metric_name": anomaly.metric_name,
                        "step": anomaly.first_abnormal_step,
                        "reason": "superseded_by_observed_finiteness_finding",
                    }
                )
                continue
            kept.append(anomaly)
        return kept

    @staticmethod
    def _multiplicity(
        anomalies: List[Anomaly],
        *,
        population_layers: int,
        population_ranks: int,
        steps_available: Optional[int],
        layer_series: Optional[Dict[int, Series]] = None,
        config: Optional[DetectorConfig] = None,
    ) -> Multiplicity:
        _apply_gradient_band(anomalies, layer_series, config or DetectorConfig())
        _apply_fusion_classes(anomalies, population_layers)
        layers = sorted({a.layer_index for a in anomalies if a.layer_index is not None})
        # Only report ranks that a finding actually names. Claiming "0 / 1 ranks"
        # when the anomaly carried no rank is worse than saying nothing, because
        # it reads as "the rank was fine" when nothing was assessed.
        ranks = sorted({a.rank for a in anomalies if a.rank is not None})
        population_ranks = population_ranks if ranks else 0
        steps = sorted({a.first_abnormal_step for a in anomalies})
        return Multiplicity(
            affected_ranks=ranks,
            population_ranks=population_ranks,
            affected_layers=layers,
            population_layers=population_layers,
            affected_steps=steps,
            population_steps=int(steps_available) if steps_available else 0,
            event_count=len(anomalies),
            population_count=int(steps_available) if steps_available else 0,
        )


# ---------------------------------------------------------------------------
# Mapping anomalies to candidate incident types
# ---------------------------------------------------------------------------

#: Which incident type a detector's finding suggests. The incident engine uses
#: this as a *starting point*, never as a verdict: LOSS_SPIKE can be caused by a
#: gradient explosion, and the taxonomy's job is to keep both in play.
DETECTOR_TO_INCIDENT: Dict[str, IncidentType] = {
    "loss.robust_zscore": IncidentType.LOSS_SPIKE,
    "loss.trend_slope": IncidentType.LOSS_DIVERGENCE,
    "grad.robust_zscore": IncidentType.GRADIENT_EXPLOSION,
    "grad.absolute_guard": IncidentType.GRADIENT_EXPLOSION,
    "numerics.finiteness": IncidentType.NAN_DETECTED,
    "throughput.regression": IncidentType.THROUGHPUT_REGRESSION,
}


def suggest_incident_type(anomaly: Anomaly) -> Optional[IncidentType]:
    if anomaly.detector == "numerics.finiteness":
        kind = anomaly.metadata.get("kind")
        if kind == "inf":
            return IncidentType.INF_DETECTED
        if kind == "nan":
            return IncidentType.NAN_DETECTED
    return DETECTOR_TO_INCIDENT.get(str(anomaly.detector))
