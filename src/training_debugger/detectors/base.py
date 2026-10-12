"""Detector contracts.

Phase 1 brief section 7 is unambiguous that detection must be deterministic:
*"禁止使用 LLM 检测异常"*. So every detector here is a pure statistical function
over a series, with:

* **no learned parameters** -- thresholds are configured, and calibrated per run;
* **robust statistics by default** -- a mean/σ baseline is self-masking when the
  outliers being hunted are *in* the baseline window;
* **a mandatory warm-up** -- a detector that fires during the first steps of a
  run is a false-alarm generator, because the baseline is not yet meaningful;
* **an inspectable result** -- every anomaly carries the detector id, the
  baseline, the threshold, the observed value, the window, and a confidence.

Confidence here is a claim about *how the number was obtained*, not about how
interesting it is:

``OBSERVED``
    measured directly, no inference. The only detector in this category is the
    finiteness check: NaN is not a statistical claim.
``INFERRED``
    produced by a named statistical procedure, which must therefore be reported.

Nothing in this package imports an LLM or an agent, and nothing here writes to
storage. Layer direction is enforced by ``tests/unit/test_layering.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..schema.enums import DetectorId, EvidenceLevel, EventPriority, Severity

__all__ = [
    "Anomaly",
    "DetectorConfig",
    "Series",
    "DetectorContext",
    "BaseDetector",
    "robust_stats",
    "theil_sen_slope",
    "trailing_median",
    "robust_local_trend",
    "detrended_residuals",
]

#: ``[(step, value)]`` in ascending step order.
Series = Sequence[Tuple[int, float]]


class ConfidenceClass(str, Enum):
    """How much a single detector fire is worth on its own (D5.1).

    Derived from the calibration data, not chosen by preference:

    * ``CERTAIN`` -- a non-finite value was measured. Not a statistical claim, so
      nothing can make it wrong. This is the only class that may open an incident
      alone.
    * ``STRONG`` -- the *aggregate* signal departed. A real gradient explosion
      moved the global norm 15.7x at the injection step; healthy tail events did
      not move it at all (3.26 against a 4.48 threshold).
    * ``WEAK`` -- a single component departed while its peers did not. Measured:
      healthy runs produce exactly this (1 of 7 layers, aggregate silent), and it
      is the mechanism behind the 50 false positives of detector v1.0.
    """

    CERTAIN = "CERTAIN"
    STRONG = "STRONG"
    WEAK = "WEAK"


@dataclass
class DetectorConfig:
    """Thresholds and windows. All of them are configuration, none are learned.

    Defaults are deliberately conservative: Phase 1 has no labelled incident
    corpus to calibrate against, so the defaults favour *not* firing. Section 4
    records the resulting false-positive behaviour on a healthy corpus.
    """

    window: int = 200
    warmup_steps: int = 50
    min_samples: int = 20

    # robust z-score threshold, in units of 1.4826*MAD
    spike_z: float = 6.0
    #: Evaluate the z-score on residuals from a robust local linear trend rather
    #: than on the raw values. Without this, a decaying loss curve is undetectable:
    #: over a 200-step window the decay itself dominates the MAD, and a 4.5x jump
    #: scored z=0.93 in the A1 scenario. Detrending makes the window measure noise
    #: rather than slope.
    detrend: bool = True
    #: Secondary, scale-free criterion: fire when the value exceeds the window
    #: median by this factor. Interpretable ("loss rose 4.5x") and immune to the
    #: MAD estimate being wrong for any reason.
    relative_factor: Optional[float] = 2.0

    # gradient guard: an absolute value that is alarming regardless of baseline
    grad_absolute_guard: Optional[float] = None
    #: --- gradient criterion (v2.0, asymmetric envelope) -------------------
    #: Measured on 30 healthy real runs (6000 observations), the healthy global
    #: gradient norm is heavy-tailed and non-stationary:
    #:
    #:     p50 1.64   p95 2.56   p99 3.18   p99.9 4.43   max 5.20
    #:     excess kurtosis 3.55 (Gaussian is 0)   Hill index 6.71
    #:     median peak / median 2.15     lag-1 autocorrelation 0.29
    #:     warm-up median is 30 % below the steady-state median
    #:
    #: A run therefore reaches 3.2x its own median with no fault at all. Any
    #: threshold that a healthy run touches by accident is a false-positive
    #: generator, which is exactly what the v1.0 z-criterion was: it fired on a
    #: healthy run at 1.51x the window peak, 0.7 % above its own guard.
    #:
    #: Eight detector families were evaluated on the same 30 healthy runs and the
    #: same injected faults (``tests/experiments/gradient_distribution.py``):
    #:
    #:     family                    healthy FP/1k   recall x2   recall x3   recall x10
    #:     A rolling quantile                  2.00        0.75        1.00         1.00
    #:     B peak relative (v1.0)              0.00        0.00        0.00         1.00
    #:     C log robust z                      0.50        0.00        0.00         1.00
    #:     D EWMA                              3.67        0.25        1.00         1.00
    #:     E CUSUM                           147.17        1.00        1.00         1.00
    #:     F asymmetric envelope               0.17        0.75        1.00         1.00
    #:     G historical max                    0.00        0.00        0.00         1.00
    #:
    #: F wins: it is the only family that reaches the FP budget while detecting a
    #: 2x perturbation. B and G are too blunt (blind below 10x); E is unusable;
    #: C is unstable on heavy-tailed healthy input (6.25 FP/1k).
    grad_envelope_quantile: float = 0.995
    #: Fire when the value exceeds `quantile(window, grad_envelope_quantile)` by
    #: this factor.
    grad_envelope_factor: float = 1.5
    #: ...or when it exceeds the window median by this multiple, whichever is the
    #: weaker requirement. The two together handle a quiet run (where the
    #: envelope is tight) and a noisy one (where the median multiple guards).
    grad_median_multiple: float = 3.0
    #: Whether a rolling baseline may include the warm-up region.
    #:
    #: Measured (30 healthy real runs): the median gradient norm over the first
    #: 20 % of training is **30 % below** the steady-state median. A baseline that
    #: includes those steps is therefore biased low, and a perfectly ordinary
    #: mid-training value can exceed ``median x 3`` against it. This was not a
    #: theoretical concern: with the contaminated baseline the detector fired at
    #: step 60 of a real 130-step run whose fault was injected at step 80 -- a
    #: pre-injection false positive produced purely by non-stationarity.
    #:
    #: Excluding the warm-up from the baseline is the phase-aware baseline the
    #: brief asks for. It is a *narrower* window, never a wider tolerance: no
    #: threshold is loosened, only the population the threshold is computed from.
    baseline_excludes_warmup: bool = True

    #: --- evidence / alert separation (Phase 1.12A) ----------------------
    #: Whether a detector cooldown may suppress an *observation*.
    #:
    #: Phase 1.11 found the contract defect: ``loss.robust_zscore`` applies
    #: ``cooldown_steps = 50`` and the ``LOSS_SPIKE`` episode rule requires **2
    #: observations within 6 steps**. The detector's own cooldown made its own
    #: confirmation rule unsatisfiable, so ``LOSS_SPIKE`` could never confirm and
    #: ``CANDIDATE_ONLY`` was unreachable.
    #:
    #: The two concepts were conflated:
    #:
    #:     evidence observation   -- every qualifying observation, feeds the episode
    #:     alert deduplication    -- one notification per episode, not per observation
    #:
    #: Deduplication belongs downstream, where it already exists: ``EpisodeEngine``
    #: groups contiguous observations into **one** episode and ``fuse()`` returns
    #: **one** verdict per anomaly type. Suppressing at the detector removes evidence
    #: the episode needs while adding no deduplication the downstream layers lack.
    #:
    #: When False, a qualifying observation during cooldown is still emitted and is
    #: marked ``cooldown_suppressed`` in its metadata, so an alert-counting consumer
    #: can still filter while the evidence chain stays complete.
    cooldown_suppresses_evidence: bool = False

    #: Quiet steps required before the protected gradient baseline resumes learning
    #: after a freeze. Exposed as configuration because it was dead code until
    #: Phase 1.13 -- `protected_ratio_series` never moved the state out of SUSPECT --
    #: so its value had never actually been exercised. Swept in W2
    #: (`outputs/alpha/W2_PRE_REGISTRATION.md`).
    baseline_resume_after_steps: int = 40

    #: Which reference the gradient ratio is measured against.
    #:
    #: ``"frozen"`` -- the Phase 1.8 protected baseline: freeze at the first hint of
    #: deviation, resume after a quiet period. **The default**, so measured behaviour is
    #: unchanged unless a caller opts in.
    #:
    #: ``"ema"`` -- an exponential moving average whose statistics are updated with
    #: *clipped* observations instead of freezing (ZClip, arXiv 2504.02507 §2.3.4).
    #: It adapts to non-stationarity, resists spike contamination, and -- because it
    #: does not need `baseline_floor + minimum_history` before its first ratio -- removes
    #: the readiness blindness that v0.1.7 root-caused and that made C3 unconstructible.
    baseline_mode: str = "frozen"
    #: Warm-up for ``baseline_mode="ema"``. The frozen baseline is blind for
    #: ``warmup_steps + minimum_history`` = 70 steps; the EMA needs only this.
    baseline_warmup_steps: int = 25
    baseline_ema_alpha: float = 0.97
    baseline_ema_z: float = 2.5
    #: Phase 1.4 measured that the univariate ``global_norm_ratio`` is **worse than
    #: chance** below x2 (AUC 0.25 at x1.5, 0.29 at x2.0) -- a mild fault produces a
    #: *lower* global ratio than a healthy heavy-tail run, because the healthy
    #: class's elevated trailing median depresses the ratio. The joint
    #: ``l1_aggregate`` -- each layer normalised by its own trailing baseline, then
    #: summed -- reaches AUC 0.96 at x2.5 and 1.00 at x3.0.
    #:
    #: The detector therefore no longer answers a yes/no question. It returns a
    #: three-way band, and the middle band is where the system abstains:
    #:
    #:     l1 <  A          -> NORMAL      (no episode)
    #:     A <= l1 < B      -> UNCERTAIN   (SUSPECT; cannot confirm)
    #:     l1 >= B          -> CONFIRMED   (strong candidate)
    #:
    #: Boundaries measured on the development corpus over BOTH healthy classes; A is
    #: the empirical maximum (33.32) plus margin, B is the fault x4.0 minimum. See
    #: ``outputs/phase1-5/gradient_boundary_registry.json``.
    gradient_band_enabled: bool = True
    #: Same frozen operating point, expressed layer-count-invariantly. The
    #: original 34.0 / 46.0 were fitted on a 4-layer model where the aggregate was a
    #: SUM; dividing by the 4 monitored layers gives exactly 8.5 / 11.5 and makes the
    #: boundary valid for any model shape. This is a correctness fix to the feature,
    #: not a retune: a 4-layer run is classified identically to before.
    gradient_agg_normal_below: float = 8.5
    gradient_agg_confirm_at: float = 11.5
    #: Minimum monitored layers before the aggregate is meaningful.
    gradient_agg_min_layers: int = 2

    # throughput
    throughput_drop_ratio: float = 0.25       # a 25% sustained drop
    throughput_consecutive: int = 10          # sustained for this many samples
    throughput_baseline_window: int = 100

    # divergence
    divergence_window: int = 100
    divergence_slope_min: float = 0.0
    divergence_min_points: int = 30

    # suppression
    cooldown_steps: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window,
            "warmup_steps": self.warmup_steps,
            "min_samples": self.min_samples,
            "spike_z": self.spike_z,
            "detrend": self.detrend,
            "relative_factor": self.relative_factor,
            "grad_absolute_guard": self.grad_absolute_guard,
            "grad_envelope_quantile": self.grad_envelope_quantile,
            "grad_envelope_factor": self.grad_envelope_factor,
            "grad_median_multiple": self.grad_median_multiple,
            "baseline_excludes_warmup": self.baseline_excludes_warmup,
            "gradient_band_enabled": self.gradient_band_enabled,
            "gradient_agg_normal_below": self.gradient_agg_normal_below,
            "gradient_agg_confirm_at": self.gradient_agg_confirm_at,
            "gradient_agg_min_layers": self.gradient_agg_min_layers,
            "cooldown_suppresses_evidence": self.cooldown_suppresses_evidence,
            "baseline_resume_after_steps": self.baseline_resume_after_steps,
            "baseline_mode": self.baseline_mode,
            "baseline_warmup_steps": self.baseline_warmup_steps,
            "baseline_ema_alpha": self.baseline_ema_alpha,
            "baseline_ema_z": self.baseline_ema_z,
            "throughput_drop_ratio": self.throughput_drop_ratio,
            "throughput_consecutive": self.throughput_consecutive,
            "throughput_baseline_window": self.throughput_baseline_window,
            "divergence_window": self.divergence_window,
            "divergence_min_points": self.divergence_min_points,
            "cooldown_steps": self.cooldown_steps,
        }


@dataclass
class DetectorContext:
    """Everything a detector may know beyond the series itself."""

    run_id: str = ""
    rank: Optional[int] = None
    metric_name: str = ""
    unit: Optional[str] = None
    layer_index: Optional[int] = None
    module_path: Optional[str] = None
    population_layers: int = 0
    population_ranks: int = 0
    other_series: Dict[str, Series] = field(default_factory=dict)


@dataclass
class Anomaly:
    """One detector firing. Everything needed to audit the decision."""

    detector: DetectorId
    detector_version: str
    metric_name: str
    first_abnormal_step: int
    observed_value: float
    baseline: Optional[float]
    threshold: Optional[float]
    score: float
    confidence: float
    level: EvidenceLevel
    severity: Severity
    window: Tuple[int, int]
    rank: Optional[int] = None
    layer_index: Optional[int] = None
    module_path: Optional[str] = None
    sample_count: int = 0
    reason: str = ""
    #: D5.1 fusion class. Defaults to STRONG so a detector that forgets to set
    #: it is never silently promoted to CERTAIN.
    confidence_class: ConfidenceClass = ConfidenceClass.STRONG
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_observed(self) -> bool:
        return self.level is EvidenceLevel.OBSERVED

    def priority(self) -> EventPriority:
        if self.level is EvidenceLevel.OBSERVED:
            return EventPriority.P0
        return EventPriority.P1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": str(self.detector),
            "detector_version": self.detector_version,
            "metric_name": self.metric_name,
            "first_abnormal_step": self.first_abnormal_step,
            "observed_value": self.observed_value,
            "baseline": self.baseline,
            "threshold": self.threshold,
            "score": self.score,
            "confidence": self.confidence,
            "level": str(self.level),
            "severity": str(self.severity),
            "window": list(self.window),
            "rank": self.rank,
            "layer_index": self.layer_index,
            "module_path": self.module_path,
            "sample_count": self.sample_count,
            "reason": self.reason,
            "confidence_class": str(self.confidence_class),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------


def robust_stats(values: Sequence[float], window: Optional[int] = None) -> Dict[str, float]:
    """Median and MAD over the trailing ``window`` values.

    MAD is scaled by 1.4826 so that it estimates σ for normal data, which makes
    the resulting z-score comparable to a familiar standard deviation -- while
    remaining insensitive to the outliers being hunted.

    Returns ``mad == 0.0`` when more than half the window is identical, which the
    callers must treat as "no dispersion information" rather than dividing by it.
    """
    usable = [v for v in values if v == v and abs(v) != float("inf")]
    if window is not None and len(usable) > window:
        usable = usable[-window:]
    if not usable:
        return {"median": float("nan"), "mad": 0.0, "count": 0, "min": float("nan"), "max": float("nan")}
    ordered = sorted(usable)
    count = len(ordered)
    median = ordered[count // 2] if count % 2 else 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])
    deviations = sorted(abs(v - median) for v in ordered)
    raw_mad = deviations[count // 2] if count % 2 else 0.5 * (deviations[count // 2 - 1] + deviations[count // 2])
    return {
        "median": median,
        "mad": 1.4826 * raw_mad,
        "count": count,
        "min": ordered[0],
        "max": ordered[-1],
    }


def robust_local_trend(steps: Sequence[int], values: Sequence[float]) -> Tuple[float, float]:
    """Robust linear trend: ``(slope_per_step, intercept)`` with ``y = intercept + slope*x``.

    Fitted through two robust anchor points -- the median of the first third and
    the median of the last third of the window -- rather than by least squares.
    Least squares would be dragged by the very deviation being tested, and a
    Theil-Sen fit over all pairs is O(n^2) per index, which is too slow to run per
    step.

    Anchoring on two real points (rather than on a slope plus a global median)
    matters: a decaying curve is convex, so a line anchored at the window centre
    extrapolates wildly at the window edge.

    Returns ``(0.0, median(values))`` when the window is too short or degenerate.
    """
    n = len(values)
    if n < 6:
        return 0.0, robust_stats(values)["median"]
    k = max(1, n // 3)
    left_x = float(robust_stats(list(steps[:k]))["median"])
    left_y = robust_stats(values[:k])["median"]
    right_x = float(robust_stats(list(steps[-k:]))["median"])
    right_y = robust_stats(values[-k:])["median"]
    dx = right_x - left_x
    if dx <= 0:
        return 0.0, robust_stats(values)["median"]
    slope = (right_y - left_y) / dx
    intercept = left_y - slope * left_x
    return slope, intercept


def detrended_residuals(
    steps: Sequence[int], values: Sequence[float]
) -> Tuple[List[float], float, float]:
    """Residuals after removing a robust local linear trend.

    Returns ``(residuals, slope_per_step, intercept)`` where the trend is
    ``y = intercept + slope * step``, so a caller can compute the expected value
    at any step in or just after the window.
    """
    slope, intercept = robust_local_trend(steps, values)
    residuals = [value - (intercept + slope * step) for step, value in zip(steps, values)]
    return residuals, slope, intercept



def trailing_median(values: Sequence[float], window: int) -> float:
    stats = robust_stats(values, window)
    return stats["median"]


def theil_sen_slope(points: Sequence[Tuple[int, float]], max_pairs: int = 2000) -> float:
    """Median of pairwise slopes. Robust to outliers, unlike least squares.

    Deterministic: pairs are enumerated in a fixed order and the median of an
    even-length list is the lower-middle element, not an average, so two runs
    produce bit-identical output.
    """
    usable = [(x, y) for x, y in points if y == y and abs(y) != float("inf")]
    if len(usable) < 2:
        return 0.0
    slopes: List[float] = []
    n = len(usable)
    for i in range(n):
        for j in range(i + 1, n):
            dx = usable[j][0] - usable[i][0]
            if dx == 0:
                continue
            slopes.append((usable[j][1] - usable[i][1]) / dx)
            if len(slopes) >= max_pairs:
                break
        if len(slopes) >= max_pairs:
            break
    if not slopes:
        return 0.0
    slopes.sort()
    return slopes[len(slopes) // 2]


class BaseDetector:
    """Common behaviour: warm-up, windowing, and suppression bookkeeping."""

    #: Overridden by subclasses.
    detector_id: DetectorId = DetectorId.LOSS_ROBUST_ZSCORE
    version: str = "1.0"
    metric_names: Tuple[str, ...] = ()

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._last_fired_step: Dict[str, int] = {}

    # -- helpers -----------------------------------------------------------

    def _in_cooldown(self, key: str, step: int) -> bool:
        previous = self._last_fired_step.get(key)
        if previous is None:
            return False
        return (step - previous) < self.config.cooldown_steps

    def _mark_fired(self, key: str, step: int) -> None:
        self._last_fired_step[key] = step

    def _enough_history(self, series: Series, index: int) -> bool:
        if index + 1 < self.config.warmup_steps:
            return False
        if index + 1 < self.config.min_samples:
            return False
        return True

    def description(self) -> Dict[str, Any]:
        return {
            "detector": str(self.detector_id),
            "version": self.version,
            "metric_names": list(self.metric_names),
            "config": self.config.to_dict(),
        }

    def detect(self, series: Series, context: DetectorContext) -> List[Anomaly]:
        raise NotImplementedError

    def reset(self) -> None:
        self._last_fired_step.clear()
