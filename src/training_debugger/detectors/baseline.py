"""L2: a contamination-resistant, state-aware gradient baseline.

Phase 1.7 measured the defect this module exists to fix, on 680 real runs:

    healthy peak l1_aggregate : p50  3.08   p95 17.07   p99 36.16   max 56.30
    fault   peak l1_aggregate : min  1.35   p50  8.77   p95 46.73
    P(CONFIRMED | magnitude)  : flat <= 0.10 from x1.5 to x12, then 1.00 at x25

The feature was ``layer_norm[step] / trailing median of that layer``. A **sustained
uniform** fault scaling drags the trailing median up with it, so the ratio decays back
toward 1 and the fault erases itself from the very statistic meant to expose it. No
threshold could fix that, and Phase 1.7 spent 680 runs proving it.

The fix is the brief's L2 design -- a baseline that is **state-aware**:

    NORMAL      -> learns
    SUSPECT     -> FREEZES; the candidate regime is never absorbed
    CONFIRMED   -> stays frozen
    RECOVERING  -> stays protected
    CLOSED      -> resumes only after the recovery policy completes

Two properties matter and are both tested:

**Freeze on first suspicion, not on confirmation.** By the time a fault is confirmed
it has already contaminated any rolling baseline. The reference must be captured at
the first *hint* of deviation, when the baseline is still clean.

**A false suspicion must not poison the baseline.** After a rejected candidate the
baseline resumes, but it must not bulk-absorb the suspect window -- otherwise a
transient excursion permanently shifts what "normal" means.

Baselines are computed offline over an emitted series, so nothing here touches the
training hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "AdaptiveBaseline",
    "adaptive_ratio_series","BaselineState", "ProtectedBaseline", "protected_ratio_series"]


class BaselineState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    CONFIRMED = "CONFIRMED"
    RECOVERING = "RECOVERING"
    CLOSED = "CLOSED"


@dataclass
class ProtectedBaseline:
    """A trailing reference that stops learning when something looks wrong.

    ``freeze_at_ratio`` is the *suspicion* threshold: a value this far above the
    current reference freezes it. It is deliberately far below any confirmation
    threshold, because its only job is to capture a clean reference early.
    """

    minimum_history: int = 20
    window: int = 150
    #: Above this multiple of the reference, the baseline stops learning.
    freeze_at_ratio: float = 1.3
    #: Steps of normal behaviour required before learning resumes after a freeze.
    resume_after_steps: int = 40

    reference: Optional[float] = None
    reference_dispersion: Optional[float] = None
    last_update_step: Optional[int] = None
    frozen: bool = False
    reason: str = "no history"
    freeze_step: Optional[int] = None
    state: str = BaselineState.NORMAL.value
    _history: List[float] = field(default_factory=list)
    _quiet_since: Optional[int] = None

    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self.reference is not None and len(self._history) >= self.minimum_history

    def ratio(self, value: float) -> Optional[float]:
        """Deviation from the reference. ``None`` when the baseline is not ready.

        Returning ``None`` rather than a number is the point: an unready baseline
        means the statistical verdict is **INSUFFICIENT_EVIDENCE**, not "normal".
        Classifying during warm-up is what the brief's L2.4 forbids.
        """
        if not self.ready or value is None or value != value:
            return None
        if self.reference is None or self.reference <= 0:
            return None
        return value / self.reference

    def update(self, step: int, value: float, *, state: Optional[str] = None) -> None:
        """Offer an observation. Whether it is absorbed depends on the state."""
        if value is None or value != value:
            return
        if state is not None:
            self.state = state

        absorbing = self.state in (BaselineState.NORMAL.value, BaselineState.CLOSED.value)

        if not absorbing:
            # SUSPECT / CONFIRMED / RECOVERING: the reference must not absorb the
            # regime under suspicion. This single guard is the whole fix.
            self.frozen = True
            self.reason = "frozen: state is {}".format(self.state)
            self.last_update_step = step
            return

        if self.frozen:
            # Learning resumes only after a quiet period, so a rejected candidate
            # cannot permanently redefine normal.
            if self._quiet_since is None:
                self._quiet_since = step
            if step - self._quiet_since < self.resume_after_steps:
                self.last_update_step = step
                return
            self.frozen = False
            self.reason = "resumed after {} quiet steps".format(self.resume_after_steps)
            self._quiet_since = None

        self._history.append(value)
        self.last_update_step = step
        reference = _median(self._history[-self.window:])
        self.reference = reference
        self.reference_dispersion = _mad(self._history[-self.window:])
        self.reason = "learning"

    def note_suspicious(self, step: int) -> None:
        """Record that this step still looks suspicious, restarting the quiet clock.

        Without this the quiet clock can be started once by a *later* normal step and
        then accumulate across intervening excursions.
        """
        self._quiet_since = None
        self.last_update_step = step

    def try_resume(self, step: int, value: float) -> bool:
        """Offer a normal-looking step. Returns True if learning resumed.

        The baseline freezes on suspicion and is supposed to resume after a quiet
        period, but ``protected_ratio_series`` never moved ``state`` out of ``SUSPECT``,
        and ``update`` only absorbs while the state is ``NORMAL`` or ``CLOSED``. The
        resume branch was therefore **unreachable**, and the reference stayed frozen at
        its first suspicion for the rest of the run -- which is how a healthy run whose
        benign excursions are closer together than one quiet period gets measured
        against an early-training reference and reported as an explosion.
        """
        if not self.frozen:
            return False
        self.update(step, value, state=BaselineState.NORMAL.value)
        return not self.frozen

    def suspect(self, step: int, reason: str) -> None:
        self.state = BaselineState.SUSPECT.value
        if self.freeze_step is None:
            self.freeze_step = step
        self.frozen = True
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "reference_dispersion": self.reference_dispersion,
            "last_update_step": self.last_update_step,
            "frozen": self.frozen,
            "reason": self.reason,
            "freeze_step": self.freeze_step,
            "state": self.state,
            "ready": self.ready,
        }


@dataclass
class AdaptiveBaseline:
    """An EMA reference whose statistics are updated with *clipped* observations.

    This is the alternative to freezing, and it comes from the external review:
    ZClip (arXiv 2504.02507, §2.3.4) solves the same problem -- a spike must not be
    absorbed into the reference -- without freezing. When an observation is a spike the
    running statistics are updated with the **clipped** value rather than the raw one or
    nothing at all:

        including the spike skews the statistics and desensitises future detection;
        excluding it entirely biases them low and makes the detector permanently
        over-aggressive.

    Freezing does only the second half. This project measured the cost of that: with the
    freeze, `resume_after_steps` never fired, precision fell 0.973 -> 0.964 and recall
    0.468 -> 0.457 (D-010). The EMA variant adapts to non-stationarity *and* resists
    contamination.

    It also removes the readiness defect measured in v0.1.7: the frozen baseline needs
    ``baseline_floor + minimum_history`` (50 + 20 = 70) steps before it produces a
    usable ratio, which is what made C3 unconstructible. This needs only ``warmup``.
    """

    warmup: int = 25
    alpha: float = 0.97          # ZClip default EMA smoothing
    z_threshold: float = 2.5     # ZClip default spike threshold
    mean: Optional[float] = None
    variance: Optional[float] = None
    ready_step: Optional[int] = None
    _warmup_values: List[float] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.mean is not None and self.mean > 0

    def ratio(self, value: float) -> Optional[float]:
        """Deviation from the current reference. ``None`` until the baseline is ready.

        Returning ``None`` rather than a number is deliberate and matches the frozen
        baseline: an unready baseline means INSUFFICIENT_EVIDENCE, never "normal".
        """
        if not self.ready or value is None or value != value:
            return None
        return value / self.mean

    def update(self, step: int, value: float) -> None:
        if value is None or value != value or not value > 0:
            return
        if self.mean is None:
            self._warmup_values.append(value)
            if len(self._warmup_values) < self.warmup:
                return
            self.mean = _median(self._warmup_values)
            self.variance = _mad(self._warmup_values) ** 2
            self.ready_step = step
            return

        sigma = math.sqrt(self.variance) if self.variance and self.variance > 0 else 0.0
        z = (value - self.mean) / sigma if sigma > 0 else 0.0
        # The clipped update: a spike contributes the clipped value, never the raw one
        # and never nothing.
        if sigma > 0 and z > self.z_threshold:
            update = self.mean + (self.z_threshold ** 2 / z) * sigma
        else:
            update = value
        previous = self.mean
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * update
        self.variance = (self.alpha * (self.variance or 0.0)
                         + (1.0 - self.alpha) * (update - self.mean) ** 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": "ema",
            "warmup": self.warmup, "alpha": self.alpha, "z_threshold": self.z_threshold,
            "reference": self.mean,
            "reference_dispersion": math.sqrt(self.variance) if self.variance else None,
            "ready_step": self.ready_step,
            "ready": self.ready,
        }


def adaptive_ratio_series(
    series: Sequence[Tuple[int, float]],
    *,
    warmup: int = 25,
    alpha: float = 0.97,
    z_threshold: float = 2.5,
) -> Dict[int, Optional[float]]:
    """Ratio-to-reference using an EMA baseline with clipped statistic updates.

    Same contract as :func:`protected_ratio_series` -- ``None`` until ready -- so the
    two are drop-in alternatives and can be compared on identical evidence.
    """
    baseline = AdaptiveBaseline(warmup=warmup, alpha=alpha, z_threshold=z_threshold)
    ratios: Dict[int, Optional[float]] = {}
    for step, value in series:
        if value is None or value != value:
            ratios[step] = None
            continue
        current = baseline.ratio(value)
        ratios[step] = current
        baseline.update(step, value)
    return ratios


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    n = len(ordered)
    return float(ordered[n // 2]) if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _mad(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    median = _median(values)
    return _median([abs(v - median) for v in values])


def protected_ratio_series(
    series: Sequence[Tuple[int, float]],
    *,
    minimum_history: int = 20,
    window: int = 150,
    freeze_at_ratio: float = 1.3,
    resume_after_steps: int = 40,
) -> Dict[int, Optional[float]]:
    """Ratio-to-reference per step, using a freeze-on-suspicion baseline.

    Walks the series once. At each step the value is compared against the reference
    built from **everything before it since the last resume**; if the value is
    suspicious the reference freezes at its pre-suspicion level, so a subsequent
    sustained shift is measured against the healthy regime rather than against
    itself.
    """
    baseline = ProtectedBaseline(
        minimum_history=minimum_history, window=window,
        freeze_at_ratio=freeze_at_ratio, resume_after_steps=resume_after_steps,
    )
    ratios: Dict[int, Optional[float]] = {}
    for step, value in series:
        if value is None or value != value:
            ratios[step] = None
            continue
        if not value > 0:
            ratios[step] = None
            baseline.update(step, value)
            continue
        current = baseline.ratio(value)
        ratios[step] = current
        if current is not None and current >= freeze_at_ratio:
            if not baseline.frozen:
                baseline.suspect(
                    step, "value {:.3f} is {:.2f}x the reference".format(value, current)
                )
            # Still suspicious: restart the quiet clock rather than letting it run.
            baseline.note_suspicious(step)
            baseline.update(step, value)
        elif baseline.frozen:
            # The value looks normal again. Try to resume learning after the quiet
            # period, so a *false* suspicion -- a benign excursion -- cannot pin the
            # reference at an early-training level for the rest of the run.
            baseline.try_resume(step, value)
        else:
            baseline.update(step, value)
    return ratios
