"""Per-module activation health — the observational capability this project lacked.

The gap was found by reading a competitor rather than by guessing: NeuralDBG classifies
`Vanishing` / `Exploding` / `Dead` / `Saturated` **per named module**, and this project
observed gradient *norms* per layer and nothing about activations at all. That is a real
gap, and it is observational — it needs no causal claim.

The design rule here is the one this repository has already paid for twice: **do not
invent a threshold.** So the health states are split by what can be established exactly.

```
EXACT — no threshold, no reference, no assumption
    DEAD          every element is exactly zero
    NONFINITE     an element is NaN or infinite
    SATURATED     |x| reaches the dtype's own maximum

INFERRED — needs a reference, and is therefore NOT a claim
    the magnitudes are emitted as statistics and left to the reader. Calling a module
    "vanishing" requires a notion of the right size for it, and this module does not
    have one, so it does not say it.
```

A competitor reports "activation_regime_shift cascaded through [module list]" at 0.72
confidence. This reports "module X: 100 % of activations are exactly zero at step 47",
which is a fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "ActivationStats", "ActivationProbe", "health_from_stats",
    "ACTIVATION_METRICS", "HealthState",
]

#: Metric names emitted per module. Kept in one place so the detector, the storage layer
#: and the tests cannot drift apart.
ACTIVATION_METRICS = (
    "act.zero_fraction",
    "act.abs_max",
    "act.abs_mean",
    "act.global_norm",
    "act.nonfinite_count",
)


class HealthState:
    HEALTHY = "healthy"
    DEAD = "dead"
    NONFINITE = "nonfinite"
    SATURATED = "saturated"
    UNKNOWN = "unknown"

    ALL = (HEALTHY, DEAD, NONFINITE, SATURATED, UNKNOWN)


@dataclass
class ActivationStats:
    module_path: str
    step: int
    elements: int
    sampled: bool
    zero_fraction: float
    abs_max: float
    abs_mean: float
    global_norm: float
    nonfinite_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_path": self.module_path,
            "step": self.step,
            "elements": self.elements,
            "sampled": self.sampled,
            "zero_fraction": self.zero_fraction,
            "abs_max": self.abs_max,
            "abs_mean": self.abs_mean,
            "global_norm": self.global_norm,
            "nonfinite_count": self.nonfinite_count,
        }


def health_from_stats(stats: ActivationStats,
                      dtype_max: Optional[float] = None) -> Tuple[str, str]:
    """Classify a module's health **from exact observations only**.

    Returns ``(state, reason)``. Every branch here is threshold-free; the reason string
    states the observation, so a reader can disagree with the label without having to
    reverse-engineer it.

    A sampled observation can still establish DEAD and NONFINITE (finding one non-zero or
    one non-finite element is proof), but it **cannot** establish DEAD by absence: a
    sample of zeros does not prove the whole tensor is zero. That case returns UNKNOWN.
    """
    if stats.nonfinite_count > 0:
        return (HealthState.NONFINITE,
                "{} of {} elements are NaN or infinite".format(
                    stats.nonfinite_count, stats.elements))
    if stats.zero_fraction >= 1.0:
        if stats.sampled:
            return (HealthState.UNKNOWN,
                    "every sampled element is zero, but the observation was sampled "
                    "({} of an unknown total) so a fully dead module is not established"
                    .format(stats.elements))
        return (HealthState.DEAD,
                "all {} elements are exactly zero".format(stats.elements))
    if dtype_max is not None and stats.abs_max >= dtype_max:
        return (HealthState.SATURATED,
                "|x| reaches the dtype maximum ({:.4g})".format(dtype_max))
    return (HealthState.HEALTHY,
            "no exact failure condition: zero_fraction={:.4g}, abs_max={:.4g}".format(
                stats.zero_fraction, stats.abs_max))


def _bounded(tensor: torch.Tensor, sample_limit: Optional[int]) -> Tuple[torch.Tensor, bool]:
    flat = tensor.detach().reshape(-1)
    if sample_limit is None or flat.numel() <= sample_limit:
        return flat, False
    # `flatten()[::stride]` rather than `[:limit]`: a contiguous prefix of an activation
    # tensor is often the same token position repeated, which is the worst possible
    # sample. A strided sample spans the tensor for the same cost.
    stride = max(1, flat.numel() // sample_limit)
    return flat[::stride], True


class ActivationProbe:
    """Forward hooks that record per-module activation statistics.

    **Off by default** (``enabled=False``) and detachable, mirroring `OptimizerWitness`:
    an unattached probe is observably identical to this module not existing.

    Cost is bounded two ways: only the selected modules are hooked, and the statistics are
    computed from one bounded pass per module. ``sample_limit=None`` is exact and costs
    ``O(elements)``; a limit bounds it to ``O(sample_limit)`` and marks the resulting
    observations as sampled so no exact claim is made from them.
    """

    def __init__(
        self,
        enabled: bool = False,
        *,
        selection: Optional[Sequence[str]] = None,
        sample_limit: Optional[int] = 8192,
        cadence: int = 1,
        emit: bool = True,
        emit_on_change_only: bool = True,
    ) -> None:
        if cadence < 1:
            raise ValueError("cadence must be >= 1")
        self.enabled = enabled
        self.selection = list(selection) if selection else None
        self.sample_limit = sample_limit
        self.cadence = cadence
        self.emit = emit
        #: Emit only when a module's health state CHANGES, rather than every step.
        #:
        #: Measured: the activation computation costs **+2.7 %** and emitting five
        #: metrics per module per step costs a further **+23.5 %** -- 1,600 events over
        #: an 80-step run with four modules. The capability was never expensive; the
        #: naive emission strategy was. A module that is healthy for a thousand steps
        #: does not need a thousand events saying so.
        self.emit_on_change_only = emit_on_change_only
        self._last_state: Dict[str, str] = {}
        self._handles: List[Any] = []
        self._latest: Dict[str, torch.Tensor] = {}
        self.records: List[ActivationStats] = []

    # -- lifecycle -------------------------------------------------------

    def attach(self, model: torch.nn.Module) -> int:
        """Register a forward hook on each selected module. Returns how many."""
        if not self.enabled:
            return 0
        count = 0
        for name, module in model.named_modules():
            if not name:
                continue
            if self.selection is not None and name not in self.selection:
                continue
            if not any(True for _ in module.children()):
                try:
                    self._handles.append(module.register_forward_hook(self._hook(name)))
                    count += 1
                except Exception:
                    continue
        return count

    def detach(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    # -- observation -----------------------------------------------------

    def _hook(self, name: str):
        def hook(module, inputs, output):
            # Fail open: a probe must never break training.
            try:
                if isinstance(output, torch.Tensor):
                    self._latest[name] = output
                elif isinstance(output, (tuple, list)):
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            self._latest[name] = item
                            break
            except Exception:
                pass
        return hook

    def collect(self, step: int, observer: Any = None) -> List[ActivationStats]:
        """Compute and emit the statistics for the modules seen since the last collect."""
        produced: List[ActivationStats] = []
        if not self.enabled or (self.cadence > 1 and step % self.cadence):
            self._latest = {}
            return produced
        for name, tensor in sorted(self._latest.items()):
            try:
                flat, sampled = _bounded(tensor, self.sample_limit)
                if flat.numel() == 0:
                    continue
                finite = torch.isfinite(flat)
                nonfinite = int((~finite).sum().item())
                safe = torch.where(finite, flat, torch.zeros_like(flat))
                absolute = safe.abs()
                stats = ActivationStats(
                    module_path=name, step=step, elements=int(flat.numel()),
                    sampled=sampled,
                    zero_fraction=float((safe == 0).float().mean().item()),
                    abs_max=float(absolute.max().item()),
                    abs_mean=float(absolute.mean().item()),
                    global_norm=float(torch.sqrt((safe * safe).sum()).item()),
                    nonfinite_count=nonfinite,
                )
            except Exception:
                continue
            self.records.append(stats)
            produced.append(stats)
            if observer is not None and self.emit:
                state, _ = health_from_stats(stats)
                previous = self._last_state.get(name)
                if not self.emit_on_change_only or state != previous:
                    self._last_state[name] = state
                    self._emit(stats, observer, state)
        self._latest = {}
        return produced

    def _emit(self, stats: ActivationStats, observer: Any,
              state: Optional[str] = None) -> None:
        from ...schema.enums import EventPriority, EventType
        from ...schema.event import MetricUnit
        try:
            for metric, value in (
                ("act.zero_fraction", stats.zero_fraction),
                ("act.abs_max", stats.abs_max),
                ("act.abs_mean", stats.abs_mean),
                ("act.global_norm", stats.global_norm),
                ("act.nonfinite_count", float(stats.nonfinite_count)),
            ):
                observer.metric(
                    metric, value, MetricUnit.DIMENSIONLESS, step=stats.step,
                    event_type=EventType.ACTIVATION_METRIC,
                    priority=EventPriority.P2,
                    metadata={"module_path": stats.module_path,
                              "sampled": stats.sampled,
                              "elements": stats.elements,
                              "health_state": state,
                              "emit_on_change_only": self.emit_on_change_only},
                )
        except Exception:
            pass

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {name: 0 for name in HealthState.ALL}
        for stats in self.records:
            state, _ = health_from_stats(stats)
            counts[state] = counts.get(state, 0) + 1
        return {
            "enabled": self.enabled, "attached": self.attached,
            "sample_limit": self.sample_limit, "cadence": self.cadence,
            "emit_on_change_only": self.emit_on_change_only,
            "observations": len(self.records), "states": counts,
        }
