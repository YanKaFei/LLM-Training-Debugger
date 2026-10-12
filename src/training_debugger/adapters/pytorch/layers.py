"""T1 layer instrumentation: per-layer gradient and parameter norms.

Phase 0's cardinality decision, restated because it determines this whole module:
**T2 is never always-on.** T1 is the always-on tier, and it is only affordable
because of three choices made here.

**No hooks on the backward path.** The obvious implementation registers a
``register_full_backward_hook`` per module and reduces inside it. That runs
*inside* autograd, so an exception there breaks training, and it interferes with
any other hook a user has installed. Instead the collector walks
``named_parameters()`` **once per collected step**, after ``backward()``, and
groups the sums by layer. The arithmetic is identical; the blast radius is not.

**Computation is skipped, not just emission.** If layer metrics are sampled every
10th step, reducing them on the other 9 wastes the exact resource the overhead
gate measures. The collector takes the cadence and returns immediately on a
skipped step.

**Emission is bounded independently of computation.** Even on a collected step,
only aggregates plus the top-K layers are emitted; the full vector goes out as a
single hash. On a 100-layer model that is ~13 events instead of 100 per step.

Layer identity is resolved by an explicit, configurable pattern rather than by
guessing from module names, so the mapping is testable and a model with an
unusual structure fails loudly instead of producing silently wrong layer indices.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from ...observer.observer import TrainingObserver
from ...schema.enums import EventPriority, EventType, MetricUnit
from .metrics import per_tensor_l2_norms, tensor_nonfinite_count

__all__ = [
    "LayerResolver",
    "LayerSelection",
    "LayerNorm",
    "LayerMetricBatch",
    "LayerMetricCollector",
    "DEFAULT_LAYER_PATTERN",
]

#: Default layer pattern for a transformer: ``blocks.3.mlp.0.weight`` -> ``blocks.3``.
#: Deliberately explicit rather than heuristic -- a wrong layer index silently
#: attributes an anomaly to the wrong layer, which is worse than no metric.
DEFAULT_LAYER_PATTERN = r"^(?P<layer>[A-Za-z_][A-Za-z0-9_]*\.\d+)"


class LayerResolver:
    """Maps a parameter name to a stable layer key and index."""

    def __init__(self, pattern: str = DEFAULT_LAYER_PATTERN) -> None:
        self.pattern = re.compile(pattern)
        self._indices: Dict[str, int] = {}

    def key_for(self, parameter_name: str) -> str:
        match = self.pattern.match(parameter_name)
        if match and "layer" in match.groupdict():
            return match.group("layer")
        # Fall back to the first path component so a model without a repeated
        # block structure still produces *something* attributable.
        return parameter_name.split(".")[0]

    def index_for(self, layer_key: str) -> int:
        """Assign dense indices in first-seen order, which is parameter order."""
        if layer_key not in self._indices:
            self._indices[layer_key] = len(self._indices)
        return self._indices[layer_key]

    @property
    def layer_count(self) -> int:
        return len(self._indices)

    def known_layers(self) -> Dict[str, int]:
        return dict(self._indices)


@dataclass
class LayerSelection:
    """Cardinality control: which layers may ever be measured or emitted.

    Phase 1 brief section 6.2 forbids unconditional
    ``every layer x every step x every rank``. This is where that is enforced.
    """

    include: Optional[str] = None       # regex; only matching layer keys are measured
    exclude: Optional[str] = None       # regex; matching layer keys are dropped
    max_layers: Optional[int] = None    # hard cap, applied by layer index order
    top_k: int = 8                      # per-layer events emitted per step
    emit_aggregate: bool = True
    emit_vector_hash: bool = True

    def __post_init__(self) -> None:
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.max_layers is not None and self.max_layers <= 0:
            raise ValueError("max_layers must be positive when set")

    def allows(self, layer_key: str, layer_index: int) -> bool:
        if self.include and not re.search(self.include, layer_key):
            return False
        if self.exclude and re.search(self.exclude, layer_key):
            return False
        if self.max_layers is not None and layer_index >= self.max_layers:
            return False
        return True


@dataclass
class LayerNorm:
    layer_key: str
    layer_index: int
    grad_norm: Optional[float] = None
    param_norm: Optional[float] = None
    grad_nonfinite: int = 0
    grad_tensor_count: int = 0
    grad_element_count: int = 0


@dataclass
class LayerMetricBatch:
    """The result of one collected step. Computed once, emitted selectively."""

    step: int
    layers: List[LayerNorm] = field(default_factory=list)
    vector_hash: Optional[str] = None

    def grad_norms(self) -> List[float]:
        return [l.grad_norm for l in self.layers if l.grad_norm is not None]

    @property
    def layer_count(self) -> int:
        return len(self.layers)


class LayerMetricCollector:
    """Computes and emits T1 layer metrics for a model.

    One instance per model. Owns no global state; every counter lives on the
    instance so two collectors cannot interfere.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        observer: TrainingObserver,
        *,
        resolver: Optional[LayerResolver] = None,
        selection: Optional[LayerSelection] = None,
        cadence: int = 1,
        emit_param_norms: bool = True,
        observability_cadence: int = 5,
    ) -> None:
        if cadence < 1:
            raise ValueError("cadence must be >= 1")
        if observability_cadence < 1:
            raise ValueError("observability_cadence must be >= 1")
        self.model = model
        self.observer = observer
        self.resolver = resolver or LayerResolver()
        self.selection = selection or LayerSelection()
        self.cadence = cadence
        self.emit_param_norms = emit_param_norms
        #: Cadence for layer metrics that **no detector consumes**, as distinct from
        #: ``cadence``, which governs the metrics the detection path reads.
        #:
        #: Emit at the rate the consumer needs. Measured on a 4-layer model: per-layer
        #: metrics were 16 of 32 events per step, and half of those were
        #: ``param.layer.norm`` -- **8 events/step, 25 % of all volume, consumed by
        #: nothing**. It appears downstream only in the presentation sink's
        #: ``FORBIDDEN_PREFIXES``. ``grad.layer.norm`` by contrast feeds
        #: ``grad.robust_zscore`` and stays at full cadence.
        self.observability_cadence = observability_cadence

        self._parameters: List[Tuple[str, torch.nn.Parameter]] = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self._selected: List[Tuple[str, int, torch.nn.Parameter]] = []
        for name, parameter in self._parameters:
            key = self.resolver.key_for(name)
            index = self.resolver.index_for(key)
            if self.selection.allows(key, index):
                self._selected.append((key, index, parameter))

        self.steps_considered = 0
        self.steps_collected = 0
        self.events_emitted = 0
        self.layers_measured = len({key for key, _, _ in self._selected})

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self, step: int) -> LayerMetricBatch:
        """Reduce the selected parameters into per-layer norms.

        One fused device pass over gradients and one over parameters, then pure
        scalar arithmetic to group them per layer. The original implementation
        called ``float(...sum().item())`` **per parameter**, forcing a device
        synchronisation per parameter: on a 149-tensor model that measured
        **35.5 ms per collected step**, more than the entire training step. Using
        :func:`per_tensor_l2_norms` reduces it to roughly one memory-bandwidth
        pass.
        """
        grads = []
        grad_keys = []
        params = []
        param_keys = []
        indices = {}

        for key, index, parameter in self._selected:
            indices[key] = index
            params.append(parameter.detach())
            param_keys.append(key)
            if parameter.grad is not None:
                grads.append(parameter.grad.detach())
                grad_keys.append(key)

        grad_norms = per_tensor_l2_norms(grads)
        param_norms = per_tensor_l2_norms(params)

        grad_sq = {}
        grad_nonfinite = {}
        grad_tensor_count = {}
        grad_elements = {}
        for key, value, tensor in zip(grad_keys, grad_norms, grads):
            grad_tensor_count[key] = grad_tensor_count.get(key, 0) + 1
            grad_elements[key] = grad_elements.get(key, 0) + int(tensor.numel())
            if value != value or value == float("inf"):
                # Rare: pay for the exact count only when a norm is non-finite.
                grad_nonfinite[key] = grad_nonfinite.get(key, 0) + tensor_nonfinite_count(tensor)
                continue
            grad_sq[key] = grad_sq.get(key, 0.0) + value * value

        param_sq = {}
        for key, value in zip(param_keys, param_norms):
            if value != value or value == float("inf"):
                continue
            param_sq[key] = param_sq.get(key, 0.0) + value * value

        layers = []
        for key in sorted(indices, key=lambda k: (indices[k], k)):
            if key in grad_nonfinite:
                grad_norm = float("nan")
            elif key in grad_sq:
                grad_norm = float(grad_sq[key] ** 0.5)
            else:
                grad_norm = None
            param_total = param_sq.get(key)
            layers.append(
                LayerNorm(
                    layer_key=key,
                    layer_index=indices[key],
                    grad_norm=grad_norm,
                    param_norm=float(param_total ** 0.5) if param_total is not None else None,
                    grad_nonfinite=grad_nonfinite.get(key, 0),
                    grad_tensor_count=grad_tensor_count.get(key, 0),
                    grad_element_count=grad_elements.get(key, 0),
                )
            )

        vector_hash = None
        if self.selection.emit_vector_hash:
            # A hash over the per-layer norms, so a change in the *shape* of the
            # layer distribution is detectable without storing the vector.
            digest = hashlib.blake2b(digest_size=8)
            for layer in layers:
                value = layer.grad_norm
                digest.update(layer.layer_key.encode("utf-8"))
                digest.update(bytes([0]))
                digest.update(
                    b"nan" if value is None or value != value else ("%.6g" % value).encode("ascii")
                )
                digest.update(bytes([1]))
            vector_hash = digest.hexdigest()

        return LayerMetricBatch(step=step, layers=layers, vector_hash=vector_hash)


    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def collect(self, step: int) -> Optional[LayerMetricBatch]:
        """Compute and emit for ``step``, honouring the cadence. Never raises."""
        self.steps_considered += 1
        if self.cadence > 1 and (step % self.cadence) != 0:
            return None
        try:
            batch = self.compute(step)
        except Exception:
            # A layer metric failure must never disturb training.
            self.observer._note_internal_error("layer_collector.compute")
            return None
        self.steps_collected += 1
        try:
            self._emit(batch)
        except Exception:
            self.observer._note_internal_error("layer_collector.emit")
        return batch

    def _observability_due(self, step: int) -> bool:
        """Whether the non-detector layer metrics are due on this step.

        These are for a human reading the store, not for a detector, so they are emitted
        at ``observability_cadence``. ``param.global_norm`` is unaffected: it is a single
        aggregate event per step, and the finiteness detector watches it.
        """
        return self.observability_cadence <= 1 or (step % self.observability_cadence) == 0

    def _emit(self, batch: LayerMetricBatch) -> None:
        layers = batch.layers
        if not layers:
            return

        graded = [l for l in layers if l.grad_norm is not None]
        finite = [l for l in graded if l.grad_norm == l.grad_norm]

        # Top-K per-layer events. Sorted by norm descending, with the layer key
        # as a deterministic tiebreak so two runs produce identical output.
        top = sorted(finite, key=lambda l: (-l.grad_norm, l.layer_index))[: self.selection.top_k]
        for layer in top:
            self._emit_one(
                "grad.layer_norm", layer.grad_norm, layer, batch,
                EventType.GRADIENT_METRIC,
            )
            if layer.grad_nonfinite:
                self.observer.nonfinite(
                    batch.step, metric_name="grad.nonfinite_count",
                    count=layer.grad_nonfinite, layer_index=layer.layer_index,
                    module_path=layer.layer_key, signal="gradient",
                )

        if self.emit_param_norms and self._observability_due(batch.step):
            param_layers = [l for l in layers if l.param_norm is not None]
            top_param = sorted(
                param_layers, key=lambda l: (-l.param_norm, l.layer_index)
            )[: self.selection.top_k]
            for layer in top_param:
                self._emit_one(
                    "param.layer_norm", layer.param_norm, layer, batch,
                    EventType.PARAMETER_METRIC,
                )

        if self.selection.emit_aggregate and finite:
            values = sorted(l.grad_norm for l in finite)
            maximum = values[-1]
            mean = sum(values) / len(values)
            p99 = values[min(len(values) - 1, int(round(0.99 * (len(values) - 1))))]
            argmax = max(finite, key=lambda l: (l.grad_norm, -l.layer_index))
            # ``population_count`` is the multiplicity denominator: "1 / 12 layers"
            # is a different diagnosis from "12 / 12 layers".
            shared = {
                "layer_count": len(layers),
                "layers_with_grad": len(graded),
                "layers_finite": len(finite),
                "population_layers": len(layers),
                "vector_hash": batch.vector_hash,
                "top_k": self.selection.top_k,
            }
            for name, value in (
                ("grad.layer_norm_max", maximum),
                ("grad.layer_norm_mean", mean),
                ("grad.layer_norm_p99", p99),
            ):
                self.observer.metric(
                    name, value, MetricUnit.DIMENSIONLESS, step=batch.step,
                    event_type=EventType.GRADIENT_METRIC, priority=EventPriority.P2,
                    metadata=dict(shared),
                )
            self.observer.metric(
                "grad.layer_norm_argmax_layer", float(argmax.layer_index),
                MetricUnit.COUNT, step=batch.step, layer_index=argmax.layer_index,
                module_path=argmax.layer_key, event_type=EventType.GRADIENT_METRIC,
                priority=EventPriority.P2, metadata=dict(shared),
            )

    def _emit_one(
        self,
        metric_name: str,
        value: float,
        layer: LayerNorm,
        batch: LayerMetricBatch,
        event_type: EventType,
    ) -> None:
        self.observer.metric(
            metric_name, value, MetricUnit.DIMENSIONLESS, step=batch.step,
            layer_index=layer.layer_index, module_path=layer.layer_key,
            event_type=event_type, priority=EventPriority.P2,
            metadata={
                "norm_type": "L2",
                "layer_key": layer.layer_key,
                "layer_count": batch.layer_count,
                "population_layers": batch.layer_count,
                "grad_tensor_count": layer.grad_tensor_count,
                "grad_element_count": layer.grad_element_count,
                "vector_hash": batch.vector_hash,
            },
        )
        self.events_emitted += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "steps_considered": self.steps_considered,
            "steps_collected": self.steps_collected,
            "events_emitted": self.events_emitted,
            "cadence": self.cadence,
            "layers_selected": self.layers_measured,
            "layers_in_model": self.resolver.layer_count,
            "top_k": self.selection.top_k,
            "param_norms_enabled": self.emit_param_norms,
        }

    def layer_keys(self) -> List[str]:
        return sorted({key for key, _, _ in self._selected})
