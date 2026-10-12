"""PyTorch adapter for the Phase 1 single-process / single-GPU scope."""

from __future__ import annotations

from .optimizer_witness import (
    WITNESS_SCHEMA_VERSION,
    OptimizerStepStatus,
    OptimizerWitness,
)
from .adapter import COUNT_METRICS, PyTorchTrainingAdapter, StepTracker, resolve_device_type
from .layers import (
    DEFAULT_LAYER_PATTERN,
    LayerMetricBatch,
    LayerMetricCollector,
    LayerNorm,
    LayerResolver,
    LayerSelection,
)
from .metrics import (
    NORM_TYPE_L2,
    NormResult,
    global_gradient_norm,
    global_parameter_norm,
    memory_snapshot,
    nonfinite_gradient_count,
    tensor_nonfinite_count,
)

__all__ = [
    "PyTorchTrainingAdapter",
    "OptimizerWitness",
    "OptimizerStepStatus",
    "WITNESS_SCHEMA_VERSION",
    "StepTracker",
    "resolve_device_type",
    "COUNT_METRICS",
    "LayerMetricCollector",
    "LayerSelection",
    "LayerResolver",
    "LayerMetricBatch",
    "LayerNorm",
    "DEFAULT_LAYER_PATTERN",
    "global_gradient_norm",
    "global_parameter_norm",
    "nonfinite_gradient_count",
    "tensor_nonfinite_count",
    "memory_snapshot",
    "NormResult",
    "NORM_TYPE_L2",
]
