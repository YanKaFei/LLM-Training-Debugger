"""Event construction helpers for tests and scenario generation.

Kept in one place so that scenario definitions read as data rather than as
twenty-line dataclass constructions, and so that a schema change breaks one
file instead of every test.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from training_debugger.schema import (
    DeviceType,
    EventPriority,
    EventType,
    MetricUnit,
    Severity,
    SourceKind,
    TrainingEvent,
    new_event_id,
)

DEFAULT_RUN = "run-test-0001"
DEFAULT_PRODUCER = "pytorch-adapter:r0:pid4242"

_SEQ_COUNTER: Dict[str, int] = {}


def reset_seq() -> None:
    _SEQ_COUNTER.clear()


def next_seq(producer: str = DEFAULT_PRODUCER) -> int:
    _SEQ_COUNTER[producer] = _SEQ_COUNTER.get(producer, 0) + 1
    return _SEQ_COUNTER[producer]


def event(
    event_type: EventType = EventType.LOSS,
    *,
    run_id: str = DEFAULT_RUN,
    producer: str = DEFAULT_PRODUCER,
    source: str = str(SourceKind.PYTORCH_ADAPTER),
    step: Optional[int] = None,
    microstep: Optional[int] = None,
    rank: Optional[int] = 0,
    source_seq: Optional[int] = None,
    metric_name: Optional[str] = None,
    metric_value: Optional[float] = None,
    metric_unit: Optional[MetricUnit] = MetricUnit.DIMENSIONLESS,
    layer_index: Optional[int] = None,
    module_path: Optional[str] = None,
    module: Optional[str] = None,
    severity: Severity = Severity.INFO,
    priority: EventPriority = EventPriority.P1,
    confidence: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parent_event_id: Optional[str] = None,
    device_type: Optional[DeviceType] = DeviceType.CPU,
    **extra: Any
) -> TrainingEvent:
    """Build a valid event with sensible defaults.

    ``module`` is accepted as an alias for ``module``-style naming used in some
    scenario files and maps onto ``module_path``.
    """
    if module is not None and module_path is None:
        module_path = module
    return TrainingEvent(
        event_id=new_event_id(),
        run_id=run_id,
        producer=producer,
        source=source,
        event_type=event_type,
        step=step,
        microstep=microstep,
        rank=rank,
        source_seq=source_seq if source_seq is not None else next_seq(producer),
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        layer_index=layer_index,
        module_path=module_path,
        severity=severity,
        priority=priority,
        confidence=confidence,
        metadata=metadata or {},
        parent_event_id=parent_event_id,
        device_type=device_type,
        **extra
    )


def loss(step: int, value: float, **kw: Any) -> TrainingEvent:
    return event(
        EventType.LOSS, step=step, metric_name="train.loss", metric_value=value, **kw
    )


def grad_norm(step: int, value: float, layer_index: Optional[int] = None, **kw: Any) -> TrainingEvent:
    return event(
        EventType.GRADIENT_METRIC,
        step=step,
        metric_name="grad.layer_norm" if layer_index is not None else "grad.global_norm",
        metric_value=value,
        layer_index=layer_index,
        module_path=("layers.%d" % layer_index) if layer_index is not None else None,
        priority=EventPriority.P2 if layer_index is not None else EventPriority.P1,
        **kw
    )


def lr(step: int, value: float, **kw: Any) -> TrainingEvent:
    return event(
        EventType.LR_CHANGE, step=step, metric_name="train.lr", metric_value=value, **kw
    )


def step_end(step: int, duration_ms: float, **kw: Any) -> TrainingEvent:
    return event(
        EventType.STEP_END,
        step=step,
        metric_name="train.step_time",
        metric_value=duration_ms,
        metric_unit=MetricUnit.MILLISECONDS,
        **kw
    )


# ---------------------------------------------------------------------------
# A synthetic training loop
# ---------------------------------------------------------------------------


def synthetic_training(
    observer: Any,
    steps: int = 50,
    *,
    base_loss: float = 2.0,
    lr: float = 1e-3,
    layer_count: int = 4,
    spike_at: Optional[int] = None,
    spike_factor: float = 5.0,
    grad_explode_at: Optional[int] = None,
    grad_explode_factor: float = 20.0,
    nan_at: Optional[int] = None,
    inf_at: Optional[int] = None,
    throughput_drop_at: Optional[int] = None,
    throughput_drop_factor: float = 3.0,
) -> dict:
    """Drive ``observer`` through a deterministic synthetic run.

    Used by the fault-injection suite (where the point is that this *completes*
    despite a broken debugger) and by the scenario corpus (where the point is
    that the injected anomaly is detected). Keeping one loop for both means the
    fault tests exercise the same emission pattern as the real scenarios.

    Returns a summary dict with the ground truth, so a test can assert against
    what was injected rather than against what was observed.
    """
    import math

    observer.run_start(model_id="synthetic", steps=steps)
    losses: List[float] = []
    grad_norms: List[float] = []
    step_times: List[float] = []

    for step in range(1, steps + 1):
        loss = base_loss * (0.99 ** step)
        grad = 1.0 + 0.05 * math.sin(step / 3.0)
        duration = 100.0

        if spike_at is not None and step >= spike_at:
            loss *= spike_factor
        if grad_explode_at is not None and step >= grad_explode_at:
            grad *= grad_explode_factor
        if throughput_drop_at is not None and step >= throughput_drop_at:
            duration *= throughput_drop_factor
        if nan_at is not None and step == nan_at:
            loss = float("nan")
        if inf_at is not None and step == inf_at:
            # A REAL non-finite element makes `global_gradient_norm` return **nan**,
            # not inf: it counts non-finite elements and substitutes nan so the two
            # cases stay distinguishable. This harness used to emit `inf` alongside
            # `grad.nonfinite_count = 1`, a pair the real function never produces --
            # and `test_A3b_inf_is_observed` then asserted that `inf` meant "a
            # non-finite gradient was observed". That is the
            # "a non-finite norm is not necessarily a non-finite tensor" error,
            # encoded in a test.
            grad = float("nan")

        losses.append(loss)
        grad_norms.append(grad)
        step_times.append(duration)

        observer.metric(
            "train.loss", loss, MetricUnit.DIMENSIONLESS, step=step,
            event_type=EventType.LOSS, priority=EventPriority.P1,
        )
        observer.metric(
            "train.lr", lr, MetricUnit.DIMENSIONLESS, step=step,
            event_type=EventType.LR_CHANGE, priority=EventPriority.P1,
        )
        observer.metric(
            "grad.global_norm", grad, MetricUnit.DIMENSIONLESS, step=step,
            event_type=EventType.GRADIENT_METRIC, priority=EventPriority.P1,
        )
        for layer in range(layer_count):
            observer.metric(
                "grad.layer_norm", grad * (1.0 + layer * 0.1),
                MetricUnit.DIMENSIONLESS, step=step, layer_index=layer,
                module_path="layers.%d" % layer,
                event_type=EventType.GRADIENT_METRIC, priority=EventPriority.P2,
            )
        observer.step_end(step, duration)
        # The non-finite *count* goes to its own metric name. Emitting it under
        # "train.loss" would overwrite the NaN with the count -- the same
        # collision bug that was found and fixed in the PyTorch adapter (W2).
        if nan_at is not None and step == nan_at:
            observer.nonfinite(step, metric_name="train.loss_nonfinite_count", count=1)
        if inf_at is not None and step == inf_at:
            observer.nonfinite(step, metric_name="grad.nonfinite_count", count=1)

    observer.run_end(steps=steps)
    return {
        "steps": steps,
        "losses": losses,
        "grad_norms": grad_norms,
        "step_times": step_times,
        "layer_count": layer_count,
        "spike_at": spike_at,
        "grad_explode_at": grad_explode_at,
        "nan_at": nan_at,
        "inf_at": inf_at,
        "throughput_drop_at": throughput_drop_at,
        "final_loss": losses[-1],
    }
