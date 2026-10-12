"""PyTorch training adapter.

Phase 1 scope (brief section 4): PyTorch only, single process, single GPU, but
architected so a single-node multi-GPU run needs no redesign. That constraint
shows up in three places:

* every metric carries ``rank``/``world_size`` when the observer knows them;
* the producer identity is per-process, so N ranks are N producers and the
  partial-order model applies unchanged;
* nothing here assumes one GPU — the device is passed in, and CUDA/MPS/CPU
  memory reporting degrades to "not reported" rather than to a fake zero.

The adapter is **not** a training framework. It observes a loop the user already
owns, which is what keeps it framework-agnostic and keeps the debugger out of the
training semantics.

Usage::

    adapter = PyTorchTrainingAdapter(observer, model, optimizer)
    adapter.run_start(config={"lr": 1e-3})
    for step, batch in enumerate(loader, start=1):
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=batch["input_ids"].shape[0],
                              tokens=batch["input_ids"].numel())
            with tracker.phase("forward"):
                out = model(**batch)
                loss = out.loss
            tracker.record_loss(loss)
            with tracker.phase("backward"):
                loss.backward()
            tracker.record_gradients()          # BEFORE optimizer.step()
            with tracker.phase("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    adapter.run_end()
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import torch

from ...observer.observer import TrainingObserver
from ...schema.enums import (
    DeviceType,
    EventPriority,
    EventType,
    MetricUnit,
    Severity,
    SpanKind,
)
from .metrics import (
    NORM_TYPE_L2,
    global_gradient_norm,
    global_parameter_norm,
    memory_snapshot,
    nonfinite_gradient_count,
    tensor_nonfinite_count,
)

__all__ = ["PyTorchTrainingAdapter", "StepTracker", "resolve_device_type", "COUNT_METRICS"]

#: Non-finite observations are recorded under their own metric names.
#:
#: This mapping exists because of a real bug caught during W2: the first version
#: emitted the non-finite *count* under the same name as the measurement it
#: qualified (``grad.global_norm``), so a consumer reading that name for the
#: affected step saw the count (``794.0``) instead of the norm (``nan``). Two
#: events sharing one metric name at one step is silent corruption of exactly the
#: kind Phase 0 warned about, so names are now allocated from one place and a
#: regression test asserts per-step metric-name uniqueness.
COUNT_METRICS: Dict[str, str] = {
    "loss": "train.loss_nonfinite_count",
    "gradient": "grad.nonfinite_count",
}


def resolve_device_type(device: Optional[torch.device]) -> Optional[DeviceType]:
    if device is None:
        return None
    kind = device.type
    return {
        "cpu": DeviceType.CPU,
        "cuda": DeviceType.CUDA,
        "mps": DeviceType.MPS,
    }.get(kind, DeviceType.UNKNOWN)


@dataclass
class StepTracker:
    """Per-step recorder. Obtained from :meth:`PyTorchTrainingAdapter.step`."""

    adapter: "PyTorchTrainingAdapter"
    step: int
    microstep: Optional[int] = None
    _phase_start_ns: Dict[str, int] = field(default_factory=dict)
    _phase_durations_ms: Dict[str, float] = field(default_factory=dict)
    _loss_recorded: bool = False
    _gradients_recorded: bool = False
    _parameters_recorded: bool = False
    _samples: Optional[int] = None
    _tokens: Optional[int] = None
    _start_ns: int = 0

    # -- batch description -------------------------------------------------

    def set_batch(self, *, samples: Optional[int] = None, tokens: Optional[int] = None) -> None:
        """Declare the batch size so throughput can be derived."""
        self._samples = samples
        self._tokens = tokens

    # -- phase timing ------------------------------------------------------

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time a phase (``forward``, ``backward``, ``optimizer``, ...).

        Nested phases are supported but each name is recorded once; a repeated
        name accumulates, which is what gradient accumulation needs.
        """
        start = time.monotonic_ns()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic_ns() - start) / 1_000_000.0
            self._phase_durations_ms[name] = self._phase_durations_ms.get(name, 0.0) + elapsed_ms

    # -- signals -----------------------------------------------------------

    def record_loss(self, loss: torch.Tensor) -> float:
        """Record the loss, including the non-finite case.

        Returns the Python float so a caller can assert on exactly what was
        recorded rather than on a re-read of the tensor.
        """
        value = float(loss.detach().to(torch.float32).item())
        self.adapter.observer.metric(
            "train.loss", value, MetricUnit.DIMENSIONLESS,
            step=self.step, microstep=self.microstep,
            event_type=EventType.LOSS,
            priority=EventPriority.P1,
            severity=Severity.INFO if value == value and abs(value) != float("inf") else Severity.ERROR,
        )
        if value != value or abs(value) == float("inf"):
            self.adapter.observer.nonfinite(
                self.step, metric_name=COUNT_METRICS["loss"], count=1,
                event_type=EventType.LOSS, microstep=self.microstep, signal="loss",
            )
            self.adapter._nonfinite_losses += 1
        self._loss_recorded = True
        return value

    def record_gradients(self) -> None:
        """Compute and record the global gradient norm.

        Must be called **after** ``backward()`` and, for the standard definition,
        **before** ``optimizer.step()``: some optimizers modify gradients in
        place, and clipping would scale them. Both would corrupt the baseline.
        """
        # Skip the reduction entirely when sampling would discard it: the cost
        # is in the computation, not the emission.
        if not self.adapter.observer.would_emit(
            "grad.global_norm", self.step, EventPriority.P1
        ):
            self._gradients_recorded = True
            return
        result = global_gradient_norm(self.adapter._parameters())
        if result.nonfinite_count:
            self.adapter.observer.nonfinite(
                self.step, metric_name=COUNT_METRICS["gradient"],
                count=result.nonfinite_count, event_type=EventType.GRADIENT_METRIC,
                signal="gradient",
            )
            self.adapter._nonfinite_grad_steps += 1
        self.adapter.observer.metric(
            "grad.global_norm", result.value, MetricUnit.DIMENSIONLESS,
            step=self.step, microstep=self.microstep,
            event_type=EventType.GRADIENT_METRIC, priority=EventPriority.P1,
            # ``nonfinite_count`` is descriptive provenance on the measurement.
            # The boolean ``nonfinite`` flag belongs only to the dedicated P0
            # count event, so the two are never confusable.
            metadata=result.as_metadata(),
        )
        self._gradients_recorded = True

    def record_parameters(self) -> None:
        """Compute and record the global parameter norm."""
        if not self.adapter.observer.would_emit(
            "param.global_norm", self.step, EventPriority.P2
        ):
            self._parameters_recorded = True
            return
        result = global_parameter_norm(self.adapter._parameters())
        self.adapter.observer.metric(
            "param.global_norm", result.value, MetricUnit.DIMENSIONLESS,
            step=self.step, microstep=self.microstep,
            event_type=EventType.PARAMETER_METRIC, priority=EventPriority.P2,
            metadata=dict(result.as_metadata(), norm_type=NORM_TYPE_L2),
        )
        self._parameters_recorded = True

    def record_learning_rate(self, value: Optional[float] = None) -> None:
        """Record the learning rate. Reads it from the optimizer when omitted."""
        if value is None:
            value = self.adapter.current_learning_rate()
        if value is None:
            return
        self.adapter.observer.metric(
            "train.lr", float(value), MetricUnit.DIMENSIONLESS,
            step=self.step, microstep=self.microstep,
            event_type=EventType.LR_CHANGE, priority=EventPriority.P1,
        )

    # -- finalisation ------------------------------------------------------

    def _finish(self) -> None:
        elapsed_ns = time.monotonic_ns() - self._start_ns
        elapsed_ms = elapsed_ns / 1_000_000.0

        self.adapter.observer.emit(
            event_type=EventType.STEP_END, step=self.step, microstep=self.microstep,
            metric_name="train.step_time", metric_value=elapsed_ms,
            metric_unit=MetricUnit.MILLISECONDS, priority=EventPriority.P1,
            span_kind=SpanKind.END,
            metadata={"step_time_ms": elapsed_ms},
        )

        for name, duration_ms in self._phase_durations_ms.items():
            self.adapter.observer.metric(
                "train.{}_time_ms".format(name), duration_ms, MetricUnit.MILLISECONDS,
                step=self.step, microstep=self.microstep,
                event_type=EventType.STEP_END, priority=EventPriority.P2,
            )

        if self._samples:
            self.adapter.observer.metric(
                "train.samples_per_sec", self._samples / (elapsed_ms / 1000.0),
                MetricUnit.SAMPLES_PER_SEC, step=self.step,
                event_type=EventType.STEP_END, priority=EventPriority.P2,
            )
        if self._tokens:
            self.adapter.observer.metric(
                "train.tokens_per_sec", self._tokens / (elapsed_ms / 1000.0),
                MetricUnit.TOKENS_PER_SEC, step=self.step,
                event_type=EventType.STEP_END, priority=EventPriority.P1,
            )

        if self.adapter.observer.would_emit("mem.allocated", self.step, EventPriority.P2):
            for key, value in memory_snapshot(self.adapter.device).items():
                self.adapter.observer.metric(
                    "mem.{}".format(key), float(value), MetricUnit.BYTES,
                    step=self.step, event_type=EventType.SYSTEM_METRIC,
                    priority=EventPriority.P2,
                )

        self.adapter.steps_recorded += 1


class PyTorchTrainingAdapter:
    """Observes a PyTorch training loop without owning it."""

    def __init__(
        self,
        observer: TrainingObserver,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        device: Optional[torch.device] = None,
        model_id: Optional[str] = None,
        layer_filter: Optional[Callable[[str, torch.nn.Module], bool]] = None,
    ) -> None:
        self.observer = observer
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_id = model_id or getattr(model, "_model_id", type(model).__name__)
        self.layer_filter = layer_filter

        self.steps_recorded = 0
        self._nonfinite_losses = 0
        self._nonfinite_grad_steps = 0
        self._use_mps = device is not None and device.type == "mps"

    # ------------------------------------------------------------------
    # Introspection -- kept explicit so a data-parallel wrapper is a local change
    # ------------------------------------------------------------------

    def _parameters(self) -> List[torch.nn.Parameter]:
        """Parameters whose gradients this process owns.

        A single ``.parameters()`` call is correct for the Phase 1 scope and for
        FSDP, where gradients are local shards of the same logical tensor. When
        Phase 2 adds gradient accumulation across replicas this method is the one
        place that needs to change.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def current_learning_rate(self) -> Optional[float]:
        if self.optimizer is None or not self.optimizer.param_groups:
            return None
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run_start(self, **metadata: Any) -> None:
        payload = {
            "model_id": self.model_id,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameter_count": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "device": str(self.device) if self.device is not None else None,
            "torch_version": torch.__version__,
            "learning_rate": self.current_learning_rate(),
        }
        payload.update(metadata)
        self.observer.emit(
            event_type=EventType.RUN_START, priority=EventPriority.P1,
            model_id=self.model_id, metadata=payload,
        )

    def run_end(self, **metadata: Any) -> None:
        payload = {
            "steps_recorded": self.steps_recorded,
            "nonfinite_losses": self._nonfinite_losses,
            "nonfinite_gradient_steps": self._nonfinite_grad_steps,
            "observer": self.observer.health()["observer"],
        }
        payload.update(metadata)
        self.observer.emit(event_type=EventType.RUN_END, priority=EventPriority.P1,
                           model_id=self.model_id, metadata=payload)

    @contextmanager
    def step(self, step: int, microstep: Optional[int] = None) -> Iterator[StepTracker]:
        """Time and finalise one optimisation step.

        The tracker is finished in a ``finally`` block, so a step that raises
        still produces its timing and memory events and leaves the accounting
        consistent.
        """
        tracker = StepTracker(
            adapter=self, step=step, microstep=microstep, _start_ns=time.monotonic_ns()
        )
        self.observer.emit(
            event_type=EventType.STEP_START, step=step, microstep=microstep,
            span_kind=SpanKind.START,
            priority=EventPriority.P2,
        )
        self.observer.emit(
            event_type=EventType.OPTIMIZER_STEP, step=step, microstep=microstep,
            priority=EventPriority.P3,
        )
        try:
            yield tracker
        finally:
            try:
                tracker._finish()
            except Exception:
                # Finishing the tracker must never turn a training error into a
                # debugger error, and must never mask the original exception.
                self.observer._note_internal_error("adapter.step.finish")

    # ------------------------------------------------------------------
    # Non-finite helpers exposed for callers that inspect tensors themselves
    # ------------------------------------------------------------------

    def count_nonfinite(self, tensor: torch.Tensor) -> int:
        return tensor_nonfinite_count(tensor)

    def nonfinite_gradients(self) -> Any:
        return nonfinite_gradient_count(self._parameters())

    def stats(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "steps_recorded": self.steps_recorded,
            "nonfinite_losses": self._nonfinite_losses,
            "nonfinite_gradient_steps": self._nonfinite_grad_steps,
        }
