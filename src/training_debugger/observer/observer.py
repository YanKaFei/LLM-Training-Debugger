"""The observer: the training process's only connection to the debugger.

Everything in this module is written under one rule, restated from Phase 1 brief
section 3.2 because it determines the shape of every method:

    Observability must never make training less reliable.

In practice that means:

**Fail-open.** :meth:`TrainingObserver.emit` returns a bool and never raises for
*any* reason -- a bad event, a full queue, a dead writer, a broken disk. The
worst outcome of a debugger bug is a missing diagnostic, never a failed step.

**Non-blocking.** The hot path is: sample, validate, stamp, enqueue. All I/O
happens on a writer thread. The only lock held on the hot path guards a counter.

**Bounded.** The queue has a hard maximum and a priority-aware eviction rule. A
P0 event is never dropped silently; if it cannot be queued *or* written it goes
to the emergency sink, which writes to stderr and keeps a recoverable copy.

**Isolated.** A writer that keeps failing trips a circuit breaker rather than
becoming a retry loop inside the training process.

The overhead this design costs is measured, not asserted: see
``tests/performance`` and W4's overhead gate.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from ..schema.enums import (
    DeviceType,
    EventPriority,
    EventType,
    MetricUnit,
    Severity,
    SourceKind,
    SpanKind,
)
from ..schema.event import TrainingEvent, make_producer_id
from ..schema.validation import (
    ProducerSequenceTracker,
    ValidationContext,
    validate_event,
)
from .health import EmergencySink, ObserverStats
from .queue import BoundedEventQueue
from .sampling import DEFAULT_POLICY_PHASE1, SamplingPolicy
from .writer_protocol import EventWriter, NullWriter

__all__ = ["TrainingObserver", "ObserverConfig"]


class ObserverConfig:
    """Tunables in one place so a benchmark can vary exactly one of them."""

    def __init__(
        self,
        *,
        max_queue: int = 10_000,
        batch_size: int = 256,
        flush_interval_s: float = 0.5,
        validate: bool = True,
        max_consecutive_writer_failures: int = 5,
        drain_timeout_s: float = 5.0,
        emergency_buffer: int = 1000,
    ) -> None:
        if max_queue <= 0:
            raise ValueError("max_queue must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.max_queue = max_queue
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.validate = validate
        self.max_consecutive_writer_failures = max_consecutive_writer_failures
        self.drain_timeout_s = drain_timeout_s
        self.emergency_buffer = emergency_buffer


class TrainingObserver:
    """Collects ``TrainingEvent``s from a training process and hands them to a writer.

    Typical use::

        observer = TrainingObserver(run_id="run-123")
        observer.run_start(model_id="tiny-transformer")
        for step in range(steps):
            observer.metric("train.loss", loss, unit=MetricUnit.DIMENSIONLESS, step=step)
            observer.metric("grad.global_norm", norm, step=step)
        observer.run_end()
        observer.close()
    """

    def __init__(
        self,
        run_id: str,
        *,
        writer: Optional[EventWriter] = None,
        config: Optional[ObserverConfig] = None,
        sampling: Optional[SamplingPolicy] = None,
        source: Union[SourceKind, str] = SourceKind.PYTORCH_ADAPTER,
        producer: Optional[str] = None,
        producer_version: str = "0.1",
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        model_id: Optional[str] = None,
        device_type: Optional[DeviceType] = None,
        device_id: Optional[str] = None,
        emergency_sink: Optional[EmergencySink] = None,
        autostart: bool = True,
    ) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        self.run_id = run_id
        self._source = str(source)
        self._config = config or ObserverConfig()
        self._sampling = sampling.clone() if sampling is not None else DEFAULT_POLICY_PHASE1.clone()
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.model_id = model_id
        self._device_type = device_type
        self._device_id = device_id
        self.producer_version = producer_version
        self.producer = producer or make_producer_id(self._source, rank)

        self._writer: EventWriter = writer if writer is not None else NullWriter()
        self._emergency = emergency_sink or EmergencySink(maxlen=self._config.emergency_buffer)
        self._queue = BoundedEventQueue(
            maxlen=self._config.max_queue, emergency_sink=self._emergency
        )

        self._stats = ObserverStats()
        self._stats_lock = threading.Lock()
        self._seq_lock = threading.Lock()
        self._source_seq = 0
        self._seq_tracker = ProducerSequenceTracker()
        self._validation_context = ValidationContext(known_world_size=world_size)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._writer_disabled = False
        self._consecutive_writer_failures = 0
        #: Batches handed to the writer but not yet recorded as written.
        #: Without this, ``flush()`` could return while the writer thread was
        #: mid-batch, so ``written_total`` momentarily under-counted and an
        #: accounting check right after a flush saw events that were neither
        #: written nor dropped. One writer thread mutates it; readers only read.
        self._in_flight = 0
        self._lock = threading.Lock()
        self._closed = False

        if autostart:
            self.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the writer thread. Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._writer_loop, name="training-debugger-writer", daemon=True
            )
            self._thread.start()
            self._stats.started += 1

    def close(self, timeout: Optional[float] = None) -> None:
        """Stop the writer thread, drain the queue, and close the writer.

        Never raises. Safe to call twice. Safe to call when the writer is dead.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._stop_event.set()
            thread = self._thread
            if thread is not None and thread.is_alive():
                thread.join(timeout if timeout is not None else self._config.drain_timeout_s)
            self._drain_once(final=True)
        except Exception:
            self._note_internal_error("close")
        finally:
            try:
                self._writer.close()
            except Exception:
                self._note_writer_failure("close")
            with self._stats_lock:
                self._stats.closed += 1
                self._stats.emergency_buffer_depth = len(self._emergency)

    def __enter__(self) -> "TrainingObserver":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Emission -- the hot path
    # ------------------------------------------------------------------

    def emit(
        self,
        event: Optional[TrainingEvent] = None,
        **fields: Any,
    ) -> bool:
        """Record an event. Returns True if it was accepted.

        **This method never raises.** Any failure is isolated, counted, and
        reported as ``False``. Pass either a constructed ``TrainingEvent`` or the
        keyword fields to build one.
        """
        started_ns = time.monotonic_ns()
        try:
            # Ownership decides whether context can be filled in place. When the
            # observer builds the event itself nothing else can hold a reference to
            # it, so cloning it to set fields is pure waste. See `_accept`.
            owned = event is None
            if event is None:
                event = TrainingEvent(**fields)
            elif fields:
                event = event.copy_with(**fields)
            accepted = self._accept(event, owned=owned)
        except Exception:
            self._note_internal_error("emit")
            accepted = False
        finally:
            elapsed = time.monotonic_ns() - started_ns
            with self._stats_lock:
                self._stats.emit_time_total_ns += elapsed
        return accepted

    def _accept(self, event: TrainingEvent, owned: bool = False) -> bool:
        """Sample, stamp, validate, enqueue. Caller guarantees exception safety.

        ``owned`` means the observer constructed this event and no other reference to it
        exists, so context can be filled **in place**. Otherwise the event belongs to the
        caller and is cloned, because mutating it would be a visible side effect on
        someone else's object.

        This matters more than it looks. ``copy_with`` is ``dataclasses.replace`` on a
        38-field dataclass: measured at **11.3 us per call**, and the context fill plus the
        sequence stamp made it **two calls per event -- 22.6 us of a 39.9 us emit, 57 %**.
        The earlier fix in this file collapsed ten calls into one; this removes the
        remaining two for the path that every ``metric()`` call takes.
        """
        with self._stats_lock:
            self._stats.emitted_total += 1

        # Fill context the caller is not expected to repeat on every event.
        #
        # Only *absent* fields are filled. An explicitly empty string is a
        # producer error and is left alone so validation rejects it: healing a
        # wrong value would hide the bug, whereas filling a missing one is the
        # whole point of the observer owning run identity.
        #
        # Collected into ONE dict and applied with a single ``replace``. The
        # original code called ``copy_with`` up to ten times, and each call copies
        # every field of a 39-field dataclass -- measured at 8.6 us per call. That
        # chain was the second-largest contributor to the W4 overhead failure.
        changes: Dict[str, Any] = {}
        if event.run_id is None:
            changes["run_id"] = self.run_id
        if event.producer is None:
            changes["producer"] = self.producer
        if event.source is None:
            changes["source"] = self._source
        if event.rank is None and self.rank is not None:
            changes["rank"] = self.rank
        if event.local_rank is None and self.local_rank is not None:
            changes["local_rank"] = self.local_rank
        if event.world_size is None and self.world_size is not None:
            changes["world_size"] = self.world_size
        if event.model_id is None and self.model_id is not None:
            changes["model_id"] = self.model_id
        if event.device_type is None and self._device_type is not None:
            changes["device_type"] = self._device_type
        if event.device_id is None and self._device_id is not None:
            changes["device_id"] = self._device_id
        if event.producer_version is None:
            changes["producer_version"] = self.producer_version
        if changes:
            if owned:
                for field_name, value in changes.items():
                    setattr(event, field_name, value)
            else:
                event = event.copy_with(**changes)

        # Sampling is the observer's job, not the caller's (brief section 3.4).
        if not self._sampling.should_emit(event):
            with self._stats_lock:
                self._stats.sampled_out_total += 1
            return False

        # Stamp the producer-local sequence if the caller did not supply one.
        if event.source_seq is None:
            with self._seq_lock:
                self._source_seq += 1
                if owned:
                    event.source_seq = self._source_seq
                else:
                    event = event.copy_with(source_seq=self._source_seq)

        if self._config.validate:
            result = validate_event(
                event, context=self._validation_context, seq_tracker=self._seq_tracker
            )
            if not result.ok:
                with self._stats_lock:
                    self._stats.validation_failures_total += 1
                    self._stats.events_dropped_total += 1
                    self._stats.last_error = "validation: " + result.reason()
                self._emergency.record(event, "validation_failed")
                return False

        if self._writer_disabled:
            with self._stats_lock:
                self._stats.events_dropped_total += 1
            self._emergency.record(event, "writer_disabled")
            return False

        accepted = self._queue.put(event)
        if not accepted:
            with self._stats_lock:
                self._stats.queue_overflow_total += 1
                self._stats.events_dropped_total += 1
        return accepted

    def would_emit(
        self,
        metric_name: Optional[str],
        step: Optional[int],
        priority: EventPriority = EventPriority.P1,
    ) -> bool:
        """Whether a metric would survive sampling. Pure -- no counters change.

        Exposed so an adapter can skip an **expensive reduction** rather than
        compute a number only to have the observer discard it. In the W4
        benchmark, the per-parameter norm cost 880 us and the default policy
        samples it every 10th step, so nine of every ten computations were
        thrown away. Sampling has to gate computation, not merely emission, for
        the cadence to mean anything.
        """
        if priority in self._sampling.never_sample_priorities:
            return True
        if step is None:
            return True
        rule = self._sampling.rule_for(metric_name)
        every_n = rule.every_n_steps if rule else self._sampling.default_every_n_steps
        warmup = rule.always_first_steps if rule else self._sampling.default_always_first_steps
        if warmup and step < warmup:
            return True
        if every_n <= 1:
            return True
        return (step % every_n) == 0

    def metric(
        self,
        name: str,
        value: float,
        unit: Optional[MetricUnit] = None,
        *,
        step: Optional[int] = None,
        microstep: Optional[int] = None,
        layer_index: Optional[int] = None,
        module_path: Optional[str] = None,
        event_type: EventType = EventType.GRADIENT_METRIC,
        priority: EventPriority = EventPriority.P1,
        severity: Severity = Severity.INFO,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Emit a metric sample. The common case, so it gets a short signature."""
        return self.emit(
            event_type=event_type,
            step=step,
            microstep=microstep,
            layer_index=layer_index,
            module_path=module_path,
            metric_name=name,
            metric_value=value,
            metric_unit=unit,
            priority=priority,
            severity=severity,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Lifecycle conveniences
    # ------------------------------------------------------------------

    def run_start(self, **metadata: Any) -> bool:
        return self.emit(event_type=EventType.RUN_START, priority=EventPriority.P1,
                         metadata=dict(metadata))

    def run_end(self, **metadata: Any) -> bool:
        return self.emit(event_type=EventType.RUN_END, priority=EventPriority.P1,
                         metadata=dict(metadata))

    def step_start(self, step: int) -> bool:
        return self.emit(event_type=EventType.STEP_START, step=step,
                         span_kind=SpanKind.START, priority=EventPriority.P2)

    def step_end(self, step: int, duration_ms: float, **metadata: Any) -> bool:
        return self.emit(
            event_type=EventType.STEP_END, step=step, span_kind=SpanKind.END,
            metric_name="train.step_time", metric_value=duration_ms,
            metric_unit=MetricUnit.MILLISECONDS, priority=EventPriority.P1,
            metadata=dict(metadata),
        )

    def warning(self, message: str, **metadata: Any) -> bool:
        return self.emit(event_type=EventType.WARNING, severity=Severity.WARN,
                         priority=EventPriority.P0, message=message,
                         metadata=dict(metadata))

    def error(self, message: str, **metadata: Any) -> bool:
        return self.emit(event_type=EventType.ERROR, severity=Severity.ERROR,
                         priority=EventPriority.P0, message=message,
                         metadata=dict(metadata))

    def nonfinite(
        self,
        step: int,
        *,
        metric_name: str,
        count: int,
        event_type: EventType = EventType.LOSS,
        **metadata: Any
    ) -> bool:
        """Record that a tensor contained non-finite values.

        A P0 event even though it is a metric: ``NAN_DETECTED`` is a required
        Phase 1 incident, and the observation that produced it must not be the
        thing that gets sampled away.
        """
        return self.emit(
            event_type=event_type, step=step, metric_name=metric_name,
            metric_value=float(count), metric_unit=MetricUnit.COUNT,
            severity=Severity.ERROR, priority=EventPriority.P0,
            metadata=dict(metadata, nonfinite=True),
        )

    # ------------------------------------------------------------------
    # Convenience aliases required by the brief
    # ------------------------------------------------------------------

    @property
    def sampling(self) -> SamplingPolicy:
        return self._sampling

    @property
    def writer(self) -> EventWriter:
        return self._writer

    def stats(self) -> ObserverStats:
        with self._stats_lock:
            snapshot = ObserverStats(**self._stats.to_dict())
        snapshot.emergency_buffer_depth = len(self._emergency)
        return snapshot

    def health(self) -> Dict[str, Any]:
        queue_stats = self._queue.stats()
        return {
            "run_id": self.run_id,
            "producer": self.producer,
            "writer": getattr(self._writer, "name", "unknown"),
            "writer_disabled": self._writer_disabled,
            "closed": self._closed,
            "queue": queue_stats.to_dict(),
            "observer": self.stats().to_dict(),
            "sampling": self._sampling.summary(),
        }

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Drain the queue and flush the writer. Returns True if fully drained.

        Never raises. Does **not** stop the writer thread. Unlike TorchTitan's
        equivalent, this is safe to call at any time; it is simply expensive, so
        it belongs at phase boundaries rather than per event.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                drained = self._drain_once()
                # An empty queue is NOT sufficient: the writer thread may still
                # hold a batch. Only when both are quiet is the flush complete.
                if len(self._queue) == 0 and self._in_flight == 0:
                    break
                if deadline is not None and time.monotonic() > deadline:
                    break
                if not drained:
                    time.sleep(0.0005)
            try:
                self._writer.flush()
            except Exception:
                self._note_writer_failure("flush")
            with self._stats_lock:
                self._stats.flush_count += 1
            return len(self._queue) == 0 and self._in_flight == 0
        except Exception:
            self._note_internal_error("flush")
            return False

    # ------------------------------------------------------------------
    # Writer thread
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        interval = self._config.flush_interval_s
        while not self._stop_event.is_set():
            wrote_anything = self._drain_once()
            if not wrote_anything:
                self._stop_event.wait(interval)
        self._drain_once(final=True)

    def _drain_once(self, final: bool = False) -> bool:
        """Move one batch from the queue to the writer. Returns True if it wrote."""
        try:
            batch = self._queue.drain(self._config.batch_size)
        except Exception:
            self._note_internal_error("drain")
            return False
        if not batch:
            return False
        if self._writer_disabled:
            with self._stats_lock:
                self._stats.events_dropped_total += len(batch)
            for event in batch:
                if event.priority is EventPriority.P0:
                    self._emergency.record(event, "writer_disabled")
            return False

        started_ns = time.monotonic_ns()
        self._in_flight += 1
        try:
            self._writer.write_batch(batch)
        except Exception as exc:
            self._note_writer_failure(str(exc))
            with self._stats_lock:
                self._stats.events_dropped_total += len(batch)
            for event in batch:
                if event.priority is EventPriority.P0:
                    self._emergency.record(event, "writer_failed")
            return False
        finally:
            elapsed = time.monotonic_ns() - started_ns
            self._in_flight -= 1
            with self._stats_lock:
                self._stats.writer_time_total_ns += elapsed
        with self._stats_lock:
            self._stats.written_total += len(batch)
        self._consecutive_writer_failures = 0
        return True

    # ------------------------------------------------------------------
    # Failure accounting -- all of these must be non-throwing
    # ------------------------------------------------------------------

    def _note_internal_error(self, where: str) -> None:
        """Record an internal observer error.

        Deliberately lock-free. This is the last-resort accounting path, so it
        must not itself be able to fail by waiting on or acquiring a lock --
        a counter that is occasionally off by one under a racing failure is
        strictly better than an accounting path that can deadlock or throw.
        """
        self._stats.observer_internal_errors_total += 1
        self._stats.last_error = "internal error in " + where

    def _note_writer_failure(self, message: str) -> None:
        """Record a writer failure and trip the breaker if it keeps happening."""
        self._consecutive_writer_failures += 1
        self._stats.writer_failures_total += 1
        self._stats.last_error = "writer: " + str(message)
        if self._consecutive_writer_failures >= self._config.max_consecutive_writer_failures:
            # Circuit breaker: a persistently broken destination must not
            # become a retry loop competing with training for CPU.
            self._writer_disabled = True

    # ------------------------------------------------------------------
    # Spans
    # ------------------------------------------------------------------

    @contextmanager
    def span(self, event_type: EventType, step: Optional[int] = None, **metadata: Any) -> Iterator[None]:
        """Time a block, emitting a start and a guaranteed-close end.

        Adopted from TorchTitan's ``log_trace_span`` (Phase 0 finding): the
        closing event is emitted even on exception, so every start has a match
        and analysis never has to special-case an interrupted span.
        """
        start_ns = time.monotonic_ns()
        self.emit(event_type=event_type, step=step, span_kind=SpanKind.START,
                  priority=EventPriority.P2, metadata=dict(metadata))
        try:
            yield
        except Exception as exc:
            self.emit(event_type=EventType.ERROR, step=step, priority=EventPriority.P0,
                      severity=Severity.ERROR, message="{}".format(exc),
                      metadata=dict(metadata, phase=str(event_type)))
            raise
        finally:
            duration_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
            self.emit(
                event_type=event_type, step=step, span_kind=SpanKind.END,
                duration_ns=time.monotonic_ns() - start_ns,
                metric_name="train.{}_time_ms".format(str(event_type).lower()),
                metric_value=duration_ms, metric_unit=MetricUnit.MILLISECONDS,
                priority=EventPriority.P2, metadata=dict(metadata),
            )

    def accounting_identity(self) -> Dict[str, int]:
        """Check that every emitted event is accounted for.

        ``written + dropped == emitted - sampled_out - validation_failures``
        must hold after :meth:`flush`. A violation means an event vanished,
        which is the one bug class this system cannot tolerate silently.
        """
        stats = self.stats()
        expected = (
            stats.emitted_total - stats.sampled_out_total - stats.validation_failures_total
        )
        accounted = stats.written_total + stats.events_dropped_total
        return {
            "emitted": stats.emitted_total,
            "sampled_out": stats.sampled_out_total,
            "validation_failures": stats.validation_failures_total,
            "written": stats.written_total,
            "dropped": stats.events_dropped_total,
            "expected_accounted": expected,
            "accounted": accounted,
            "unaccounted": expected - accounted,
            "in_flight": self._in_flight,
            "queue_depth": len(self._queue),
        }

    def emergency_events(self) -> List[TrainingEvent]:
        """Retrieve events that could not be persisted. For tests and recovery."""
        return self._emergency.drain()
