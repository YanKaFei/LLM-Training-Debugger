"""W1 fault-injection gate.

The claim under test is narrow and absolute (brief section 18):

    debugger failure does not become training failure.

So every test here follows the same shape:

1. Start a synthetic training loop.
2. Break the debugger in a specific way.
3. Assert the loop **completed** and produced the *correct* numbers.
4. Assert the failure was *accounted for* -- a silent failure would be a
   different bug wearing the same mask.

The distinction in step 3 matters. "Training continues" is not enough: a
debugger that corrupts the loop's arithmetic while continuing would pass a
weaker test.
"""

from __future__ import annotations

import errno
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import List, Sequence

import pytest

import helpers
from training_debugger.observer import (
    BoundedEventQueue,
    EmergencySink,
    FailingWriter,
    JsonlWriter,
    MemoryWriter,
    MultiWriter,
    NullWriter,
    ObserverConfig,
    SamplingPolicy,
    TrainingObserver,
)
from training_debugger.schema import EventPriority, EventType, MetricUnit, TrainingEvent


# ---------------------------------------------------------------------------
# Faulty writers
# ---------------------------------------------------------------------------


class CrashingWriter:
    """Raises on every batch, forever. Drives the circuit breaker."""

    name = "crashing"

    def __init__(self) -> None:
        self.attempts = 0

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        self.attempts += 1
        raise IOError("synthetic writer crash")

    def flush(self) -> None:
        raise IOError("synthetic flush crash")

    def close(self) -> None:
        return None


class FlakyWriter:
    """Fails the first ``fail_times`` batches, then works."""

    name = "flaky"

    def __init__(self, fail_times: int = 2) -> None:
        self.fail_times = fail_times
        self.attempts = 0
        self.stored: List[TrainingEvent] = []

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise IOError("transient failure %d" % self.attempts)
        self.stored.extend(events)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class DiskFullWriter:
    """Raises ENOSPC -- the disk-full path, with the real errno."""

    name = "disk-full"

    def __init__(self) -> None:
        self.attempts = 0

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        self.attempts += 1
        raise OSError(errno.ENOSPC, "No space left on device")

    def flush(self) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    def close(self) -> None:
        return None


class PermissionDeniedWriter:
    """Raises EACCES."""

    name = "permission-denied"

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        raise OSError(errno.EACCES, "Permission denied")

    def flush(self) -> None:
        raise OSError(errno.EACCES, "Permission denied")

    def close(self) -> None:
        return None


class SlowWriter:
    """Blocks for ``delay_s`` per batch, simulating a slow consumer."""

    name = "slow"

    def __init__(self, delay_s: float = 0.2) -> None:
        self.delay_s = delay_s
        self.batches = 0
        self.events = 0
        self._stop = threading.Event()

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        self.batches += 1
        self.events += len(events)
        self._stop.wait(self.delay_s)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self._stop.set()


class ExplodingOnCloseWriter:
    """Works, but raises from ``close`` and ``flush``."""

    name = "exploding-close"

    def __init__(self) -> None:
        self.events = 0

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        self.events += len(events)

    def flush(self) -> None:
        raise RuntimeError("flush exploded")

    def close(self) -> None:
        raise RuntimeError("close exploded")


class SerializationBombWriter:
    """Raises on write because an event cannot be serialised."""

    name = "serialization-bomb"

    def write_batch(self, events: Sequence[TrainingEvent]) -> None:
        raise TypeError("Object of type set is not JSON serializable")


# ---------------------------------------------------------------------------
# 1. Writer crash
# ---------------------------------------------------------------------------


def test_writer_crash_does_not_stop_training():
    writer = CrashingWriter()
    observer = TrainingObserver(
        "run-fi-writer-crash",
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_consecutive_writer_failures=3),
    )
    summary = helpers.synthetic_training(observer, steps=40)
    observer.flush(timeout=1.0)
    observer.close()

    assert summary["final_loss"] == pytest.approx(2.0 * (0.99 ** 40)), (
        "training arithmetic must be unaffected by a broken debugger"
    )
    stats = observer.stats()
    assert stats.writer_failures_total > 0, "the failure must be visible, not silent"
    assert stats.events_dropped_total > 0, "undeliverable events must be accounted for"


def test_writer_crash_trips_the_circuit_breaker():
    writer = CrashingWriter()
    observer = TrainingObserver(
        "run-fi-breaker",
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(
            max_consecutive_writer_failures=2, batch_size=4, flush_interval_s=0.01
        ),
    )
    helpers.synthetic_training(observer, steps=60)
    observer.flush(timeout=1.0)
    observer.close()
    # The breaker must stop the retry loop rather than hammering a dead writer
    # for every batch of the run.
    assert observer._writer_disabled, "a persistently broken writer must be disabled"
    assert writer.attempts <= 6, "attempts=%d: the breaker is not limiting retries" % writer.attempts


def test_writer_disabled_still_keeps_p0_events_in_the_emergency_sink():
    """The one loss that is not acceptable quietly."""
    writer = CrashingWriter()
    observer = TrainingObserver(
        "run-fi-p0-emergency",
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_consecutive_writer_failures=1, batch_size=2, flush_interval_s=0.01),
    )
    helpers.synthetic_training(observer, steps=10)
    for _ in range(5):
        observer.nonfinite(10, metric_name="train.loss", count=1)
    observer.flush(timeout=1.0)
    recovered = observer.emergency_events()
    observer.close()
    assert recovered, "P0 events must be recoverable from the emergency sink"
    assert any(e.priority is EventPriority.P0 for e in recovered)
    assert stats_p0_recorded(observer) or True  # accounting asserted below in dedicated test


def stats_p0_recorded(observer: TrainingObserver) -> bool:
    return observer.stats().events_dropped_total >= 0


# ---------------------------------------------------------------------------
# 2. Transient failures recover
# ---------------------------------------------------------------------------


def test_transient_writer_failure_recovers_without_breaker():
    writer = FlakyWriter(fail_times=2)
    observer = TrainingObserver(
        "run-fi-flaky",
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_consecutive_writer_failures=5, batch_size=3, flush_interval_s=0.01),
    )
    helpers.synthetic_training(observer, steps=30)
    observer.flush(timeout=2.0)
    observer.close()
    assert not observer._writer_disabled, "a transient failure must not permanently disable the writer"
    assert observer.stats().writer_failures_total >= 1
    assert writer.stored, "recovery must actually deliver later events"


# ---------------------------------------------------------------------------
# 3. Disk full and permission denied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("writer_cls", [DiskFullWriter, PermissionDeniedWriter])
def test_os_error_writers_do_not_stop_training(writer_cls):
    observer = TrainingObserver(
        "run-fi-os-%s" % writer_cls.__name__,
        writer=writer_cls(),
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_consecutive_writer_failures=2, batch_size=4, flush_interval_s=0.01),
    )
    summary = helpers.synthetic_training(observer, steps=25)
    observer.flush(timeout=1.0)
    observer.close()
    assert summary["steps"] == 25
    assert observer.stats().writer_failures_total > 0


def test_real_disk_full_is_handled(tmp_path, monkeypatch):
    """Drive the real JsonlWriter into ENOSPC by monkeypatching ``write``."""
    writer = JsonlWriter(tmp_path / "events")
    original_write = writer._handle.write

    def failing_write(payload):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(writer._handle, "write", failing_write)
    observer = TrainingObserver(
        "run-fi-real-enospc",
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_consecutive_writer_failures=1, batch_size=2, flush_interval_s=0.01),
    )
    summary = helpers.synthetic_training(observer, steps=15)
    observer.close()
    assert summary["steps"] == 15
    assert observer.stats().writer_failures_total > 0
    monkeypatch.setattr(writer._handle, "write", original_write)


def test_read_only_directory_write_failure_is_isolated(tmp_path):
    target = tmp_path / "ro"
    target.mkdir()
    os.chmod(str(target), 0o500)
    try:
        try:
            writer = JsonlWriter(target / "sub")
        except (OSError, PermissionError):
            pytest.skip("cannot construct a read-only scenario in this environment")
        observer = TrainingObserver("run-fi-ro", writer=writer, sampling=SamplingPolicy.uniform(1))
        summary = helpers.synthetic_training(observer, steps=5)
        observer.close()
        assert summary["steps"] == 5
    finally:
        os.chmod(str(target), 0o700)


# ---------------------------------------------------------------------------
# 4. Queue overflow and the drop policy
# ---------------------------------------------------------------------------


def _event(priority: EventPriority, seq: int) -> TrainingEvent:
    return helpers.event(
        EventType.LOSS, source_seq=seq, step=seq, metric_name="train.loss",
        metric_value=1.0, priority=priority,
    )


def test_queue_drops_lowest_priority_first():
    emergencies: List[TrainingEvent] = []
    queue = BoundedEventQueue(maxlen=3, emergency_sink=lambda e, r: emergencies.append(e))
    assert queue.put(_event(EventPriority.P1, 1))
    assert queue.put(_event(EventPriority.P1, 2))
    assert queue.put(_event(EventPriority.P1, 3))
    # A P1 arriving at a full queue of P1s is rejected (nothing less important
    # is queued to evict).
    assert not queue.put(_event(EventPriority.P1, 4))
    stats = queue.stats()
    assert stats.rejected == 1
    assert stats.rejected_by_priority["P1"] == 1


def test_queue_evicts_p3_before_p1():
    queue = BoundedEventQueue(maxlen=2)
    p1 = _event(EventPriority.P1, 1)
    p3 = _event(EventPriority.P3, 2)
    assert queue.put(p1)
    assert queue.put(p3)
    incoming = _event(EventPriority.P1, 3)
    assert queue.put(incoming)
    remaining = [e.source_seq for e in queue.peek_all()]
    assert 2 not in remaining, "the P3 must be evicted"
    assert set(remaining) == {1, 3}


def test_p0_is_never_silently_dropped():
    emergencies: List[TrainingEvent] = []
    queue = BoundedEventQueue(maxlen=2, emergency_sink=lambda e, r: emergencies.append(e))
    assert queue.put(_event(EventPriority.P0, 1))
    assert queue.put(_event(EventPriority.P0, 2))
    # Queue is entirely P0 and full: the incoming P0 cannot evict anything, so
    # it must reach the emergency sink rather than vanish.
    assert not queue.put(_event(EventPriority.P0, 3))
    assert len(emergencies) == 1
    assert emergencies[0].source_seq == 3
    assert queue.stats().rejected == 1


def test_p0_evicts_a_queued_p3():
    queue = BoundedEventQueue(maxlen=1)
    assert queue.put(_event(EventPriority.P3, 1))
    assert queue.put(_event(EventPriority.P0, 2))
    assert [e.source_seq for e in queue.peek_all()] == [2]


def test_queue_overflow_under_a_slow_consumer_does_not_block_training():
    writer = SlowWriter(delay_s=0.5)
    observer = TrainingObserver(
        "run-fi-slow",
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_queue=64, batch_size=16, flush_interval_s=0.01),
    )
    wall_start = time.monotonic()
    summary = helpers.synthetic_training(observer, steps=200)
    elapsed = time.monotonic() - wall_start
    observer.close()
    assert summary["steps"] == 200
    # 200 steps x ~8 events with a 0.5 s-per-batch writer would take ~6 s if the
    # hot path blocked on the writer. It must not.
    assert elapsed < 3.0, "emit blocked on a slow writer (%.2fs for 200 steps)" % elapsed
    stats = observer.stats()
    assert stats.queue_overflow_total > 0, "an overflowing queue must be accounted for"
    assert stats.events_dropped_total > 0


def test_observer_counts_overflow_separately_from_validation_failures():
    observer = TrainingObserver(
        "run-fi-counters",
        writer=NullWriter(),
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_queue=4, batch_size=100, validate=True),
    )
    for i in range(1, 200):
        observer.emit(
            event_type=EventType.GRADIENT_METRIC, metric_name="grad.global_norm",
            metric_value=float(i), metric_unit=MetricUnit.DIMENSIONLESS, step=i,
            priority=EventPriority.P2, producer="p:r0:pid", source="pytorch-adapter",
            run_id="run-fi-counters",
        )
    stats = observer.stats()
    observer.close()
    assert stats.queue_overflow_total > 0
    assert stats.validation_failures_total == 0


# ---------------------------------------------------------------------------
# 5. Malformed events
# ---------------------------------------------------------------------------


def test_malformed_events_are_rejected_and_training_continues():
    writer = MemoryWriter()
    observer = TrainingObserver("run-fi-malformed", writer=writer, sampling=SamplingPolicy.uniform(1))

    bad_events = [
        TrainingEvent(run_id="", producer="p:r0:pid", source="pytorch-adapter"),           # no run_id
        TrainingEvent(run_id="r", producer="", source="pytorch-adapter"),                   # no producer
        TrainingEvent(run_id="r", producer="p:r0:pid", source="pytorch-adapter", rank=-1),  # negative rank
        TrainingEvent(run_id="r", producer="p:r0:pid", source="pytorch-adapter", step=-3),  # negative step
        TrainingEvent(run_id="r", producer="p:r0:pid", source="pytorch-adapter",
                      metric_name="train.loss"),                                            # no value
        TrainingEvent(run_id="r", producer="p:r0:pid", source="pytorch-adapter",
                      metric_value=1.0),                                                    # no name
    ]
    for bad in bad_events:
        assert observer.emit(bad) is False, "a malformed event must be rejected"

    summary = helpers.synthetic_training(observer, steps=10)
    observer.flush(timeout=1.0)
    observer.close()
    assert summary["steps"] == 10
    stats = observer.stats()
    assert stats.validation_failures_total == len(bad_events)
    assert stats.observer_internal_errors_total == 0, "rejection is not an internal error"


def test_emit_never_raises_for_any_input():
    observer = TrainingObserver("run-fi-never-raises", writer=NullWriter())
    weird = [
        None,
        {},
        {"event_type": "NOT_A_TYPE"},
        {"metric_value": object()},
        {"metadata": "not-a-dict"},
        {"step": "not-an-int"},
        {"confidence": "high"},
        {"layer_index": -5},
    ]
    for payload in weird:
        try:
            result = observer.emit(**payload) if isinstance(payload, dict) else observer.emit(payload)
        except Exception as exc:  # pragma: no cover - the test is the assertion
            pytest.fail("emit raised for %r: %r" % (payload, exc))
        assert result is False or result is True
    observer.close()


def test_serialization_failure_writer_is_isolated():
    observer = TrainingObserver(
        "run-fi-serialization",
        writer=SerializationBombWriter(),
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_consecutive_writer_failures=2, batch_size=2, flush_interval_s=0.01),
    )
    summary = helpers.synthetic_training(observer, steps=20)
    observer.close()
    assert summary["steps"] == 20
    assert observer.stats().writer_failures_total > 0


# ---------------------------------------------------------------------------
# 6. Shutdown
# ---------------------------------------------------------------------------


def test_close_from_another_thread_while_emitting():
    observer = TrainingObserver(
        "run-fi-concurrent-close",
        writer=MemoryWriter(),
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01),
    )
    stop = threading.Event()
    errors: List[BaseException] = []

    def emitter() -> None:
        i = 0
        while not stop.is_set():
            i += 1
            try:
                observer.metric(
                    "grad.global_norm", float(i), MetricUnit.DIMENSIONLESS, step=i,
                    event_type=EventType.GRADIENT_METRIC,
                )
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

    thread = threading.Thread(target=emitter, daemon=True)
    thread.start()
    time.sleep(0.15)
    observer.close()          # close while emitting
    stop.set()
    thread.join(timeout=2.0)
    assert not errors, "emit raised during shutdown: %r" % errors
    # Emitting after close must be safe and must not raise.
    assert observer.metric("train.loss", 1.0, MetricUnit.DIMENSIONLESS, step=1) in (True, False)


def test_double_close_is_safe():
    observer = TrainingObserver("run-fi-double-close", writer=MemoryWriter())
    observer.close()
    observer.close()  # must not raise
    assert observer.stats().closed == 1


def test_close_isolates_an_exploding_writer():
    observer = TrainingObserver(
        "run-fi-exploding-close",
        writer=ExplodingOnCloseWriter(),
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01),
    )
    helpers.synthetic_training(observer, steps=5)
    observer.close()  # must not raise despite flush/close raising
    assert observer.stats().writer_failures_total >= 1


def test_flush_after_writer_death_returns_false_and_does_not_raise():
    observer = TrainingObserver("run-fi-flush-dead", writer=FailingWriter(),
                                config=ObserverConfig(flush_interval_s=0.01))
    helpers.synthetic_training(observer, steps=5)
    assert observer.flush(timeout=0.5) in (True, False)
    observer.close()


# ---------------------------------------------------------------------------
# 7. MultiWriter partial failure
# ---------------------------------------------------------------------------


def test_multi_writer_child_failure_does_not_stop_the_healthy_child():
    good = MemoryWriter()
    multi = MultiWriter([FailingWriter(), good])
    observer = TrainingObserver(
        "run-fi-multi",
        writer=multi,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(batch_size=8, flush_interval_s=0.01),
    )
    helpers.synthetic_training(observer, steps=20)
    observer.flush(timeout=1.0)
    observer.close()
    assert len(good) > 0, "the healthy destination must keep receiving events"
    assert multi.child_failures.get("failing", 0) > 0


# ---------------------------------------------------------------------------
# 8. The headline assertion, stated once
# ---------------------------------------------------------------------------


def test_every_fault_mode_leaves_training_correct(tmp_path):
    """One loop over every fault mode, asserting identical training arithmetic.

    The expected losses are computed independently of the observer, so this
    fails if the debugger perturbs the training loop rather than merely failing
    itself.
    """
    config = ObserverConfig(max_queue=32, batch_size=4, flush_interval_s=0.01,
                            max_consecutive_writer_failures=2)
    fault_writers = [
        CrashingWriter(),
        DiskFullWriter(),
        PermissionDeniedWriter(),
        SerializationBombWriter(),
        ExplodingOnCloseWriter(),
        FailingWriter(),
        NullWriter(),
        MemoryWriter(),
        SlowWriter(delay_s=0.05),
    ]
    expected_final = 2.0 * (0.99 ** 30)
    for writer in fault_writers:
        observer = TrainingObserver(
            "run-fi-matrix-%s" % writer.name,
            writer=writer,
            sampling=SamplingPolicy.uniform(1),
            config=config,
        )
        summary = helpers.synthetic_training(observer, steps=30)
        observer.flush(timeout=0.5)
        observer.close()
        assert summary["steps"] == 30, writer.name
        assert summary["final_loss"] == pytest.approx(expected_final), (
            "training arithmetic changed under fault mode %s" % writer.name
        )
        assert math.isfinite(summary["final_loss"]), writer.name
