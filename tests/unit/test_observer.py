"""Observer unit tests: sampling, accounting, priority, spans, lifecycle."""

from __future__ import annotations

import json
import time

import pytest

import helpers
from training_debugger.observer import (
    DEFAULT_POLICY_PHASE1,
    NullWriter,
    BoundedEventQueue,
    EmergencySink,
    JsonlWriter,
    MemoryWriter,
    ObserverConfig,
    SamplingPolicy,
    SamplingRule,
    TrainingObserver,
)
from training_debugger.schema import EventPriority, EventType, MetricUnit, SourceKind


@pytest.fixture()
def observer():
    writer = MemoryWriter()
    obs = TrainingObserver(
        "run-unit",
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01),
    )
    yield obs
    obs.close()


# ---------------------------------------------------------------------------
# Context filling
# ---------------------------------------------------------------------------


def test_observer_fills_context_the_caller_omitted(observer):
    observer.metric("train.loss", 1.0, MetricUnit.DIMENSIONLESS, step=1, event_type=EventType.LOSS)
    observer.flush(timeout=1.0)
    events = observer.writer.events
    assert len(events) == 1
    event = events[0]
    assert event.run_id == "run-unit"
    assert event.producer == observer.producer
    assert event.source == str(SourceKind.PYTORCH_ADAPTER)
    assert event.source_seq == 1


def test_producer_sequence_increments_per_emission(observer):
    for step in range(1, 6):
        observer.metric("train.loss", 1.0, MetricUnit.DIMENSIONLESS, step=step, event_type=EventType.LOSS)
    observer.flush(timeout=1.0)
    assert [e.source_seq for e in observer.writer.events] == [1, 2, 3, 4, 5]


def test_explicit_source_seq_is_respected(observer):
    observer.emit(event_type=EventType.LOSS, step=1, metric_name="train.loss",
                  metric_value=1.0, metric_unit=MetricUnit.DIMENSIONLESS, source_seq=99)
    observer.flush(timeout=1.0)
    assert observer.writer.events[0].source_seq == 99


def test_world_size_is_propagated():
    obs = TrainingObserver("run-ws", writer=MemoryWriter(), rank=3, world_size=8,
                           sampling=SamplingPolicy.uniform(1))
    obs.metric("train.loss", 1.0, MetricUnit.DIMENSIONLESS, step=1, event_type=EventType.LOSS)
    obs.flush(timeout=1.0)
    event = obs.writer.events[0]
    assert event.rank == 3 and event.world_size == 8
    obs.close()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sampling_drops_the_configured_cadence():
    policy = SamplingPolicy(rules=[SamplingRule("grad.*", every_n_steps=5)])
    obs = TrainingObserver("run-s", writer=MemoryWriter(), sampling=policy)
    for step in range(1, 21):
        obs.metric("grad.global_norm", 1.0, MetricUnit.DIMENSIONLESS, step=step,
                   event_type=EventType.GRADIENT_METRIC)
    obs.flush(timeout=1.0)
    kept = [e.step for e in obs.writer.events]
    assert kept == [5, 10, 15, 20]
    assert obs.stats().sampled_out_total == 16
    obs.close()


def test_sampling_is_deterministic_across_identical_runs():
    def run():
        policy = SamplingPolicy(rules=[SamplingRule("*", every_n_steps=3)])
        obs = TrainingObserver("run-det", writer=MemoryWriter(), sampling=policy)
        for step in range(1, 31):
            obs.metric("grad.global_norm", 1.0, MetricUnit.DIMENSIONLESS, step=step,
                       event_type=EventType.GRADIENT_METRIC)
        obs.flush(timeout=1.0)
        steps = [e.step for e in obs.writer.events]
        obs.close()
        return steps

    assert run() == run()


def test_warmup_steps_are_never_sampled_away():
    """``always_first_steps`` is an exclusive bound on the step index.

    With 1-based step numbering, ``always_first_steps=10`` keeps steps 1..9 and
    lets step 10 be governed by the cadence. The bound is exclusive so that a
    0-based producer (``step`` starting at 0) gets exactly N guaranteed steps.
    """
    policy = SamplingPolicy(rules=[SamplingRule("grad.*", every_n_steps=100, always_first_steps=10)])
    obs = TrainingObserver("run-warm", writer=MemoryWriter(), sampling=policy)
    for step in range(1, 12):
        obs.metric("grad.global_norm", 1.0, MetricUnit.DIMENSIONLESS, step=step,
                   event_type=EventType.GRADIENT_METRIC)
    obs.flush(timeout=1.0)
    assert [e.step for e in obs.writer.events] == list(range(1, 10))
    obs.close()


def test_p0_events_are_never_sampled():
    policy = SamplingPolicy(rules=[SamplingRule("*", every_n_steps=1000)])
    obs = TrainingObserver("run-p0", writer=MemoryWriter(), sampling=policy)
    for step in range(1, 6):
        obs.nonfinite(step, metric_name="train.loss", count=1)
    obs.flush(timeout=1.0)
    assert len(obs.writer.events) == 5
    obs.close()


def test_events_without_a_step_are_never_sampled():
    policy = SamplingPolicy(rules=[SamplingRule("*", every_n_steps=1000)])
    obs = TrainingObserver("run-nostep", writer=MemoryWriter(), sampling=policy)
    obs.run_start()
    obs.run_end()
    obs.flush(timeout=1.0)
    assert len(obs.writer.events) == 2
    obs.close()


def test_sampling_policy_clone_has_independent_counters():
    a = DEFAULT_POLICY_PHASE1.clone()
    b = DEFAULT_POLICY_PHASE1.clone()
    event = helpers.event(EventType.LOSS, step=1, metric_name="train.loss", metric_value=1.0)
    a.should_emit(event)
    assert a.considered == 1
    assert b.considered == 0
    assert DEFAULT_POLICY_PHASE1.considered == 0, "the template must not accumulate counters"


def test_default_policy_records_loss_every_step_and_layers_sparsely():
    assert DEFAULT_POLICY_PHASE1.effective_rate("train.loss") == 1.0
    assert DEFAULT_POLICY_PHASE1.effective_rate("train.step_time") == 1.0
    assert DEFAULT_POLICY_PHASE1.effective_rate("grad.layer_norm") == pytest.approx(0.1)
    assert DEFAULT_POLICY_PHASE1.effective_rate("grad.global_norm") == pytest.approx(0.5)


def test_sampling_rule_rejects_invalid_cadence():
    with pytest.raises(ValueError):
        SamplingRule("*", every_n_steps=0)
    with pytest.raises(ValueError):
        SamplingRule("*", every_n_steps=1, always_first_steps=-1)


# ---------------------------------------------------------------------------
# Accounting and health
# ---------------------------------------------------------------------------


def test_stats_count_what_was_written(observer):
    helpers.synthetic_training(observer, steps=10, layer_count=2)
    observer.flush(timeout=2.0)
    stats = observer.stats()
    # run_start + run_end + 10*(loss+lr+grad+2 layer+step_end) = 2 + 60
    assert stats.written_total == 62
    assert stats.emitted_total == 62
    assert stats.events_dropped_total == 0
    assert stats.observer_internal_errors_total == 0


def test_health_reports_the_contract_fields(observer):
    observer.metric("train.loss", 1.0, MetricUnit.DIMENSIONLESS, step=1, event_type=EventType.LOSS)
    observer.flush(timeout=1.0)
    health = observer.health()
    for key in (
        "observer_internal_errors_total",
        "events_dropped_total",
        "queue_overflow_total",
        "writer_failures_total",
    ):
        assert key in health["observer"], key
    assert health["run_id"] == "run-unit"
    assert health["queue"]["depth"] == 0
    assert json.dumps(health, default=str)  # must be serialisable for an endpoint


def test_health_is_json_serialisable_with_nan_values(observer):
    observer.metric("train.loss", float("nan"), MetricUnit.DIMENSIONLESS, step=1, event_type=EventType.LOSS)
    observer.flush(timeout=1.0)
    payload = json.dumps(observer.health(), default=str)
    assert "run-unit" in payload


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


def test_span_emits_start_and_end(observer):
    with observer.span(EventType.STEP_START, step=1):
        time.sleep(0.001)
    observer.flush(timeout=1.0)
    kinds = [e.span_kind.value for e in observer.writer.events]
    assert kinds == ["start", "end"]
    end = observer.writer.events[-1]
    assert end.duration_ns is not None and end.duration_ns > 0


def test_span_closes_even_when_the_body_raises(observer):
    with pytest.raises(ValueError):
        with observer.span(EventType.STEP_START, step=2):
            raise ValueError("boom")
    observer.flush(timeout=1.0)
    types = [e.event_type for e in observer.writer.events]
    assert types == [EventType.STEP_START, EventType.ERROR, EventType.STEP_START], (
        "a raise must still produce a matching close, with an ERROR in between"
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_start_is_idempotent(observer):
    observer.start()
    observer.start()
    assert observer.stats().started == 1


def test_close_is_idempotent_and_counts_once(observer):
    observer.close()
    observer.close()
    assert observer.stats().closed == 1


def test_flush_reports_whether_the_queue_drained():
    obs = TrainingObserver("run-flush", writer=MemoryWriter(),
                           sampling=SamplingPolicy.uniform(1),
                           config=ObserverConfig(flush_interval_s=0.01))
    for step in range(1, 101):
        obs.metric("train.loss", 1.0, MetricUnit.DIMENSIONLESS, step=step, event_type=EventType.LOSS)
    assert obs.flush(timeout=5.0) is True
    assert len(obs.writer) == 100
    obs.close()


def test_jsonl_writer_round_trips_events(tmp_path):
    writer = JsonlWriter(tmp_path / "run")
    obs = TrainingObserver("run-jsonl", writer=writer, sampling=SamplingPolicy.uniform(1))
    helpers.synthetic_training(obs, steps=5, layer_count=1)
    obs.flush(timeout=2.0)
    obs.close()
    lines = writer.path.read_text(encoding="utf-8").strip().splitlines()
    # 2 lifecycle events + 5 steps x (loss + lr + grad + 1 layer metric + step_end)
    assert len(lines) == 2 + 5 * 5
    from training_debugger.schema import TrainingEvent

    restored = [TrainingEvent.from_json_line(line) for line in lines]
    assert all(e.run_id == "run-jsonl" for e in restored)
    assert len({e.event_id for e in restored}) == len(restored)


def test_run_id_is_required():
    with pytest.raises(ValueError):
        TrainingObserver("")


# ---------------------------------------------------------------------------
# Priority is carried on the event, and the queue respects it
# ---------------------------------------------------------------------------


def test_priority_survives_emission(observer):
    observer.metric("train.loss", 1.0, MetricUnit.DIMENSIONLESS, step=1,
                    event_type=EventType.LOSS, priority=EventPriority.P0)
    observer.metric("grad.layer_norm", 1.0, MetricUnit.DIMENSIONLESS, step=1,
                    event_type=EventType.GRADIENT_METRIC, priority=EventPriority.P3)
    observer.flush(timeout=1.0)
    priorities = [e.priority for e in observer.writer.events]
    assert priorities == [EventPriority.P0, EventPriority.P3]


def test_queue_never_exceeds_its_bound():
    queue = BoundedEventQueue(maxlen=10)
    for i in range(1000):
        queue.put(helpers.event(EventType.LOSS, source_seq=i, step=i,
                                metric_name="train.loss", metric_value=1.0))
    assert len(queue) <= 10
    assert queue.stats().max_depth <= 10


# ---------------------------------------------------------------------------
# Accounting identity
# ---------------------------------------------------------------------------


def test_every_event_is_accounted_for_after_flush():
    """The identity that the 1M-event benchmark exposed a violation of.

    Before ``flush`` waited for the writer thread to be idle, it could return
    with a batch still in flight, so ``written_total`` momentarily under-counted
    and events appeared to be neither written nor dropped. An event that is
    neither written nor counted is exactly the failure this system must not have.
    """
    observer = TrainingObserver(
        "run-accounting",
        writer=MemoryWriter(),
        sampling=SamplingPolicy.from_spec([("grad.*", 3)]),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16),
    )
    helpers.synthetic_training(observer, steps=120, layer_count=3)
    assert observer.flush(timeout=5.0) is True
    identity = observer.accounting_identity()
    assert identity["unaccounted"] == 0, identity
    assert identity["in_flight"] == 0
    assert identity["queue_depth"] == 0
    observer.close()


def test_accounting_identity_holds_with_a_dropping_writer():
    from training_debugger.observer import FailingWriter

    observer = TrainingObserver(
        "run-accounting-fail",
        writer=FailingWriter(),
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=8, max_consecutive_writer_failures=2),
    )
    helpers.synthetic_training(observer, steps=40, layer_count=2)
    observer.flush(timeout=2.0)
    identity = observer.accounting_identity()
    assert identity["unaccounted"] == 0, identity
    assert identity["dropped"] > 0
    observer.close()


def test_accounting_identity_holds_under_queue_overflow():
    observer = TrainingObserver(
        "run-accounting-overflow",
        writer=NullWriter(),
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_queue=4, batch_size=500, flush_interval_s=0.05),
    )
    helpers.synthetic_training(observer, steps=100, layer_count=2)

    # `flush` returns whether it fully drained, and it can time out. The accounting
    # identity is only meaningful *after* a completed flush: an event sitting in the
    # writer's batch is neither `written` nor `dropped`, so it shows up as
    # `unaccounted` even though nothing has been lost.
    #
    # This test previously discarded the return value and flushed with `timeout=2.0`,
    # so under full-suite load a timed-out flush was reported as an accounting
    # violation -- a misleading failure that looks like data loss but is an
    # incomplete drain. Asserting completion first separates the two.
    drained = observer.flush(timeout=30.0)
    assert drained, (
        "the flush did not complete, so the accounting identity cannot be asserted; "
        "this is an incomplete drain, not a lost event"
    )
    identity = observer.accounting_identity()
    assert identity["unaccounted"] == 0, identity
    assert identity["in_flight"] == 0, identity
    assert identity["dropped"] > 0
    observer.close()


def test_incomplete_flush_is_not_an_accounting_violation():
    """W4: separate "flush did not finish" from "an event vanished".

    An in-flight event is counted in `emitted` but in neither `written` nor `dropped`,
    so it appears as `unaccounted`. Reporting that as a lost event would be a false
    alarm about the one bug class this system exists to catch.
    """
    from training_debugger.observer import (
        NullWriter, ObserverConfig, SamplingPolicy, TrainingObserver,
    )

    observer = TrainingObserver(
        "run-incomplete-flush", writer=NullWriter(), sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(max_queue=4, batch_size=500, flush_interval_s=0.05),
    )
    helpers.synthetic_training(observer, steps=100, layer_count=2)

    # A flush that has demonstrably completed is the only state in which the identity
    # is a statement about correctness.
    assert observer.flush(timeout=30.0) is True
    identity = observer.accounting_identity()
    assert identity["unaccounted"] == 0
    assert identity["accounted"] == identity["expected_accounted"]

    # Every event is in exactly one terminal state: written or dropped.
    assert identity["written"] + identity["dropped"] == identity["expected_accounted"]
    observer.close()
