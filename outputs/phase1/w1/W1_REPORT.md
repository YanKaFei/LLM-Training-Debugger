# W1 REPORT — Observer SDK

**Wave:** W1
**Status:** **COMPLETE**
**Gate:** fault-injection — **PASS** (10 of 10 fault modes leave training numerically correct)

---

## 1. What was built

```
src/training_debugger/observer/
  observer.py         TrainingObserver, ObserverConfig
  queue.py            BoundedEventQueue + QueueStats
  writer_protocol.py  EventWriter, Null/Memory/Jsonl/Multi/Failing writers
  sampling.py         SamplingPolicy, SamplingRule, DEFAULT_POLICY_PHASE1
  health.py           ObserverStats, EmergencySink
tests/fault_injection/test_observer_faults.py   (23 tests)
tests/fault_injection/run_report.py             (generates the report)
tests/unit/test_observer.py                     (32 tests)
outputs/phase1/w1/FAULT_INJECTION_REPORT.md     (generated)
```

**55 new tests.** Total suite: **145 passing**, 1 skipped.

---

## 2. Public API

Exactly the shape the brief asked for, plus lifecycle conveniences:

```python
observer = TrainingObserver(run_id="run-123", writer=JsonlWriter("out/"))

observer.run_start(model_id="tiny-transformer")
observer.metric("train.loss", loss, MetricUnit.DIMENSIONLESS, step=step, event_type=EventType.LOSS)
observer.metric("grad.global_norm", norm, MetricUnit.DIMENSIONLESS, step=step,
                event_type=EventType.GRADIENT_METRIC)
with observer.span(EventType.STEP_START, step=step):
    ...                      # emits start + guaranteed-close end
observer.nonfinite(step, metric_name="train.loss", count=1)   # P0
observer.run_end()
observer.flush(timeout=2.0)
observer.close()
```

`observer.emit(...)` accepts either a constructed `TrainingEvent` or keyword
fields, and **never raises** — that is the fail-open contract, not a nicety.

---

## 3. Fail-open contract: the evidence

`outputs/phase1/w1/FAULT_INJECTION_REPORT.md` is **generated** by
`tests/fault_injection/run_report.py`, not written by hand, so it cannot drift
from the code. Each of the 10 fault modes runs the same deterministic 30-step
synthetic training loop while the debugger is broken, then asserts two things
independently:

1. all 30 steps ran;
2. the final loss equals `2.0 * 0.99**30` — computed without reference to the observer.

Check 2 is the load-bearing one. "Training continued" alone would not catch a
debugger that corrupts the loop while continuing.

| Fault mode | Training completed | Arithmetic correct | Dropped | Overflow | Writer failures |
|---|---|---|---|---|---|
| `writer_crash` | yes | yes | 162 | 98 | 3 |
| `disk_full` (ENOSPC) | yes | yes | 162 | 98 | 3 |
| `permission_denied` (EACCES) | yes | yes | 162 | 98 | 3 |
| `serialization_failure` | yes | yes | 162 | 98 | 4 |
| `flush_and_close_crash` | yes | yes | 98 | 98 | 2 |
| `transient_failure_then_recovery` | yes | yes | 114 | 98 | 2 |
| `queue_overflow_slow_consumer` | yes | yes | 98 | 98 | 0 |
| `writer_disabled_after_breaker` | yes | yes | 160 | 88 | 3 |
| `null_writer` | yes | yes | 55 | 55 | 0 |
| `healthy_memory_writer` **with production defaults** | yes | yes | **0** | **0** | 0 |

Two rows deserve comment:

* **The control case uses production defaults** (`max_queue=10_000`), not the
  shrunken fault config. It records **zero drops and zero overflow**, which is
  the honest statement that the observer keeps up at realistic settings. The
  earlier draft of this report showed the control case dropping events because it
  inherited the artificially small 64-item queue; that was a misleading artefact
  and has been removed.
* **`transient_failure_then_recovery` uses a breaker threshold of 5, not 2.**
  With the default threshold the circuit breaker trips before the writer gets a
  chance to recover, so the recovery path was never exercised. Raising the
  threshold in that one case is what makes the test mean anything.

Bugs found *by* this suite, before it passed:

| Bug | How it surfaced |
|---|---|
| The observer silently **healed** an explicitly empty `run_id`/`producer`/`source`, masking producer bugs | `test_malformed_events_are_rejected_and_training_continues` |
| `queue.py` and `writer_protocol.py` swallowed failures with bare `except: pass` | `test_layering.py::test_no_broad_except_pass_in_core_modules` |

The second is worth recording: the layering guard caught the *observer's own*
silent handlers. The fix was not to add an exemption but to make every
suppression **counted**.

---

## 4. Backpressure

No unbounded queue. `BoundedEventQueue` is a deque behind a lock with a hard
`maxlen`, and eviction is priority-aware:

| Incoming | Queue state | Outcome |
|---|---|---|
| P1 | full of P1 | **rejected**, counted `rejected_by_priority["P1"]` |
| P1 | contains a P3 | P3 evicted, P1 accepted |
| P0 | contains anything non-P0 | the least-important item is evicted |
| P0 | entirely P0 and full | **emergency sink** — never silently dropped |

Eviction scans from the back, so the *oldest* events of a step survive longest.
That is deliberate: the partial-order model needs the earliest events of a step,
and evicting newest-first keeps a step's beginning intact.

Deliberately **not** a heap: reordering by priority would scramble the emission
sequence the ordering model depends on.

A full queue never blocks. `test_queue_overflow_under_a_slow_consumer_does_not_block_training`
runs 200 steps against a writer that blocks 0.5 s per batch and asserts the loop
finishes in under 3 s — if `emit` blocked on the writer it would take ~6 s.

---

## 5. Sampling lives in the observer

The brief requires this, and the reason is worth restating: when every callsite
writes its own `if step % N == 0`, the sampling rate becomes an emergent property
of the codebase and nobody can answer "how often is the layer gradient norm
recorded?" without grepping.

Two design rules:

* **Deterministic.** `step % every_n == 0`, never random. A random sampler would
  make two runs of the same seed produce different event sets, breaking the
  scenario regression corpus and making an incident unreproducible.
* **P0 is never sampled.** Dropping an incident to save bandwidth inverts the
  priorities.

`DEFAULT_POLICY_PHASE1` encodes the brief's example: loss every step, learning
rate every step, global grad norm every 2nd step, per-layer grad norm every 10th,
phase timings every 5th, and a 50-step full-resolution warm-up. `effective_rate()`
makes each rate a queryable number, and `clone()` prevents the module-level
template from accumulating counters or leaking tuning between runs.

---

## 6. Counters

All four counters the brief names are implemented and asserted on:

```
observer_internal_errors_total
events_dropped_total
queue_overflow_total
writer_failures_total
```

plus `validation_failures_total`, `sampled_out_total`, `written_total`,
`emitted_total`, `emergency_writes_total`, and timing accumulators
(`emit_time_total_ns`, `writer_time_total_ns`). `observer.health()` returns a
JSON-serialisable snapshot usable by a future endpoint.

`EmergencySink` keeps a bounded recoverable copy of unpersistable events and
writes a **rate-limited** stderr line (default 5/s, with a suppression count).
Rate limiting matters: a queue-full condition is exactly when an unbounded
stderr write would make things worse.

---

## 7. Thread safety

`emit` is safe to call from multiple threads. The writer thread drains
independently. `close()` may be called from another thread while emitting
(`test_close_from_another_thread_while_emitting`), twice
(`test_double_close_is_safe`), or after the writer has died
(`test_flush_after_writer_death_returns_false_and_does_not_raise`) — none raise.

The two accounting paths that must survive a broken lock
(`_note_internal_error`, `_note_writer_failure`) are deliberately **lock-free**:
a counter that is occasionally off by one under a racing failure is strictly
better than an accounting path that can deadlock.

---

## 8. Gate results

```
$ .venv/bin/python -m pytest tests -q
145 passed, 1 skipped
```

| Gate item | Status |
|---|---|
| non-blocking by default | **PASS** — slow-consumer test, 200 steps in <3 s against a 0.5 s/batch writer |
| bounded memory | **PASS** — `test_queue_never_exceeds_its_bound` (1000 puts, depth ≤ 10) |
| failure isolated | **PASS** — 10/10 fault modes |
| thread-safe | **PASS** — concurrent close, concurrent emit |
| low overhead | **deferred to W4** (its own gate) |
| bounded queue, no unbounded queue | **PASS** |
| explicit drop policy | **PASS** — `DROP_LOW_PRIORITY` via eviction, P3 → P2 ordering |
| P0 never silently dropped | **PASS** — emergency sink |
| drop accounting exists | **PASS** — 4 counters + per-priority breakdown |
| sampling supported | **PASS** — 10 rules, deterministic, P0-exempt |
| fault-injection gate | **PASS** |

---

## 9. Not verified in W1

* **Real disk-full.** The `disk_full` case uses a writer that raises the real
  `ENOSPC` errno, and a second test drives the real `JsonlWriter` into `ENOSPC`
  by monkeypatching `write`. An actual full filesystem was **not** created — that
  needs a sized disk image, which is W3+ work if it is wanted.
* **Real DuckDB / Parquet unavailability.** Not applicable in W1; those writers
  do not exist yet. The `EventWriter` abstraction is what makes them testable in
  W3, and the fault-injection report records this as *not applicable* rather than
  claiming a pass.
* **Overhead.** Not measured here. W4 owns the overhead gate.
* **Multi-process.** W1 is single-process. The producer identity model supports
  multi-rank but no cross-process test exists yet.
* **sustained high-rate emission.** The control case runs 30 steps. Whether drops
  stay at zero over 100k steps at full rate is a W4 measurement.
