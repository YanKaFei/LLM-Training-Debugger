# The observer contract

## The one rule

> Observability must never make training less reliable.

## Guarantees

| Guarantee | Mechanism | Test |
|---|---|---|
| `emit()` never raises | every path is exception-isolated | `test_emit_never_raises_for_any_input` |
| never blocks | bounded queue, no I/O on the hot path | `test_queue_overflow_under_a_slow_consumer_does_not_block_training` |
| bounded memory | `maxlen` with priority-aware eviction | `test_queue_never_exceeds_its_bound` |
| P0 never silently dropped | emergency sink (stderr + recoverable buffer) | `test_p0_is_never_silently_dropped` |
| failures are visible | four counters plus `last_error` | `test_health_reports_the_contract_fields` |
| a dead writer cannot spin | circuit breaker | `test_writer_crash_trips_the_circuit_breaker` |
| accounting is exact | `accounting_identity()` | `test_every_event_is_accounted_for_after_flush` |

## Drop policy

A full queue accepts an incoming event only by evicting one of **strictly lower**
importance (P0 > P1 > P2 > P3). A P3 arrival is dropped rather than displacing a
queued P1; a P0 arrival evicts anything non-P0; a P0 arrival into an all-P0 full
queue goes to the emergency sink. Eviction scans from the back, so a step's
earliest events survive longest — the partial order needs them.

## Sampling

`SamplingPolicy` is owned by the observer, never by callers. Deterministic
(`step % every_n`), P0-exempt, and — importantly — it gates **computation**, not
just emission: `would_emit()` lets an adapter skip an expensive reduction that
would be discarded anyway.

## The identity

After `flush()`, `written + dropped == emitted - sampled_out - validation_failures`.
A violation means an event vanished. `flush()` waits for the queue to be empty
**and** the writer thread to be idle, because queue-empty alone is not enough.
