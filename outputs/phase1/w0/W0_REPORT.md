# W0 REPORT — Freeze TrainingEvent v0.1

**Wave:** W0
**Status:** **COMPLETE**
**Gate:** schema serialization · deserialization · unknown-field behaviour · old-schema read · idempotency · producer sequence · partial ordering · invalid-event rejection — **all PASS**

---

## 1. What was built

```
src/training_debugger/schema/
  enums.py       closed vocabularies + SCHEMA_VERSION
  event.py       the TrainingEvent envelope
  ordering.py    the partial-order algebra (EventOrdering)
  evidence.py    Evidence, Multiplicity, EvidenceBuckets
  incident.py    Incident, Hypothesis
  validation.py  strict validation that never raises for data problems
docs/SCHEMA_EVOLUTION.md
```

**62 new unit tests** across five files (`test_schema_serialization`, `test_schema_evolution`, `test_ordering`, `test_validation`, `test_layering`).

---

## 2. The envelope

Every field required by the Phase 1 brief §2.1 is present, plus four that the
brief's W1/W6 requirements make necessary:

| Added field | Why it could not be omitted | Required by |
|---|---|---|
| `priority` | The drop policy is defined over it, and the queue must not parse `metadata` to decide what to evict | W1 §3.3 |
| `parent_event_id` | The only *provable* cross-producer happens-before witness besides a shared producer sequence | W0 §2.3 |
| `span_kind` + `duration_ns` | Span pairing without inferring it from a name suffix | event model |
| `idempotency_key` | Derived from `(run_id, producer, source_seq)`; a retry of the same logical event must not look like a new event | W0 §2.6 |

`message` is also a first-class field (bounded to 1000 chars with middle
truncation), because `WARNING`/`ERROR` need somewhere to put text that is not
`metadata`.

**`run_id`, `producer` and `source` default to `None`, not `""`.** This is
load-bearing: `None` means *absent, fill from context*; `""` means *the producer
supplied an explicitly wrong value*. The observer fills the former and rejects
the latter, so an integration bug surfaces instead of being silently healed.

---

## 3. The four ordering concepts are separate fields

Implemented exactly as the brief requires. `ordering.py`'s module docstring
states the table, and the code enforces it:

| Concept | Fields | Establishes |
|---|---|---|
| logical position | `step`, `microstep` | which step is described — **not** who ran first |
| producer-local order | `rank`, `source_seq` | a total order *within one emitting stream* |
| observation time | `wall_time_ns`, `monotonic_time_ns` | when we observed it — **never** a happens-before witness |
| storage order | `ingest_seq` | physical append order — **never** causal |

`EventOrdering.compare` returns `BEFORE` **only** with one of three witnesses:

1. `producer_sequence` — same producer, ordered by `source_seq`.
2. `causal_edge` — `b.parent_event_id == a.event_id`.
3. `barrier` — a caller-supplied barrier witness. **Phase 1 never fabricates one**; the parameter exists so the model is complete, and `test_barrier_witness_orders_across_producers_but_is_never_invented` proves an unrecognised witness is ignored.

Everything else is `CONCURRENT_OR_UNKNOWN`. `event_id` and `wall_time_ns` are
never used as witnesses — `event_id` appears only as a deterministic tiebreak in
`storage_sort_key`.

### 3.1 A producer ordering-violation guard, added during implementation

`source_seq` records *emission* order, which equals *training* order only if the
producer emits each step's observations in order. A producer that defers or
batches — flushing layer metrics on a different cadence, say — can emit step
812's loss before step 810's gradient metric. Trusting `source_seq` there would
manufacture a false "X preceded Y" story with full confidence.

`compare()` therefore detects the conflict and **abstains**:

```python
if pa != pb and position_says is not seq_relation:
    return CONCURRENT_OR_UNKNOWN, LOW, "producer_sequence_position_conflict"
```

This was not in the brief. It falls directly out of taking §2.3 seriously, and
`test_emission_order_contradicting_step_order_is_refused` pins it.

---

## 4. `first_abnormal_event` handling

The brief asked for `earliest_abnormal_candidate_ids[]` plus
`ordering_confidence`, and for `first_abnormal_event_id` to be populated only
when uniqueness is provable. Implemented as the **minimal elements of the partial
order** — the set of events that no other event provably precedes.

`Incident.first_abnormal_event_id` is a property that returns `None` unless
`ordering_confidence is PROVEN` **and** exactly one minimum exists, so a consumer
*cannot* read a singular answer where only a set is defensible.

Confidence semantics, verified by test:

| Situation | `first_abnormal_event_id` | `ordering_confidence` |
|---|---|---|
| Exactly one minimum, no unsettled comparisons | set | `PROVEN` |
| One minimum, but some comparisons had no witness | set | `LOW` |
| Several minima, all pairs settled (same logical position) | `None` | `PROVEN` — *proved concurrent* |
| Several minima, some pairs unsettled | `None` | `UNKNOWN` |

Confirmed behaviour on the headline case:

```
2 abnormal events are mutually unordered, so no single event can be named first:
  - grad.global_norm = 9.2 on pytorch-adapter, step 810
  - train.loss = 4.7 on pytorch-adapter, step 812
Their exact ordering cannot be established from the available evidence.
```

The brief's required wording — *"Their exact ordering cannot be established"* —
is produced verbatim, by the deterministic renderer, from the data.

**One design decision worth flagging:** cross-producer events at *different*
steps are classified as a knowledge **gap** (`UNKNOWN`), not as proven
concurrency. Two ranks at steps 810 and 812 are almost certainly ordered in
reality; we simply cannot prove which way. Reporting them as "concurrent" would
be a stronger claim than the evidence supports, so the engine says the weaker,
true thing.

---

## 5. Schema evolution

`docs/SCHEMA_EVOLUTION.md` defines the `MAJOR.MINOR` policy, the compatibility
matrix, the unknown-field policy, the migration policy for immutable Parquet, and
the non-finite-value interop caveat.

| Event version | Reader `0.1` behaviour |
|---|---|
| `0.≤1` | Accept |
| `0.>1` | Accept with preservation under `metadata["_unrecognized"]`; non-fatal issue recorded |
| `≠0.x` | Reject; fatal `schema_major_mismatch`; observer drops and counts |
| malformed | Reject |

Two guard tests protect the easiest mistake in this area: `KNOWN_FIELDS` must
equal the dataclass's fields, and `to_dict()` must emit exactly `KNOWN_FIELDS`.
(That test immediately caught `KNOWN_FIELDS` itself leaking into the dataclass as
a field, which needed `ClassVar`.)

---

## 6. Fail-open validation

`validate_event` returns a `ValidationResult` and **never raises for data
problems**. 34 issue codes, each machine-assertable. `validate_event_or_raise`
exists separately and is documented as *not for the observer*.

Non-finite metric values are permitted only on `LOSS`, `GRADIENT_METRIC`,
`PARAMETER_METRIC`, `ACTIVATION_METRIC` — because `NAN_DETECTED` is a required
Phase 1 incident and rejecting NaN would reject the product requirement. On any
other event type a NaN is a producer defect.

Unit handling has a second purpose: a metric named `*_ms` declared in `SECONDS`
is flagged (`unit_name_mismatch`), because Phase 0 named a silent 1000× scale
error as a correlation hazard. `normalize_unit` converts to canonical scales but
deliberately leaves `PERCENT` alone — converting percent to a ratio would
silently rescale every published threshold by 100×.

---

## 7. Gate results

```
$ .venv/bin/python -m pytest tests/unit -q
98 passed
```

| Gate item | Status | Test |
|---|---|---|
| schema serialization | **PASS** | `test_schema_serialization.py` (13 tests) |
| schema deserialization | **PASS** | same, incl. non-finite round-trip |
| unknown field behavior | **PASS** | `test_schema_evolution.py` (3 modes + preservation across a second round-trip) |
| old schema read | **PASS** | minimal hand-written v0.1 payload loads; minor/major policy |
| idempotency | **PASS** | `test_validation.py` (derived key, stability, duplicate detection) |
| producer sequence | **PASS** | `test_validation.py` (monotonic, regression, duplicate, per-producer isolation) |
| partial ordering | **PASS** | `test_ordering.py` (17 tests, all three §8.4 cases) |
| invalid event rejection | **PASS** | `test_validation.py` (parametrised across 5 index codes + 6 field codes) |

---

## 8. Deviations from the brief, and why

| Brief | Actual | Reason |
|---|---|---|
| `schema/{event,enums,incident,evidence,validation}.py` | plus `ordering.py` | The brief puts `EventOrdering` in W0 §2.3 *and* lists `ordering.py` under `incidents/` in W6. It is one algebra, so it lives once, in `schema/`, and W6 imports it. |
| `first_abnormal_event_id` as a field | a **property** | A stored field can be populated inconsistently with `ordering_confidence`. A property makes the invariant impossible to violate. |
| — | `KNOWN_FIELDS` + two guard tests | Not requested; it is the highest-frequency schema-change bug. |

---

## 9. Not verified in W0

* **No cross-language round-trip.** Python only. A consumer in another language reading the JSONL would need its own `NaN`/`Infinity` handling — documented in `docs/SCHEMA_EVOLUTION.md` §7 rather than tested.
* **Multiple simultaneous producers.** The ordering algebra is tested with two producers and a conflict case, but not under real thread contention; that arrives with W1's concurrency tests and W2's adapter.
* **Storage round-trip.** Parquet is W3. W0 verifies dict and JSONL round-trips only.
