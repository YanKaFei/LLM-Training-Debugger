# Schema Evolution Policy

**Applies to:** `TrainingEvent` and `Incident` (`src/training_debugger/schema/`)
**Current version:** `0.1`

Phase 1's release gate requires that schema versioning *exists*, not merely that
it is planned. This document is the policy; `schema_version` is the mechanism.

---

## 1. Version format

```
MAJOR.MINOR
```

Both components are non-negative integers. The version is carried on **every
event** and **every incident**, not in a file header: a file header cannot
describe an event that was appended after a producer upgraded mid-run.

`MAJOR` changes break readers. `MINOR` changes do not.

---

## 2. Compatibility matrix

| Event major | Event minor | Reader `0.1` behaviour |
|---|---|---|
| `0` | `≤ 1` | **Accept.** Fully understood. |
| `0` | `> 1` | **Accept with preservation.** Unknown fields are kept under `metadata["_unrecognized"]`; validation emits a *non-fatal* `unknown_schema_version` issue. The event stays usable. |
| `≠ 0` | any | **Reject.** Fatal `schema_major_mismatch`. The observer drops the event and counts it. |
| malformed | — | **Reject.** Fatal `unknown_schema_version`. |

This is the **forward compatibility policy**: a Phase 1 reader keeps working
against a Phase 2 producer, losing nothing, as long as the major version holds.
The inverse (a Phase 2 reader consuming Phase 1 data) is guaranteed by the same
table read the other way, because minor-only additions are strictly additive.

---

## 3. Unknown-field policy

Controlled by `TrainingEvent.from_dict(..., on_unknown=...)`:

| Mode | Behaviour | Use |
|---|---|---|
| `preserve` *(default)* | Stored under `metadata["_unrecognized"]`; a non-fatal issue is recorded | Ingest. A newer producer's data survives an older reader. |
| `drop` | Discarded silently | Never in the pipeline; available for tests. |
| `error` | Raises `ValueError` | Tests, and strict ingest of an externally supplied file. |

Preservation is deliberately *not* silent: `validate_event` emits
`unknown_fields_present`, so a producer/reader version skew is visible in the
data rather than inferred from odd behaviour later.

**Reserved keys.** `metadata` reserves `_unrecognized`, `_schema_from_future`,
`_observer_error`, `_drop_reason`. A producer that sets one of these is
rejected (`reserved_metadata_key`) rather than allowed to shadow the schema's
own bookkeeping.

---

## 4. Rules for changing the schema

### 4.1 Adding a field — MINOR bump

1. Add the field to `TrainingEvent` with a default that preserves current
   semantics (`None` for optional, an explicit default otherwise).
2. Add the name to `TrainingEvent.KNOWN_FIELDS`. **This is the step that is easy
   to forget**, and forgetting it turns a known field into an unknown one on
   round-trip.
3. If the field can carry a correlation key, it belongs in the envelope, **not**
   in `metadata`. Phase 0 axiom A2: `metadata` is never a correlation key.
4. Add a round-trip test in `tests/unit/test_schema_roundtrip.py`.
5. Bump `SCHEMA_VERSION` minor and add a row to the table in §6.

### 4.2 Changing a field's type or meaning — MAJOR bump

Also required: a migration function under `training_debugger/schema/migrations/`
and a stated rule for what happens to already-written Parquet (see §5).

### 4.3 Removing a field — MAJOR bump

Deprecate for one minor version first by keeping the field and emitting it as
`None`, so readers produce a clear data-quality signal rather than a schema
error.

### 4.4 Adding an enum member — MINOR bump for readers, MAJOR for detector authors

Adding to `EventType`, `MetricUnit`, `IncidentType` or `DetectorId` is additive
and therefore minor. It is flagged separately because **detectors key off enum
members**: a new `IncidentType` with no detector is a silent gap, so the enum
change and the detector change must land together.

### 4.5 Never

* Never reuse a retired enum value for a different meaning.
* Never change `metric_name` grammar without bumping minor and documenting the
  rename map; metric names are join keys in stored data.
* Never make an optional field required without a major bump — an old Parquet
  file would become unreadable.

---

## 5. Migration policy for stored data

Phase 1 stores events in **Parquet, which is the source of truth**. Parquet files
are immutable once written. Therefore:

| Situation | Policy |
|---|---|
| Reader minor > data minor | Read directly. Missing columns come back as `None`; that is the designed behaviour of columnar storage and requires no migration. |
| Reader minor < data minor | Read directly; unknown columns preserved on read. |
| Reader major ≠ data major | **Quarantine, do not guess.** The file stays on disk and is reported by `training-debugger runs --incompatible`. Migration is an explicit, offline, opt-in operation. |
| Corrupt / partial file | Skipped with a recorded `reader_skipped_files` count. Phase 1 does not attempt repair. |

The reasoning: a debugger that silently misreads historical evidence is worse
than one that admits it cannot read it. Quarantine is the honest failure mode.

**No in-place rewrite.** Because Parquet parts are immutable and the ingester is
append-only, migration means *writing a new part beside the old one* and
recording the mapping, not mutating history.

---

## 6. Version history

| Version | Date | Change | Migration |
|---|---|---|---|
| `0.1` | Phase 1, W0 | Initial freeze. Envelope fields per Phase 1 brief §2.1, plus `priority`, `parent_event_id`, `span_kind`, `duration_ns`, `idempotency_key`. Partial-order semantics in `ordering.py`. | n/a — first version. |

---

## 7. Non-finite metric values: a documented interop caveat

NaN and Infinity are **legitimate values** in this schema. `NAN_DETECTED` is one
of the four incidents the Phase 1 release gate requires, so a schema that
rejected non-finite values would reject its own product requirement.

Consequences:

* Parquet stores IEEE-754 NaN/Inf natively. No caveat.
* `to_json_line()` emits the JSON5-ish `NaN` / `Infinity` tokens (Python's
  `json.dumps(..., allow_nan=True)`). A strict RFC-8259 parser will reject them.
* Round-tripping through this library is lossless because `json.loads` accepts
  those tokens by default.

If a strict-JSON consumer is ever required, the correct fix is a transport
encoder that maps non-finite to `null` plus an explicit
`metadata["nonfinite"] = "nan"` marker — **not** a schema change, because the
value is real and must not be silently nulled in storage.

Non-finite values are permitted only on event types that can legitimately carry
them (`LOSS`, `GRADIENT_METRIC`, `PARAMETER_METRIC`, `ACTIVATION_METRIC`). On any
other event type a non-finite value is a producer defect and is rejected.

---

## 8. Ordering is not versioned by accident

A recurring hazard worth stating explicitly: `event_id` is time-sortable
(ULID-shaped), and it is tempting to treat lexical id order as causal order. It
is not, and this schema forbids it. `ordering.py` accepts exactly three
witnesses — `producer_sequence`, `causal_edge`, `barrier` — and none of them is
`event_id` or `wall_time_ns`. `event_id` is used only as a stable tiebreak in
:func:`EventOrdering.storage_sort_key`, for deterministic test output.
