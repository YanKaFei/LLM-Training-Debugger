# The TrainingEvent schema

Frozen at `schema_version = "0.1"`. See `SCHEMA_EVOLUTION.md` for the versioning
policy; this file is the field reference.

## Four ordering concepts, four separate fields

| Concept | Fields | Establishes |
|---|---|---|
| logical training position | `step`, `microstep` | which step is described — **not** who ran first |
| producer-local order | `rank`, `source_seq` | a total order **within one emitting stream** |
| observation time | `wall_time_ns`, `monotonic_time_ns` | when it was observed — **never** a happens-before witness |
| storage order | `ingest_seq` | physical append order — **never** causal |

Conflating these is the mistake the schema exists to prevent.

## Envelope

Identity `schema_version`, `event_id`, `idempotency_key`, `run_id` · provenance
`producer`, `producer_version`, `event_type`, `source` · distributed `rank`,
`local_rank`, `world_size` · position `step`, `microstep` · ordering
`source_seq`, `ingest_seq`, `parent_event_id`, `span_kind`, `duration_ns` · time
`wall_time_ns`, `monotonic_time_ns` · model `model_id`, `module_path`,
`layer_index` · metric `metric_name`, `metric_value`, `metric_unit` · device
`device_type`, `device_id` · data `dataset_id`, `shard_id`, `batch_id` ·
classification `severity`, `confidence`, `priority` · trace `trace_id`, `span_id`
· text `message` · escape hatch `metadata`.

`None` means *absent, fill from context*; `""` means *explicitly wrong*. The
observer fills the former and validation rejects the latter, so an integration bug
surfaces instead of being healed.

## Invariants

* One metric name, one value, per step per layer. Two events sharing a name at one
  step is silent corruption; it happened once (see `W2_REPORT.md`) and is now
  regression-tested.
* Non-finite values are permitted only on `LOSS`, `GRADIENT_METRIC`,
  `PARAMETER_METRIC` and `ACTIVATION_METRIC`, because `NAN_DETECTED` is a product
  requirement.
* A non-finite *count* always has its own metric name
  (`grad.nonfinite_count`, `train.loss_nonfinite_count`), never the name of the
  measurement it qualifies.
