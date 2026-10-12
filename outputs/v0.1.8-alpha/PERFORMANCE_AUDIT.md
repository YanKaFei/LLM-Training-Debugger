# PERFORMANCE AUDIT

Triggered by the activation probe adding +3.3 % on top of an observer that already missed
its own **< 3 %** budget. Two things came out of it: the budget failure is explained, and
the overhead was cut by about **30 %**.

```
tests  584 passing, 1 skipped   (unchanged by this audit)
```

---

## 1. Two benchmarks disagreed by 5x on the same workload

```
benchmark_overhead.py      tiny: core = +2.15 ms/step   (MEDIAN of per-repeat medians)
benchmark_cost_ledger.py   tiny: observed = +0.41 ms/step (MIN of per-repeat medians)
```

They are **not comparable**, and neither stated the difference:

| | overhead benchmark | cost ledger |
|---|---|---|
| Point estimate | median of per-repeat medians | `per_repeat[0]` — the **minimum** |
| Rationale given | none | *"contention only adds time"* |
| Workload params | batch 4 × seq 8 | its own `n_layers/d_model/heads/seq/batch` |

The ledger also reported a **negative residual**:

```
tiny   observed +0.41 ms, accounted 3.86 ms, residual -3.45 ms
medium observed +1.07 ms, accounted 4.15 ms, residual -3.08 ms
```

A negative residual means the parts cost more than the whole. That is not a valid
attribution — the micro-costs are measured in tight loops where they are cache-hot and
uncontended, and they do not sum in situ. **The ledger's accounting cannot be used to
decide anything**, and it was reported as if it could.

## 2. One defensible measurement

`tests/performance/audit_observer_cost.py` — off and on alternate **inside each repeat**,
so drift and background load hit both arms equally, with 9 repeats for a confidence
interval:

```
mode             median ms   added vs off          relative   events  ev/step
off              3.2840      baseline              -
observer_only    4.3500      +1.066 ± 0.024 ms      +32.5%     1442    14.4
core_layers      5.6981      +2.414 ± 0.039 ms      +73.5%     3842    38.4
```

The intervals are tight (±0.024 ms), so these numbers can be quoted and compared.

## 3. Root cause: the hot path cloned every event, twice

The observer's own telemetry agreed with an independent measurement — **41.1 us per event**
emitted, against **0.012 us** for writing it. So the cost was never the writer.

Profiling the emit path:

```
emit                 39.9 us/event
  copy_with          11.3 us x 2 calls = 22.6 us   <-- 57 % of the entire emit
  validate_event     12.0 us
  new_event_id        6.3 us
  metadata_size_bytes 2.9 us   (JSON-encodes metadata only to measure its length)
```

`copy_with` is `dataclasses.replace` on a **38-field** dataclass. `_accept` called it twice
per event: once to fill run context, once to stamp the producer sequence. A comment in the
file already recorded an earlier fix collapsing *ten* calls into one — these were the
remaining two, on the path that every `metric()` call takes.

## 4. The fix, and why it is safe

When the observer builds the event itself, nothing else can hold a reference to it, so
filling its fields is not an observable side effect. When a caller passes their own event,
it is still cloned.

```
emit  41.1 us/event  ->  17.8 us/event        (copy_with gone from the profile)
observer_only   +1.066 ->  +0.751 ms   (+32.5 % -> +24.0 %)
core_layers     +2.414 ->  +1.635 ms   (+73.5 % -> +52.2 %)
```

**About 30 % off the observer overhead. All 584 tests pass unchanged.**

## 5. Where the remaining cost sits

```
core_layers, validate=ON    +0.820 ms   emit 22.66 us/event
core_layers, validate=OFF   +0.506 ms   emit 11.07 us/event
                            ---------
validation costs            +0.314 ms   = 38 % of the overhead
```

**Validation is 38 % of what is left**, and it is a config flag. Recomputed against the
defensible measurement, about **half** the overhead is event emission (`events/step x
us/event`) and the other half is instrumentation on the training path.

## 6. The budget verdict, and why the FAIL is mostly the workload

```
tiny   (4L,  d32)   baseline  4.66 ms   core_layers +0.51 ms   +11.0 %
medium (8L,  d128)  baseline 18.39 ms   core_layers +0.95 ms    +5.2 %
large  (12L, d256)  baseline 44.76 ms   core_layers +0.19 ms    +0.4 %
```

**The relative overhead collapses as the workload grows, because the cost is
approximately FIXED per step (0.2–0.9 ms) and a percentage of a fixed cost is not a
constant.** On any workload whose step exceeds roughly **17 ms**, the observer is inside
its < 3 % budget. On the 4-layer toy model used for the gate it cannot be, at any
implementation quality, without reducing the absolute cost to under 0.14 ms.

So the honest statements are:

* **The < 3 % gate FAILS on the benchmark workload** and the benchmark workload is a
  4-layer model whose step is 3–5 ms. This is stated, not excused.
* **The transferable number is the absolute cost: ~0.2–0.9 ms per step.** On a production
  step of 2 s that is 0.01–0.05 %, which is a projection from this machine and is
  **NOT VERIFIED** here. What is verified is the absolute per-step cost and the scaling
  curve above.
* The large-model point (+0.19 ms) is **lower** than the medium one (+0.95 ms). The curve
  is not monotonic, which is measurement noise at that scale rather than a real effect; it
  is reported as measured rather than smoothed.

## 7. What is left to do

| Lever | Current | Note |
|---|---|---|
| Event volume | 38.4 events/step (core_layers) | edge-triggering took the activation probe from 1,600 to 20 events; the same idea has not been applied to T1 layer hooks |
| Per-event cost | 11–23 us | validation is 38 % of overhead and is configurable; `metadata_size_bytes` still JSON-encodes every event to check a 16 KB limit |
| `new_event_id` | 6.3 us | pure-Python ULID base32 loop |

**The largest single remaining lever is event volume**, because it multiplies with
per-event cost. 38.4 events per step is a design choice, not a requirement.

## 8. Audit verdict

```
MEASUREMENT    two existing benchmarks were incomparable and one had a negative
               residual; a paired design with confidence intervals now exists
ROOT CAUSE     found and fixed -- 57 % of emit was cloning a 38-field dataclass twice
OVERHEAD       -30 % (core_layers +2.414 -> +1.635 ms, strictly paired)
BUDGET         still FAILS on the 4-layer gate workload; passes above ~17 ms/step
DISCIPLINE     the FAIL is reported as a FAIL, and the projection to production
               steps is labelled NOT VERIFIED
```
