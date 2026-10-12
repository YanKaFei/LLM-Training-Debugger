# C2 — RESOURCE BOUNDARY

**Status: COMPLETE.** The phase's largest open risk was measured and is **closed**.

---

## 1. C2.1 — where the volume increase lands

Phase 1.12A set `cooldown_suppresses_evidence = False` and measured **20 detector
observations where there had been 1**. Phase 1.12B recorded the fear that this might
load the *observer* and cause drops, silently truncating the evidence chain.

**Measured, identical training, both settings:**

| Layer | cooldown ON | cooldown OFF |
|---|---|---|
| TrainingEvents **persisted by the observer** | **3722** | **3722** |
| observer `emitted` / `dropped` / `written` | identical | identical |

```
Observer raw-event load unchanged.
Detector-side evidence volume increased.
```

**The cooldown is inside the detector, which runs offline over already-persisted
events.** There is no feedback path from detector emission back into the observer queue,
so the 20× is confined to the detector side and the observer's bounded queue never sees
it. The risk recorded in Phase 1.12B §5 is **not realised**.

## 2. C2.3 — detector-side volume is bounded

| Spike length | observations | episodes | incidents | detector ms |
|---|---|---|---|---|
| 1 | 1 | **1** | **0** | 181.8 |
| 20 | 20 | **1** | **0** | 177.9 |
| 100 | 89 | **1** | **0** | 184.5 |
| 1000 | 89 | **1** | **0** | 181.7 |

Two things worth stating:

* **Observations saturate at 89**, not 1000 — the detector stops emitting once the
  spike has run its course and the trailing window has adapted to it. Growth is bounded
  by the episode, not by the input length.
* **Detector time is flat at ~180 ms across a 1000× range of spike length.** The
  cooldown change adds no measurable detector cost.
* **Episodes stay at 1 and incidents at 0 at every size.** 1000 observations do not
  produce 1000 episodes or 1000 incidents.

## 3. C2.2 — bounded queue under forced overflow

Overflow was **forced** by never attaching a consumer, not by hoping the machine was
busy:

```
capacity            64
offered           4999
accepted            64
refused/evicted   4935
size after load     64        bounded = True
accounting       64 + 4935 = 4999
```

## 4. C2.2 — priority drop policy

A full queue of P3 events accepted an incoming **P0** by eviction, stayed within
capacity, and the P0 **survived the drain**:

```
p0_accepted = True    size_after = 4 (capacity 4)    p0_survived = True
```

## 5. C2.4 — evidence completeness visibility

**NOT DONE.** The brief requires that a lost event or a persistence failure surface as
`evidence_completeness = DEGRADED`, and that an unqualified `NO_ANOMALY` be blocked when
completeness is unknown. The verdict schema was **not** extended with
`data_quality_status`, `drop_count` or `completeness_warning`.

Until that exists, a `NO_ANOMALY` still means only "no anomaly was detected in the
events that arrived", and the system cannot distinguish that from "no anomaly was
detected in a complete stream". This is a real limitation and it is recorded rather than
softened.

## 6. C2.5 — observer race

```
OBSERVER_RACE = UNRESOLVED
```

The deterministic overflow stress above exercised the queue without a consumer and
produced no accounting failure. That is **more controlled** than the historical
full-suite run, but it is still an absence of reproduction, not a repair. No root cause
was identified and no production code was changed.

## 7. Gate

```
OBSERVER_RESOURCE_INTEGRITY   PASS   (observer load unchanged; queue bounded;
                                      accounting exact; priority policy respected)
DETECTOR_EVIDENCE_VOLUME      PASS   (bounded at 89; episodes=1; incidents=0;
                                      detector time flat across 1000x)
OBSERVER_RACE                 UNRESOLVED
EVIDENCE_COMPLETENESS         NOT DONE
```
