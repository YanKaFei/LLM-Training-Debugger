# L2 — PROTECTED BASELINE REPORT

**Status: COMPLETE.** `src/training_debugger/detectors/baseline.py`.

---

## 1. What changed

| | before (Phase 1.4–1.7) | after (Phase 1.8) |
|---|---|---|
| reference | trailing median, always updated | **freeze-on-suspicion**, state-aware |
| under a sustained fault | median rises with it, ratio → 1, **fault invisible** | reference frozen at the healthy level, ratio stays at the fault level |
| warm-up | classified against a thin baseline | returns `None` → `INSUFFICIENT_EVIDENCE` |
| after a false alarm | baseline had already absorbed the excursion | protected; resumes only after `resume_after_steps` quiet steps |
| state | none | `NORMAL` / `SUSPECT` / `CONFIRMED` / `RECOVERING` / `CLOSED` |

## 2. The mechanism

```
NORMAL      -> learns
SUSPECT     -> FREEZES; the candidate regime is never absorbed
CONFIRMED   -> stays frozen
RECOVERING  -> stays protected
CLOSED      -> resumes only after the recovery policy completes
```

Verified directly:

```
sustained x5 fault from step 50
  step  49  ratio 1.00
  step  50  ratio 5.00
  step  60  ratio 5.00
  step 100  ratio 5.00
  step 150  ratio 5.00
  step 199  ratio 5.00
```

The old feature decayed to 1.00 by step 299. **The fault is no longer erased by the
statistic meant to expose it**, which was Phase 1.7's entire finding.

## 3. Design decisions and why

**Freeze on suspicion, not confirmation.** `freeze_at_ratio = 1.3` is far below any
confirmation threshold. Its only job is to capture a *clean* reference early; by the
time a fault is confirmable, a rolling baseline has already absorbed it.

**`ratio()` returns `None` when not ready.** An unready reference means no opinion,
not "normal". Returning a number would let the aggregator classify during warm-up.

**Resume requires quiet steps.** A single step back to normal does not unfreeze the
baseline, so a transient excursion cannot permanently shift what "normal" means.

**Preserved unchanged:** per-layer normalisation (each layer against its own
reference) and joint aggregation (`l1_aggregate` as the mean of per-layer ratios,
layer-count-invariant since Phase 1.6).

## 4. Dual baseline (L2.5) — not adopted

The brief permits a short/long dual horizon *only if measurement shows a benefit*. No
dual-horizon variant was implemented or measured, so none is claimed. The
freeze-on-suspicion reference solves the identified failure mode on its own.

## 5. Performance (L2/perf requirement)

Unchanged: the baseline is computed **offline over an emitted series** in the
detection path. No observer or adapter change, **no new training-path
synchronisation, no new raw tensor capture** — the feature reads only the T1 per-layer
gradient norms the project already emits.

## 6. Not verified

* **`freeze_at_ratio = 1.3` and `resume_after_steps = 40` were not swept.** 1.3 was chosen as a level clearly below any confirmation threshold; 40 as roughly the observation horizon. Neither is measured.
* **A dual-horizon variant is untested** (§4).
* **Interaction with the loss and numerics detectors** is not evaluated — the protected baseline is used only by the joint gradient feature.
