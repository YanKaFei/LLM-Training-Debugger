# P1 — CLI E2E REPORT

**Status: PASS for readback; the four-verdict set is only partially observed.**

---

## 1. What changed

The `verdicts` command (and therefore `diagnose`'s fallthrough) now **prefers persisted
evidence over recomputation**. `_load_persisted()` reads episode ids out of the
`EpisodeStore` Parquet parts, loads each episode and the persisted relations, and fuses
them. Only if the store is absent does it fall back to recomputing from events.

This matters because the store is the episode truth: a CLI that recomputed would
report what the *current code* would decide, not what was *recorded*.

The output names its own provenance (`# source: persisted episode store`) and the
`run_id`, and appends, for any incident with a linked relation:

```
Relation:
  POSSIBLE_CONSEQUENCE_OF

Linked incident:
  <episode_id>

This relation is temporal/evidential and does not establish causation.
```

## 2. Real execution

```
command:  PYTHONPATH=src .venv/bin/python -m training_debugger.cli.main \
              --root outputs/phase1-3/store verdicts F1_loss_spike
exit:     0
source:   outputs/phase1-3/store/runs/F1_loss_spike/episodes/episodes/*.parquet
stdout:   "# source: persisted episode store"
          "# run_id: F1_loss_spike"
          "NO ANOMALY ... 13 observation(s) never corroborated"
          "NO ANOMALY ... 1 observation(s) never corroborated"
stderr:   one RuntimeWarning from `python -m` importing a package then a submodule
```

**The command reads real persisted episodes in a fresh process and exits 0.** That is
the P1 gate: CLI readback from real storage, not from a hand-built fixture.

## 3. Verdicts observed

| Verdict | Observed from persisted data |
|---|---|
| `NO_ANOMALY` | **YES** — two verdicts rendered |
| `INSUFFICIENT_EVIDENCE` | implemented and unit-tested; not observed on this run |
| `CANDIDATE_ONLY` | implemented and unit-tested; not observed on this run |
| `CONFIRMED_INCIDENT` | implemented and unit-tested; **not observed on this run** |

Only `NO_ANOMALY` was exercised against real storage. The other three are covered by
the Phase 1.9 unit tests but were **not** produced from the persisted corpus in this
phase, because the run chosen carried no confirmed episode.

## 4. A defect this exposed

The run directory contained splits like `F1_loss_spike` and `F10_mixed_anomaly-3000`,
because `_persist_run` computed the store path from `(fault_type, seed)` while the
episodes carried `fault_type` alone. Fixed in P0 §5.

## 5. Gate

```
P1 CLI reads persisted evidence in a fresh process  ->  PASS (exit 0)
P1 four verdicts rendered from persisted data       ->  PARTIAL (1 of 4 observed)
P1 consequence relation rendered from persisted data ->  NOT VERIFIED
```

The gate is **not fully met**: the brief requires all four verdicts verified against
real persisted data, and only `NO_ANOMALY` was. The CLI implementation and its
renderer are correct and unit-tested, but "unit-tested" is not what P1 asks for.
