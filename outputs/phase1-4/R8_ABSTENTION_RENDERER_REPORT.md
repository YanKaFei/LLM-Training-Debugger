# R8 — ABSTENTION RENDERER REPORT

**Status: COMPLETE.** Abstention is now user-visible.

Phase 1.3 implemented abstention but kept it inside an object, so a reader could not
distinguish "nothing happened" from "we could not tell". `render_verdict()` in
`src/training_debugger/incidents/render.py` renders all four outcomes.

---

## 1. The four renderings

### `INSUFFICIENT_EVIDENCE`

```
INSUFFICIENT EVIDENCE

Observed:
  - GRADIENT_EXPLOSION at step 78, affecting 1/7 monitored layers

Missing:
  the corroboration needed to confirm the observation -- either a
  persistent recurrence, an aggregate-series departure, or a
  cross-layer multiplicity above the validated threshold.

Therefore no incident was opened.
Reason: 1 episode(s) remain unconfirmed with no corroboration

A candidate onset near step 78 was observed but is not defensible as an
incident onset.
```

### `CANDIDATE_ONLY`

```
No confirmed incident. One or more candidates were observed but did not meet
a promotion rule.

Observed:
  - GRADIENT_EXPLOSION at step 78, affecting 1/7 monitored layers
    onset at step 78, confirmed at step 80 (+2 steps)

Therefore no incident was opened.
Reason: 1 confirmed episode(s) but none met a promotion rule (best class: WEAK)
```

### `NO_ANOMALY`

```
NO ANOMALY

Observations were made and dismissed as not anomalous.

This is a positive statement that nothing was found, not an abstention.
```

### `CONFIRMED_INCIDENT`

```
CONFIRMED INCIDENT

  type:       GRADIENT_EXPLOSION
  severity:   warn
  confidence: 0.90
  promotion:  promote.strong.single (vphase1.3-1.0)

Evidence:
  - GRADIENT_EXPLOSION at step 78, affecting 7/7 monitored layers
    onset at step 78, confirmed at step 80 (+2 steps)

Earliest defensible onset: step 78
Confirmed at:              step 80
```

## 2. Requirements met

| Requirement | State |
|---|---|
| abstention not hidden in an internal object | **PASS** — `INSUFFICIENT_EVIDENCE` and `CANDIDATE_ONLY` render as first-class output |
| distinguish "nothing happened" from "could not tell" | **PASS** — `NO_ANOMALY` explicitly states it is a positive statement, not an abstention |
| show observed scope | **PASS** — `1/7 monitored layers` appears in the output |
| show both moments | **PASS** — onset and confirmation step are separate lines |
| no causal language | **PASS** — `test_renderer_never_uses_causal_language` runs `assert_causal_language_absent` over three verdict shapes |
| deterministic | **PASS** — `test_rendering_is_deterministic` |

## 3. A defect the renderer exposed

Writing `test_candidate_only_is_rendered_with_its_scope` revealed that
`CANDIDATE_ONLY` recorded **no episodes**, so the renderer had nothing to show. The
whole purpose of surfacing abstention was defeated by a missing field. `fuse()` now
records the confirmed-but-unpromoted episodes as `episodes_used`, and a test pins it.

## 4. Not done

* **The CLI does not call this renderer.** `training_debugger.cli.main diagnose` still renders incidents, not verdicts, so a user of the CLI cannot yet see an abstention. The renderer exists and is tested; it is not wired to the command line.
* **No abstention summary across a run** ("3 candidates, 1 confirmed, 2 abstained").
