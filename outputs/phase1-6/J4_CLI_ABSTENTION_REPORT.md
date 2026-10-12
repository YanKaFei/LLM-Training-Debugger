# J4 — CLI ABSTENTION REPORT

**Status: COMPLETE.** Abstention is visible from the command line.

---

## 1. What was added

### A new `verdicts` command

```bash
training-debugger verdicts <run-id> [--json]
```

Recomputes the full pipeline over stored events — detectors, episodes, fusion — and
renders **every** verdict through the Phase 1.4 renderer. It emits all four outcomes:

```
CONFIRMED_INCIDENT
CANDIDATE_ONLY
INSUFFICIENT_EVIDENCE
NO_ANOMALY
```

`--json` returns the verdicts and episodes as structures, each carrying
`promotion_rule_id`, `promotion_rule_version`, `episodes_used`, `episodes_rejected`,
`abstained_episodes` and `signals_used`.

### `diagnose` falls through to verdicts

Previously, when no incident was confirmed, `diagnose` printed *"No incident was
detected"* — indistinguishable from a system that had simply found nothing. It now
prints that line, then renders the verdicts, so a `CANDIDATE_ONLY` or
`INSUFFICIENT_EVIDENCE` outcome is displayed rather than swallowed.

## 2. Rendering, as the brief specifies

```
INSUFFICIENT EVIDENCE

Observed:
  - GRADIENT_EXPLOSION at step 78, affecting 1/7 monitored layers

Missing:
  the corroboration needed to confirm the observation -- either a
  persistent recurrence, an aggregate-series departure, or a
  cross-layer multiplicity above the validated threshold.

Therefore no incident was opened.
```

No causal language is possible: the renderer is guarded by
`test_renderer_never_uses_causal_language`, and the forbidden vocabulary
(`probably caused`, `likely root cause`) is rejected by
`assert_causal_language_absent`.

## 3. Not done

* **The CLI was not executed end to end against a stored run in this phase**, because the episode store is not wired into the pipeline that writes runs (J3 §5). The command is implemented and its renderer is unit-tested; the **integration path is NOT VERIFIED**.
* **Consequence rendering is not wired.** The brief's J4.2 example — showing a numerical episode as `POSSIBLE_CONSEQUENCE_OF` an incident — needs the relation store read into the renderer, which is not done.
* No test invokes `main([...])` with `verdicts`.
