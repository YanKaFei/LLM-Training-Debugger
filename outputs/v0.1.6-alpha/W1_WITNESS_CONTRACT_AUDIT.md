# W1 — WITNESS CONTRACT AUDIT

**Status: PASS.** The witness reports only what it observes, and `UNKNOWN` is never
rendered as `False`.

| Requirement | Verified |
|---|---|
| Hook confirms only the call boundary it observed | **yes** — `_on_pre` / `_on_post` set one flag, nothing more |
| `invoked` never equated with "parameters changed" | **yes** — `parameters_changed` is `None` on every record, always |
| `skipped` needs explicit upstream evidence | **yes** — only set from a caller-supplied `skip_reason` |
| No hook observed ≠ `skipped` | **yes** — it yields `UNKNOWN` |
| Multi-optimizer isolation | **yes** — each witness carries its own `optimizer_id` and records |
| Accumulation records micro-step and optimizer-step separately | **yes** — `micro_step_id` + `optimizer_step_id`, asserted different |
| Missing state stays `UNKNOWN` | **yes** — an optimizer without hooks attaches `False` and reports `UNKNOWN` |
| No parameter scan, no added sync | **yes** — nothing in the module reads parameter values |

Measured on the challenge corpus: the hooks fire on **every** `step()` call, including
ones with no gradients (C6), which is exactly why invocation alone cannot be treated as
an applied update. `OPTIMIZER_WITNESS` remains **default disabled**.
