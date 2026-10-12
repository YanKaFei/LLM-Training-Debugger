# SECURITY

## Scope

This project is an **experimental, local** training-observability tool. It runs inside your
own training process, writes to a local directory, and makes **no network calls**.

There is no server, no authentication layer, no multi-tenant surface and no remote
telemetry. Nothing is uploaded anywhere.

## Reporting a vulnerability

Open a private security advisory on the repository, or contact the maintainer directly.
Please do not open a public issue for a suspected vulnerability.

Include: affected version, a minimal reproduction, and the impact you believe it has.

## What the audit covers

The release audit checks the published artefacts for:

* credentials, API keys, tokens and passwords — **none found**
* hardcoded absolute paths in shipped source — **none found**
* accidental inclusion of virtual environments, build caches or local experiment output
* whether the sdist/wheel contain the resources needed to run

Two generated files (`tests/experiments/restart_worker.py`,
`tests/experiments/restart_continue.py`) previously embedded the local checkout path,
because `restart_smoke.py` writes them at runtime. They are now excluded from both
`.gitignore` and `MANIFEST.in`. **Path scanning alone is not treated as sufficient** — the
final check extracted the built sdist and searched its contents.

## Design properties that matter for safety

* **The observer never raises and never blocks training.** It drops rather than delaying.
* **Fail-open is bounded**: a bounded priority queue with exact accounting, and a circuit
  breaker for a dead writer.
* **Dropped evidence is visible.** A verdict whose completeness was not demonstrated
  carries a warning and is never rendered as an unqualified `NO_ANOMALY`.
* **No causal claim is representable.** `CAUSED_BY` and `ROOT_CAUSE_OF` are absent from
  the type system, asserted by test.
* **Ground truth is isolated.** Evaluation labels live in a separate tree and never reach
  a detector, the episode engine, fusion or the relation decision.

## Known unresolved reliability issue

An observer accounting race is **`UNRESOLVED`**. One full-suite failure was observed
historically and never reproduced (0 of ~70 attempts, plus deterministic forced-overflow
stress in which the accounting identity held exactly).

**Risk boundary:** because the observer is fail-open, a race would corrupt **counts**, not
training numerics — every fault-injection run left training numerically correct. It would
misreport completeness, which is surfaced rather than silent.

It is **not** described as fixed and is disclosed in [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

## Do not use for

**Do not use this tool as the sole basis for stopping, rolling back, or otherwise
automating a decision about a training run.** It is experimental, it abstains often by
design, and its detector reliability gates currently **FAIL**.
