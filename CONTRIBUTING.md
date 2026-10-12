# CONTRIBUTING

## Ground rules

These are not style preferences; they are the rules this project has had to learn.

1. **Never mark something PASS that was not executed.** Not "the code exists", not "the
   test is written", not "the metric is estimated". If it was not run, write
   `NOT VERIFIED`.
2. **Never fix a failing test by weakening it.** If a test's assumption was wrong, fix
   the assumption and record why in the test's docstring. Deleting an assertion to reach
   green is the failure mode this repository has retracted three times.
3. **Never tune a threshold to reach a target.** Where distributions overlap, the correct
   output is `INSUFFICIENT_EVIDENCE`. Phase 1.7 spent 680 runs proving that no threshold
   separates a mild gradient fault from healthy variation.
4. **Never let a cooldown suppress evidence.** A detector cooldown may deduplicate
   *alerts*; it may not starve the episode layer, which needs every observation.
5. **Never report missing evidence as evidence of normality.** If completeness was not
   demonstrated, say so.
6. **Never claim causation.** Replay does not exist in this build.

## Development loop

```
reproduce -> isolate -> explain -> fix -> regression -> verify
```

If three fixes to the same code path fail, stop editing it. Re-examine the architectural
assumption, build a smaller reproduction, and audit the upstream data and downstream
contract. Do not try a fourth threshold.

## Running things

```bash
.venv/bin/python -m pytest tests -q                 # full suite
.venv/bin/python -m pytest tests -q -m "not slow"   # skip benchmarks
.venv/bin/python examples/quickstart.py             # end-to-end demo
.venv/bin/python tests/experiments/full_pipeline_smoke.py   # 8-scenario smoke
```

## Where things live

```
src/training_debugger/     library
tests/unit|integration|scenarios|fault_injection|performance|experiments
docs/                      design and state
outputs/                   per-phase evidence; historical, do not rewrite
```

`docs/DEVELOPMENT_STATE.md` is the project memory. **Update it as you work**, not at the
end. A new session should read it first and verify against the repository.

## Adding a detector

Do not. The backlog's standing instruction is: *do not add another detector while the
existing incident evidence chain remains unverified.* If you believe one is needed,
first show a measured failure the current set cannot express.
