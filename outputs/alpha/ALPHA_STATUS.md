# ALPHA STATUS — v0.1.0-alpha

```
ALPHA_RELEASE_READY = YES
```

Every gate below was **executed**. No value is estimated, and `NOT VERIFIED` marks
anything not run.

## Gate

| Gate | State | Evidence |
|---|---|---|
| Clean environment installation | **PASS** | fresh venv, `pip install -e .` exit 0; console script runs |
| Alpha package installation (non-editable) | **PASS** | `pip install .` into a second fresh venv; imports from site-packages, **not** the source tree; CLI runs from `/tmp` |
| Real PyTorch training example | **PASS** | `examples/quickstart.py`, real autograd, 130 steps, incident produced |
| Observer / event persistence | **PASS** | 62,002 events at 2,000 steps, 0 dropped, ~25 bytes/event |
| Episode cross-process recovery | **PASS** | S8, two interpreters, same episode, onset 80, baseline restored |
| Incident and relation persistence | **PASS** | S5; relations survive a store round trip |
| Four-verdict CLI E2E | **PASS** | 4 of 4, separate process, exit 0 |
| Persisted scoring execution | **PASS** | reads `episode_relations` from disk |
| Evidence completeness surfaced | **PASS** | forced 484-event overflow → `DEGRADED` + warning |
| **Full pipeline smoke 8/8** | **PASS** | 8 executed, 8 passed |
| Relevant regression tests | **PASS** | 473 passing |
| Full test suite | **PASS** | 473 passing, 1 skipped |
| No unexplained new test failures | **PASS** | — |
| Capability registry honest | **PASS** | `validated_capabilities = []` |
| No unsupported causal claims | **PASS** | asserted by test |
| Documentation complete | **PASS** | README, QUICKSTART, ARCHITECTURE, CAPABILITIES, KNOWN_LIMITATIONS, CONTRIBUTING, CHANGELOG, RELEASE_NOTES, LICENSE |
| No final holdout consumed | **PASS** | — |
| No Agent added | **PASS** | — |
| No Phase 2 work | **PASS** | — |

## Detector reliability — deliberately NOT satisfied

```
DETECTOR_RELIABILITY = FAIL
episode recall 0.457 vs 0.85 gate        (680-run corpus, this session)
healthy false-incident rate 1/13 = 0.077 vs 0.05 gate   (this session)
qualifying operating points 0
validated_capabilities = []
```

**`ALPHA_RELEASE_READY = YES` asserts installability and engineering correctness. It
asserts nothing about detection quality**, and the reliability gates remain FAIL.

## Standing disclosure — observer stability

`OBSERVER_RACE = UNRESOLVED`. Reproduced **0 of 35** attempts (30 isolated, 5
full-suite) plus deterministic forced-overflow stress that measured exact accounting
(capacity 64, accepted 64, evicted 4,935). **Risk boundary:** fail-open means a race
would corrupt *counts*, not training numerics. It would misreport completeness — which
is now surfaced rather than silent. **Not verified:** sustained multi-threaded load.
Not described as fixed.

## What changed this session

| Item | Result |
|---|---|
| `evidence_completeness` | implemented; `from_observer_health` bug fixed (unknown had read as COMPLETE) |
| Baseline resume path | **repaired** — `resume_after_steps` was dead code; measured cost recorded |
| S1 false positive | **re-diagnosed**: not a bug; a genuine 1-step transient. Rate measured 1/13 = 7.7 % |
| Smoke S1 assertion | corrected category error (n=1 asserting a statistical property) |
| D2 scale | 62,002 events, 0 drops, ~25 bytes/event, linear |
| D3 fail-open | **closed** — a forced overflow reaches the verdict as DEGRADED |
| A3/C2/C3 | idempotency, independent-NaN protection, baseline restoration — through the store |
| E5 install | verified editable **and** non-editable |
