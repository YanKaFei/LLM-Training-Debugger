# RELEASE CANDIDATE REPORT — source version v0.1.7-alpha

```
RELEASE CANDIDATE STATUS = READY
```

```
tests            533 passing, 1 skipped   (531 -> 533)
clean install    PASS  (wheel into a fresh venv, imports from site-packages)
package build    PASS  (wheel 229,858 B + sdist 189,785 B)
full pipeline    8/8 PASS
real training    PASS
CLI              PASS
persistence      PASS (cross-process)
relation safety  PASS
research         EXPERIMENTAL  (unchanged)
performance      FAIL  (unchanged — see below)
```

---

## R2 — the safety fix that matters most

`POSSIBLE_CONSEQUENCE_OF` is now **disabled by default**. v0.1.6/v0.1.7 measured that the
mechanism clause could not separate a genuine propagation from an independently injected
fault in every case that could be constructed — C3/C12 produced **16 consequence claims**
against `INDEPENDENT` ground truth.

```
production default  -> PRECEDES only (order)
research config     -> POSSIBLE_CONSEQUENCE_OF  (opt-in, unvalidated)
```

**No evidence is deleted.** Observations, episodes, order and revisions are all retained;
only the *claim* changes. Two tests pin the property: the default never asserts a possible
consequence, and the research configuration is opt-in and reversible on identical evidence.

The CLI therefore cannot present an unvalidated inference as a causal one.

## R3 — engineering audit

```
R3.1 end-to-end        8/8 smoke · quickstart · four verdicts · cross-process restart
R3.2 numerical         tensor finiteness vs norm-reduction overflow distinguished;
                       metric identity preserved; episode grouping scoped by metric;
                       candidate onset anchored to band-abnormal evidence;
                       ground truth isolated from production paths
R3.3 completeness      emitted/written/dropped consistent; an unreported completeness
                       yields a warning, never an unqualified NO_ANOMALY
R3.4 temporal          see below
```

### R3.4 — an off-by-one, and it was in the report, not the data

```
baseline_floor (warmup_steps) = 50, minimum_history = 20
first step with a usable ratio = 70
injection at 60 -> first scoreable at 70   => latency = 10 steps, INCLUSIVE
```

v0.1.7 described this as **11 steps**. That was a wording error, now corrected to **10**.
**No timestamp was altered to make the prose consistent** — the brief explicitly forbids
that, and the real event times are unchanged.

## R4 — performance

```
observer core overhead    FAIL — the < 3 % target is still missed at the largest workload
witness increment         +0.30 % total / +0.97 % p95   (NOT a fix for the above)
GPU benchmark             NOT VERIFIED — no CUDA device available
```

The witness's small increment is **not** presented as evidence that the observer meets its
budget. No monitoring intensity was reduced to make a measurement look better.

## R5/R6 — packaging

```
README.md              rewritten around what the tool is and is not
EVALUATION_REPORT.md   every measured number with its denominator
REPRODUCIBILITY.md     exact commands, evidence kinds kept separate
CAPABILITIES / KNOWN_LIMITATIONS / ARCHITECTURE / QUICKSTART / CONTRIBUTING / CHANGELOG
LICENSE                Apache-2.0
wheel + sdist          built and the wheel verified in a clean venv
```

Verified from the installed wheel, not the source tree:

```
imported from : /private/tmp/rcvenv/lib/python3.9/site-packages/training_debugger
self-contained: True      enabled detectors: 3      validated_capabilities: []
console script exit=0
```

## Blockers and limitations carried into the candidate

1. **Performance gate FAILS.** Observer overhead exceeds the < 3 % core target at the
   largest workload. Disclosed, not hidden.
2. **No blind holdout has ever been run.** All quality figures are development data.
3. **`validated_capabilities = []`.** Nothing is validated.
4. **Mild gradient faults are missed** below roughly ×12 — a measured boundary, not a bug.
5. **Healthy-run false incidents ~7.7 %** on the smoke configuration, above the 5 % gate.
6. **C3 remains unconstructible.** The gradient magnitude needed to fire the aggregate
   detector also destabilises the model, so an invoked-but-independent case cannot be held
   to shape. Root-caused; not a matter of more effort on the same axis.
7. **C4 is missed.** A real downstream overflow with finite parameters leaves no
   observable midstream trace.
8. **Observer accounting race `UNRESOLVED`** — mechanism identified, never reproduced.
9. **Historical 23 duplicates `HISTORICAL_23_NOT_FULLY_REPRODUCED`.**

## Artifact paths

```
/tmp/rcbuild/training_debugger-0.1.0-py3-none-any.whl
/tmp/rcbuild/training_debugger-0.1.0.tar.gz
outputs/rc/V0_1_7_RC_REPORT.md
outputs/v0.1.6-alpha/   outputs/v0.1.7-alpha/   outputs/alpha/   outputs/phase1-*/
```

The package version in `pyproject.toml` reads `0.1.0` while the source is developed as
`v0.1.7-alpha`. **Recorded as a discrepancy rather than silently bumped**, because the
sprint instruction was not to create a new version number.

## Reproduction

See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md). Every command there was executed.

## Publication

```
GITHUB PUBLICATION = NOT AUTHORIZED
```

Nothing pushed, nothing published, no release created, no final holdout consumed, no
Phase 2, no agent, optimizer witness still default-disabled.
