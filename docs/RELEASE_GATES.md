# RELEASE GATES

## Alpha (v0.1.0-alpha)

```
[x] Clean environment installation PASS
[x] Real PyTorch training example PASS
[x] Observer / event persistence PASS
[x] Episode cross-process recovery PASS
[x] Incident and relation persistence PASS
[x] Four-verdict CLI E2E PASS
[x] Persisted scoring execution PASS
[x] Evidence completeness surfaced
[x] Full pipeline smoke 8/8 PASS
[x] Relevant regression tests PASS
[x] Full test suite PASS
[x] No unexplained new test failures
[x] Capability registry honest
[x] No unsupported causal claims
[x] Documentation complete
[x] Alpha package installation PASS
[x] Alpha example execution PASS
[x] No final holdout consumed
[x] No Agent added
[x] No Phase 2 work introduced
```

With one standing disclosure: the observer accounting race is `UNRESOLVED`. It must be
disclosed with its risk boundary, the stress tests actually run, and what remains
unverified — or it blocks release if it could corrupt training or data.

## Detector reliability (UNCHANGED — Alpha does not relax these)

```
episode precision  >= 0.90        episode recall  >= 0.85
incident precision >= 0.90        incident recall >= 0.85
earliest evidence precision >= 0.90
healthy-run false incident rate <= 5%
pre-injection confirmed incidents = 0
```

Status: **FAIL** (last measured: episode recall 0.341, Phase 1.9).


## Release Candidate (source v0.1.7-alpha)

```
[x] Clean install PASS            (wheel -> fresh venv, imports from site-packages)
[x] Package build PASS            (wheel 229,858 B + sdist 189,785 B)
[x] Smoke 8/8 PASS
[x] Real training example PASS
[x] CLI four verdicts PASS
[x] Cross-process persistence PASS
[x] Evidence completeness PASS
[x] Relation presentation safety PASS   (POSSIBLE_CONSEQUENCE_OF disabled by default)
[x] Full regression suite PASS         (533 passing, 1 skipped)
[x] Documentation complete
[x] No unreported critical correctness defect
```

```
ENGINEERING_RELEASE_CANDIDATE = READY
RESEARCH_VALIDATION           = EXPERIMENTAL   (not upgraded)
PERFORMANCE_VALIDATION        = FAIL           (core observer overhead > 3 % target)
```
