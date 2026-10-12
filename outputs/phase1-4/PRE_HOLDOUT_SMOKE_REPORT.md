# PRE-HOLDOUT SMOKE REPORT

**Status: NOT RUN — NOT VERIFIED.**

R10.2 requires a pre-holdout smoke corpus that verifies every detector executes,
episodes are generated, fusion executes, persistence works, the renderer works, and
every enabled incident type can fire — *before* freezing for the final holdout. Its
purpose is to prevent discovering a "pipeline does nothing" defect **inside** the
holdout, which is exactly what happened three times in Phase 1.3.

**The smoke corpus was not built and not run**, because the Phase 1.4 changes it
would have exercised (R2.4 uncertainty band, R4 consequence attribution, R5/R6/R7
detector work) were not implemented.

Consequently the R10.4 final holdout **was not started**. Running a holdout against a
code base with known-unimplemented work would have consumed it for no purpose, and
the brief is explicit that a holdout cannot be reused after a fix.

What *was* verified, by the test suite rather than by a smoke corpus:

| Smoke check | State |
|---|---|
| all enabled detectors execute | **PASS** — 337 tests, no detector errors |
| episodes are generated | **PASS** — `tests/unit/test_episode_fusion.py` |
| fusion executes | **PASS** |
| renderer works, including abstention | **PASS** — 6 renderer tests |
| disabled detectors stay disabled | **PASS** |
| persistence works (incidents) | **PASS** — Phase 1.1 tests still green |
| persistence works (episodes) | **FAIL** — not implemented |
| cross-process reconstruction | **PASS for episodes as dicts; NOT VERIFIED for the incident store** |
| every enabled incident type can fire | **NOT VERIFIED** |
| result files non-empty | **NOT VERIFIED** — no holdout run |
