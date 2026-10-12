# GITHUB EXPERIMENTAL PRE-RELEASE PREPARATION — v0.1.7-alpha

```
PUBLICATION_PREPARATION_STATUS: READY (package) / BLOCKED (repository)
```

> **Amended after a repository-level audit.** The *package* (wheel + sdist) is clean and
> publishable. The *repository* was not: the working directory holds ~1.9 GB of unrelated
> material including a personal contact database, and the draft release instructions said
> `git add -A`, which would have staged and pushed all of it. Both are now fixed. See §8.

```
SOURCE_VERSION:   v0.1.7-alpha
PACKAGE_VERSION:  0.1.7a1   (PEP 440)
GITHUB_TAG:       v0.1.7-alpha

TESTS:            533 passing, 1 skipped
INSTALLATION:     PASS  (wheel -> fresh venv, self-contained, CLI runs)
SMOKE:            8/8 PASS
QUICKSTART:       PASS

PACKAGE_AUDIT:    PASS
LICENSE_AUDIT:    PASS   (Apache-2.0, LICENSE in wheel metadata and sdist)
SECRET_AUDIT:     PASS   (no credentials, tokens or keys; no absolute paths)

PERFORMANCE_GATE: FAIL — disclosed
DETECTOR_RELIABILITY: EXPERIMENTAL
VALIDATED_CAPABILITIES: []
BLIND_HOLDOUT:    NOT EXECUTED

GITHUB_RELEASE_DRAFT: PREPARED
PUBLICATION:      NOT EXECUTED
OWNER_APPROVAL_REQUIRED: YES
```

---

## 1. Version identification — repaired

The RC sprint recorded a discrepancy: source `v0.1.7-alpha` against package `0.1.0`. Now
aligned, with an explicit mapping:

```
pyproject.toml        version    = "0.1.7a1"
__init__.py           __version__ = "0.1.7a1"
__init__.py           GITHUB_TAG  = "v0.1.7-alpha"
wheel METADATA        Version: 0.1.7a1 · License: Apache-2.0 · Requires-Python: >=3.9
```

Verified from the **installed wheel**, not the source tree:

```
version 0.1.7a1 / v0.1.7-alpha
```

## 2. Publication safety audit

| Check | Result |
|---|---|
| Credentials, API keys, tokens, passwords | **none found** (every `token` hit is tokenizer/token-count code) |
| Hardcoded absolute paths in shipped source | **none** |
| Virtual environments / build caches / local output in the artefact | **none** |
| Local paths anywhere in the built sdist | **none** (180 files extracted and searched) |
| LICENSE present and declared | **yes** — Apache-2.0, in wheel metadata and sdist |
| sdist/wheel contain the resources needed to run | **yes** |
| Test and example references resolve | **yes** |

**Two real leaks were found and fixed** — this is why string scanning alone was not
treated as sufficient:

1. `tests/experiments/restart_worker.py` and `restart_continue.py` are **generated at
   runtime** by `restart_smoke.py` and embedded the local checkout path
   (`sys.path.insert(0, '<home>/...')`). They were removed and excluded from both
   `.gitignore` and `MANIFEST.in`. Deleting them is safe: `restart_smoke.py` regenerates
   them on every run.
2. `src/training_debugger.egg-info/` — a regenerable editable-install cache. Removed
   (path verified before deletion; it is not source and not data).

`src/training_debugger.egg-info` still appears in the sdist because **setuptools
regenerates it during every build** and includes the metadata. This is standard sdist
behaviour, contains only `PKG-INFO`/`SOURCES.txt`/`requires.txt`/`entry_points.txt`, and
the final audit confirms **zero local paths** inside it.

## 3. README

Rewritten around the required positioning — *Experimental PyTorch Training Evidence
Monitor* — covering all nine required points, with the five mandatory disclosures present
verbatim (verified by grep, not by assertion).

## 4. Release materials

```
RELEASE_NOTES.md   complete, GitHub-ready
CHANGELOG.md       present
KNOWN_LIMITATIONS.md  present
SECURITY.md        new — scope, reporting, audit coverage, the unresolved race
REPRODUCIBILITY.md present
outputs/rc/GITHUB_PRERELEASE_DRAFT.md   draft body + owner commands, NOT executed
```

## 5. Final independent install test — re-executed on the NEW build

```
wheel built        training_debugger-0.1.7a1-py3-none-any.whl   229,987 B
sdist built        training_debugger-0.1.7a1.tar.gz             386,470 B
fresh venv         /tmp/rcvenv3  (created for this run)
install            Successfully installed training-debugger-0.1.7a1
import             /private/tmp/rcvenv3/lib/python3.9/site-packages/training_debugger
self-contained     True
CLI                exit 0
mechanism linker   default False   (relation safety holds in the shipped artefact)
```

No result from the previous RC build was reused.

## 6. Repository review

`.gitignore` did **not** exist and has been created (build artefacts, virtual
environments, local experiment output, training data, credentials, editor files).

**This workspace is not a git repository** (`git init` has never been run). "GitHub
pre-release" therefore presupposes initialising one — an owner action, not an autonomous
one. **No history was rewritten, no remote was added, no permission was changed.**

## 7. Blockers carried into the pre-release

1. **Performance gate FAILS** — observer overhead exceeds the 3 % core target at the
   largest workload. Disclosed in README, RELEASE_NOTES and KNOWN_LIMITATIONS.
2. **No blind holdout has ever been run.**
3. **`validated_capabilities = []`.**
4. Mild gradient faults missed below ~×12.
5. Healthy-run false incidents ≈ 7.7 % (gate ≤ 5 %).
6. Observer accounting race `UNRESOLVED`.
7. Historical 23 duplicates `HISTORICAL_23_NOT_FULLY_REPRODUCED`.
8. **Dependency installation in this environment is network-bound and slow**; the runtime
   dependencies (numpy, pyarrow, duckdb) were verified installable in the RC sprint, and
   the fresh-venv test here used `--no-deps` to isolate the package itself. Recorded
   rather than presented as a full end-to-end dependency test.

## Artifacts

```
/tmp/rc5/training_debugger-0.1.7a1-py3-none-any.whl
/tmp/rc5/training_debugger-0.1.7a1.tar.gz
outputs/rc/GITHUB_PRERELEASE_DRAFT.md
outputs/rc/V0_1_7_PRERELEASE_REPORT.md
```

## Publication

```
NOT EXECUTED
```

No `git push`, no `gh release create`, no `twine upload`, no new repository, no permission
change, no final holdout, no Phase 2, no new detector.

**Owner approval required before publication.**


---

# 8. REPOSITORY-LEVEL AUDIT — added after the file was first written

The original audit was scoped to the **built artefact**. A later pass over the **working
directory** found a defect in my own deliverable.

## What was wrong

```
.pnpm-store/               415 MB   pnpm package cache, unrelated to this project
ascii-cinema/              1.1 GB   unrelated project
puppet-global/             208 MB   unrelated project
research/                  149 MB   upstream clones + local absolute paths
ARTEX/, ascii-h3-director/  19 MB   unrelated
puppetry-contacts.json      26 KB   PERSONAL contact database (names, emails)
proxy-*  pac-*  *.sh                proxy / VPN configuration
                                    ───────
                            ~1.9 GB
```

**And `outputs/rc/GITHUB_PRERELEASE_DRAFT.md` instructed the owner to run:**

```bash
git init && git add -A && git commit … && git push origin main --follow-tags
```

That would have committed and pushed **every one of those files**, including the personal
contact database. The instruction was wrong, and it was the kind of wrong that a human
would follow without suspicion.

## What was fixed

1. **`.gitignore` rewritten.** It now ignores the unrelated directories, the personal
   files, the proxy/VPN scripts and the pnpm store, in addition to build artefacts,
   environments and generated experiment output. **Verified with real `git check-ignore`**,
   not by reading the file.
2. **The draft instructions replaced** with an explicit allow-list (`git add <named
   paths>`), a `git status --short` verification step, and a warning explaining why
   `git add -A` must not be used here.
3. **A recommendation added**: move the project into its own directory. An ignore rule is a
   guard; a dedicated checkout is safe by construction.

## Verified after the fix

```
.pnpm-store/  puppet-global/  ascii-cinema/  research/
puppetry-contacts.json  proxy-mode.sh            -> IGNORED
src/training_debugger/registry.py
outputs/**/*.md (the audit trail)                -> would be staged
```

## Consequence for the release

```
package     READY   — wheel + sdist are clean; 180 files searched, 0 local paths
repository  READY   — ONLY IF the allow-list form is used or the project is moved
```

**Owner approval required. Do not run `git add -A` in this directory.**
