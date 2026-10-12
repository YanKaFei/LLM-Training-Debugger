"""Report which committed evidence was produced by a different revision.

Provenance is only useful if something reads it. Every artifact now records the
`code_sha` that produced it; nothing was comparing that against the current revision,
so drift would still be discovered by accident or not at all.

    python tests/experiments/check_evidence_drift.py [--strict]

Two different failures, deliberately treated differently:

* **An artifact with no provenance at all** is an error. It cannot be checked, and the
  fix is to make its generator use `write_artifact`.
* **An artifact from an older revision** is a *report*, not an error. Evidence is
  regenerated in batches; between batches it is expected to lag the code. Failing on it
  would make the check something people disable.

`--strict` turns the second into a failure, for use when the corpora have just been
regenerated and a stale artifact would mean the regeneration missed one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]

#: Files that carry provenance but are not evidence: generated tooling output.
SKIP_DIRS = (".git", ".venv", "build", "dist", "node_modules", ".tools", ".venv312")

#: Artifacts that predate the provenance mechanism, listed explicitly.
#:
#: `write_artifact` was introduced after most of this repository's evidence already
#: existed, so a check that failed on any artifact without provenance would fail on 234
#: of them and be turned off within a day. An explicit ledger is the alternative: the
#: entries are real, visible, and can only ever be removed -- by regenerating the
#: artifact, at which point it gains provenance and drops off the list. Nothing may be
#: added to it.
LEDGER = ROOT / "outputs" / "PRE_PROVENANCE_ARTIFACTS.txt"


def _ledger() -> set:
    if not LEDGER.exists():
        return set()
    return {
        line.strip() for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


#: Paths that define "the code". An artifact is reproducible from its revision when
#: these are identical; evidence churn and documentation are not the code.
SOURCE_PATHS = ("src", "tests", "pyproject.toml")


def _source_differs(from_sha: str, to_sha: str) -> bool:
    """Whether the *code* changed between two revisions.

    Comparing revision strings is not the question. Regenerating evidence and
    committing it moves HEAD, so a freshly regenerated artifact is immediately
    "stale" by string comparison -- the check could never report itself satisfied,
    which makes it noise rather than signal.

    The question worth asking is whether the code that produced the evidence is the
    code that is here now. Commits that only touch `outputs/` or markdown do not
    change the answer.
    """
    if from_sha == to_sha:
        return False
    if not _looks_like_a_sha(from_sha):
        return True
    completed = subprocess.run(
        ["git", "diff", "--quiet", from_sha, to_sha, "--", *SOURCE_PATHS],
        cwd=str(ROOT), capture_output=True,
    )
    if completed.returncode not in (0, 1):
        return True                      # cannot tell -> assume it differs
    return completed.returncode == 1


def _head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                               capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _provenance_of(path: Path) -> Optional[Dict[str, Any]]:
    """Inline provenance, or a sibling `.provenance.json`, or nothing."""
    sibling = path.with_suffix(".provenance.json")
    if sibling.exists():
        try:
            return json.loads(sibling.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    if path.suffix != ".json":
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and _looks_like_a_sha(data.get("code_sha")):
        return {"code_sha": data.get("code_sha"), "code_dirty": data.get("code_dirty"),
                "generated_at": data.get("generated_at")}
    return None


def _looks_like_a_sha(value: Any) -> bool:
    """Whether a `code_sha` value is provenance rather than a hand-written field.

    `outputs/phase1-4/FINAL_HOLDOUT_LOCK.json` carries its own `code_sha` set to
    ``NOT COMPUTED`` -- a statement that the lock was deliberately not taken, not a
    record of what produced the file. Treating it as provenance made a hand-authored
    record look like a stale generated artifact.
    """
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return len(stripped) >= 7 and all(c in "0123456789abcdef" for c in stripped.lower())


def main(argv: List[str]) -> int:
    strict = "--strict" in argv
    list_unknown = "--list-unknown" in argv
    head = _head()

    artifacts: List[Path] = []
    for path in sorted(ROOT.glob("outputs/**/*.json")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name.endswith(".provenance.json"):
            continue
        if path.name.endswith("ground_truth.json"):
            continue                      # a keyed map: provenance is its sibling
        artifacts.append(path)

    known_old = _ledger()
    current: List[Path] = []
    stale: List[Tuple[Path, str]] = []
    unknown: List[Path] = []
    grandfathered: List[Path] = []

    for path in artifacts:
        note = _provenance_of(path)
        if note is None:
            rel = str(path.relative_to(ROOT))
            (grandfathered if rel in known_old else unknown).append(path)
        elif not _source_differs(str(note.get("code_sha")), head):
            current.append(path)
        else:
            stale.append((path, str(note.get("code_sha"))[:10]))

    if list_unknown:
        for path in unknown:
            print(path.relative_to(ROOT))
        return 0

    print("evidence drift check")
    print("  HEAD                        : {}".format(head[:10]))
    print("  artifacts with provenance   : {}".format(len(current) + len(stale)))
    print("  current                     : {}".format(len(current)))
    print("  from a DIFFERENT source rev : {}".format(len(stale)))
    print("  no provenance, grandfathered: {}".format(len(grandfathered)))
    print("  no provenance, UNEXPECTED   : {}".format(len(unknown)))
    print("  ledger entries not seen     : {}".format(
        len(known_old - {str(p.relative_to(ROOT)) for p in grandfathered})))
    print()
    if stale:
        print("  stale (regenerate when convenient):")
        for path, sha in stale[:25]:
            print("    {}  {}".format(sha, path.relative_to(ROOT)))
        if len(stale) > 25:
            print("    ... {} more".format(len(stale) - 25))
        print()
    if unknown:
        print("  NO PROVENANCE -- these cannot be checked:")
        for path in unknown[:25]:
            print("    {}".format(path.relative_to(ROOT)))
        if len(unknown) > 25:
            print("    ... {} more".format(len(unknown) - 25))
        print()

    orphans = known_old - {str(p.relative_to(ROOT)) for p in grandfathered}
    if orphans:
        print("  NOTE: {} ledger entr(ies) no longer exist -- likely regenerated. Remove "
              "them from {}:".format(len(orphans), LEDGER.relative_to(ROOT)))
        for rel in sorted(orphans)[:10]:
            print("    {}".format(rel))
        print()
    if unknown:
        print("FAIL: {} artifact(s) carry no provenance and are not in the ledger. "
              "Make their generator call write_artifact(), or add them to {} if they "
              "predate it.".format(len(unknown), LEDGER.relative_to(ROOT)))
        return 1
    if strict and stale:
        print("FAIL (--strict): {} artifact(s) predate HEAD.".format(len(stale)))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
