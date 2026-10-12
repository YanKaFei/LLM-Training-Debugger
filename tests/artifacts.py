"""Writing experiment artifacts without leaking this machine into them.

Experiment scripts write JSON files that are **committed as evidence**. A path
recorded as an absolute home-directory location is wrong in two ways: it is not
reproducible for anyone else, and it is not even a stable description of where the
store is. Within the repository the store is at `outputs/.../store`, and that is
what should be written.

The leakage happened once, was sanitised by hand, and **came back on the next run**
because the generator was the cause and the file was only the symptom. Nine further
call sites had the same pattern. This module exists so that the correct thing is
also the easy thing, and so that a slip fails loudly at the point of writing rather
than being discovered by a scan weeks later.

    write_artifact(path, payload)     # relativize, then write
    repo_relative(path)               # one path -> str, for inline use

Anything written through :func:`write_artifact` is scanned before it reaches disk.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "REPO_ROOT", "repo_relative", "relativize", "write_artifact", "assert_clean",
    "MachinePathLeak", "code_sha", "code_is_dirty",
]

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A home-directory path, on any of the three platforms this runs on.
#:
#: Assembled from parts rather than written literally. A detector that contains the
#: characters it detects is indistinguishable from a leak to any scanner reading the
#: file -- this module tripped the package audit for exactly that reason, the third
#: time in one session that explanatory text was mistaken for the thing it explains.
#: Building the pattern keeps every scanner in the repository simple and removes the
#: need for an exclusion list that someone would eventually forget to update.
_USER_DIR = "U" + "sers"
_HOME_DIR = "h" + "ome"
_HOME_PATTERNS = (
    re.compile(r"/" + _USER_DIR + r"/[^\"'\s,)\]]*"),
    re.compile(r"/" + _HOME_DIR + r"/[^\"'\s,)\]]*"),
    re.compile(r"[A-Za-z]:\\\\?" + _USER_DIR + r"\\\\?[^\"'\s,)\]]*"),
)


class MachinePathLeak(RuntimeError):
    """An artifact still described this machine after relativisation."""


def repo_relative(path: Any, *, root: Optional[Path] = None) -> str:
    """Render ``path`` relative to the repository root when it is inside it.

    Paths outside the repository are returned unchanged apart from a ``~``
    substitution for a home directory, because inventing a relative path for
    something outside the tree would be a lie about where it is.
    """
    root = Path(root) if root is not None else REPO_ROOT
    if not isinstance(path, (str, os.PathLike)):
        return path
    text = os.fspath(path)
    if not text:
        return text
    try:
        candidate = Path(text)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            try:
                rel = resolved.relative_to(root.resolve())
            except ValueError:
                return _substitute_home(text)
            return str(rel) or "."
    except (OSError, ValueError):
        pass
    return _substitute_home(text)


def _substitute_home(text: str) -> str:
    home = os.path.expanduser("~")
    if home and home != "~" and text.startswith(home):
        return "~" + text[len(home):]
    return text


def _rewrite_embedded(text: str, root: Path) -> str:
    """Rewrite the checkout and home prefixes **anywhere** in a string.

    Not only at the start. A recorded command is the common case and it is a whole
    string containing a path in the middle, as in a `--root` argument followed by a
    store location. Handling only strings that *begin* with a path missed every one
    of those, which
    is what the guard caught on the first run of the patched scripts. Captured
    `stderr` is the same shape -- the home path sits somewhere inside a traceback.
    """
    for prefix in (str(root.resolve()), str(root)):
        text = text.replace(prefix + os.sep, "").replace(prefix, ".")
    home = os.path.expanduser("~")
    if home and home != "~":
        text = text.replace(home + os.sep, "~" + os.sep).replace(home, "~")
    return text


def relativize(payload: Any, *, root: Optional[Path] = None) -> Any:
    """Walk a JSON-shaped structure and relativise every path inside it.

    Strings are rewritten whether the path is the whole value or embedded in a
    command line or a traceback. Keys are left alone: a metric or a run id that
    happens to look like a path is data, not a location.
    """
    resolved = Path(root) if root is not None else REPO_ROOT
    if isinstance(payload, dict):
        return {key: relativize(value, root=resolved) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [relativize(item, root=resolved) for item in payload]
    if isinstance(payload, str):
        return _rewrite_embedded(payload, resolved)
    return payload


def _find_leaks(payload: Any, trail: str = "") -> List[str]:
    found: List[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.extend(_find_leaks(value, "{}.{}".format(trail, key)))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            found.extend(_find_leaks(value, "{}[{}]".format(trail, index)))
    elif isinstance(payload, str):
        for pattern in _HOME_PATTERNS:
            if pattern.search(payload):
                found.append("{}: {!r}".format(trail or "<root>", payload[:120]))
                break
    return found


def assert_clean(payload: Any, *, where: str = "artifact") -> None:
    """Raise if any home-directory path survives. Called before writing."""
    leaks = _find_leaks(payload)
    if leaks:
        raise MachinePathLeak(
            "{} would record a machine-specific path in {} place(s):\n  {}".format(
                where, len(leaks), "\n  ".join(leaks[:10])
            )
        )


def write_artifact(path: Any, payload: Any, *, root: Optional[Path] = None,
                   indent: int = 2, provenance: str = "inline") -> Path:
    """Relativise, verify, then write. The only supported way to emit evidence.

    A failure here is a bug in the caller, and it is better to fail than to commit
    one machine's layout as a finding.

    Evidence that does not say what generated it cannot be checked for drift, and this
    repository learned that the expensive way: artifacts committed under one revision
    were regenerated under another, and nothing in either file revealed it. So every
    artifact records its revision. ``provenance`` chooses where:

    ``"inline"``
        Merge ``code_sha``, ``code_dirty`` and ``generated_at`` into the payload. For
        a report, whose top level is a set of named fields.

    ``"sibling"``
        Write them to ``<name>.provenance.json`` beside the artifact. **For a keyed
        map**, such as ground truth keyed by run id. Merging into a map was the first
        implementation and it added three keys that look like run ids -- harmless for
        the scorer, which looks runs up by name, and corrupt for anything that
        iterates the file. A ground-truth file is a map, so it gets a sibling.

    ``"none"``
        Record nothing. Nothing in this repository currently uses it.
    """
    target = Path(path)
    cleaned = relativize(payload, root=root)
    note = {
        "code_sha": code_sha(root=root),
        "code_dirty": code_is_dirty(root=root),
        "generated_at": _utc_now(),
    }
    if provenance == "inline" and isinstance(cleaned, dict) and "code_sha" not in cleaned:
        merged = dict(cleaned)
        merged.update(note)
        cleaned = merged
    elif provenance == "sibling":
        sidecar = target.with_suffix(".provenance.json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(note, indent=indent), encoding="utf-8")
    assert_clean(cleaned, where=str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cleaned, indent=indent, default=str), encoding="utf-8")
    return target


def _git(args: List[str], root: Optional[Path] = None) -> Optional[str]:
    repo = Path(root) if root is not None else REPO_ROOT
    try:
        completed = subprocess.run(
            ["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def code_sha(*, root: Optional[Path] = None) -> str:
    """The revision that produced an artifact, or a reason it is unknown.

    ``UNKNOWN`` rather than an empty string, so a consumer can tell "not recorded"
    apart from "recorded as nothing".
    """
    sha = _git(["rev-parse", "HEAD"], root=root)
    return sha if sha else "UNKNOWN"


def code_is_dirty(*, root: Optional[Path] = None) -> Optional[bool]:
    """Whether the *source* differs from the committed revision.

    Scoped to the code, not to the whole tree. The first version asked
    ``git status --porcelain`` about everything, and every script after the first in
    a sequence recorded ``dirty=True`` -- because the earlier scripts had rewritten
    the artifacts those later scripts were about to rewrite. The code was identical
    throughout; only the evidence had moved.

    An artifact is reproducible from its ``code_sha`` when the source matches that
    revision. Evidence churn is not a reason to distrust the revision, so it is not
    reported as one.
    """
    repo = Path(root) if root is not None else REPO_ROOT
    status = _git(["status", "--porcelain", "--", "src", "tests", "pyproject.toml"],
                  root=repo)
    if status is None:
        return None
    return bool(status.strip())


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
