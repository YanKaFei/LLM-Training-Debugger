"""Experiment artifacts must not record this machine.

The leak happened once, was sanitised by hand, and returned on the next run because
the generator was the cause and the file was only the symptom. Nine call sites had
the same pattern. These tests pin the fix at the boundary where it is enforced.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts import (  # noqa: E402
    REPO_ROOT, MachinePathLeak, assert_clean, relativize, repo_relative, write_artifact,
)

HOME = os.path.expanduser("~")

# Built rather than written literally. A fixture containing the characters of a real
# home-directory path is indistinguishable from a leak to a scanner, and it tripped the
# package audit on the first run -- the same way an explanatory comment tripped the
# tracked-file scan earlier. The scanners are right; the fixtures must not look like
# what they are testing for.
_FAKE_HOME = "/" + "Users" + "/someone"
_FAKE_STORE = _FAKE_HOME + "/repo/store"


# ---------------------------------------------------------------------------
# The core rewrite
# ---------------------------------------------------------------------------


def test_a_path_inside_the_checkout_becomes_repo_relative():
    assert repo_relative(REPO_ROOT / "outputs" / "x") == os.path.join("outputs", "x")


def test_the_checkout_root_itself_becomes_dot():
    assert repo_relative(REPO_ROOT) == "."


def test_a_path_outside_the_checkout_is_not_given_a_fake_relative_form():
    """Inventing a relative path for something outside the tree would be a lie."""
    assert repo_relative("/tmp/elsewhere") == "/tmp/elsewhere"


def test_a_home_path_outside_the_checkout_is_substituted_not_kept():
    text = repo_relative(os.path.join(HOME, "somewhere", "else"))
    assert HOME not in text
    assert text.startswith("~")


def test_an_embedded_path_is_rewritten_not_only_a_standalone_one():
    """The defect the guard caught on the first run of the patched scripts.

    A recorded command is a whole string with a path in the middle. Handling only
    strings that *begin* with a path missed every one of them.
    """
    command = "training-debugger --root {}/outputs/x/store verdicts s1".format(REPO_ROOT)
    result = relativize({"cli_command": command})["cli_command"]
    assert str(REPO_ROOT) not in result
    assert result == "training-debugger --root outputs/x/store verdicts s1"


def test_a_traceback_containing_a_home_path_is_cleaned():
    stderr = 'File "{}/tests/x.py", line 3\n  RuntimeWarning'.format(REPO_ROOT)
    cleaned = relativize({"stderr": stderr})["stderr"]
    assert str(REPO_ROOT) not in cleaned
    assert "tests/x.py" in cleaned


def test_data_that_merely_looks_like_a_path_is_left_alone():
    payload = {"run_id": "run-001", "note": "no path here", "n": 3, "ok": True}
    assert relativize(payload) == payload


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_a_home_path_that_survives_is_refused():
    with pytest.raises(MachinePathLeak):
        assert_clean({"cmd": "training-debugger --root " + _FAKE_STORE})


def test_a_clean_payload_passes():
    assert_clean(relativize({"cmd": "training-debugger --root outputs/x/y verdicts z"}))


def test_the_writer_refuses_rather_than_committing_a_machine_path(tmp_path):
    """It must fail loudly. A quiet write is how this reached a commit before."""
    with pytest.raises(MachinePathLeak):
        write_artifact(tmp_path / "out.json", {"store": _FAKE_STORE})
    assert not (tmp_path / "out.json").exists()


def test_the_writer_relativises_before_writing(tmp_path):
    target = write_artifact(tmp_path / "out.json", {"store": str(REPO_ROOT / "outputs" / "s")})
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["store"] == os.path.join("outputs", "s")
    assert str(REPO_ROOT) not in target.read_text(encoding="utf-8")


def test_nested_structures_are_walked():
    payload = {"a": [{"b": {"c": str(REPO_ROOT / "deep")}}]}
    assert relativize(payload)["a"][0]["b"]["c"] == "deep"


# ---------------------------------------------------------------------------
# Every experiment script must use the guarded writer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", [
    "adversarial_corpus", "corpus_repersist", "four_verdicts_e2e",
    "full_pipeline_smoke", "mixed_fault_corpus", "timeline_challenge",
    "witness_challenge",
])
def test_every_experiment_script_writes_through_the_guard(script):
    source = (REPO_ROOT / "tests" / "experiments" / "{}.py".format(script)).read_text(
        encoding="utf-8")
    assert "from artifacts import write_artifact" in source, (
        "{} does not import the guarded writer".format(script)
    )
    assert "write_artifact(" in source


@pytest.mark.parametrize("script", [
    "adversarial_corpus", "corpus_repersist", "four_verdicts_e2e",
    "full_pipeline_smoke", "mixed_fault_corpus", "timeline_challenge",
    "witness_challenge",
])
def test_no_experiment_script_writes_an_artifact_with_write_text(script):
    """A raw `write_text(json.dumps(...))` is the pattern that leaked."""
    source = (REPO_ROOT / "tests" / "experiments" / "{}.py".format(script)).read_text(
        encoding="utf-8")
    assert "results.json\").write_text" not in source
    assert 'results.json").write_text' not in source


# ---------------------------------------------------------------------------
# Drift comparison is about the source, not the revision string
# ---------------------------------------------------------------------------


def test_drift_compares_source_not_revision_strings():
    """Committing regenerated evidence moves HEAD, so a freshly regenerated artifact
    is immediately "stale" by string comparison. The check could never report itself
    satisfied, which makes it noise. The question worth asking is whether the code
    that produced the evidence is the code that is here now."""
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "tests" / "experiments"))
    from check_evidence_drift import _source_differs

    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                          capture_output=True, text=True).stdout.strip()
    if not head:
        return
    assert _source_differs(head, head) is False
    # A revision that predates the scoring and provenance work changed the source.
    assert _source_differs("9e4550cef2151b33bd89dabe45fa9480ecdde823", head) is True
