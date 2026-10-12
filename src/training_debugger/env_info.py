"""Environment capture for reproducibility (Phase 1 brief section 13).

A performance number without its environment is not a measurement. Every
benchmark and report written by this project embeds :func:`capture_environment`
so that "median step time overhead 1.8%" is attributable to a specific machine,
torch build and commit.

Deliberately tolerant: every probe is optional and reports ``None`` rather than
raising, because a report must still be producible on a machine without a GPU.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["capture_environment", "git_commit_sha", "write_environment"]


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def git_commit_sha(repo_root: Optional[Path] = None) -> Optional[str]:
    """Current commit, or ``None`` outside a git checkout."""
    root = repo_root or Path(__file__).resolve().parents[3]

    def run() -> str:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip())
        return out.stdout.strip()

    return _safe(run)


def git_dirty(repo_root: Optional[Path] = None) -> Optional[bool]:
    """Whether the tree has uncommitted changes, or ``None`` outside a repo.

    Returning ``None`` rather than ``False`` for a non-repository matters: a
    report that says "clean tree" about a directory with no git history is
    claiming a verification it did not perform.
    """
    if git_commit_sha(repo_root) is None:
        return None
    root = repo_root or Path(__file__).resolve().parents[3]

    def run() -> bool:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip())
        return bool(out.stdout.strip())

    return _safe(run)


def _torch_info() -> Dict[str, Any]:
    import torch

    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
        "num_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def capture_environment(*, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Snapshot everything needed to interpret a performance number."""
    env: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _safe(lambda: platform.processor() or None),
        "cpu_count": os.cpu_count(),
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version()},
        "git_commit_sha": git_commit_sha(),
        "git_dirty": git_dirty(),
        "packages": {},
        "torch": None,
    }
    for module_name in ("numpy", "pyarrow", "duckdb"):
        def probe(name=module_name):
            module = __import__(name)
            return getattr(module, "__version__", "unknown")

        env["packages"][module_name] = _safe(probe)

    env["torch"] = _safe(_torch_info)

    if extra:
        env["extra"] = extra
    return env


def write_environment(path, *, extra: Optional[Dict[str, Any]] = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = capture_environment(extra=extra)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return target


if __name__ == "__main__":  # pragma: no cover - operator convenience
    print(json.dumps(capture_environment(), indent=2, sort_keys=True, default=str))
