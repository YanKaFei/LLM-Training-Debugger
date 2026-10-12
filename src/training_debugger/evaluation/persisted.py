"""B1: score from persisted evidence, not from an in-memory result dict.

Phase 1.9–1.12C repeatedly scored the *in-memory* result dictionary produced by the run
script. That is not the same artefact a user can inspect later, and it is why the
persisted scorer kept being reported as "implemented but not executed".

This module reads the stores directly:

    EpisodeStore.episodes        -> episodes
    EpisodeStore.episode_relations -> relations
    IncidentStore.incidents      -> incidents
    run manifest / ground truth  -> evaluation-only labels

**Ground truth is read only here.** It never enters the detector, the episode engine,
fusion or the relation decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq

from ..persistence import EpisodeStore, IncidentStore

__all__ = ["load_persisted_runs", "build_scenarios_from_store", "persisted_run_ids"]


#: Artifacts that make a run directory a real persisted run.
_RUN_ARTIFACTS = ("episodes", "analysis_state", "incidents")


def persisted_run_ids(root: Any) -> List[str]:
    """Run ids that have **any** store artifact on disk.

    The criterion used to be the existence of ``episodes/episodes``. A healthy run
    that produced **zero episodes** therefore has no such directory and was silently
    excluded -- ``dev-healthy-L4d32-s80002`` was reported as unreadable when its
    analysis state was persisted and complete. A run with no episodes is a valid,
    informative run: it is the strongest evidence of correct abstention, and dropping
    it biases every healthy-run statistic.
    """
    base = Path(root) / "runs"
    if not base.exists():
        return []
    found = []
    for directory in sorted(base.iterdir()):
        if not directory.is_dir():
            continue
        if any((directory / name).exists() for name in _RUN_ARTIFACTS):
            found.append(directory.name)
    return found


def load_persisted_runs(root: Any, run_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Load episodes, relations and incidents for every run in the store."""
    root_path = Path(root)
    episode_store = EpisodeStore(root_path)
    incident_store = IncidentStore(root_path)
    loaded: List[Dict[str, Any]] = []

    for run_id in (run_ids if run_ids is not None else persisted_run_ids(root_path)):
        try:
            ids = _episode_ids(root_path, run_id)
            episodes = [episode_store.load(run_id, episode_id) for episode_id in ids]
        except Exception as exc:
            loaded.append({"run_id": run_id, "error": "{}: {}".format(type(exc).__name__, exc)})
            continue
        # A run with no episodes is valid, not broken: note it explicitly so the
        # distinction between "no anomaly" and "could not read" is never lost.
        zero_episodes = not episodes
        try:
            relations = episode_store.load_relations(run_id)
        except Exception as exc:
            relations = []
            loaded.append({"run_id": run_id, "error": "relations: {}".format(exc)})
            continue
        try:
            incidents = incident_store.list_incidents(run_id)
        except Exception:
            incidents = []
        loaded.append(
            {
                "run_id": run_id,
                "episodes": [e.to_dict() for e in episodes],
                "relations": [r.to_dict() for r in relations],
                "incidents": incidents,
                "analysis_state_revisions": _analysis_revisions(root_path, run_id),
                "zero_episodes": zero_episodes,
            }
        )
    return loaded


def _episode_ids(root: Path, run_id: str) -> List[str]:
    directory = root / "runs" / run_id / "episodes" / "episodes"
    ids = set()
    if not directory.exists():
        return []
    for path in sorted(directory.glob("*.parquet")):
        for row in pq.read_table(str(path)).to_pylist():
            ids.add(row["episode_id"])
    return sorted(ids)


def _analysis_revisions(root: Path, run_id: str) -> List[int]:
    directory = root / "runs" / run_id / "analysis_state" / "analysis_state"
    if not directory.exists():
        return []
    revisions = []
    for path in sorted(directory.glob("*.parquet")):
        for row in pq.read_table(str(path)).to_pylist():
            revisions.append(row["revision"])
    return sorted(revisions)


def build_scenarios_from_store(
    root: Any,
    ground_truth: Optional[Dict[str, Dict[str, Any]]] = None,
    run_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Turn persisted runs into scorer scenarios.

    ``ground_truth`` is keyed by run id and is **evaluation-only**. A run with no
    ground-truth entry is treated as a healthy control, which is the conservative
    choice: an unlabelled run cannot be credited as a true positive.
    """
    scenarios: List[Dict[str, Any]] = []
    for run in load_persisted_runs(root, run_ids):
        if run.get("error"):
            scenarios.append({"run_id": run["run_id"], "run_error": run["error"]})
            continue
        truth = (ground_truth or {}).get(run["run_id"], {})
        scenarios.append(
            {
                "run_id": run["run_id"],
                "injected": bool(truth.get("injected", False)),
                "fault_type": truth.get("fault_type"),
                "primary_fault_type": truth.get("primary_fault_type"),
                "injection_step": truth.get("injection_step"),
                "acceptable_window": truth.get("acceptable_window"),
                "allowed_secondary_manifestations": truth.get("allowed_secondary_manifestations"),
                # Pass the evaluation-only truth through under the key the scorer
                # reads. Without this the scorer falls back to
                # `primary_anomaly_type(fault_type)`, which cannot resolve a scenario
                # label like "grad_large" and silently scores every primary as a miss.
                "ground_truth": truth.get("ground_truth"),
                "episodes": run["episodes"],
                "relations": run["relations"],
                "incidents": run["incidents"],
                "verdicts": _verdicts_from_episodes(run["episodes"], run["relations"]),
                "source": "persisted",
                # Guard against a silent-zero scoreboard. If a fault run is marked
                # injected but carries no `acceptable_window`, `classify_observation`
                # cannot place the onset in the expected interval, so EVERY correct
                # detection is filed as UNRELATED_FALSE_POSITIVE and precision reads
                # 0.0 with no error. That happened twice during this work.
                "scoring_warning": (
                    "injected run has no acceptable_window: every true detection will "
                    "be classified UNRELATED_FALSE_POSITIVE and precision will read 0.0"
                    if truth.get("injected") and not truth.get("acceptable_window")
                    else None
                ),
                "analysis_state_revisions": run["analysis_state_revisions"],
                "zero_episodes": run.get("zero_episodes", False),
            }
        )
    return scenarios


def _verdicts_from_episodes(episodes, relations) -> List[Dict[str, Any]]:
    """Re-derive verdicts with the production fusion rules over persisted episodes.

    Fusion is deterministic given episodes and the frozen rules, so re-deriving is
    faithful; it is recorded as ``source: persisted`` because the *episodes* came from
    disk rather than from the run's memory.
    """
    from ..episodes.model import AnomalyEpisode
    from ..fusion import fuse

    rebuilt = [AnomalyEpisode.from_dict(e) for e in episodes]
    run_id = rebuilt[0].run_id if rebuilt else ""
    return [v.to_dict() for v in fuse(run_id, rebuilt)]
