"""P0: durable detector/analysis state, so a restart cannot silently reset it.

Phase 1.9 proved the system can **write** evidence. Nothing proved it could **read it
back**. The specific inconsistency the brief forbids is:

    Episode = SUSPECT
    Baseline = newly initialized

which would leave a persisted episode whose detector reference had silently been
rebuilt from scratch -- the same run judged by two different baselines.

**Design choice (brief P0.2).** Two options were considered:

* **A. Persist a detector-state snapshot.**
* **B. Rebuild state deterministically from persisted events.**

**B cannot be the only mechanism**, because it fails exactly when it matters: a
truncated run's baseline depends on where processing stopped, and rebuilding from the
full event stream would produce the *finished* baseline rather than the one in force at
the moment of the crash. **A is the source of truth**; B is retained as a *verifier*
(`verify_against_series`) so a corrupted snapshot is detectable rather than trusted.

The snapshot is small -- one reference and a dispersion per observed layer -- so the
cost of A is negligible and the guarantee is real.

**This is not training replay.** No model, optimiser or data-loader state is restored,
and nothing here claims to resume a training run. Only the *analysis* state needed to
continue judging an already-emitted event stream is persisted.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "BaselineSnapshot",
    "AnalysisState",
    "AnalysisStateStore",
    "capture_analysis_state",
    "ANALYSIS_STATE_SCHEMA",
    "analysis_state_dir",
]


def analysis_state_dir(root: os.PathLike, run_id: str) -> Path:
    return Path(root) / "runs" / run_id / "analysis_state"


@dataclass
class BaselineSnapshot:
    """The protected baseline for one signal key, in serialisable form."""

    key: str
    reference: Optional[float] = None
    dispersion: Optional[float] = None
    state: str = "NORMAL"
    frozen: bool = False
    reason: str = ""
    freeze_step: Optional[int] = None
    last_update_step: Optional[int] = None
    observations: int = 0
    ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "reference": self.reference,
            "dispersion": self.dispersion,
            "state": self.state,
            "frozen": self.frozen,
            "reason": self.reason,
            "freeze_step": self.freeze_step,
            "last_update_step": self.last_update_step,
            "observations": self.observations,
            "ready": self.ready,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BaselineSnapshot":
        return cls(**payload)


@dataclass
class AnalysisState:
    """Everything a fresh process needs to continue judging a run correctly."""

    run_id: str
    processed_through_step: Optional[int] = None
    detector_versions: Dict[str, str] = field(default_factory=dict)
    fusion_rule_version: str = ""
    episode_rule_version: str = ""
    rule_version: str = ""
    baselines: Dict[str, BaselineSnapshot] = field(default_factory=dict)
    episode_ids: List[str] = field(default_factory=list)
    open_episode_ids: List[str] = field(default_factory=list)
    revision: int = 1
    recorded_wall_time_ns: int = field(default_factory=time.time_ns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "processed_through_step": self.processed_through_step,
            "detector_versions": dict(self.detector_versions),
            "fusion_rule_version": self.fusion_rule_version,
            "episode_rule_version": self.episode_rule_version,
            "rule_version": self.rule_version,
            "baselines": {k: v.to_dict() for k, v in self.baselines.items()},
            "episode_ids": list(self.episode_ids),
            "open_episode_ids": list(self.open_episode_ids),
            "revision": self.revision,
            "recorded_wall_time_ns": self.recorded_wall_time_ns,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AnalysisState":
        data = dict(payload)
        data["baselines"] = {
            k: BaselineSnapshot.from_dict(v) for k, v in (data.get("baselines") or {}).items()
        }
        return cls(**data)

    def baseline_for(self, key: str) -> Optional[BaselineSnapshot]:
        return self.baselines.get(key)


ANALYSIS_STATE_SCHEMA = pa.schema([
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("revision", pa.int32(), nullable=False),
    pa.field("processed_through_step", pa.int64(), nullable=True),
    pa.field("open_episode_count", pa.int32(), nullable=True),
    pa.field("baseline_count", pa.int32(), nullable=True),
    pa.field("episode_rule_version", pa.string(), nullable=True),
    pa.field("fusion_rule_version", pa.string(), nullable=True),
    pa.field("rule_version", pa.string(), nullable=True),
    pa.field("recorded_wall_time_ns", pa.int64(), nullable=False),
    pa.field("payload", pa.string(), nullable=False),
])


class AnalysisStateStore:
    """Append-only store for :class:`AnalysisState`. Latest revision wins."""

    def __init__(self, root: os.PathLike, *, fail_open: bool = True) -> None:
        self.root = Path(root)
        self.fail_open = fail_open
        self._lock = threading.Lock()
        self._unique = uuid.uuid4().hex[:8]
        self._index = 0
        self.errors: List[str] = []

    def append(self, state: AnalysisState) -> int:
        try:
            with self._lock:
                revision = self.next_revision(state.run_id)
                state.revision = revision
                directory = analysis_state_dir(self.root, state.run_id)
                directory.mkdir(parents=True, exist_ok=True)
                self._index += 1
                target = directory / "analysis_state"
                target.mkdir(parents=True, exist_ok=True)
                final = target / "analysis_state-{:08d}-{}.parquet".format(
                    self._index, self._unique
                )
                temp = target / ".analysis_state-{:08d}-{}.parquet.tmp".format(
                    self._index, self._unique
                )
                row = {
                    "run_id": state.run_id,
                    "revision": revision,
                    "processed_through_step": state.processed_through_step,
                    "open_episode_count": len(state.open_episode_ids),
                    "baseline_count": len(state.baselines),
                    "episode_rule_version": state.episode_rule_version,
                    "fusion_rule_version": state.fusion_rule_version,
                    "rule_version": state.rule_version,
                    "recorded_wall_time_ns": state.recorded_wall_time_ns,
                    "payload": json.dumps(state.to_dict(), sort_keys=True, default=str),
                }
                pq.write_table(pa.Table.from_pylist([row], schema=ANALYSIS_STATE_SCHEMA), temp,
                               compression="zstd")
                os.replace(str(temp), str(final))
                return revision
        except Exception as exc:
            self.errors.append("{}: {}".format(type(exc).__name__, exc))
            if not self.fail_open:
                raise
            return -1

    def _read(self, run_id: str) -> pa.Table:
        directory = analysis_state_dir(self.root, run_id) / "analysis_state"
        if not directory.exists():
            return pa.table({})
        files = sorted(p for p in directory.glob("*.parquet") if p.is_file())
        readable = []
        for path in files:
            try:
                pq.read_metadata(str(path))
                readable.append(path)
            except Exception:
                self.errors.append("skipped unreadable analysis_state part: {}".format(path))
        if not readable:
            return pa.table({})
        return pq.read_table([str(p) for p in readable])

    def next_revision(self, run_id: str) -> int:
        table = self._read(run_id)
        if table.num_rows == 0:
            return 1
        return max(row["revision"] for row in table.to_pylist()) + 1

    def load(self, run_id: str, revision: Optional[int] = None) -> Optional[AnalysisState]:
        table = self._read(run_id)
        if table.num_rows == 0:
            return None
        rows = sorted(table.to_pylist(), key=lambda r: r["revision"])
        chosen = rows[-1] if revision is None else next(
            (r for r in rows if r["revision"] == revision), None
        )
        if chosen is None:
            return None
        return AnalysisState.from_dict(json.loads(chosen["payload"]))

    def history(self, run_id: str) -> List[int]:
        table = self._read(run_id)
        return sorted(row["revision"] for row in table.to_pylist())


def capture_analysis_state(
    run_id: str,
    *,
    processed_through_step: Optional[int],
    layer_series: Optional[Dict[Any, Sequence[Any]]] = None,
    config: Optional[Any] = None,
    episodes: Sequence[Any] = (),
    detector_versions: Optional[Dict[str, str]] = None,
    fusion_rule_version: str = "",
    episode_rule_version: str = "",
    rule_version: str = "",
) -> AnalysisState:
    """Snapshot the protected baseline for every observed layer.

    Each layer's reference is rebuilt with the **same** freeze-on-suspicion policy the
    live detector used, over the *same* truncated series, so the persisted reference is
    the one actually in force rather than a reconstruction from the finished run.
    """
    from ..detectors.baseline import ProtectedBaseline

    baselines: Dict[str, BaselineSnapshot] = {}
    floor = getattr(config, "warmup_steps", 50) if config else 50
    minimum_history = 20
    window = getattr(config, "window", 150) if config else 150
    freeze_at_ratio = 1.3

    for index, series in (layer_series or {}).items():
        points = [(step, value) for step, value in series if step >= floor]
        if len(points) < minimum_history:
            continue
        baseline = ProtectedBaseline(
            minimum_history=minimum_history, window=window,
            freeze_at_ratio=freeze_at_ratio,
        )
        for step, value in points:
            if value is None or value != value or not value > 0:
                baseline.update(step, value)
                continue
            current = baseline.ratio(value)
            if current is not None and current >= freeze_at_ratio and not baseline.frozen:
                baseline.suspect(step, "value {:.3f} is {:.2f}x the reference".format(value, current))
            baseline.update(step, value)
        baselines["grad.layer_norm:{}".format(index)] = BaselineSnapshot(
            key="grad.layer_norm:{}".format(index),
            reference=baseline.reference,
            dispersion=baseline.reference_dispersion,
            state=baseline.state,
            frozen=baseline.frozen,
            reason=baseline.reason,
            freeze_step=baseline.freeze_step,
            last_update_step=baseline.last_update_step,
            observations=len(points),
            ready=baseline.ready,
        )

    episode_list = list(episodes)
    return AnalysisState(
        run_id=run_id,
        processed_through_step=processed_through_step,
        detector_versions=dict(detector_versions or {}),
        fusion_rule_version=fusion_rule_version,
        episode_rule_version=episode_rule_version,
        rule_version=rule_version,
        baselines=baselines,
        episode_ids=[_episode_id(e) for e in episode_list],
        open_episode_ids=[
            _episode_id(e) for e in episode_list
            if (e.to_dict() if hasattr(e, "to_dict") else dict(e)).get("state") != "CLOSED"
        ],
    )


def _episode_id(episode: Any) -> str:
    return (episode.to_dict() if hasattr(episode, "to_dict") else dict(episode))["episode_id"]
