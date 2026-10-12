"""J3: append-only episode persistence.

Phase 1.5 reported `episode persistence = absent` and `restart reconstruction =
absent`. That is the gap this closes: an episode that cannot survive a process
boundary is not durable evidence, and the Phase 1.1 incident store's restart
guarantee did not extend to it.

Same design as the incident store, for the same reasons:

**Append-only.** Parquet is immutable, so a revision is a new row in a new part
file. `SUSPECT -> CONFIRMED -> RECOVERING -> CLOSED` is a *history*, and the
`SUSPECT` state that was superseded is the evidence that the onset was observed
before the confirmation was available.

**Three tables.** ``episodes`` for the queryable summary plus a lossless
``payload``; ``episode_revisions`` for the state history; ``episode_relations`` for
the consequence links, so a relation survives a restart with the rest.

**Cross-process continuation is the test that matters.** A process must be able to
persist a `SUSPECT` episode, exit, and have a second process load it and *promote
the same episode* -- same id, same onset, correct signal count.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..episodes.model import AnomalyEpisode
from ..incidents.relations import IncidentRelation

__all__ = ["EpisodeStore", "EpisodeSnapshot", "episodes_dir", "EPISODE_SCHEMA",
           "EPISODE_REVISION_SCHEMA", "EPISODE_RELATION_SCHEMA"]


def episodes_dir(root: os.PathLike, run_id: str) -> Path:
    return Path(root) / "runs" / run_id / "episodes"


EPISODE_SCHEMA = pa.schema([
    pa.field("episode_id", pa.string(), nullable=False),
    pa.field("revision", pa.int32(), nullable=False),
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("state", pa.string(), nullable=False),
    pa.field("outcome", pa.string(), nullable=True),
    pa.field("detector_id", pa.string(), nullable=False),
    pa.field("detector_version", pa.string(), nullable=True),
    pa.field("rule_version", pa.string(), nullable=True),
    pa.field("anomaly_type", pa.string(), nullable=False),
    pa.field("signal_key", pa.string(), nullable=False),
    pa.field("candidate_onset_step", pa.int64(), nullable=True),
    pa.field("confirmed_at_step", pa.int64(), nullable=True),
    pa.field("last_abnormal_step", pa.int64(), nullable=True),
    pa.field("recovered_at_step", pa.int64(), nullable=True),
    pa.field("closed_at_step", pa.int64(), nullable=True),
    pa.field("signal_count", pa.int32(), nullable=True),
    pa.field("affected_layers", pa.string(), nullable=True),
    pa.field("affected_ranks", pa.string(), nullable=True),
    pa.field("supporting_event_ids", pa.string(), nullable=True),
    pa.field("confidence", pa.float64(), nullable=True),
    pa.field("confidence_class", pa.string(), nullable=True),
    pa.field("gradient_band", pa.string(), nullable=True),
    pa.field("signature", pa.string(), nullable=True),
    pa.field("payload", pa.string(), nullable=False),
])

EPISODE_REVISION_SCHEMA = pa.schema([
    pa.field("episode_id", pa.string(), nullable=False),
    pa.field("revision", pa.int32(), nullable=False),
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("state", pa.string(), nullable=False),
    pa.field("outcome", pa.string(), nullable=True),
    pa.field("signal_count", pa.int32(), nullable=True),
    pa.field("confirmed_at_step", pa.int64(), nullable=True),
    pa.field("recorded_wall_time_ns", pa.int64(), nullable=False),
])

EPISODE_RELATION_SCHEMA = pa.schema([
    pa.field("relation_id", pa.string(), nullable=False),
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("source_incident_id", pa.string(), nullable=False),
    pa.field("target_incident_id", pa.string(), nullable=False),
    pa.field("relation_type", pa.string(), nullable=False),
    pa.field("source_anomaly_type", pa.string(), nullable=True),
    pa.field("target_anomaly_type", pa.string(), nullable=True),
    pa.field("gap_steps", pa.int64(), nullable=True),
    pa.field("rule_id", pa.string(), nullable=True),
    pa.field("rule_version", pa.string(), nullable=True),
    pa.field("confidence", pa.float64(), nullable=True),
    pa.field("payload", pa.string(), nullable=False),
])


@dataclass
class EpisodeSnapshot:
    """What a restarted process needs to continue an episode."""

    episode: AnomalyEpisode
    revision: int
    signal_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision": self.revision,
            "signal_count": self.signal_count,
            "episode": self.episode.to_dict(),
        }


class EpisodeStore:
    """Append-only episode store for one dataset root."""

    def __init__(self, root: os.PathLike, *, fail_open: bool = True) -> None:
        self.root = Path(root)
        self.fail_open = fail_open
        self._lock = threading.Lock()
        self._unique = uuid.uuid4().hex[:8]
        self._index = 0
        self.errors: List[str] = []
        self.revisions_written = 0

    # -- writing -------------------------------------------------------

    def append(self, episode: AnomalyEpisode, *, revision: Optional[int] = None) -> int:
        try:
            return self._append(episode, revision)
        except Exception as exc:
            self.errors.append("{}: {}".format(type(exc).__name__, exc))
            if not self.fail_open:
                raise
            return -1

    def _append(self, episode: AnomalyEpisode, revision: Optional[int]) -> int:
        with self._lock:
            if revision is None:
                revision = self.next_revision(episode.run_id, episode.episode_id)
            directory = episodes_dir(self.root, episode.run_id)
            directory.mkdir(parents=True, exist_ok=True)
            payload = episode.to_dict()
            row = {
                "episode_id": episode.episode_id,
                "revision": revision,
                "run_id": episode.run_id,
                "state": episode.state,
                "outcome": episode.outcome,
                "detector_id": episode.detector_id,
                "detector_version": episode.detector_version,
                "rule_version": episode.rule_version,
                "anomaly_type": episode.anomaly_type,
                "signal_key": episode.signal_key,
                "candidate_onset_step": episode.candidate_onset_step,
                "confirmed_at_step": episode.confirmed_at_step,
                "last_abnormal_step": episode.last_abnormal_step,
                "recovered_at_step": episode.recovered_at_step,
                "closed_at_step": None,
                "signal_count": episode.signal_count,
                "affected_layers": json.dumps(episode.affected_layers),
                "affected_ranks": json.dumps(episode.affected_ranks),
                "supporting_event_ids": json.dumps(episode.supporting_event_ids),
                "confidence": episode.confidence,
                "confidence_class": episode.confidence_class,
                "gradient_band": (episode.metadata or {}).get("gradient_band"),
                "signature": episode.signature(),
                "payload": json.dumps(payload, sort_keys=True, default=str),
            }
            self._write(directory, "episodes", pa.Table.from_pylist([row], schema=EPISODE_SCHEMA))
            revision_row = {
                "episode_id": episode.episode_id,
                "revision": revision,
                "run_id": episode.run_id,
                "state": episode.state,
                "outcome": episode.outcome,
                "signal_count": episode.signal_count,
                "confirmed_at_step": episode.confirmed_at_step,
                "recorded_wall_time_ns": time.time_ns(),
            }
            self._write(
                directory, "episode_revisions",
                pa.Table.from_pylist([revision_row], schema=EPISODE_REVISION_SCHEMA),
            )
            self.revisions_written += 1
            return revision

    def append_relations(self, run_id: str, relations: Sequence[IncidentRelation]) -> int:
        if not relations:
            return 0
        try:
            directory = episodes_dir(self.root, run_id)
            directory.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "relation_id": r.relation_id,
                    "run_id": r.run_id,
                    "source_incident_id": r.source_incident_id,
                    "target_incident_id": r.target_incident_id,
                    "relation_type": r.relation_type,
                    "source_anomaly_type": r.source_anomaly_type,
                    "target_anomaly_type": r.target_anomaly_type,
                    "gap_steps": r.gap_steps,
                    "rule_id": r.rule_id,
                    "rule_version": r.rule_version,
                    "confidence": r.confidence,
                    "payload": json.dumps(r.to_dict(), sort_keys=True, default=str),
                }
                for r in relations
            ]
            self._write(
                directory, "episode_relations",
                pa.Table.from_pylist(rows, schema=EPISODE_RELATION_SCHEMA),
            )
            return len(rows)
        except Exception as exc:
            self.errors.append("{}: {}".format(type(exc).__name__, exc))
            if not self.fail_open:
                raise
            return 0

    def _write(self, directory: Path, name: str, table: pa.Table) -> None:
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        self._index += 1
        final = target / "{}-{:08d}-{}.parquet".format(name, self._index, self._unique)
        temp = target / ".{}-{:08d}-{}.parquet.tmp".format(name, self._index, self._unique)
        pq.write_table(table, temp, compression="zstd")
        os.replace(str(temp), str(final))

    # -- reading -------------------------------------------------------

    def _read(self, run_id: str, name: str) -> pa.Table:
        directory = episodes_dir(self.root, run_id) / name
        if not directory.exists():
            return pa.table({})
        files = sorted(p for p in directory.glob("*.parquet") if p.is_file())
        readable = []
        for path in files:
            try:
                pq.read_metadata(str(path))
                readable.append(path)
            except Exception:
                self.errors.append("skipped unreadable episode part: {}".format(path))
        if not readable:
            return pa.table({})
        return pq.read_table([str(p) for p in readable])

    def next_revision(self, run_id: str, episode_id: str) -> int:
        rows = self.history(run_id, episode_id)
        return (rows[-1]["revision"] + 1) if rows else 1

    def history(self, run_id: str, episode_id: str) -> List[Dict[str, Any]]:
        table = self._read(run_id, "episode_revisions")
        if table.num_rows == 0:
            return []
        rows = [r for r in table.to_pylist() if r["episode_id"] == episode_id]
        return sorted(rows, key=lambda r: r["revision"])

    def load(self, run_id: str, episode_id: str, revision: Optional[int] = None) -> AnomalyEpisode:
        """Reconstruct an episode. Latest revision by default."""
        table = self._read(run_id, "episodes")
        if table.num_rows == 0:
            raise KeyError("no episodes stored for run %s" % run_id)
        rows = [r for r in table.to_pylist() if r["episode_id"] == episode_id]
        if not rows:
            raise KeyError("no episode %s in run %s" % (episode_id, run_id))
        rows.sort(key=lambda r: r["revision"])
        chosen = rows[-1] if revision is None else next(
            (r for r in rows if r["revision"] == revision), None
        )
        if chosen is None:
            raise KeyError("no revision %s of %s" % (revision, episode_id))
        return AnomalyEpisode.from_dict(json.loads(chosen["payload"]))

    def open_episodes(self, run_id: str) -> List[EpisodeSnapshot]:
        """Every episode whose latest revision is not CLOSED.

        This is what a restarted process loads to continue: the episode that was
        SUSPECT when the previous process exited.
        """
        table = self._read(run_id, "episodes")
        if table.num_rows == 0:
            return []
        latest: Dict[str, Dict[str, Any]] = {}
        for row in table.to_pylist():
            current = latest.get(row["episode_id"])
            if current is None or row["revision"] > current["revision"]:
                latest[row["episode_id"]] = row
        snapshots = []
        for row in latest.values():
            if row["state"] == "CLOSED":
                continue
            snapshots.append(
                EpisodeSnapshot(
                    episode=AnomalyEpisode.from_dict(json.loads(row["payload"])),
                    revision=row["revision"],
                    signal_count=row["signal_count"] or 0,
                )
            )
        return sorted(snapshots, key=lambda s: s.episode.episode_id)

    def load_relations(self, run_id: str) -> List[IncidentRelation]:
        table = self._read(run_id, "episode_relations")
        if table.num_rows == 0:
            return []
        return [IncidentRelation.from_dict(json.loads(r["payload"])) for r in table.to_pylist()]

    def stats(self) -> Dict[str, Any]:
        return {"revisions_written": self.revisions_written, "errors": list(self.errors)}
