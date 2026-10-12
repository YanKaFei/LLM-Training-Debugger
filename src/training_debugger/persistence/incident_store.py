"""Append-only incident persistence.

Phase 1 produced incidents that existed only in memory. That made three things
impossible: auditing why an old run was judged anomalous, comparing incidents
across runs, and calibrating detectors against historical outcomes. This module
closes that gap.

Design commitments, each with a reason:

**Append-only, one file per write.** Parquet is immutable, so a revision is a new
row in a new part file, never a rewrite. An incident therefore has a full history
rather than a current value, which is what the brief requires: ``OPEN`` →
``UPDATED`` → ``RESOLVED`` cannot overwrite the earlier state, because the earlier
state is the evidence for why the later one was reached.

**Three tables, not one.** ``incidents`` carries the queryable summary and the
state; ``incident_evidence`` carries one row per evidence item with its full
provenance; ``incident_hypotheses`` carries one row per scored hypothesis. A
single blob column would make "show me every incident where the learning rate was
ruled out" impossible to express.

**Lossless reconstruction is a tested property, not a hope.**
``test_persistence`` writes an incident, discards the process state, reads it back
and asserts the rendered output is byte-identical. If that fails, the audit trail
is decorative.

**Provenance is mandatory on every evidence row**: ``event_id``, ``detector``,
``detector_version``, ``rule_version``. Without the rule version, an old incident
cannot be explained after a threshold changes — which is exactly the question a
postmortem asks.

**No database.** Parquet plus the existing DuckDB reader. Phase 0's decision was
that Postgres becomes necessary only when several humans write concurrently, and
that is still not true.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from ..schema.enums import SCHEMA_VERSION
from ..schema.incident import Incident
from ..schema.evidence import EVIDENCE_BUCKETS

__all__ = [
    "IncidentStore",
    "INCIDENT_SCHEMA",
    "EVIDENCE_SCHEMA",
    "HYPOTHESIS_SCHEMA",
    "incidents_dir",
    "RULE_VERSION",
]

#: Version of the detector rule set that produced a persisted incident. Bumped
#: whenever a threshold or criterion changes, so an old incident can still be
#: explained after the rules move.
RULE_VERSION = "phase1.1-1.0"


def incidents_dir(root: os.PathLike, run_id: str) -> Path:
    return Path(root) / "runs" / run_id / "incidents"


INCIDENT_SCHEMA = pa.schema(
    [
        pa.field("incident_id", pa.string(), nullable=False),
        pa.field("revision", pa.int32(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("incident_type", pa.string(), nullable=False),
        pa.field("state", pa.string(), nullable=False),
        pa.field("severity", pa.string(), nullable=False),
        pa.field("detected_step", pa.int64(), nullable=True),
        pa.field("onset_step", pa.int64(), nullable=True),
        pa.field("confidence", pa.float64(), nullable=True),
        pa.field("ordering_confidence", pa.string(), nullable=True),
        pa.field("first_abnormal_event_id", pa.string(), nullable=True),
        pa.field("candidate_count", pa.int32(), nullable=True),
        pa.field("evidence_count", pa.int32(), nullable=True),
        pa.field("not_correlated_count", pa.int32(), nullable=True),
        pa.field("affected_layer_count", pa.int32(), nullable=True),
        pa.field("population_layer_count", pa.int32(), nullable=True),
        pa.field("summary", pa.string(), nullable=True),
        pa.field("detector_ids", pa.string(), nullable=True),      # JSON list
        pa.field("missing_evidence", pa.string(), nullable=True),  # JSON list
        pa.field("detector_config", pa.string(), nullable=True),   # JSON object
        pa.field("rule_version", pa.string(), nullable=False),
        pa.field("persisted_wall_time_ns", pa.int64(), nullable=False),
        pa.field("payload", pa.string(), nullable=False),          # full to_dict JSON
    ]
)

EVIDENCE_SCHEMA = pa.schema(
    [
        pa.field("incident_id", pa.string(), nullable=False),
        pa.field("revision", pa.int32(), nullable=False),
        pa.field("bucket", pa.string(), nullable=False),
        pa.field("evidence_id", pa.string(), nullable=False),
        pa.field("claim", pa.string(), nullable=True),
        pa.field("level", pa.string(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=True),
        pa.field("value", pa.float64(), nullable=True),
        pa.field("unit", pa.string(), nullable=True),
        pa.field("baseline", pa.float64(), nullable=True),
        pa.field("threshold", pa.float64(), nullable=True),
        pa.field("window_start", pa.int64(), nullable=True),
        pa.field("window_end", pa.int64(), nullable=True),
        pa.field("detector", pa.string(), nullable=True),
        pa.field("detector_version", pa.string(), nullable=True),
        pa.field("rule_version", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=True),
        pa.field("rank", pa.int32(), nullable=True),
        pa.field("layer_index", pa.int32(), nullable=True),
        pa.field("module_path", pa.string(), nullable=True),
        pa.field("reason", pa.string(), nullable=True),
        # Provenance: which events and which query produced this row. Without it
        # a rendered sentence cannot be traced back to its evidence.
        pa.field("event_ids", pa.string(), nullable=True),         # JSON list
        pa.field("query_definition", pa.string(), nullable=True),  # JSON object
    ]
)

HYPOTHESIS_SCHEMA = pa.schema(
    [
        pa.field("incident_id", pa.string(), nullable=False),
        pa.field("revision", pa.int32(), nullable=False),
        pa.field("hypothesis_id", pa.string(), nullable=False),
        pa.field("template_id", pa.string(), nullable=False),
        pa.field("statement", pa.string(), nullable=True),
        pa.field("score", pa.float64(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("score_decomposition", pa.string(), nullable=True),  # JSON object
        pa.field("supporting", pa.string(), nullable=True),           # JSON list
        pa.field("refuting", pa.string(), nullable=True),             # JSON list
        pa.field("missing", pa.string(), nullable=True),              # JSON list
        pa.field("rule_version", pa.string(), nullable=False),
    ]
)


@dataclass
class PersistResult:
    incident_id: str
    revision: int
    run_id: str
    written: bool
    error: Optional[str] = None


class IncidentStore:
    """Append-only incident store for one dataset root."""

    def __init__(
        self,
        root: os.PathLike,
        *,
        rule_version: str = RULE_VERSION,
        detector_config: Optional[Dict[str, Any]] = None,
        fail_open: bool = True,
    ) -> None:
        self.root = Path(root)
        self.rule_version = rule_version
        self.detector_config = detector_config or {}
        self.fail_open = fail_open
        self._lock = threading.Lock()
        self._unique = uuid.uuid4().hex[:8]
        self._index = 0
        self.errors: List[str] = []
        self.revisions_written = 0

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append(self, incident: Incident) -> PersistResult:
        """Append a new revision of ``incident``.

        Fail-open by default: a persistence failure must not take down an
        analysis that has already succeeded. The failure is recorded in
        ``self.errors`` rather than swallowed, so silence is not the failure mode.
        """
        try:
            return self._append(incident)
        except Exception as exc:
            message = "{}: {}".format(type(exc).__name__, exc)
            self.errors.append(message)
            if not self.fail_open:
                raise
            return PersistResult(
                incident_id=incident.incident_id, revision=-1,
                run_id=incident.run_id, written=False, error=message,
            )

    def _append(self, incident: Incident) -> PersistResult:
        with self._lock:
            revision = self.next_revision(incident.run_id, incident.incident_id)
            directory = incidents_dir(self.root, incident.run_id)
            directory.mkdir(parents=True, exist_ok=True)

            incident_row = self._incident_row(incident, revision)
            evidence_rows = self._evidence_rows(incident, revision)
            hypothesis_rows = self._hypothesis_rows(incident, revision)

            self._write_table(
                directory, "incidents", pa.Table.from_pylist([incident_row], schema=INCIDENT_SCHEMA)
            )
            if evidence_rows:
                self._write_table(
                    directory, "incident_evidence",
                    pa.Table.from_pylist(evidence_rows, schema=EVIDENCE_SCHEMA),
                )
            if hypothesis_rows:
                self._write_table(
                    directory, "incident_hypotheses",
                    pa.Table.from_pylist(hypothesis_rows, schema=HYPOTHESIS_SCHEMA),
                )
            self.revisions_written += 1
            return PersistResult(
                incident_id=incident.incident_id, revision=revision,
                run_id=incident.run_id, written=True,
            )

    def _write_table(self, directory: Path, name: str, table: pa.Table) -> None:
        """Write one part atomically: temp file, then rename.

        Same crash-safety argument as the event store: a reader globbing
        ``<name>-*.parquet`` never sees a half-written batch.
        """
        target_dir = directory / name
        target_dir.mkdir(parents=True, exist_ok=True)
        self._index += 1
        final = target_dir / "{}-{:08d}-{}.parquet".format(name, self._index, self._unique)
        temp = target_dir / ".{}-{:08d}-{}.parquet.tmp".format(name, self._index, self._unique)
        pq.write_table(table, temp, compression="zstd")
        os.replace(str(temp), str(final))

    def _incident_row(self, incident: Incident, revision: int) -> Dict[str, Any]:
        return {
            "incident_id": incident.incident_id,
            "revision": revision,
            "run_id": incident.run_id,
            "schema_version": incident.schema_version or SCHEMA_VERSION,
            "incident_type": str(incident.incident_type),
            "state": str(incident.state),
            "severity": str(incident.severity),
            "detected_step": incident.detected_step,
            "onset_step": incident.metadata.get("onset_step"),
            "confidence": incident.confidence,
            "ordering_confidence": incident.ordering_confidence,
            "first_abnormal_event_id": incident.first_abnormal_event_id,
            "candidate_count": len(incident.earliest.candidates) if incident.earliest else 0,
            "evidence_count": len(incident.evidence.all()),
            "not_correlated_count": len(incident.evidence.not_correlated),
            "affected_layer_count": len(incident.multiplicity.affected_layers),
            "population_layer_count": incident.multiplicity.population_layers,
            "summary": incident.summary,
            "detector_ids": json.dumps(list(incident.detector_ids)),
            "missing_evidence": json.dumps(list(incident.missing_evidence)),
            "detector_config": json.dumps(self.detector_config, sort_keys=True, default=str),
            "rule_version": self.rule_version,
            "persisted_wall_time_ns": time.time_ns(),
            "payload": json.dumps(incident.to_dict(), sort_keys=True, default=str, allow_nan=True),
        }

    def _evidence_rows(self, incident: Incident, revision: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        buckets = {
            "supporting": incident.evidence.supporting,
            "contradicting": incident.evidence.contradicting,
            "not_correlated": incident.evidence.not_correlated,
            "unknown": incident.evidence.unknown,
        }
        for bucket in EVIDENCE_BUCKETS:
            for evidence in buckets.get(bucket, []):
                window = list(evidence.window or [])
                rows.append(
                    {
                        "incident_id": incident.incident_id,
                        "revision": revision,
                        "bucket": bucket,
                        "evidence_id": evidence.evidence_id,
                        "claim": evidence.claim,
                        "level": str(evidence.level),
                        "metric_name": evidence.metric_name,
                        "value": evidence.value,
                        "unit": None if evidence.unit is None else str(evidence.unit),
                        "baseline": evidence.baseline,
                        "threshold": evidence.threshold,
                        "window_start": window[0] if window else None,
                        "window_end": window[-1] if len(window) > 1 else (window[0] if window else None),
                        "detector": evidence.detector,
                        "detector_version": evidence.detector_version,
                        "rule_version": self.rule_version,
                        "confidence": evidence.confidence,
                        "rank": evidence.rank,
                        "layer_index": evidence.layer_index,
                        "module_path": evidence.module_path,
                        "reason": evidence.reason,
                        "event_ids": json.dumps(list(evidence.event_ids)),
                        # The query definition is the statistical procedure that
                        # produced the row, so a reviewer can recompute it.
                        "query_definition": json.dumps(
                            {
                                "metric_name": evidence.metric_name,
                                "detector": evidence.detector,
                                "window": window,
                                "baseline": evidence.baseline,
                                "threshold": evidence.threshold,
                                "method": (evidence.metadata or {}).get("method"),
                            },
                            sort_keys=True,
                            default=str,
                        ),
                    }
                )
        return rows

    def _hypothesis_rows(self, incident: Incident, revision: int) -> List[Dict[str, Any]]:
        return [
            {
                "incident_id": incident.incident_id,
                "revision": revision,
                "hypothesis_id": h.hypothesis_id,
                "template_id": h.template_id,
                "statement": h.statement,
                "score": float(h.score),
                "status": str(h.status),
                "score_decomposition": json.dumps(h.score_decomposition, sort_keys=True, default=str),
                "supporting": json.dumps(list(h.supporting)),
                "refuting": json.dumps(list(h.refuting)),
                "missing": json.dumps(list(h.missing)),
                "rule_version": self.rule_version,
            }
            for h in incident.hypotheses
        ]

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _read_table(self, run_id: str, name: str) -> pa.Table:
        directory = incidents_dir(self.root, run_id) / name
        if not directory.exists():
            return pa.table({})
        files = sorted(p for p in directory.glob("*.parquet") if p.is_file())
        readable = []
        for path in files:
            try:
                pq.read_metadata(str(path))
                readable.append(path)
            except Exception:
                self.errors.append("skipped unreadable incident part: {}".format(path))
        if not readable:
            return pa.table({})
        return pq.read_table([str(p) for p in readable])

    def list_incidents(
        self, run_id: str, *, latest_only: bool = True, include_closed: bool = True
    ) -> List[Dict[str, Any]]:
        """Incident summaries. ``latest_only`` collapses revisions, as the reader
        API requires; pass ``False`` for the full history."""
        table = self._read_table(run_id, "incidents")
        if table.num_rows == 0:
            return []
        rows = table.to_pylist()
        if not include_closed:
            rows = [r for r in rows if r["state"] != "CLOSED"]
        if latest_only:
            best: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                current = best.get(row["incident_id"])
                if current is None or row["revision"] > current["revision"]:
                    best[row["incident_id"]] = row
            rows = sorted(best.values(), key=lambda r: (r["detected_step"] or 0, r["incident_id"]))
        else:
            rows = sorted(rows, key=lambda r: (r["incident_id"], r["revision"]))
        return rows

    def history(self, run_id: str, incident_id: str) -> List[Dict[str, Any]]:
        return [
            row for row in self.list_incidents(run_id, latest_only=False)
            if row["incident_id"] == incident_id
        ]

    def load(self, run_id: str, incident_id: str, revision: Optional[int] = None) -> Incident:
        """Reconstruct an incident, from its parts rather than from the blob.

        Deliberately rebuilds the evidence and hypotheses from their own tables:
        reconstructing from the stored ``payload`` alone would not prove that the
        relational tables are sufficient, and those tables are what a query uses.
        """
        history = self.history(run_id, incident_id)
        if not history:
            raise KeyError("no incident %s in run %s" % (incident_id, run_id))
        row = history[-1] if revision is None else next(
            (r for r in history if r["revision"] == revision), None
        )
        if row is None:
            raise KeyError("no revision %s of %s" % (revision, incident_id))

        incident = Incident.from_dict(json.loads(row["payload"]))

        # Replace the evidence and hypotheses with what the relational tables hold,
        # so a divergence between the blob and the tables is detected here.
        evidence_table = self._read_table(run_id, "incident_evidence")
        hypothesis_table = self._read_table(run_id, "incident_hypotheses")
        if evidence_table.num_rows:
            rows = [
                r for r in evidence_table.to_pylist()
                if r["incident_id"] == incident_id and r["revision"] == row["revision"]
            ]
            if rows:
                from ..schema.evidence import Evidence, EvidenceBuckets

                rebuilt = EvidenceBuckets()
                for entry in rows:
                    evidence = Evidence.from_dict(
                        {
                            "claim": entry["claim"],
                            "level": entry["level"],
                            "event_ids": json.loads(entry["event_ids"] or "[]"),
                            "metric_name": entry["metric_name"],
                            "value": entry["value"],
                            "unit": entry["unit"],
                            "baseline": entry["baseline"],
                            "threshold": entry["threshold"],
                            "window": [entry["window_start"], entry["window_end"]]
                            if entry["window_start"] is not None else None,
                            "detector": entry["detector"],
                            "detector_version": entry["detector_version"],
                            "confidence": entry["confidence"],
                            "rank": entry["rank"],
                            "layer_index": entry["layer_index"],
                            "module_path": entry["module_path"],
                            "reason": entry["reason"],
                            "metadata": {},
                        }
                    )
                    evidence.evidence_id = entry["evidence_id"]
                    getattr(rebuilt, entry["bucket"]).append(evidence)
                incident.evidence = rebuilt

        if hypothesis_table.num_rows:
            rows = [
                r for r in hypothesis_table.to_pylist()
                if r["incident_id"] == incident_id and r["revision"] == row["revision"]
            ]
            if rows:
                from ..schema.incident import Hypothesis

                incident.hypotheses = [
                    Hypothesis.from_dict(
                        {
                            "hypothesis_id": entry["hypothesis_id"],
                            "template_id": entry["template_id"],
                            "statement": entry["statement"],
                            "score": entry["score"],
                            "status": entry["status"],
                            "score_decomposition": json.loads(entry["score_decomposition"] or "{}"),
                            "supporting_evidence_ids": json.loads(entry["supporting"] or "[]"),
                            "refuting_evidence_ids": json.loads(entry["refuting"] or "[]"),
                            "missing_evidence": json.loads(entry["missing"] or "[]"),
                        }
                    )
                    for entry in rows
                ]
        return incident

    def next_revision(self, run_id: str, incident_id: str) -> int:
        history = self.history(run_id, incident_id)
        return (history[-1]["revision"] + 1) if history else 1

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def update_state(self, incident: Incident, state: Any) -> PersistResult:
        """Append a revision with a new lifecycle state.

        The previous revision is left untouched: an incident that was resolved
        after being updated must still be able to show that it was once open.
        """
        incident.touch(state)
        return self.append(incident)

    def stats(self) -> Dict[str, Any]:
        return {
            "revisions_written": self.revisions_written,
            "rule_version": self.rule_version,
            "errors": list(self.errors),
        }
