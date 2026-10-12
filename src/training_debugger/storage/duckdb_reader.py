"""DuckDB query layer over the Parquet dataset.

Parquet is the truth; DuckDB is a query engine pointed at it. Nothing here
writes, and nothing here duplicates state -- that is the point of the Phase 0
decision to keep Trackio out of the telemetry path.

Two robustness behaviours matter more than speed:

**A corrupt or truncated part is skipped, not fatal.** One bad file must not make
a run unreadable, because the most likely moment to read a run is right after it
crashed. Skipped files are counted and reported, so the omission is visible
rather than silent.

**Every query returns its own coverage.** :meth:`DuckDBReader.coverage` reports
which partitions and step range actually exist, so a caller can distinguish "no
anomaly" from "no data" -- the failure mode Phase 0 called out as the most common
way an automated diagnosis goes wrong.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .layout import INDEXED_COLUMNS, events_dir, list_part_files, run_dir

__all__ = ["DuckDBReader", "ReaderCoverage"]


class ReaderCoverage:
    """What data actually exists for a run. Answers "was it absent or unread?"."""

    def __init__(
        self,
        run_id: str,
        *,
        files_total: int,
        files_readable: int,
        files_skipped: int,
        skipped_paths: Sequence[str],
        step_min: Optional[int],
        step_max: Optional[int],
        event_types: Sequence[str],
        row_count: int,
        ranks: Sequence[int],
    ) -> None:
        self.run_id = run_id
        self.files_total = files_total
        self.files_readable = files_readable
        self.files_skipped = files_skipped
        self.skipped_paths = list(skipped_paths)
        self.step_min = step_min
        self.step_max = step_max
        self.event_types = list(event_types)
        self.row_count = row_count
        self.ranks = list(ranks)

    def to_dict(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "files_total": self.files_total,
            "files_readable": self.files_readable,
            "files_skipped": self.files_skipped,
            "skipped_paths": list(self.skipped_paths),
            "step_min": self.step_min,
            "step_max": self.step_max,
            "event_types": list(self.event_types),
            "row_count": self.row_count,
            "ranks": list(self.ranks),
        }

    def covers_step(self, step: int) -> bool:
        if self.step_min is None or self.step_max is None:
            return False
        return self.step_min <= step <= self.step_max


class DuckDBReader:
    """Read-only query API. One connection, one view per run, cached."""

    def __init__(self, root: Any, *, validate_files: bool = True) -> None:
        self.root = Path(root)
        self._connection = duckdb.connect(database=":memory:")
        self._lock = threading.Lock()
        self._views: Dict[str, Optional[str]] = {}
        self._skipped: Dict[str, List[str]] = {}
        self._validate_files = validate_files
        self.extension_status = "unknown"
        self.extension_notes: List[str] = []
        self.close_failed = False
        self._read_parquet_available = self._check_read_parquet()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _check_read_parquet(self) -> bool:
        # INSTALL/LOAD are best-effort: DuckDB ships read_parquet built in on
        # most distributions, so a failure here is informational. It is recorded
        # rather than swallowed, because "the extension could not be loaded" and
        # "the extension was unnecessary" are different states and a silent
        # handler would make them indistinguishable.
        self.extension_status = "builtin"
        try:
            self._connection.execute("INSTALL 'parquet'")
            self.extension_status = "installed"
        except Exception as exc:
            self.extension_notes.append("install: {}".format(exc))
        try:
            self._connection.execute("LOAD 'parquet'")
            self.extension_status = "loaded"
        except Exception as exc:
            self.extension_notes.append("load: {}".format(exc))
        try:
            self._connection.execute("SELECT 1 FROM (SELECT 1) t LIMIT 0")
            return True
        except Exception:
            return False

    def _readable_files(self, run_id: str) -> List[Path]:
        """Part files whose Parquet footer parses. Corrupt files are excluded.

        Reading the footer is far cheaper than reading the data, and it is what
        lets a truncated file be skipped instead of poisoning the whole scan.
        """
        files = list_part_files(self.root, run_id)
        if not self._validate_files:
            return files
        good: List[Path] = []
        skipped: List[str] = []
        for path in files:
            try:
                pq.read_metadata(str(path))
                good.append(path)
            except Exception:
                skipped.append(str(path))
        self._skipped[run_id] = skipped
        return good

    def _view_name(self, run_id: str) -> Optional[str]:
        """Create (or reuse) a view over the run's files. None when empty."""
        with self._lock:
            if run_id in self._views:
                return self._views[run_id]
            files = self._readable_files(run_id)
            if not files:
                # Cache an explicit None. Caching "" here was a real bug: the
                # empty string is falsy but is not None, so `if view is None`
                # in the callers missed it and built `SELECT * FROM  WHERE ...`.
                self._views[run_id] = None
                return None
            name = "run_{}".format(abs(hash(run_id)) % (10 ** 12))
            file_list = ", ".join("'" + str(p).replace("'", "''") + "'" for p in files)
            self._connection.execute(
                "CREATE OR REPLACE VIEW {name} AS "
                "SELECT * FROM read_parquet([{files}], union_by_name=true)".format(
                    name=name, files=file_list
                )
            )
            self._views[run_id] = name
            return name

    def refresh(self, run_id: Optional[str] = None) -> None:
        """Forget cached views so newly written files are picked up."""
        with self._lock:
            if run_id is None:
                self._views.clear()
                self._skipped.clear()
            else:
                self._views.pop(run_id, None)

    def close(self) -> None:
        try:
            self._connection.close()
        except Exception:
            # Recorded rather than raised: closing a read-only handle must not
            # be able to fail a caller that has already got its answer.
            self.close_failed = True

    def __enter__(self) -> "DuckDBReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Query primitives
    # ------------------------------------------------------------------

    def _query(self, sql: str, params: Optional[Sequence[Any]] = None) -> pa.Table:
        with self._lock:
            cursor = self._connection.cursor()
            try:
                result = cursor.execute(sql, list(params) if params else None)
                return result.fetch_arrow_table()
            finally:
                cursor.close()

    def _run_table(self, run_id: str) -> Optional[str]:
        return self._view_name(run_id)

    # ------------------------------------------------------------------
    # The queries the brief requires (section 5.4)
    # ------------------------------------------------------------------

    def metrics(
        self,
        run_id: str,
        metric_names: Optional[Iterable[str]] = None,
        *,
        step_range: Optional[Tuple[int, int]] = None,
        ranks: Optional[Iterable[int]] = None,
        layer_index: Optional[int] = None,
        order_by_step: bool = True,
        limit: Optional[int] = None,
    ) -> pa.Table:
        """Metric samples for a run, with optional filters."""
        view = self._run_table(run_id)
        if view is None:
            return self._empty_metrics()
        clauses = ["metric_name IS NOT NULL", "metric_value IS NOT NULL"]
        params: List[Any] = []
        if metric_names:
            names = list(metric_names)
            clauses.append("metric_name IN ({})".format(", ".join(["?"] * len(names))))
            params.extend(names)
        if step_range:
            clauses.append("step BETWEEN ? AND ?")
            params.extend([step_range[0], step_range[1]])
        if ranks:
            rank_list = list(ranks)
            clauses.append("rank IN ({})".format(", ".join(["?"] * len(rank_list))))
            params.extend(rank_list)
        if layer_index is not None:
            clauses.append("layer_index = ?")
            params.append(layer_index)
        sql = "SELECT * FROM {view} WHERE {where}".format(view=view, where=" AND ".join(clauses))
        if order_by_step:
            sql += " ORDER BY step NULLS FIRST, microstep NULLS FIRST, source_seq NULLS LAST"
        if limit:
            sql += " LIMIT {}".format(int(limit))
        return self._query(sql, params)

    def metrics_around(
        self, run_id: str, step: int, window: int, metric_names: Optional[Iterable[str]] = None
    ) -> pa.Table:
        return self.metrics(
            run_id, metric_names, step_range=(max(0, step - window), step + window)
        )

    def events(
        self,
        run_id: str,
        *,
        event_types: Optional[Iterable[str]] = None,
        step_range: Optional[Tuple[int, int]] = None,
        rank: Optional[int] = None,
        layer_index: Optional[int] = None,
        metric_name: Optional[str] = None,
        priority: Optional[str] = None,
        order_by_step: bool = True,
        limit: Optional[int] = None,
    ) -> pa.Table:
        """Events of any type, filtered. The generic escape hatch."""
        view = self._run_table(run_id)
        if view is None:
            return self._empty_events()
        clauses: List[str] = []
        params: List[Any] = []
        if event_types:
            types = list(event_types)
            clauses.append("event_type IN ({})".format(", ".join(["?"] * len(types))))
            params.extend(types)
        if step_range:
            clauses.append("step BETWEEN ? AND ?")
            params.extend([step_range[0], step_range[1]])
        if rank is not None:
            clauses.append("rank = ?")
            params.append(rank)
        if layer_index is not None:
            clauses.append("layer_index = ?")
            params.append(layer_index)
        if metric_name is not None:
            clauses.append("metric_name = ?")
            params.append(metric_name)
        if priority is not None:
            clauses.append("priority = ?")
            params.append(priority)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = "SELECT * FROM {view}{where}".format(view=view, where=where)
        if order_by_step:
            sql += " ORDER BY step NULLS FIRST, microstep NULLS FIRST, source_seq NULLS LAST"
        if limit:
            sql += " LIMIT {}".format(int(limit))
        return self._query(sql, params)

    def events_by_rank(self, run_id: str, rank: int, **kwargs: Any) -> pa.Table:
        return self.events(run_id, rank=rank, **kwargs)

    def events_by_layer(self, run_id: str, layer_index: int, **kwargs: Any) -> pa.Table:
        return self.events(run_id, layer_index=layer_index, **kwargs)

    def event_type_counts(self, run_id: str) -> pa.Table:
        view = self._run_table(run_id)
        if view is None:
            return pa.table({"event_type": pa.array([], type=pa.string()),
                             "n": pa.array([], type=pa.int64())})
        return self._query(
            "SELECT event_type, count(*) AS n FROM {view} GROUP BY event_type ORDER BY n DESC".format(
                view=view
            )
        )

    def incident_window(
        self, run_id: str, step: int, back: int, forward: int = 0, *, limit: Optional[int] = None
    ) -> pa.Table:
        """All events in ``[step - back, step + forward]``, in deterministic order.

        The correlation engine's primary input, so it deliberately returns *all*
        event types rather than only metrics: the causal story usually lives in
        the lifecycle events (a shard change, a checkpoint load), not in the
        numbers.
        """
        return self.events(
            run_id, step_range=(max(0, step - back), step + forward), limit=limit
        )

    # ------------------------------------------------------------------
    # Coverage and summaries
    # ------------------------------------------------------------------

    def coverage(self, run_id: str) -> ReaderCoverage:
        files_total = len(list_part_files(self.root, run_id))
        view = self._run_table(run_id)
        skipped = self._skipped.get(run_id, [])
        readable = files_total - len(skipped)
        if view is None:
            return ReaderCoverage(
                run_id, files_total=files_total, files_readable=0,
                files_skipped=len(skipped), skipped_paths=skipped,
                step_min=None, step_max=None, event_types=[], row_count=0, ranks=[],
            )
        summary = self._query(
            "SELECT min(step) AS step_min, max(step) AS step_max, count(*) AS n, "
            "count(DISTINCT rank) AS n_ranks FROM {view} WHERE step IS NOT NULL".format(view=view)
        ).to_pylist()[0]
        types = [r["event_type"] for r in
                 self._query("SELECT DISTINCT event_type FROM {view}".format(view=view)).to_pylist()]
        ranks = [r["rank"] for r in
                 self._query(
                     "SELECT DISTINCT rank FROM {view} WHERE rank IS NOT NULL ORDER BY rank".format(view=view)
                 ).to_pylist()]
        total = self._query("SELECT count(*) AS n FROM {view}".format(view=view)).to_pylist()[0]["n"]
        return ReaderCoverage(
            run_id,
            files_total=files_total,
            files_readable=readable,
            files_skipped=len(skipped),
            skipped_paths=skipped,
            step_min=summary["step_min"],
            step_max=summary["step_max"],
            event_types=sorted(types),
            row_count=int(total),
            ranks=[r for r in ranks if r is not None],
        )

    def list_runs(self) -> List[str]:
        base = self.root / "runs"
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    def series(
        self, run_id: str, metric_name: str, *, rank: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """``[(step, value)]`` for one metric. The detector layer's input shape.

        Non-finite values are preserved (they are signals); steps with no value
        are dropped rather than filled, because interpolating a missing sample
        would invent evidence.
        """
        table = self.metrics(run_id, [metric_name], ranks=[rank] if rank is not None else None)
        steps = table.column("step").to_pylist()
        values = table.column("metric_value").to_pylist()
        return [(int(s), float(v)) for s, v in zip(steps, values) if s is not None and v is not None]

    def series_by_layer(
        self, run_id: str, metric_name: str
    ) -> Dict[int, List[Tuple[int, float]]]:
        """``{layer_index: [(step, value)]}`` for a per-layer metric."""
        table = self.metrics(run_id, [metric_name])
        layers = table.column("layer_index").to_pylist()
        steps = table.column("step").to_pylist()
        values = table.column("metric_value").to_pylist()
        out: Dict[int, List[Tuple[int, float]]] = {}
        for layer, step, value in zip(layers, steps, values):
            if layer is None or step is None or value is None:
                continue
            out.setdefault(int(layer), []).append((int(step), float(value)))
        for layer in out:
            out[layer].sort()
        return out

    def step_range_present(self, run_id: str) -> Optional[Tuple[int, int]]:
        coverage = self.coverage(run_id)
        if coverage.step_min is None:
            return None
        return (int(coverage.step_min), int(coverage.step_max))

    # ------------------------------------------------------------------
    # Empty results with the right schema
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_metrics() -> pa.Table:
        return pa.table(
            {
                "step": pa.array([], type=pa.int64()),
                "metric_name": pa.array([], type=pa.string()),
                "metric_value": pa.array([], type=pa.float64()),
                "metric_unit": pa.array([], type=pa.string()),
            }
        )

    @staticmethod
    def _empty_events() -> pa.Table:
        return pa.table(
            {
                "event_id": pa.array([], type=pa.string()),
                "event_type": pa.array([], type=pa.string()),
                "step": pa.array([], type=pa.int64()),
            }
        )
