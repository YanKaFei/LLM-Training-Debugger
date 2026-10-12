"""Phase 1 CLI: `runs`, `incidents`, `incident`, `metrics`, `diagnose`.

The commands read the Parquet dataset and print either text or ``--json``. Every
command is deterministic: the same store always produces the same output, which
is what makes the golden-file regression possible.

``diagnose`` is a **deterministic renderer**. It is explicitly *not* an LLM agent
(Phase 1 brief section 9): it re-runs the detectors and the incident engine over
the stored events and prints the resulting incident. The agent arrives in a later
phase and will explain this output rather than replace it.

One design note on why this file is thin: the CLI owns no analysis logic. It
loads, calls the library, and formats. A CLI that computed an anomaly would be a
second implementation of the product, and the two would drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["main", "build_parser"]


def _default_root() -> Path:
    return Path.cwd() / "training-debugger-data"


# ---------------------------------------------------------------------------
# Shared loading
# ---------------------------------------------------------------------------


def _reader(root: Path):
    from ..storage import DuckDBReader

    return DuckDBReader(root)


def _series(reader, run_id: str) -> Tuple[Dict[str, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]], Dict[int, str]]:
    """Extract the detector input series from the store."""
    table = reader.events(run_id)
    by_metric: Dict[str, Dict[int, float]] = {}
    layers: Dict[int, Dict[int, float]] = {}
    module_paths: Dict[int, str] = {}
    names = table.column("metric_name").to_pylist()
    steps = table.column("step").to_pylist()
    values = table.column("metric_value").to_pylist()
    layer_index = table.column("layer_index").to_pylist()
    module_path = table.column("module_path").to_pylist()
    for name, step, value, layer, module in zip(names, steps, values, layer_index, module_path):
        if name is None or step is None or value is None:
            continue
        if name == "grad.layer_norm" and layer is not None:
            layers.setdefault(int(layer), {})[int(step)] = float(value)
            if module:
                module_paths[int(layer)] = str(module)
            continue
        by_metric.setdefault(str(name), {})[int(step)] = float(value)
    return (
        {name: sorted(v.items()) for name, v in by_metric.items()},
        {k: sorted(v.items()) for k, v in layers.items()},
        module_paths,
    )


def _analyze(root: Path, run_id: str):
    """Run detectors and the incident engine over a stored run."""
    from ..detectors import DetectorConfig, DetectorRunner
    from ..incidents import IncidentEngine

    reader = _reader(root)
    try:
        coverage = reader.coverage(run_id)
        by_metric, layer_series, module_paths = _series(reader, run_id)
        events_table = reader.events(run_id)
        events = _events_from_table(events_table)
        runner = DetectorRunner(DetectorConfig())
        detection = runner.run(
            run_id,
            by_metric,
            rank=None,
            layer_series=layer_series,
            module_paths=module_paths,
            population_layers=len(layer_series),
            population_ranks=max(1, len(coverage.ranks)),
            coverage=coverage.to_dict(),
        )
        incidents = IncidentEngine().build(
            run_id, detection, events, by_metric, layer_series=layer_series
        )
        return reader, coverage, detection, incidents
    except Exception:
        reader.close()
        raise


def _events_from_table(table) -> List[Any]:
    from ..schema.event import TrainingEvent

    events: List[TrainingEvent] = []
    for row in table.to_pylist():
        payload = {k: v for k, v in row.items()}
        if payload.get("metadata"):
            try:
                payload["metadata"] = json.loads(payload["metadata"])
            except (TypeError, ValueError):
                payload["metadata"] = {}
        try:
            events.append(TrainingEvent.from_dict(payload, on_unknown="drop"))
        except Exception:
            continue
    return events


def _emit(payload: Any, as_json: bool, text_renderer=None) -> int:
    if as_json:
        print(json.dumps(payload, indent=2, default=str, allow_nan=True))
    else:
        if text_renderer is not None:
            print(text_renderer(payload))
        else:
            print(payload)
    return 0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_runs(args: argparse.Namespace) -> int:
    reader = _reader(Path(args.root))
    try:
        runs = reader.list_runs()
        payload = []
        for run_id in runs:
            coverage = reader.coverage(run_id)
            payload.append({"run_id": run_id, **coverage.to_dict()})
    finally:
        reader.close()

    def render(items: Sequence[Dict[str, Any]]) -> str:
        if not items:
            return "no runs found in {}".format(args.root)
        lines = ["{:<32} {:>10} {:>10} {:>10} {:>8}".format(
            "run_id", "rows", "files", "steps", "skipped")]
        for item in items:
            lines.append("{:<32} {:>10} {:>10} {:>10} {:>8}".format(
                item["run_id"], item["row_count"], item["files_total"],
                "{}..{}".format(item["step_min"], item["step_max"]), item["files_skipped"]))
        return "\n".join(lines)

    return _emit(payload, args.json, render)


def cmd_incidents(args: argparse.Namespace) -> int:
    from ..incidents import render_incidents_table

    reader, coverage, detection, incidents = _analyze(Path(args.root), args.run_id)
    try:
        if args.json:
            payload = {
                "run_id": args.run_id,
                "coverage": coverage.to_dict(),
                "incidents": [i.to_dict() for i in incidents],
                "detector_errors": detection.detector_errors,
                "series_examined": detection.series_examined,
            }
            return _emit(payload, True)
        return _emit(incidents, False, render_incidents_table)
    finally:
        reader.close()


def cmd_incident(args: argparse.Namespace) -> int:
    from ..incidents import render_incident

    reader, coverage, detection, incidents = _analyze(Path(args.root), args.run_id)
    try:
        match = [i for i in incidents if i.incident_id == args.incident_id]
        if not match:
            print("no incident {} in run {}".format(args.incident_id, args.run_id), file=sys.stderr)
            return 2
        incident = match[0]
        if args.json:
            return _emit({"incident": incident.to_dict()}, True)
        return _emit(incident, False, render_incident)
    finally:
        reader.close()


def cmd_metrics(args: argparse.Namespace) -> int:
    reader = _reader(Path(args.root))
    try:
        names = args.metric or None
        table = reader.metrics(
            args.run_id, names,
            step_range=(args.from_step, args.to_step) if args.from_step is not None else None,
            limit=args.limit,
        )
        rows = table.to_pylist()
        if args.json:
            return _emit({"run_id": args.run_id, "rows": rows}, True)
        if not rows:
            print("no metrics matched")
            return 0
        lines = ["{:<28} {:>8} {:>16} {:<16}".format("metric_name", "step", "value", "unit")]
        for row in rows:
            name = row.get("metric_name")
            if name is None:
                continue
            lines.append("{:<28} {:>8} {:>16.6g} {:<16}".format(
                name, row.get("step"), row.get("metric_value") or float("nan"),
                row.get("metric_unit") or ""))
        return _emit(None, False, lambda _: "\n".join(lines))
    finally:
        reader.close()


def _run_verdicts(root: Path, run_id: str):
    """Recompute episodes and fusion verdicts for a stored run.

    The pipeline up to a verdict is deterministic given the events and the frozen
    rules, so a verdict can always be re-derived. Episodes are additionally read
    from the episode store when one exists, so a restart does not change the answer.
    """
    from ..detectors import DetectorConfig, DetectorRunner
    from ..episodes import EpisodeEngine
    from ..fusion import fuse
    from ..registry import detector_registry

    reader = _reader(root)
    try:
        coverage = reader.coverage(run_id)
        by_metric, layer_series, module_paths = _series(reader, run_id)
        table = reader.events(run_id)
        events = _events_from_table(table)
        runner = DetectorRunner(DetectorConfig())
        detection = runner.run(
            run_id, by_metric, rank=None, layer_series=layer_series,
            module_paths=module_paths, population_layers=len(layer_series),
            population_ranks=max(1, len(coverage.ranks)), coverage=coverage.to_dict(),
        )
        engine = EpisodeEngine(
            enabled_detectors=detector_registry()["enabled_detector_ids"]
        )
        engine.consume(detection.anomalies, run_id)
        engine.finish(run_id, final_step=coverage.step_max or 0)
        episodes = engine.all_episodes()
        return reader, detection, episodes, fuse(run_id, episodes), events
    except Exception:
        reader.close()
        raise


def _load_persisted(root: Path, run_id: str):
    """P1: prefer persisted evidence over recomputation.

    Returns ``(episodes, verdicts, relations, source)``. The store is the episode and
    incident truth, so a CLI reading it reports what was actually recorded rather
    than what the current code would decide if re-run.
    """
    from ..episodes import AnomalyEpisode
    from ..fusion import fuse
    from ..persistence import EpisodeStore

    store = EpisodeStore(root)
    history = set()
    try:
        directory = root / "runs" / run_id / "episodes" / "episodes"
        if not directory.exists():
            return None
        ids = set()
        for path in sorted(directory.glob("*.parquet")):
            import pyarrow.parquet as pq

            for row in pq.read_table(str(path)).to_pylist():
                ids.add(row["episode_id"])
        for episode_id in sorted(ids):
            history.add(episode_id)
        if not history:
            return None
        episodes = [store.load(run_id, episode_id) for episode_id in sorted(history)]
        relations = store.load_relations(run_id)
        return episodes, fuse(run_id, episodes), relations, "persisted"
    except Exception:
        return None


def cmd_verdicts(args: argparse.Namespace) -> int:
    """P1: render every fusion verdict, including the abstentions, from the store."""
    from ..incidents import render_verdict, render_verdicts

    root = Path(args.root)
    persisted = _load_persisted(root, args.run_id)
    if persisted is not None:
        episodes, verdicts, relations, source = persisted
        if args.json:
            return _emit(
                {
                    "run_id": args.run_id,
                    "source": source,
                    "verdicts": [v.to_dict() for v in verdicts],
                    "episodes": [e.to_dict() for e in episodes],
                    "relations": [r.to_dict() for r in relations],
                },
                True,
            )
        print("# source: {} episode store".format(source))
        print("# run_id: {}".format(args.run_id))
        blocks = []
        for verdict in verdicts:
            text = render_verdict(verdict, episodes)
            linked = [
                r for r in relations
                if r.source_incident_id in set(verdict.episodes_used)
            ]
            for relation in linked:
                text += (
                    "\n\nRelation:\n  {}\n\nLinked incident:\n  {}\n\n"
                    "This relation is temporal/evidential and does not establish "
                    "causation.".format(relation.relation_type, relation.target_incident_id)
                )
            blocks.append(text)
        print(("\n" + "=" * 72 + "\n").join(blocks))
        return 0

    reader, detection, episodes, verdicts, _ = _run_verdicts(root, args.run_id)
    try:
        if args.json:
            return _emit(
                {
                    "run_id": args.run_id,
                    "verdicts": [v.to_dict() for v in verdicts],
                    "episodes": [e.to_dict() for e in episodes],
                    "detector_errors": detection.detector_errors,
                },
                True,
            )
        print(render_verdicts(verdicts, episodes))
        if detection.detector_errors:
            print("\ndetector errors: {}".format(detection.detector_errors))
        return 0
    finally:
        reader.close()


def _series_for(root: Path, run_id: str) -> Dict[str, List[Tuple[int, float]]]:
    """Load one run's metric series. Raises if the run cannot be read."""
    reader = _reader(root)
    try:
        table = reader.events(run_id)
        per_metric: Dict[str, List[Tuple[int, float]]] = {}
        for name, step, value in zip(
            table.column("metric_name").to_pylist(),
            table.column("step").to_pylist(),
            table.column("metric_value").to_pylist(),
        ):
            if name is None or step is None or value is None:
                continue
            per_metric.setdefault(str(name), []).append((int(step), float(value)))
        return {k: sorted(v) for k, v in per_metric.items()}
    finally:
        reader.close()


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan a batch of runs and report which one disagrees with the others."""
    from ..comparison import find_outlier, render_consensus

    root = Path(args.root)
    try:
        series = {run_id: _series_for(root, run_id) for run_id in args.run_ids}
    except Exception as exc:
        print("scan failed: {}".format(exc), file=sys.stderr)
        return 1
    report = find_outlier(
        series,
        baseline_window=args.baseline_window,
        z_threshold=args.z_threshold,
        persistence=args.persistence,
        include_non_deterministic=args.include_non_deterministic,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(render_consensus(report))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Localize the first step at which two runs stop agreeing."""
    from ..comparison import localize_divergence, render_divergence
    from ..storage import DuckDBReader

    root = Path(args.root)
    try:
        series = {run_id: _series_for(root, run_id)
                  for run_id in (args.run_a, args.run_b)}
    except Exception as exc:
        print("comparison failed: {}".format(exc), file=sys.stderr)
        return 1

    report = localize_divergence(
        args.run_a, args.run_b, series[args.run_a], series[args.run_b],
        baseline_window=args.baseline_window,
        z_threshold=args.z_threshold,
        persistence=args.persistence,
        include_non_deterministic=args.include_non_deterministic,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(render_divergence(report))
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Deterministic diagnosis. Not an agent -- a renderer over the incident."""
    from ..incidents import render_incident

    reader, coverage, detection, incidents = _analyze(Path(args.root), args.run_id)
    try:
        if not incidents:
            # J4: an abstention must be visible, not silent. Fall through to the
            # verdict renderer so a candidate or an insufficient-evidence outcome is
            # shown rather than a bare "no incident".
            from ..incidents import render_verdicts as _render_verdicts

            print("No confirmed incident in run {}".format(args.run_id))
            print("Series examined: {}".format(len(detection.series_examined)))
            print("Steps covered: {}..{}".format(coverage.step_min, coverage.step_max))
            print("")
            try:
                verdict_reader, _, episodes, verdicts, _ = _run_verdicts(
                    Path(args.root), args.run_id
                )
                try:
                    print(_render_verdicts(verdicts, episodes))
                finally:
                    verdict_reader.close()
            except Exception as exc:
                print("verdict reconstruction unavailable: {}".format(exc))
            print("")
            print("This is a statement about the data that exists, not a claim that "
                  "the run is healthy.")
            return 0
        selected = incidents
        if args.incident_id:
            selected = [i for i in incidents if i.incident_id == args.incident_id]
            if not selected:
                print("no incident {} in run {}".format(args.incident_id, args.run_id), file=sys.stderr)
                return 2
        if getattr(args, "against", None):
            from ..comparison import localize_divergence, render_divergence_diagnosis
            from ..storage import DuckDBReader

            try:
                reference = _series_for(Path(args.root), args.against)
                suspect = _series_for(Path(args.root), args.run_id)
            except Exception as exc:
                print("cannot compare against {}: {}".format(args.against, exc),
                      file=sys.stderr)
                return 1
            report = localize_divergence(args.against, args.run_id, reference, suspect)
            if args.json:
                return _emit({
                    "run_id": args.run_id,
                    "against": args.against,
                    "divergence": report.to_dict(),
                    "incidents": [i.to_dict() for i in selected],
                }, True)
            print(render_divergence_diagnosis(report, selected))
            return 0

        if args.json:
            return _emit({"run_id": args.run_id, "incidents": [i.to_dict() for i in selected]}, True)
        blocks = []
        for incident in selected:
            blocks.append(render_incident(incident))
        print(("\n" + "=" * 72 + "\n").join(blocks))
        return 0
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training-debugger",
        description=(
            "Training observability, incident detection and earliest-abnormal-evidence "
            "analysis. Phase 1: single process, Parquet storage."
        ),
    )
    parser.add_argument("--root", default=str(_default_root()),
                        help="dataset root (default: ./training-debugger-data)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    p_runs = sub.add_parser("runs", help="list runs in the dataset")
    p_runs.set_defaults(func=cmd_runs)

    def add_run_arg(p):
        p.add_argument("run_id", help="run identifier")

    p_incidents = sub.add_parser("incidents", help="list incidents for a run")
    add_run_arg(p_incidents)
    p_incidents.set_defaults(func=cmd_incidents)

    p_incident = sub.add_parser("incident", help="show one incident")
    add_run_arg(p_incident)
    p_incident.add_argument("incident_id", help="incident identifier")
    p_incident.set_defaults(func=cmd_incident)

    p_metrics = sub.add_parser("metrics", help="query metric rows")
    add_run_arg(p_metrics)
    p_metrics.add_argument("metric", nargs="*", help="metric name(s); omit for all")
    p_metrics.add_argument("--from-step", type=int, default=None)
    p_metrics.add_argument("--to-step", type=int, default=None)
    p_metrics.add_argument("--limit", type=int, default=200)
    p_metrics.set_defaults(func=cmd_metrics)

    p_verdicts = sub.add_parser(
        "verdicts", help="render all fusion verdicts, including abstentions"
    )
    add_run_arg(p_verdicts)
    p_verdicts.set_defaults(func=cmd_verdicts)

    p_scan = sub.add_parser(
        "scan",
        help="compare a batch of runs against each other and name the outlier",
    )
    p_scan.add_argument("run_ids", nargs="+", help="two or more run ids")
    p_scan.add_argument("--baseline-window", type=int, default=30)
    p_scan.add_argument("--z-threshold", type=float, default=8.0)
    p_scan.add_argument("--persistence", type=int, default=3)
    p_scan.add_argument("--include-nondeterministic", action="store_true",
                        dest="include_non_deterministic")
    p_scan.set_defaults(func=cmd_scan)

    p_compare = sub.add_parser(
        "compare",
        help="localize the first step at which two runs stop agreeing",
    )
    p_compare.add_argument("run_a", help="reference run id")
    p_compare.add_argument("run_b", help="suspect run id")
    p_compare.add_argument("--baseline-window", type=int, default=30,
                           help="steps of agreement used as the null model (default 30)")
    p_compare.add_argument("--z-threshold", type=float, default=8.0,
                           help="how far outside the agreement a difference must sit")
    p_compare.add_argument("--persistence", type=int, default=3,
                           help="consecutive steps required before it counts")
    p_compare.add_argument("--include-nondeterministic", action="store_true",
                           dest="include_non_deterministic",
                           help="also compare wall-clock/timing metrics, which are NOT "
                                "reproducible between runs")
    p_compare.set_defaults(func=cmd_compare)

    p_diagnose = sub.add_parser("diagnose", help="deterministic incident diagnosis")
    add_run_arg(p_diagnose)
    p_diagnose.add_argument("--incident-id", default=None)
    p_diagnose.add_argument(
        "--against", default=None, metavar="REFERENCE_RUN",
        help="also localize the first step at which this run diverged from a reference "
             "run, and date the incidents against that pivot",
    )
    p_diagnose.set_defaults(func=cmd_diagnose)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print("dataset not found: {}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
