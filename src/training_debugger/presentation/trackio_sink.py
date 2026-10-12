"""Experiment presentation sinks.

Phase 0's decision, restated because this module exists to enforce it:

    Parquet               = telemetry truth
    our incident store    = incident truth
    Trackio               = experiment presentation

Trackio must never become a detector input, an incident source of truth, or a
layer-telemetry database. That is not a convention here, it is a **closed
allow-list** (:data:`PRESENTED_METRICS`). A metric that is not on the list is not
forwarded, and a test asserts that no per-layer or per-event metric can reach a
sink even if someone asks for it.

Two further properties are non-negotiable:

**Fail-open.** Trackio unavailable, slow, throwing, or failing to initialise must
not affect training, the Parquet store, the detectors, or the incidents. Every
call into a sink is isolated and counted.

**No dual truth.** Deleting every Trackio artefact must leave the product fully
functional. A test asserts exactly that.

The sink interface is deliberately tiny, because everything in it must be
correctly isolated under failure.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

__all__ = [
    "PresentationSink",
    "NullSink",
    "RecordingSink",
    "TrackioSink",
    "PresentationBridge",
    "PRESENTED_METRICS",
    "PRESENTED_SUMMARY_KEYS",
    "is_presentable",
    "build_sink",
    "trackio_available",
]

#: The closed allow-list. Exactly the high-level signals the brief names.
PRESENTED_METRICS = frozenset(
    {
        "train.loss",
        "train.lr",
        "grad.global_norm",
        "param.global_norm",
        "train.step_time",
        "train.tokens_per_sec",
    }
)

#: Scalar summary keys presented for a run or an incident.
PRESENTED_SUMMARY_KEYS = frozenset(
    {
        "incident_count",
        "highest_incident_severity",
        "earliest_incident_step",
        "not_correlated_count",
    }
)

#: Metric-name prefixes that must NEVER reach a presentation sink, regardless of
#: what a caller asks for. Belt and braces on top of the allow-list, because a
#: per-layer metric slipping into a dashboard is exactly the dual-truth failure
#: this module exists to prevent.
FORBIDDEN_PREFIXES = ("grad.layer_norm", "param.layer_norm", "test.", "profile.", "event.")


def is_presentable(metric_name: str) -> bool:
    """Whether a metric may be forwarded to a presentation sink."""
    if metric_name not in PRESENTED_METRICS:
        return False
    for prefix in FORBIDDEN_PREFIXES:
        if metric_name.startswith(prefix):
            return False
    return True


@runtime_checkable
class PresentationSink(Protocol):
    """Minimal sink contract. Everything here must be safe to call under failure."""

    name: str

    def start(self, run_id: str, config: Dict[str, Any]) -> None: ...

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None: ...

    def log_summary(self, summary: Dict[str, Any]) -> None: ...

    def finish(self) -> None: ...


class NullSink:
    """Discards everything. The default when no sink is configured, and the
    fallback when Trackio cannot be used.

    Having an explicit null sink rather than ``sink=None`` keeps the bridge's code
    path identical whether or not a destination exists, which is what the
    fail-open tests need to exercise.
    """

    name = "null"

    def __init__(self) -> None:
        self.calls = 0

    def start(self, run_id: str, config: Dict[str, Any]) -> None:
        self.calls += 1

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        self.calls += 1

    def log_summary(self, summary: Dict[str, Any]) -> None:
        self.calls += 1

    def finish(self) -> None:
        self.calls += 1


class RecordingSink:
    """Keeps what it was given, so a test can assert on precisely what was forwarded."""

    name = "recording"

    def __init__(self) -> None:
        self.started: List[Dict[str, Any]] = []
        self.metrics: List[Dict[str, Any]] = []
        self.summaries: List[Dict[str, Any]] = []
        self.finished = 0

    def start(self, run_id: str, config: Dict[str, Any]) -> None:
        self.started.append({"run_id": run_id, "config": dict(config)})

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        self.metrics.append({"step": step, "metrics": dict(metrics)})

    def log_summary(self, summary: Dict[str, Any]) -> None:
        self.summaries.append(dict(summary))

    def finish(self) -> None:
        self.finished += 1

    def metric_names(self) -> set:
        names = set()
        for entry in self.metrics:
            names.update(entry["metrics"])
        return names


class FaultySink:
    """Raises from every method. Drives the isolation path without mocking."""

    def __init__(self, name: str = "faulty") -> None:
        self.name = name
        self.calls = 0

    def start(self, run_id: str, config: Dict[str, Any]) -> None:
        self.calls += 1
        raise RuntimeError("synthetic sink start failure")

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        self.calls += 1
        raise RuntimeError("synthetic sink metric failure")

    def log_summary(self, summary: Dict[str, Any]) -> None:
        self.calls += 1
        raise RuntimeError("synthetic sink summary failure")

    def finish(self) -> None:
        self.calls += 1
        raise RuntimeError("synthetic sink finish failure")


class SlowSink:
    """Blocks for ``delay_s`` per call, simulating a slow remote."""

    name = "slow"

    def __init__(self, delay_s: float = 0.2) -> None:
        self.delay_s = delay_s
        self.calls = 0

    def start(self, run_id: str, config: Dict[str, Any]) -> None:
        self.calls += 1
        time.sleep(self.delay_s)

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        self.calls += 1
        time.sleep(self.delay_s)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        self.calls += 1
        time.sleep(self.delay_s)

    def finish(self) -> None:
        self.calls += 1


class TrackioSink:
    """Real Trackio, imported lazily and optionally.

    ``trackio`` requires Python >= 3.10 (0.40.0 declares ``requires_python
    >=3.10``). This project's primary environment is 3.9, so the import is
    attempted at construction and a failure is reported through
    :attr:`available` rather than raised -- which makes "Trackio is not installed"
    a first-class, tested state instead of an import error.
    """

    name = "trackio"

    def __init__(
        self,
        *,
        project: str = "training-debugger",
        run_name: Optional[str] = None,
        module: Any = None,
        settings: Any = None,
        directory: Optional[str] = None,
    ) -> None:
        # NOTE ON THE REAL API (verified by running trackio 0.40.0, not by reading
        # its README): ``init`` has NO ``mode`` parameter, and ``log`` takes the
        # step as a keyword rather than inside the metrics dict. Phase 0's notes
        # recorded a ``mode="local"`` argument; that was wrong for this version and
        # the mistake was only discoverable by executing the library. Trackio is
        # local-first by default, so no mode selection is needed.
        self.project = project
        self.run_name = run_name
        self.settings = settings
        self.directory = directory
        # ``trackio`` resolves ``TRACKIO_DIR`` from the environment *at import
        # time* (``trackio/utils.py``: ``TRACKIO_DIR = _get_trackio_dir()``), so it
        # must be set before the import below. Without this, Trackio writes to
        # ``$HF_HOME/trackio`` and a restricted environment silently fails to
        # persist anything -- which is what happened on the first real
        # verification run.
        if directory and not os.environ.get("TRACKIO_DIR"):
            os.environ["TRACKIO_DIR"] = str(directory)
        self.available = False
        self.import_error: Optional[str] = None
        self._trackio = module
        self._run = None
        self.events_sent = 0

        if self._trackio is None:
            try:
                import trackio  # type: ignore

                self._trackio = trackio
            except Exception as exc:
                self.import_error = "{}: {}".format(type(exc).__name__, exc)
                return
        self.available = True

    def start(self, run_id: str, config: Dict[str, Any]) -> None:
        if not self.available:
            raise RuntimeError("trackio unavailable: {}".format(self.import_error))
        kwargs: Dict[str, Any] = {
            "project": self.project,
            "name": self.run_name or run_id,
            "config": dict(config),
        }
        if self.settings is not None:
            kwargs["settings"] = self.settings
        self._run = self._trackio.init(**kwargs)

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        if self._run is None:
            raise RuntimeError("trackio run not started")
        # Real signature: ``log(metrics, step=None)``.
        self._trackio.log(dict(metrics), step=step)
        self.events_sent += len(metrics)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        if self._run is None:
            raise RuntimeError("trackio run not started")
        self._trackio.log(dict(summary))

    def finish(self) -> None:
        if not self.available:
            return
        try:
            self._trackio.finish()
        finally:
            self._run = None


def trackio_available() -> Dict[str, Any]:
    """Probe without importing the caller's process into a failure state."""
    try:
        import trackio  # type: ignore

        return {"available": True, "version": getattr(trackio, "__version__", "unknown")}
    except Exception as exc:
        return {"available": False, "error": "{}: {}".format(type(exc).__name__, exc)}


def build_sink(
    prefer: str = "trackio",
    *,
    recording: Optional[RecordingSink] = None,
    **kwargs: Any
) -> PresentationSink:
    """Return a usable sink, falling back to :class:`NullSink`.

    A construction failure is not fatal: the caller gets a null sink and the
    reason is available via :meth:`PresentationBridge.health`.
    """
    if prefer == "recording" and recording is not None:
        return recording
    if prefer == "none":
        return NullSink()
    if prefer == "trackio":
        sink = TrackioSink(**kwargs)
        if sink.available:
            return sink
        return NullSink()
    return NullSink()


@dataclass
class PresentationBridge:
    """Forwards a *filtered* subset of metrics to a sink, isolating all failure.

    The bridge is the only component that talks to a presentation sink. It owns:

    * the allow-list filter (presentation-only, enforced in one place);
    * batching, so a slow remote costs one call per interval rather than per metric;
    * failure isolation and counters, so a broken dashboard is visible without
      being fatal.
    """

    sink: PresentationSink
    flush_interval_s: float = 2.0
    #: Maximum metrics held before a forced flush. Bounded, because an unreachable
    #: sink must not grow an unbounded buffer inside the training process.
    max_buffer: int = 512

    started: int = 0
    metrics_forwarded: int = 0
    metrics_filtered_out: int = 0
    summaries_forwarded: int = 0
    failures: int = 0
    buffer_overflows: int = 0
    last_error: Optional[str] = None
    last_flush_monotonic: float = field(default_factory=time.monotonic)

    _buffer: Dict[int, Dict[str, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------

    def start(self, run_id: str, config: Dict[str, Any]) -> bool:
        """Forward the run configuration. Returns True on success."""
        try:
            self.sink.start(run_id, self._safe_config(config))
            self.started += 1
            return True
        except Exception as exc:
            self._note_failure("start", exc)
            return False

    @staticmethod
    def _safe_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Only JSON-scalar config values. A dashboard must not receive the model."""
        out: Dict[str, Any] = {}
        for key, value in (config or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[str(key)] = value
        return out

    def observe_metric(
        self, metric_name: Optional[str], value: Optional[float], step: Optional[int]
    ) -> bool:
        """Offer a metric. Returns True if it was accepted for presentation.

        Called from the observer's hot path, so it must be cheap and must never
        raise. Filtering happens before any lock is taken.
        """
        if metric_name is None or value is None or step is None:
            return False
        if not is_presentable(metric_name):
            self.metrics_filtered_out += 1
            return False
        try:
            with self._lock:
                bucket = self._buffer.setdefault(int(step), {})
                bucket[metric_name] = float(value)
                size = sum(len(v) for v in self._buffer.values())
                overflow = size > self.max_buffer
                if overflow:
                    self._buffer.clear()
                    self.buffer_overflows += 1
            if overflow or (time.monotonic() - self.last_flush_monotonic) >= self.flush_interval_s:
                self.flush()
            return True
        except Exception as exc:  # pragma: no cover - fatal only if a lock breaks
            self._note_failure("observe_metric", exc)
            return False

    def flush(self) -> bool:
        """Push buffered metrics to the sink. Returns True if nothing failed."""
        try:
            with self._lock:
                pending = self._buffer
                self._buffer = {}
                self.last_flush_monotonic = time.monotonic()
            ok = True
            for step in sorted(pending):
                try:
                    self.sink.log_metrics(step, pending[step])
                    self.metrics_forwarded += len(pending[step])
                except Exception as exc:
                    self._note_failure("log_metrics", exc)
                    ok = False
            return ok
        except Exception as exc:  # pragma: no cover
            self._note_failure("flush", exc)
            return False

    def summarise_incidents(self, incidents: Sequence[Any]) -> bool:
        """Forward incident *counts and severity*, never incidents themselves.

        The incident store remains the source of truth; a dashboard showing
        incident counts is presentation, a dashboard holding incidents would be a
        second copy of the truth.
        """
        ranks = {"debug": 0, "info": 1, "warn": 2, "error": 3, "fatal": 4}
        highest = None
        for incident in incidents:
            name = str(getattr(incident, "severity", "")).lower()
            if name in ranks and (highest is None or ranks[name] > ranks[highest]):
                highest = name
        from typing import Any as _Any  # local import keeps the module import list short

        summary: Dict[str, _Any] = {
            "incident_count": len(incidents),
            "highest_incident_severity": highest,
            "earliest_incident_step": min(
                (i.detected_step for i in incidents if i.detected_step is not None), default=None
            ),
            "not_correlated_count": sum(
                len(i.evidence.not_correlated) for i in incidents
            ),
        }
        summary = {k: v for k, v in summary.items() if k in PRESENTED_SUMMARY_KEYS}
        try:
            self.sink.log_summary(summary)
            self.summaries_forwarded += 1
            return True
        except Exception as exc:
            self._note_failure("log_summary", exc)
            return False

    def finish(self) -> bool:
        self.flush()
        try:
            self.sink.finish()
            return True
        except Exception as exc:
            self._note_failure("finish", exc)
            return False

    # ------------------------------------------------------------------

    def _note_failure(self, where: str, exc: BaseException) -> None:
        """Counted, never raised. A dashboard failure is not a training failure."""
        self.failures += 1
        self.last_error = "{}: {}: {}".format(where, type(exc).__name__, exc)

    def health(self) -> Dict[str, Any]:
        return {
            "sink": getattr(self.sink, "name", "unknown"),
            "started": self.started,
            "metrics_forwarded": self.metrics_forwarded,
            "metrics_filtered_out": self.metrics_filtered_out,
            "summaries_forwarded": self.summaries_forwarded,
            "failures": self.failures,
            "buffer_overflows": self.buffer_overflows,
            "last_error": self.last_error,
            "allow_list": sorted(PRESENTED_METRICS),
        }
