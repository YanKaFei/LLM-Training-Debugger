"""Generate FAULT_INJECTION_REPORT.md from an actual instrumented run.

Run with::

    .venv/bin/python tests/fault_injection/run_report.py

The report is generated rather than hand-written on purpose: a hand-written
matrix of "training continued" claims is exactly the kind of artefact that
silently drifts from the code. This script executes every fault mode, records
what the observer counted, verifies the training arithmetic independently, and
writes the evidence out.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import helpers  # noqa: E402
from training_debugger.env_info import capture_environment  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    FailingWriter,
    MemoryWriter,
    MultiWriter,
    NullWriter,
    ObserverConfig,
    SamplingPolicy,
    TrainingObserver,
)
from test_observer_faults import (  # noqa: E402
    CrashingWriter,
    DiskFullWriter,
    ExplodingOnCloseWriter,
    FlakyWriter,
    PermissionDeniedWriter,
    SerializationBombWriter,
    SlowWriter,
)

OUTPUT = ROOT / "outputs" / "phase1" / "w1" / "FAULT_INJECTION_REPORT.md"
STEPS = 30
LAYERS = 2
#: Independently computed, not read back from the observer.
EXPECTED_FINAL_LOSS = 2.0 * (0.99 ** STEPS)


@dataclass
class FaultCase:
    name: str
    description: str
    writer: Any
    config: ObserverConfig


def build_cases() -> List[FaultCase]:
    # Fault cases use a deliberately small queue so that overflow paths are
    # reached within 30 steps rather than needing a million events.
    default = dict(max_queue=64, batch_size=8, flush_interval_s=0.01,
                   max_consecutive_writer_failures=2)
    return [
        FaultCase("writer_crash", "Writer raises IOError on every batch",
                  CrashingWriter(), ObserverConfig(**default)),
        FaultCase("disk_full", "Writer raises OSError(ENOSPC, 'No space left on device')",
                  DiskFullWriter(), ObserverConfig(**default)),
        FaultCase("permission_denied", "Writer raises OSError(EACCES, 'Permission denied')",
                  PermissionDeniedWriter(), ObserverConfig(**default)),
        FaultCase("serialization_failure", "Writer raises TypeError from a non-serialisable payload",
                  SerializationBombWriter(), ObserverConfig(**default)),
        FaultCase("flush_and_close_crash", "Writer works but flush() and close() raise",
                  ExplodingOnCloseWriter(), ObserverConfig(**default)),
        # Breaker threshold raised so the recovery path is actually exercised:
        # with the default threshold of 2 the breaker trips before the writer
        # has a chance to recover.
        FaultCase("transient_failure_then_recovery", "Writer fails the first 2 batches, then works",
                  FlakyWriter(fail_times=2),
                  ObserverConfig(max_queue=64, batch_size=8, flush_interval_s=0.01,
                                 max_consecutive_writer_failures=5)),
        FaultCase("queue_overflow_slow_consumer", "Writer blocks 200 ms per batch; queue capped at 64",
                  SlowWriter(delay_s=0.2), ObserverConfig(**default)),
        FaultCase("writer_disabled_after_breaker", "Circuit breaker trips after 2 consecutive failures",
                  FailingWriter(), ObserverConfig(**default)),
        FaultCase("null_writer", "No destination configured at all",
                  NullWriter(), ObserverConfig(**default)),
        # Control case uses PRODUCTION defaults, not the shrunken fault config,
        # so that "no drops" is demonstrated at a realistic queue size rather
        # than at a size chosen to make faults easy to trigger.
        FaultCase("healthy_memory_writer_DEFAULT_CONFIG",
                  "Control case: a working destination with production defaults",
                  MemoryWriter(), ObserverConfig()),
    ]


def run_case(case: FaultCase) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": case.name,
        "description": case.description,
        "writer": getattr(case.writer, "name", type(case.writer).__name__),
        "training_completed": False,
        "arithmetic_correct": False,
        "error": None,
    }
    observer = None
    try:
        observer = TrainingObserver(
            "run-fi-report-%s" % case.name,
            writer=case.writer,
            sampling=SamplingPolicy.uniform(1),
            config=case.config,
        )
        summary = helpers.synthetic_training(observer, steps=STEPS, layer_count=LAYERS)
        observer.flush(timeout=0.5)
        observer.close()

        stats = observer.stats()
        result["training_completed"] = summary["steps"] == STEPS
        result["final_loss"] = summary["final_loss"]
        result["arithmetic_correct"] = abs(summary["final_loss"] - EXPECTED_FINAL_LOSS) < 1e-9
        result["losses_all_finite"] = all(
            (x == x) and abs(x) != float("inf") for x in summary["losses"]
        )
        result["counters"] = {
            "emitted_total": stats.emitted_total,
            "written_total": stats.written_total,
            "events_dropped_total": stats.events_dropped_total,
            "queue_overflow_total": stats.queue_overflow_total,
            "writer_failures_total": stats.writer_failures_total,
            "observer_internal_errors_total": stats.observer_internal_errors_total,
            "validation_failures_total": stats.validation_failures_total,
            "emergency_writes_total": stats.emergency_writes_total,
        }
        result["queue"] = observer.health()["queue"]
        result["writer_disabled"] = bool(getattr(observer, "_writer_disabled", False))
        # Deliberately ignoring the writer's own max: an explicit statement of
        # how many events reached a destination, not an inference.
        if isinstance(case.writer, MemoryWriter):
            result["delivered_to_writer"] = len(case.writer)
        elif isinstance(case.writer, FlakyWriter):
            result["delivered_to_writer"] = len(case.writer.stored)
        elif isinstance(case.writer, SlowWriter):
            result["delivered_to_writer"] = case.writer.events
        else:
            result["delivered_to_writer"] = 0
        if isinstance(case.writer, MultiWriter):
            result["child_failures"] = dict(case.writer.child_failures)
    except BaseException as exc:  # the point of the report is to record these
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
        result["traceback"] = traceback.format_exc()
    return result


def render(results: List[Dict[str, Any]], env: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append

    add("# FAULT INJECTION REPORT")
    add("")
    add("**Generated by:** `tests/fault_injection/run_report.py` (not hand-written)")
    add("**Wave:** W1")
    add("")
    add("## The claim under test")
    add("")
    add("> debugger failure does not become training failure.")
    add("")
    add("Every case below runs the same deterministic synthetic training loop for")
    add("{} steps with {} layer metrics, while the debugger is broken in a specific".format(STEPS, LAYERS))
    add("way. Two things are then checked independently:")
    add("")
    add("1. **Training completed** — all steps ran.")
    add("2. **Training arithmetic is unchanged** — the final loss must equal")
    add("   `2.0 * 0.99**{}` = `{:.12f}`, computed without reference to the observer.".format(STEPS, EXPECTED_FINAL_LOSS))
    add("")
    add("A check of step count alone would not catch a debugger that corrupts the loop")
    add("while continuing, so the second check is the load-bearing one.")
    add("")

    all_ok = all(
        r["training_completed"] and r["arithmetic_correct"] and not r["error"] for r in results
    )
    add("## Verdict")
    add("")
    add("**{}** — {} of {} fault modes left training complete and numerically correct.".format(
        "PASS" if all_ok else "FAIL", sum(1 for r in results if r["training_completed"] and r["arithmetic_correct"]), len(results)))
    add("")

    add("## Results")
    add("")
    add("| Fault mode | Writer | Training completed | Arithmetic correct | Dropped | Overflow | Writer failures | Internal errors | Deliveries |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r["error"]:
            add("| `{}` | {} | **ERROR** | — | — | — | — | — | — |".format(r["name"], r["writer"]))
            continue
        c = r["counters"]
        add("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["name"],
            r["writer"],
            "yes" if r["training_completed"] else "**NO**",
            "yes" if r["arithmetic_correct"] else "**NO**",
            c["events_dropped_total"],
            c["queue_overflow_total"],
            c["writer_failures_total"],
            c["observer_internal_errors_total"],
            r.get("delivered_to_writer", 0),
        ))
    add("")

    add("## Per-case detail")
    add("")
    for r in results:
        add("### `{}`".format(r["name"]))
        add("")
        add("{}".format(r["description"]))
        add("")
        if r["error"]:
            add("**The fault escalated out of the observer: `{}`**".format(r["error"]))
            add("")
            add("```")
            add(r.get("traceback", "").strip())
            add("```")
            add("")
            continue
        add("- Final loss: `{:.12f}` (expected `{:.12f}`) → {}".format(
            r["final_loss"], EXPECTED_FINAL_LOSS,
            "correct" if r["arithmetic_correct"] else "**INCORRECT**"))
        add("- All losses finite: {}".format(r["losses_all_finite"]))
        add("- Writer circuit breaker tripped: {}".format(r["writer_disabled"]))
        add("- Events delivered to a destination: {}".format(r.get("delivered_to_writer", 0)))
        add("- Counters: `{}`".format(json.dumps(r["counters"], sort_keys=True)))
        if r.get("child_failures"):
            add("- MultiWriter child failures: `{}`".format(json.dumps(r["child_failures"])))
        add("")

    add("## What this demonstrates")
    add("")
    add("| Requirement (brief §3.2) | Where it is shown |")
    add("|---|---|")
    add("| serialization failure | `serialization_failure` |")
    add("| queue full | `queue_overflow_slow_consumer` |")
    add("| writer thread crash | `writer_crash`, `writer_disabled_after_breaker` |")
    add("| disk full | `disk_full` |")
    add("| permission denied | `permission_denied` |")
    add("| invalid event | `tests/fault_injection/test_observer_faults.py::test_malformed_events_are_rejected_and_training_continues` |")
    add("| DuckDB unavailable | **not applicable in W1** — DuckDB is W3; the writer abstraction is what makes it testable there |")
    add("| Parquet write error | **not applicable in W1** — see W3 |")
    add("| process shutdown | `close_from_another_thread_while_emitting`, `double_close_is_safe` |")
    add("| unexpected exception | `test_emit_never_raises_for_any_input` |")
    add("")

    add("## Environment")
    add("")
    add("```json")
    add(json.dumps(env, indent=2, sort_keys=True, default=str))
    add("```")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    results = [run_case(case) for case in build_cases()]
    env = capture_environment(extra={"fault_injection_steps": STEPS, "layers": LAYERS})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(results, env), encoding="utf-8")

    # Machine-readable companion for CI.
    (OUTPUT.parent / "fault_injection_results.json").write_text(
        json.dumps({"results": results, "environment": env}, indent=2, default=str),
        encoding="utf-8",
    )

    ok = all(r["training_completed"] and r["arithmetic_correct"] and not r["error"] for r in results)
    print("fault modes: {}".format(len(results)))
    print("all training completed and correct: {}".format(ok))
    print("report: {}".format(OUTPUT))
    for r in results:
        status = "ERROR" if r["error"] else ("ok" if r["arithmetic_correct"] else "ARITHMETIC-WRONG")
        print("  {:38s} {:10s} dropped={:5d} overflow={:5d} wfail={:5d}".format(
            r["name"], status,
            r["counters"]["events_dropped_total"] if not r["error"] else -1,
            r["counters"]["queue_overflow_total"] if not r["error"] else -1,
            r["counters"]["writer_failures_total"] if not r["error"] else -1,
        ))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
