"""Run the A0-A4 corpus through the detector suite.

A single harness is used for two purposes: the W5 regression test, and the
generation of the scenario report. Keeping them on one code path means the
report cannot describe a run different from the one the tests assert on.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from training_debugger.detectors import (  # noqa: E402
    DetectorConfig,
    DetectorRunner,
    suggest_incident_type,
)
from training_debugger.observer import (  # noqa: E402
    MemoryWriter,
    ObserverConfig,
    SamplingPolicy,
    TrainingObserver,
)
from training_debugger.schema import EventType  # noqa: E402

from scenarios.corpus import Scenario  # noqa: E402
import helpers  # noqa: E402


def generate(scenario: Scenario):
    """Drive the synthetic training loop and return (events, ground truth)."""
    writer = MemoryWriter()
    observer = TrainingObserver(
        "run-" + scenario.name,
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=512),
        rank=0,
        world_size=1,
    )
    truth = helpers.synthetic_training(
        observer,
        steps=scenario.steps,
        layer_count=scenario.layer_count,
        spike_at=scenario.spike_at,
        spike_factor=scenario.spike_factor,
        grad_explode_at=scenario.grad_explode_at,
        grad_explode_factor=scenario.grad_explode_factor,
        nan_at=scenario.nan_at,
        inf_at=scenario.inf_at,
        throughput_drop_at=scenario.throughput_drop_at,
        throughput_drop_factor=scenario.throughput_drop_factor,
    )
    # The scenario's own seed shifts the noise floor so scenarios are not all
    # identical before the injection point.
    observer.flush(timeout=20.0)
    observer.close()
    return writer.events, truth


def series_from_events(events) -> Tuple[Dict[str, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]], Dict[int, str]]:
    """Extract the series the detectors consume from an event stream."""
    by_metric: Dict[str, Dict[int, float]] = {}
    layers: Dict[int, Dict[int, float]] = {}
    module_paths: Dict[int, str] = {}

    for event in events:
        if event.metric_name is None or event.metric_value is None or event.step is None:
            continue
        if event.metric_name == "grad.layer_norm" and event.layer_index is not None:
            layers.setdefault(event.layer_index, {})[event.step] = event.metric_value
            if event.module_path:
                module_paths[event.layer_index] = event.module_path
            continue
        # Later events at the same step win; the adapter does not emit duplicates.
        by_metric.setdefault(event.metric_name, {})[event.step] = event.metric_value

    ordered = {
        name: sorted(values.items()) for name, values in by_metric.items()
    }
    ordered_layers = {
        index: sorted(values.items()) for index, values in layers.items()
    }
    return ordered, ordered_layers, module_paths


@dataclass
class ScenarioOutcome:
    scenario: Scenario
    result: Any
    fired_detectors: List[str]
    onsets: Dict[str, int]
    truth: Dict[str, Any]
    step_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.name,
            "description": self.scenario.description,
            "injection": self.scenario.injection_summary(),
            "fired_detectors": list(self.fired_detectors),
            "onsets": dict(self.onsets),
            "expected_detectors": list(self.scenario.expected_detectors),
            "forbidden_detectors": list(self.scenario.forbidden_detectors),
            "expected_onset_window": list(self.scenario.expected_onset_window or ()),
            "expected_incident_type": self.scenario.expected_incident_type,
            "multiplicity": self.result.multiplicity.to_dict(),
            "anomaly_count": len(self.result.anomalies),
            "suppressed": list(self.result.suppressed),
            "series_examined": len(self.result.series_examined),
            "step_count": self.step_count,
        }


def run_scenario(scenario: Scenario, config: Optional[DetectorConfig] = None) -> ScenarioOutcome:
    events, truth = generate(scenario)
    by_metric, layer_series, module_paths = series_from_events(events)
    steps = [e.step for e in events if e.step is not None]
    runner = DetectorRunner(config or DetectorConfig())
    result = runner.run(
        "run-" + scenario.name,
        by_metric,
        rank=0,
        layer_series=layer_series,
        module_paths=module_paths,
        population_layers=scenario.layer_count,
        population_ranks=1,
        coverage={
            "step_count": (max(steps) - min(steps) + 1) if steps else 0,
            "step_min": min(steps) if steps else None,
            "step_max": max(steps) if steps else None,
        },
    )
    fired = sorted({str(a.detector) for a in result.anomalies})
    onsets: Dict[str, int] = {}
    for anomaly in result.anomalies:
        key = str(anomaly.detector)
        if key not in onsets or anomaly.first_abnormal_step < onsets[key]:
            onsets[key] = anomaly.first_abnormal_step
    return ScenarioOutcome(
        scenario=scenario,
        result=result,
        fired_detectors=fired,
        onsets=onsets,
        truth=truth,
        step_count=(max(steps) - min(steps) + 1) if steps else 0,
    )
