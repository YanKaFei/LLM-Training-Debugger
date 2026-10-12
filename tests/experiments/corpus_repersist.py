"""W3: re-persist the development corpus under stable, reproducible run ids.

The existing development store holds only **14 of 169** runs and **0 relations**,
because a `run_id` defect wrote episodes under a directory their own identifier could
not find, and because relations did not exist when it was written.

This rebuilds a corpus with:

    stable id      {split}-{scenario}-L{layers}d{d_model}-s{seed}
    every artefact TrainingEvents, AnomalyEpisodes, AnalysisState, Incidents,
                   IncidentRelations
    isolated truth ground truth goes to a SEPARATE tree, never into the run directory

Run ids are a pure function of the scenario, so re-running reproduces them exactly.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
from artifacts import write_artifact  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tiny_model import make_tiny_lm, synthetic_batch  # noqa: E402

from training_debugger.adapters.pytorch import (  # noqa: E402
    LayerMetricCollector, LayerSelection, PyTorchTrainingAdapter,
)
from training_debugger.detectors import DetectorConfig, DetectorRunner  # noqa: E402
from training_debugger.episodes import RULE_VERSION, EpisodeEngine  # noqa: E402
from training_debugger.fusion import FUSION_RULE_VERSION, fuse  # noqa: E402
from training_debugger.fusion.quality import DataQuality  # noqa: E402
from training_debugger.incidents.relations import (  # noqa: E402
    RELATION_RULE_VERSION, link_consequences,
)
from training_debugger.observer import (  # noqa: E402
    MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver,
)
from training_debugger.persistence import (  # noqa: E402
    AnalysisStateStore, EpisodeStore, IncidentStore, capture_analysis_state,
)
from training_debugger.registry import detector_registry  # noqa: E402

OUT = ROOT / "outputs" / "alpha" / "corpus"
STORE = OUT / "store"
TRUTH = OUT / "ground_truth"        # deliberately OUTSIDE the run tree
STEPS = 200
INJECTION = 110

#: (scenario, kind, magnitude)
SCENARIOS: Tuple[Tuple[str, str, Optional[float]], ...] = (
    ("healthy", "healthy", None),
    ("loss_spike", "fault", 20.0),
    ("grad_small", "fault", 2.0),
    ("grad_mid", "fault", 5.0),
    ("grad_large", "fault", 20.0),
    ("grad_extreme", "fault", 40.0),
    ("nan_loss", "fault", -1.0),          # -1 marks the NaN scenario
)
CONFIGS = ((4, 32),)
SEEDS = tuple(range(80000, 80006))


def run_id_for(scenario: str, layers: int, d_model: int, seed: int) -> str:
    """Stable and reproducible: a pure function of the scenario."""
    return "dev-{}-L{}d{}-s{}".format(scenario, layers, d_model, seed)


def produce(scenario: str, magnitude: Optional[float], layers: int, d_model: int,
            seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model, config = make_tiny_lm(seed=seed, n_layers=layers, d_model=d_model, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "w3", writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384), rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="w3")
    collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=32), cadence=1)
    adapter.run_start(steps=STEPS)
    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            tracker.record_loss(loss)
            loss.backward()
            if magnitude is not None and step == INJECTION:
                if magnitude == -1.0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.fill_(float("nan"))
                else:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(magnitude)
            tracker.record_gradients()
            collector.collect(step)
            with tracker.phase("optimizer"):
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            tracker.record_learning_rate()
    adapter.run_end()
    observer.flush(timeout=60.0)
    health = observer.health()
    observer.close()
    by_metric: Dict[str, Dict[int, float]] = {}
    layer_series: Dict[int, Dict[int, float]] = {}
    for event in writer.events:
        if event.metric_name is None or event.step is None or event.metric_value is None:
            continue
        if event.metric_name == "grad.layer_norm" and event.layer_index is not None:
            layer_series.setdefault(int(event.layer_index), {})[event.step] = event.metric_value
        else:
            by_metric.setdefault(str(event.metric_name), {})[event.step] = event.metric_value
    del model, optimizer, observer, adapter, collector, writer
    gc.collect()
    return {
        "by_metric": {k: sorted(v.items()) for k, v in by_metric.items()},
        "layer_series": {k: sorted(v.items()) for k, v in layer_series.items()},
        "health": health,
    }


def persist(run_id: str, scenario: str, layers: int, evidence: Dict[str, Any]) -> Dict[str, Any]:
    detection = DetectorRunner(DetectorConfig()).run(
        run_id, evidence["by_metric"], rank=0, layer_series=evidence["layer_series"],
        module_paths={}, population_layers=layers, population_ranks=1,
        coverage={"step_count": STEPS, "step_min": 1, "step_max": STEPS},
    )
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    engine.consume(detection.anomalies, run_id)
    engine.finish(run_id, final_step=STEPS)
    episodes = engine.all_episodes()
    quality = DataQuality.from_observer_health(evidence["health"])
    verdicts = fuse(run_id, episodes, data_quality=quality)

    episode_store = EpisodeStore(STORE)
    for episode in episodes:
        episode_store.append(episode)
    relations, linked = link_consequences(run_id, episodes)
    episode_store.append_relations(run_id, relations)
    AnalysisStateStore(STORE).append(capture_analysis_state(
        run_id, processed_through_step=STEPS,
        layer_series=evidence["layer_series"], config=DetectorConfig(),
        episodes=episodes,
        detector_versions={str(a.detector): str(a.detector_version) for a in detection.anomalies},
        fusion_rule_version=FUSION_RULE_VERSION, episode_rule_version=RULE_VERSION,
        rule_version="v0.1.1-alpha",
    ))

    incident_store = IncidentStore(STORE, detector_config=DetectorConfig().to_dict())
    incident_ids: List[str] = []
    for verdict in verdicts:
        if verdict.verdict != "CONFIRMED_INCIDENT":
            continue
        match = [e for e in episodes if e.episode_id in set(verdict.episodes_used)]
        if not match:
            continue
        incident = _incident_from(verdict, match[0])
        if incident is not None:
            incident_store.append(incident)
            incident_ids.append(incident.incident_id)

    return {
        "run_id": run_id, "scenario": scenario,
        "episodes": len(episodes), "relations": len(relations),
        "incidents": len(incident_ids), "incident_ids": incident_ids,
        "episode_ids": [e.episode_id for e in episodes],
        "verdicts": [v.verdict for v in verdicts],
        "data_quality": quality.status,
    }


def _incident_from(verdict, episode):
    from training_debugger.schema.enums import IncidentState, IncidentType, Severity
    from training_debugger.schema.evidence import Evidence, EvidenceBuckets, Multiplicity
    from training_debugger.schema.incident import Hypothesis, Incident

    try:
        incident_type = IncidentType(episode.anomaly_type)
    except ValueError:
        return None
    return Incident(
        run_id=episode.run_id, incident_type=incident_type, state=IncidentState.OPEN,
        severity=Severity.WARN,
        detected_step=episode.last_abnormal_step or episode.candidate_onset_step,
        confidence=verdict.confidence,
        evidence=EvidenceBuckets(
            supporting=[Evidence(
                claim="confirmed {} episode".format(episode.anomaly_type),
                level="OBSERVED", detector=episode.detector_id,
                detector_version=episode.detector_version,
                metric_name=episode.metric_name, value=episode.peak_value,
                baseline=episode.baseline, threshold=episode.threshold,
                window=[episode.candidate_onset_step, episode.last_abnormal_step]
                if episode.candidate_onset_step is not None else None,
                confidence=episode.confidence, event_ids=list(episode.supporting_event_ids),
                metadata={"onset_step": episode.candidate_onset_step},
            )],
            not_correlated=[Evidence(
                claim="no secondary manifestation was linked to this incident",
                level="INFERRED", detector="fusion.consequence_linkage", confidence=0.6,
            )],
        ),
        multiplicity=Multiplicity(
            affected_layers=list(episode.affected_layers),
            population_layers=(episode.metadata or {}).get("population_layers") or 0,
            affected_ranks=list(episode.affected_ranks), population_ranks=1,
            event_count=episode.signal_count, population_count=episode.signal_count,
        ),
        hypotheses=[Hypothesis(
            template_id="promotion." + (verdict.promotion_rule_id or "unknown"),
            statement=verdict.reason or "promoted by a deterministic rule",
            score=verdict.confidence, status="INFERRED",
            score_decomposition={"rule": verdict.promotion_rule_id},
        )],
        detector_ids=[episode.detector_id],
        metadata={"onset_step": episode.candidate_onset_step,
                  "promotion_rule_id": verdict.promotion_rule_id},
    )


def main() -> int:
    STORE.mkdir(parents=True, exist_ok=True)
    TRUTH.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    truth: Dict[str, Any] = {}

    for scenario, kind, magnitude in SCENARIOS:
        for layers, d_model in CONFIGS:
            for seed in SEEDS:
                run_id = run_id_for(scenario, layers, d_model, seed)
                print("[W3] {} ...".format(run_id))
                evidence = produce(scenario, magnitude, layers, d_model, seed)
                row = persist(run_id, scenario, layers, evidence)
                row["kind"] = kind
                rows.append(row)
                # Ground truth lives in a SEPARATE tree and never enters the run dir.
                truth[run_id] = {
                    "kind": kind, "scenario": scenario,
                    "injected": kind == "fault",
                    "injection_step": INJECTION if kind == "fault" else None,
                    "magnitude": magnitude, "layers": layers, "d_model": d_model,
                    "seed": seed,
                }

    (TRUTH / "development_ground_truth.json").write_text(
        json.dumps(truth, indent=2), encoding="utf-8")
    payload = {
        "runs": len(rows),
        "store": str(STORE),
        "ground_truth": str(TRUTH / "development_ground_truth.json"),
        "run_id_scheme": "{split}-{scenario}-L{layers}d{d_model}-s{seed}",
        "results": rows,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    write_artifact(OUT / "corpus_repersist_results.json", payload)
    total_ep = sum(r["episodes"] for r in rows)
    total_rel = sum(r["relations"] for r in rows)
    total_inc = sum(r["incidents"] for r in rows)
    print("")
    print("persisted {} runs | {} episodes | {} relations | {} incidents".format(
        len(rows), total_ep, total_rel, total_inc))
    print("ground truth written to a separate tree: {}".format(TRUTH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
