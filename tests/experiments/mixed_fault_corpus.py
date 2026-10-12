"""WS3: a mixed-fault development corpus, so relations exist and can be scored.

The v0.1.1 corpus injected exactly one fault type per run, so **0 relations were
persisted** and relation precision/recall were permanently `NOT ESTIMABLE`. The
relation path -- the part of the evidence chain that says "this manifestation may be a
consequence of that fault" -- has never been measured.

Five scenarios, each a real PyTorch training run:

```
M1  gradient fault -> genuinely non-finite loss   (a REAL consequence chain)
M2  standalone non-finite loss, no gradient fault
M3  gradient fault + an INDEPENDENT numerical fault, injected deliberately
M4  gradient anomaly + ordinary loss noise        (must NOT manufacture a relation)
M5  healthy heavy-tail gradients                  (no fault at all)
```

M1's chain is real, not asserted: filling the gradients with a huge **finite** value
makes `optimizer.step()` overflow the parameters, and the *next* forward pass produces a
non-finite loss. The gradient fault causes the numerical manifestation.

**Ground truth is written to a separate tree** and never enters a run directory, so
production code cannot read the labels it is being scored against.

DEVELOPMENT DATA. Not a blind holdout.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
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

OUT = ROOT / "outputs" / "alpha" / "mixed" / "v015"
STORE = OUT / "store"
TRUTH = OUT / "ground_truth"

STEPS = 200
INJECTION = 80
INDEPENDENT_AT = 120
CONFIGS: Tuple[Tuple[int, int], ...] = ((4, 32), (6, 48))
SEEDS = tuple(range(90000, 90005))          # 5 seeds x 2 configs = 10 runs per scenario

#: scenario -> (kind, ground-truth description)
SCENARIOS: Tuple[str, ...] = ("M1_gradient_then_nonfinite", "M2_standalone_nonfinite",
                              "M3_two_independent_faults", "M4_gradient_plus_loss_noise",
                              "M5_healthy_heavy_tail")


def run_id_for(scenario: str, layers: int, d_model: int, seed: int) -> str:
    return "mix-{}-L{}d{}-s{}".format(scenario, layers, d_model, seed)


def train(scenario: str, layers: int, d_model: int, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    lr = 0.25 if scenario == "M5_healthy_heavy_tail" else 0.05
    model, config = make_tiny_lm(seed=seed, n_layers=layers, d_model=d_model, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "mix", writer=writer, sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384), rank=0, world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="mix")
    collector = LayerMetricCollector(model, observer, selection=LayerSelection(top_k=32), cadence=1)
    adapter.run_start(steps=STEPS)
    nonfinite_steps: List[int] = []
    collapsed = False

    for step in range(1, STEPS + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            loss = model(**batch)["loss"]
            effective = loss
            if scenario == "M2_standalone_nonfinite" and step == INJECTION:
                effective = effective * float("nan")
            if scenario == "M3_two_independent_faults" and step == INDEPENDENT_AT:
                effective = effective * float("nan")
            if not math.isfinite(float(effective.detach())):
                nonfinite_steps.append(step)
            tracker.record_loss(effective)
            if math.isfinite(float(effective.detach())) and not collapsed:
                effective.backward()
                if scenario == "M1_gradient_then_nonfinite" and step == INJECTION:
                    # Huge but FINITE: the parameter update overflows, so the next
                    # forward pass is non-finite. A real consequence, not a claim.
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.fill_(1e28)
                if scenario == "M3_two_independent_faults" and step == INJECTION:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(20.0)
                if scenario == "M4_gradient_plus_loss_noise" and step == INJECTION:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(6.0)
            tracker.record_gradients()
            collector.collect(step)
            with tracker.phase("optimizer"):
                if math.isfinite(float(effective.detach())) and not collapsed:
                    try:
                        optimizer.step()
                    except Exception:
                        collapsed = True
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
        "nonfinite_steps": nonfinite_steps,
    }


def persist(run_id: str, layers: int, evidence: Dict[str, Any]) -> Dict[str, Any]:
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
        run_id, processed_through_step=STEPS, layer_series=evidence["layer_series"],
        config=DetectorConfig(), episodes=episodes,
        detector_versions={str(a.detector): str(a.detector_version) for a in detection.anomalies},
        fusion_rule_version=FUSION_RULE_VERSION, episode_rule_version=RULE_VERSION,
        rule_version="v0.1.5-alpha",
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
        "run_id": run_id,
        "episodes": [
            {"episode_id": e.episode_id, "anomaly_type": e.anomaly_type,
             "state": e.state, "outcome": e.outcome,
             "candidate_onset_step": e.candidate_onset_step,
             "confirmed_at_step": e.confirmed_at_step,
             "affected_layers": list(e.affected_layers),
             "confidence_class": e.confidence_class}
            for e in episodes
        ],
        "relations": [
            {"relation_id": r.relation_id, "relation_type": r.relation_type,
             "source_incident_id": r.source_incident_id,
             "target_incident_id": r.target_incident_id,
             "target_anomaly_type": r.target_anomaly_type,
             "gap_steps": r.gap_steps}
            for r in relations
        ],
        "incident_ids": incident_ids,
        "verdicts": [v.verdict for v in verdicts],
        "data_quality": quality.status,
        "nonfinite_steps": evidence["nonfinite_steps"],
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


def ground_truth(scenario: str, seed: int, layers: int, d_model: int,
                 result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluation-only labels. Distinguishes the five required classes."""
    base = {"scenario": scenario, "seed": seed, "layers": layers, "d_model": d_model,
            "run_id": run_id_for(scenario, layers, d_model, seed)}
    if scenario == "M1_gradient_then_nonfinite":
        base.update({
            "injected": True,
            "injected_primary_fault": "GRADIENT_EXPLOSION",
            "injection_step": INJECTION,
            "acceptable_window": [INJECTION, INJECTION + 12],
            "observed_secondary_manifestation": "NAN_DETECTED" if result["nonfinite_steps"] else None,
            "secondary_window": [INJECTION, INJECTION + 15],
            "independent_second_fault": None,
            "expected_relation": True,
        })
    elif scenario == "M2_standalone_nonfinite":
        base.update({
            "injected": True,
            "injected_primary_fault": "NAN_DETECTED",
            "injection_step": INJECTION,
            "acceptable_window": [INJECTION, INJECTION + 12],
            "observed_secondary_manifestation": None,
            "secondary_window": None,
            "independent_second_fault": None,
            "expected_relation": False,
        })
    elif scenario == "M3_two_independent_faults":
        base.update({
            "injected": True,
            "injected_primary_fault": "GRADIENT_EXPLOSION",
            "injection_step": INJECTION,
            "acceptable_window": [INJECTION, INJECTION + 12],
            "observed_secondary_manifestation": None,
            "secondary_window": None,
            # A second, deliberately independent fault. Linking it to the gradient
            # fault as a consequence would be a fabricated causal claim.
            "independent_second_fault": {"anomaly_type": "NAN_DETECTED",
                                         "step": INDEPENDENT_AT},
            "expected_relation": False,
        })
    elif scenario == "M4_gradient_plus_loss_noise":
        base.update({
            "injected": True,
            "injected_primary_fault": "GRADIENT_EXPLOSION",
            "injection_step": INJECTION,
            "acceptable_window": [INJECTION, INJECTION + 12],
            "observed_secondary_manifestation": None,
            "secondary_window": None,
            "independent_second_fault": None,
            "expected_relation": False,
        })
    else:                                    # M5
        base.update({
            "injected": False,
            "injected_primary_fault": None,
            "injection_step": None,
            "acceptable_window": None,
            "observed_secondary_manifestation": None,
            "independent_second_fault": None,
            "expected_relation": False,
        })
    return base


def main() -> int:
    STORE.mkdir(parents=True, exist_ok=True)
    TRUTH.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    truth: Dict[str, Any] = {}

    for scenario in SCENARIOS:
        for layers, d_model in CONFIGS:
            for seed in SEEDS:
                run_id = run_id_for(scenario, layers, d_model, seed)
                print("[WS3] {} ...".format(run_id))
                evidence = train(scenario, layers, d_model, seed)
                row = persist(run_id, layers, evidence)
                row["scenario"] = scenario
                row["nonfinite_steps"] = evidence["nonfinite_steps"]
                results.append(row)
                truth[run_id] = ground_truth(scenario, seed, layers, d_model,
                                             {"nonfinite_steps": evidence["nonfinite_steps"]})

    (TRUTH / "mixed_fault_ground_truth.json").write_text(
        json.dumps(truth, indent=2), encoding="utf-8")
    payload = {
        "runs": len(results),
        "store": str(STORE),
        "ground_truth": str(TRUTH / "mixed_fault_ground_truth.json"),
        "scenarios": list(SCENARIOS),
        "configs": [{"layers": a, "d_model": b} for a, b in CONFIGS],
        "seeds": list(SEEDS),
        "results": results,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    (OUT / "mixed_fault_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    total_rel = sum(len(r["relations"]) for r in results)
    print("")
    print("persisted {} runs | {} episodes | {} relations".format(
        len(results), sum(len(r["episodes"]) for r in results), total_rel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
