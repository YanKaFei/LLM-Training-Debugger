"""E9/E10: development corpus and the FINAL HOLDOUT.

Statistical rule this file exists to enforce (brief section 7): the Phase 1.2
90-run holdout has been **examined, analysed, and used to choose the Phase 1.3
design**. It is therefore development data and can never again be quoted as an
unbiased final estimate. Its numbers remain valid as *development* numbers.

The final holdout is generated fresh, and frozen **before it is run**:

1. detector implementation frozen (recorded as a hash of the relevant sources)
2. thresholds frozen (``DetectorConfig`` serialised into the manifest)
3. promotion and fusion rules frozen
4. ``FINAL_HOLDOUT_MANIFEST.json`` written
5. its SHA256 written to ``FINAL_HOLDOUT_MANIFEST.sha256``
6. **only then** the holdout runs

After the first look at the results, no further tuning is permitted. If the gates
fail, Phase 1.3 is PARTIAL -- the same data cannot be re-labelled as a holdout
after being used for design.

Run::

    .venv/bin/python tests/final_holdout/run_final_holdout.py --split final
    .venv/bin/python tests/final_holdout/run_final_holdout.py --split development
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tiny_model import make_tiny_lm, synthetic_batch  # noqa: E402

from training_debugger.adapters.pytorch import (  # noqa: E402
    LayerMetricCollector,
    LayerSelection,
    PyTorchTrainingAdapter,
)
from training_debugger.detectors import DetectorConfig, DetectorRunner  # noqa: E402
from training_debugger.env_info import capture_environment  # noqa: E402
from training_debugger.episodes import EpisodeEngine, EpisodeOutcome  # noqa: E402
from training_debugger.evaluation.layers import evaluate_layers  # noqa: E402
from training_debugger.episodes import RULE_VERSION  # noqa: E402
from training_debugger.fusion import FUSION_RULE_VERSION, fusion_rule_registry, fuse  # noqa: E402
from training_debugger.fusion.quality import DataQuality  # noqa: E402
from training_debugger.observer import (  # noqa: E402
    MemoryWriter,
    ObserverConfig,
    SamplingPolicy,
    TrainingObserver,
)
from training_debugger.incidents.relations import link_consequences  # noqa: E402
from training_debugger.persistence import (  # noqa: E402
    AnalysisStateStore,
    EpisodeStore,
    IncidentStore,
    capture_analysis_state,
)
from training_debugger.registry import detector_registry  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "phase1-3"

#: The final holdout deliberately differs from Phase 1.2 in **model shape, seeds,
#: step count and injection step**, so agreement cannot come from re-measuring the
#: same runs.
SPLITS: Dict[str, Dict[str, Any]] = {
    "development": {
        # Phase 1.2's two splits, now demoted to development data by brief section 7.
        "configs": [{"layers": 4, "d_model": 32}, {"layers": 6, "d_model": 48}],
        "seeds": list(range(3000, 3040)) + list(range(4000, 4040)),
        "steps": 200,
        "injection_steps": [120],
    },
    "final": {
        "configs": [{"layers": 5, "d_model": 40}, {"layers": 3, "d_model": 64}],
        "seeds": list(range(5000, 5200)),
        "steps": 240,
        "injection_steps": [100, 150],
    },
}


@dataclass
class Fault:
    fault_type: str
    seed: int
    split: str
    layers: int
    d_model: int
    steps: int
    magnitude: Optional[float] = None
    injection_step: Optional[int] = None
    expected_detectors: Tuple[str, ...] = ()
    healthy: bool = False
    effect_lag: int = 0
    note: str = ""

    def window(self) -> Optional[Tuple[int, int]]:
        if self.injection_step is None:
            return None
        return (self.injection_step + self.effect_lag, self.injection_step + self.effect_lag + 10)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_type": self.fault_type,
            "seed": self.seed,
            "split": self.split,
            "layers": self.layers,
            "d_model": self.d_model,
            "steps": self.steps,
            "magnitude": self.magnitude,
            "injection_step": self.injection_step,
            "acceptable_window": list(self.window() or []),
            "expected_detectors": list(self.expected_detectors),
            "healthy": self.healthy,
            "effect_lag": self.effect_lag,
            "note": self.note,
        }


#: ``(name, magnitudes, healthy, expected detectors, note)``
SPECS: Tuple[Tuple[str, Tuple[Optional[float], ...], bool, Tuple[str, ...], str], ...] = (
    ("H0_stable_healthy", (None,), True, (), ""),
    ("H1_heavy_tail_gradients", (None,), True, (), ""),
    ("H2_noisy_healthy_loss", (None,), True, (), ""),
    ("H3_scheduler_transition", (None,), True, (), ""),
    ("H4_gradient_clipping", (None,), True, (), ""),
    ("F7_lr_jump_no_failure", (None,), True, (), "benign LR discontinuity"),
    ("F1_loss_spike", (0.4, 1.5, 6.0), False, ("loss.robust_zscore",), ""),
    ("F2_sustained_divergence", (1.008, 1.018, 1.04), False,
     ("loss.trend_slope",), "detector DISABLED: expect no incident"),
    ("F3_gradient_explosion", (2.5, 4.0, 7.0, 12.0, 25.0, 50.0), False,
     ("grad.robust_zscore",), ""),
    ("F4_nan_loss", (None,), False, ("numerics.finiteness",), ""),
    ("F5_inf_gradient", (None,), False, ("numerics.finiteness",), ""),
    ("F5b_nan_gradient", (None,), False, ("numerics.finiteness",), ""),
    ("F6_throughput_drop", (0.006,), False, (), "detector DISABLED: expect no incident"),
    ("F10_mixed_anomaly", (None,), False,
     ("numerics.finiteness", "grad.robust_zscore"), ""),
)


def build_manifest(split: str) -> List[Fault]:
    config = SPLITS[split]
    seeds = config["seeds"]
    configs = config["configs"]
    steps = config["steps"]
    injection_steps = config["injection_steps"]
    healthy_specs = [s for s in SPECS if s[2]]
    fault_specs = [s for s in SPECS if not s[2]]

    faults: List[Fault] = []
    index = 0
    # Healthy: 51 % of the corpus, alternating model configuration.
    for name, magnitudes, healthy, detectors, note in healthy_specs:
        for seed in seeds[:14]:
            shape = configs[index % len(configs)]
            faults.append(Fault(
                fault_type=name, seed=seed, split=split,
                layers=shape["layers"], d_model=shape["d_model"], steps=steps,
                magnitude=None, injection_step=None,
                healthy=True, note=note,
            ))
            index += 1
    for name, magnitudes, healthy, detectors, note in fault_specs:
        # Magnitude sweeps spread a smaller seed budget across magnitudes.
        take = 5 if len(magnitudes) > 1 else 5
        for magnitude in magnitudes:
            for seed in seeds[:take]:
                shape = configs[index % len(configs)]
                injection = injection_steps[index % len(injection_steps)]
                faults.append(Fault(
                    fault_type=name, seed=seed, split=split,
                    layers=shape["layers"], d_model=shape["d_model"], steps=steps,
                    magnitude=magnitude, injection_step=injection,
                    expected_detectors=detectors,
                    effect_lag=1 if name == "F6_throughput_drop" else 0,
                    note=note,
                ))
                index += 1
    return faults


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_one(fault: Fault, root: Optional[Path] = None) -> Dict[str, Any]:
    torch.manual_seed(fault.seed)
    model, config = make_tiny_lm(
        seed=fault.seed, n_layers=fault.layers, d_model=fault.d_model, n_heads=4
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    writer = MemoryWriter()
    observer = TrainingObserver(
        "holdout-{}-{}".format(fault.fault_type, fault.seed),
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=16384),
        rank=0,
        world_size=1,
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="holdout")
    collector = LayerMetricCollector(
        model, observer, selection=LayerSelection(top_k=32), cadence=1
    )
    adapter.run_start(steps=fault.steps)

    name = fault.fault_type
    injection = fault.injection_step
    scale = 1.0
    for step in range(1, fault.steps + 1):
        batch = synthetic_batch(batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step)
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            with tracker.phase("forward"):
                loss = model(**batch)["loss"]
            effective = loss

            if name == "H2_noisy_healthy_loss":
                effective = effective * (1.0 + 0.35 * math.sin(step * 1.7))
            if name == "F1_loss_spike" and injection and step == injection and fault.magnitude:
                effective = effective + fault.magnitude
            if name == "F2_sustained_divergence" and injection and step >= injection:
                scale *= fault.magnitude or 1.012
                effective = effective * scale
            if name == "F4_nan_loss" and injection and step == injection:
                effective = effective * float("nan")
            if name == "F10_mixed_anomaly" and injection and step == injection:
                effective = effective * float("nan")
            if name == "F6_throughput_drop" and injection and step >= injection:
                time.sleep(fault.magnitude or 0.006)

            tracker.record_loss(effective)
            with tracker.phase("backward"):
                if effective == effective:
                    effective.backward()

            if name == "F3_gradient_explosion" and injection and step >= injection:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(fault.magnitude or 1.0)
            if name == "F10_mixed_anomaly" and injection and step >= injection:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(20.0)
            if name == "F5_inf_gradient" and injection and step == injection:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.flatten()[0] = float("inf")
            if name == "F5b_nan_gradient" and injection and step == injection:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.flatten()[0] = float("nan")
            if name == "H1_heavy_tail_gradients" and step % 17 == 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(2.6)
            if name == "H4_gradient_clipping":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)

            tracker.record_gradients()
            collector.collect(step)
            with tracker.phase("optimizer"):
                if effective == effective:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tracker.record_parameters()
            if name in ("H3_scheduler_transition", "F7_lr_jump_no_failure") and step == injection:
                for group in optimizer.param_groups:
                    group["lr"] = 0.25 if name == "F7_lr_jump_no_failure" else 0.15
            tracker.record_learning_rate()

    adapter.run_end()
    observer.flush(timeout=30.0)
    observer_health = observer.health()
    observer.close()

    by_metric, layer_series, module_paths = _series(writer.events)
    detection = DetectorRunner(DetectorConfig()).run(
        "holdout",
        by_metric,
        rank=0,
        layer_series=layer_series,
        module_paths=module_paths,
        population_layers=fault.layers,
        population_ranks=1,
        coverage={"step_count": fault.steps, "step_min": 1, "step_max": fault.steps},
    )

    # -- episodes and fusion -------------------------------------------
    engine = EpisodeEngine(enabled_detectors=detector_registry()["enabled_detector_ids"])
    engine.consume(detection.anomalies, fault.fault_type)
    engine.finish(fault.fault_type, final_step=fault.steps)
    episodes = engine.all_episodes()
    # A1: a verdict must know whether the evidence it rests on is complete.
    verdicts = fuse(
        fault.fault_type, episodes,
        data_quality=DataQuality.from_observer_health(observer_health),
    )

    fires = [
        (str(a.detector), int(a.first_abnormal_step))
        for a in detection.anomalies
        if a.first_abnormal_step is not None
    ]

    # -- M2: real pipeline persistence ------------------------------------
    #
    # The stores become the episode and incident truth for this run. The ad-hoc
    # Parquet artefacts written below are derived summaries of the same objects,
    # not a second source of truth. Ordering is defined and enforced: every
    # episode revision is persisted BEFORE any incident that consumes it, so a
    # crash between the two can never leave an incident whose episode is missing.
    persistence = _persist_run(root, fault, episodes, verdicts, detection,
                               layer_series=layer_series)

    result = {
        "run_id": "{}-{}-{}".format(fault.fault_type, fault.seed, fault.injection_step),
        "fault_type": fault.fault_type,
        "seed": fault.seed,
        "split": fault.split,
        "layers": fault.layers,
        "d_model": fault.d_model,
        "steps": fault.steps,
        "magnitude": fault.magnitude,
        "injection_step": injection,
        "injected": not fault.healthy,
        "acceptable_window": list(fault.window() or []),
        "expected_detectors": list(fault.expected_detectors),
        "fires": fires,
        "episodes": [e.to_dict() for e in episodes],
        "verdicts": [v.to_dict() for v in verdicts],
        "detector_errors": detection.detector_errors,
        "persistence": persistence,
        "run_error": None,
    }
    del model, optimizer, observer, adapter, collector
    gc.collect()
    return result


def _persist_run(root, fault, episodes, verdicts, detection, layer_series=None) -> Dict[str, Any]:
    """M2: write episodes, relations and incidents from a real run.

    Fail-open, per the brief: a persistence failure must not fail training or the
    analysis, but it must be *surfaced* rather than swallowed, so ``errors`` is
    returned and reported.
    """
    if root is None:
        return {"status": "disabled"}
    from pathlib import Path as _Path

    store_root = _Path(root)
    # The run_id must match the one the episodes were grouped under. An earlier
    # version computed "{fault_type}-{seed}" here while ``engine.consume`` had used
    # ``fault.fault_type``, so episodes carried one id and were stored under another
    # -- a directory that cannot be looked up by the identifier inside it.
    run_id = fault.fault_type
    episode_store = EpisodeStore(store_root)
    incident_store = IncidentStore(
        store_root, detector_config=DetectorConfig().to_dict()
    )
    revisions = {}
    for episode in episodes:
        revisions[episode.episode_id] = episode_store.append(episode)

    # P0: the detector reference must survive a restart, or a resumed episode would
    # be judged by a different baseline than the one that raised it.
    detector_versions = {}
    for anomaly in detection.anomalies:
        detector_versions[str(anomaly.detector)] = str(anomaly.detector_version)
    state = capture_analysis_state(
        run_id,
        processed_through_step=fault.steps,
        layer_series=layer_series,
        config=DetectorConfig(),
        episodes=episodes,
        detector_versions=detector_versions,
        fusion_rule_version=FUSION_RULE_VERSION,
        episode_rule_version=RULE_VERSION,
        rule_version="phase1.10-1.0",
    )
    state_revision = AnalysisStateStore(store_root).append(state)

    relations, linked = link_consequences(run_id, episodes)
    relations_written = episode_store.append_relations(run_id, relations)

    incidents_written = 0
    for verdict in verdicts:
        if verdict.verdict != "CONFIRMED_INCIDENT":
            continue
        for episode_id in verdict.episodes_used:
            match = [e for e in episodes if e.episode_id == episode_id]
            if not match:
                continue
            incident = _incident_from_verdict(verdict, match[0], run_id, relations, linked)
            if incident is not None:
                incident_store.append(incident)
                incidents_written += 1

    return {
        "status": "ok",
        "run_id": run_id,
        "analysis_state_revision": state_revision,
        "baselines": len(state.baselines),
        "episodes": len(episodes),
        "episode_revisions": revisions,
        "relations": relations_written,
        "linked_episodes": sorted(linked),
        "incidents": incidents_written,
        "episode_store_root": str(store_root),
        "errors": list(episode_store.errors) + list(incident_store.errors),
    }


def _incident_from_verdict(verdict, episode, run_id, relations, linked):
    """Build an Incident from a fusion verdict, preserving promotion provenance."""
    from training_debugger.schema.enums import IncidentState, IncidentType, Severity
    from training_debugger.schema.evidence import Evidence, EvidenceBuckets, Multiplicity
    from training_debugger.schema.incident import Hypothesis, Incident

    try:
        incident_type = IncidentType(episode.anomaly_type)
    except ValueError:
        return None
    severity = {
        "error": Severity.ERROR, "warn": Severity.WARN,
        "info": Severity.INFO, "debug": Severity.DEBUG,
    }.get(str(verdict.severity), Severity.WARN)

    evidence = EvidenceBuckets(
        supporting=[
            Evidence(
                claim="confirmed {} episode at step {}".format(
                    episode.anomaly_type, episode.candidate_onset_step),
                level="OBSERVED" if episode.confidence_class == "CERTAIN" else "INFERRED",
                detector=episode.detector_id,
                detector_version=episode.detector_version,
                metric_name=episode.metric_name,
                value=episode.peak_value,
                baseline=episode.baseline,
                threshold=episode.threshold,
                window=[episode.candidate_onset_step, episode.last_abnormal_step]
                if episode.candidate_onset_step is not None else None,
                confidence=episode.confidence,
                event_ids=list(episode.supporting_event_ids),
                metadata={
                    "onset_step": episode.candidate_onset_step,
                    "gradient_band": (episode.metadata or {}).get("gradient_band"),
                },
            )
        ],
        not_correlated=[
            Evidence(
                claim="no secondary numerical manifestation was linked to this incident",
                level="INFERRED",
                detector="fusion.consequence_linkage",
                confidence=0.6,
                metadata={"onset_step": episode.candidate_onset_step},
            )
        ],
    )
    relations_for = [r for r in relations if r.source_incident_id == episode.episode_id]
    secondary = [
        {
            "episode_id": r.target_incident_id,
            "relation_type": r.relation_type,
            "gap_steps": r.gap_steps,
        }
        for r in relations_for
    ]
    metadata = {
        "onset_step": episode.candidate_onset_step,
        "confirmed_at_step": episode.confirmed_at_step,
        "promotion_rule_id": verdict.promotion_rule_id,
        "promotion_rule_version": verdict.promotion_rule_version,
        "episodes_used": list(verdict.episodes_used),
        "episodes_rejected": list(verdict.episodes_rejected),
        "abstained_episodes": list(verdict.abstained_episodes),
        "signals_used": list(verdict.signals_used),
        "secondary_manifestations": secondary,
    }
    return Incident(
        run_id=run_id,
        incident_type=incident_type,
        state=IncidentState.OPEN,
        severity=severity,
        detected_step=episode.last_abnormal_step or episode.candidate_onset_step,
        confidence=verdict.confidence,
        evidence=evidence,
        multiplicity=Multiplicity(
            affected_layers=list(episode.affected_layers),
            population_layers=(episode.metadata or {}).get("population_layers") or 0,
            affected_ranks=list(episode.affected_ranks),
            population_ranks=1,
            event_count=episode.signal_count,
            population_count=episode.signal_count,
        ),
        hypotheses=[
            Hypothesis(
                template_id="promotion." + (verdict.promotion_rule_id or "unknown"),
                statement=verdict.reason or "promoted by a deterministic rule",
                score=verdict.confidence,
                status="INFERRED",
                score_decomposition={
                    "rule": verdict.promotion_rule_id,
                    "confidence_class": episode.confidence_class,
                },
            )
        ],
        detector_ids=[episode.detector_id],
        metadata=metadata,
    )


def _series(events):
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
        by_metric.setdefault(event.metric_name, {})[event.step] = event.metric_value
    return (
        {k: sorted(v.items()) for k, v in by_metric.items()},
        {k: sorted(v.items()) for k, v in layers.items()},
        module_paths,
    )


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------


def _hash_sources() -> Dict[str, str]:
    """Hash the frozen implementation: detectors, episodes, fusion, registry."""
    targets = [
        "src/training_debugger/detectors/base.py",
        "src/training_debugger/detectors/gradient.py",
        "src/training_debugger/detectors/loss.py",
        "src/training_debugger/detectors/throughput.py",
        "src/training_debugger/detectors/runner.py",
        "src/training_debugger/episodes/model.py",
        "src/training_debugger/episodes/engine.py",
        "src/training_debugger/fusion/rules.py",
        "src/training_debugger/registry.py",
    ]
    digests: Dict[str, str] = {}
    for target in targets:
        path = ROOT / target
        if path.exists():
            digests[target] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def build_freeze_manifest(split: str, faults: Sequence[Fault]) -> Dict[str, Any]:
    return {
        "phase": "1.3",
        "split": split,
        "frozen_before_first_run": True,
        "fusion_rule_version": FUSION_RULE_VERSION,
        "detector_config": DetectorConfig().to_dict(),
        "source_hashes": _hash_sources(),
        "detector_registry": detector_registry(),
        "fusion_rules": fusion_rule_registry(),
        "environment": capture_environment(),
        "splits": {k: {"configs": v["configs"], "steps": v["steps"],
                       "injection_steps": v["injection_steps"],
                       "seed_range": [v["seeds"][0], v["seeds"][-1]]}
                   for k, v in SPLITS.items()},
        "scenarios": [f.to_dict() for f in faults],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1.3 holdout")
    parser.add_argument("--split", default="final", choices=sorted(SPLITS))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    faults = build_manifest(args.split)
    if args.limit:
        faults = faults[: args.limit]

    # ---- FREEZE FIRST, THEN RUN (brief section 18.1) -------------------
    manifest = build_freeze_manifest(args.split, faults)
    manifest_path = OUTPUT_DIR / (
        "FINAL_HOLDOUT_MANIFEST.json" if args.split == "final" else "DEVELOPMENT_MANIFEST.json"
    )
    payload = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    manifest_path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if args.split == "final":
        (OUTPUT_DIR / "FINAL_HOLDOUT_MANIFEST.sha256").write_text(digest + "\n", encoding="utf-8")
    print("manifest frozen: {} sha256={}".format(manifest_path.name, digest[:16]))

    healthy = sum(1 for f in faults if f.healthy)
    print("corpus: {} runs ({} healthy = {:.0f}%, {} fault)".format(
        len(faults), healthy, 100.0 * healthy / max(1, len(faults)), len(faults) - healthy))
    print("configs:", [c for c in SPLITS[args.split]["configs"]])
    started = time.perf_counter()

    results: List[Dict[str, Any]] = []
    for index, fault in enumerate(faults, start=1):
        try:
            results.append(run_one(fault, root=OUTPUT_DIR / 'store'))
        except Exception as exc:
            results.append({
                "run_id": "{}-{}".format(fault.fault_type, fault.seed),
                "fault_type": fault.fault_type, "seed": fault.seed, "split": fault.split,
                "layers": fault.layers, "d_model": fault.d_model, "steps": fault.steps,
                "magnitude": fault.magnitude, "injection_step": fault.injection_step,
                "injected": not fault.healthy,
                "acceptable_window": list(fault.window() or []),
                "expected_detectors": list(fault.expected_detectors),
                "fires": [], "episodes": [], "verdicts": [], "detector_errors": [],
                "run_error": "{}: {}".format(type(exc).__name__, exc),
            })
        if index % 20 == 0 or index == len(faults):
            print("   {}/{} ({:.0f}s)".format(index, len(faults), time.perf_counter() - started))

    elapsed = time.perf_counter() - started
    scores = evaluate_layers(results)
    _write_artifacts(args.split, results, scores, elapsed, digest)
    _print_summary(scores)
    return 0


def _write_artifacts(split: str, results, scores, elapsed: float, digest: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    prefix = "final_holdout" if split == "final" else "development"

    # episodes
    episode_rows = []
    for result in results:
        for episode in result["episodes"]:
            episode_rows.append({
                "run_id": result["run_id"],
                "fault_type": result["fault_type"],
                "split": result["split"],
                "injected": result["injected"],
                "magnitude": result["magnitude"],
                "episode_id": episode["episode_id"],
                "detector_id": episode["detector_id"],
                "anomaly_type": episode["anomaly_type"],
                "signal_key": episode["signal_key"],
                "state": episode["state"],
                "outcome": episode["outcome"],
                "candidate_onset_step": episode["candidate_onset_step"],
                "confirmed_at_step": episode["confirmed_at_step"],
                "last_abnormal_step": episode["last_abnormal_step"],
                "signal_count": episode["signal_count"],
                "affected_layer_count": len(episode["affected_layers"]),
                "cross_layer_multiplicity": episode["cross_layer_multiplicity"],
                "confidence_class": episode["confidence_class"],
                "peak_ratio": episode["peak_ratio"],
                "confirmation_delay_steps": episode["confirmation_delay_steps"],
            })
    pq.write_table(pa.Table.from_pylist(episode_rows), OUTPUT_DIR / "episode_results.parquet",
                   compression="zstd")

    # incidents (verdicts)
    incident_rows = []
    for result in results:
        for verdict in result["verdicts"]:
            incident_rows.append({
                "run_id": result["run_id"],
                "fault_type": result["fault_type"],
                "split": result["split"],
                "injected": result["injected"],
                "magnitude": result["magnitude"],
                "verdict": verdict["verdict"],
                "anomaly_type": verdict["anomaly_type"],
                "candidate_onset_step": verdict["candidate_onset_step"],
                "confirmed_at_step": verdict["confirmed_at_step"],
                "promotion_rule_id": verdict["promotion_rule_id"],
                "promotion_rule_version": verdict["promotion_rule_version"],
                "episodes_used": ",".join(verdict["episodes_used"]),
                "episodes_rejected": ",".join(verdict["episodes_rejected"]),
                "abstained_episodes": ",".join(verdict["abstained_episodes"]),
                "signals_used": ",".join(verdict["signals_used"]),
                "confidence": verdict["confidence"],
                "severity": verdict["severity"],
            })
    pq.write_table(pa.Table.from_pylist(incident_rows), OUTPUT_DIR / "incident_results.parquet",
                   compression="zstd")

    # runs
    run_rows = []
    for result in results:
        confirmed = [v for v in result["verdicts"]
                     if v["verdict"] == "CONFIRMED_INCIDENT"]
        abstained = [v for v in result["verdicts"]
                     if v["verdict"] in ("INSUFFICIENT_EVIDENCE", "CANDIDATE_ONLY")]
        run_rows.append({
            "run_id": result["run_id"],
            "fault_type": result["fault_type"],
            "split": result["split"],
            "injected": result["injected"],
            "layers": result["layers"],
            "d_model": result["d_model"],
            "steps": result["steps"],
            "magnitude": result["magnitude"],
            "injection_step": result["injection_step"],
            "acceptable_start": (result["acceptable_window"] or [None, None])[0],
            "acceptable_end": (result["acceptable_window"] or [None, None])[1],
            "confirmed_incidents": len(confirmed),
            "abstained_verdicts": len(abstained),
            "episode_count": len(result["episodes"]),
            "observation_count": sum(e["signal_count"] for e in result["episodes"]),
            "first_confirmed_onset": min(
                (v["candidate_onset_step"] for v in confirmed
                 if v["candidate_onset_step"] is not None), default=None),
            "run_error": result["run_error"] or "",
        })
    run_path = OUTPUT_DIR / ("final_holdout_results.parquet" if split == "final"
                             else "run_results.parquet")
    pq.write_table(pa.Table.from_pylist(run_rows), run_path, compression="zstd")
    if split == "final":
        pq.write_table(pa.Table.from_pylist(run_rows), OUTPUT_DIR / "run_results.parquet",
                       compression="zstd")

    (OUTPUT_DIR / "{}_confusion_matrix.json".format("episode")).write_text(
        json.dumps(scores["episode"], indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "incident_confusion_matrix.json").write_text(
        json.dumps(scores["incident"], indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "run_confusion_matrix.json").write_text(
        json.dumps(scores["run"], indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "detector_registry.json").write_text(
        json.dumps(detector_registry(), indent=2), encoding="utf-8")
    (OUTPUT_DIR / "fusion_rule_registry.json").write_text(
        json.dumps(fusion_rule_registry(), indent=2), encoding="utf-8")
    (OUTPUT_DIR / "{}_scores.json".format(prefix)).write_text(
        json.dumps({"scores": scores, "elapsed_s": elapsed, "manifest_sha256": digest},
                   indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "{}_raw.json".format(prefix)).write_text(
        json.dumps({"results": results, "scores": scores}, indent=2, default=str),
        encoding="utf-8")


def _print_summary(scores) -> None:
    print("")
    for layer in ("signal", "episode", "incident", "run"):
        entry = scores[layer]
        p = entry.get("precision")
        r = entry.get("recall")
        print("{:<9} P={:<7} R={:<7} TP={:<3} FP={:<3} preFP={:<3} FN={:<3} abstain={:.2f}".format(
            layer,
            "n/a" if p is None else "{:.3f}".format(p),
            "n/a" if r is None else "{:.3f}".format(r),
            entry.get("true_positives", 0), entry.get("false_positives", 0),
            entry.get("pre_injection", 0), entry.get("false_negatives", 0),
            entry.get("abstention_rate") or 0.0))
    run = scores["run"]
    print("healthy-run FP rate: {}  false incidents/1k: {}  abstention: {}".format(
        run.get("healthy_run_false_incident_rate"),
        run.get("false_incidents_per_1k_steps"),
        run.get("abstention_rate_overall")))


if __name__ == "__main__":
    raise SystemExit(main())
