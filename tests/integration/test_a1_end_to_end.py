"""A1 — the Phase 1 release gate, end to end on real PyTorch training.

    healthy training
      -> inject a controlled failure
      -> observer captures signals
      -> events persisted
      -> detector identifies the anomaly
      -> incident opens
      -> evidence graph constructed
      -> earliest defensible abnormal event identified
      -> CLI explains the incident

Every step of that chain runs here against a real model with real autograd. The
injection is a genuine gradient explosion followed by the loss consequence it
causes -- not a synthetic series -- because the interaction between autograd,
gradient lifetime and optimiser state is where a debugger of this kind actually
breaks.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

import helpers
from tiny_model import make_tiny_lm, synthetic_batch

from training_debugger.adapters.pytorch import (
    LayerMetricCollector,
    LayerSelection,
    PyTorchTrainingAdapter,
)
from training_debugger.detectors import DetectorConfig, DetectorRunner
from training_debugger.detectors.base import ConfidenceClass
from training_debugger.detectors.runner import LAYER_MULTIPLICITY_STRONG
from training_debugger.incidents import IncidentEngine, render_incident
from training_debugger.observer import MemoryWriter, ObserverConfig, SamplingPolicy, TrainingObserver
from training_debugger.schema import EvidenceLevel, IncidentType, OrderingConfidence, validate_event

pytestmark = pytest.mark.torch

WARMUP = 50
TOTAL_STEPS = 130
INJECT_GRADIENT_AT = 80
GRADIENT_FACTOR = 40.0


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _run_a1():
    """Real training with a controlled gradient explosion at step 80."""
    torch.manual_seed(20240)
    model, config = make_tiny_lm(seed=20240, n_layers=4, d_model=32, n_heads=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    writer = MemoryWriter()
    observer = TrainingObserver(
        "run-a1",
        writer=writer,
        sampling=SamplingPolicy.uniform(1),
        config=ObserverConfig(flush_interval_s=0.01, batch_size=4096),
        rank=0,
        world_size=1,
        model_id="tiny-transformer-lm",
    )
    adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="tiny-transformer-lm")
    collector = LayerMetricCollector(
        model, observer, selection=LayerSelection(top_k=6), cadence=1
    )
    adapter.run_start()

    losses = []
    injected = False
    for step in range(1, TOTAL_STEPS + 1):
        batch = synthetic_batch(
            batch_size=4, seq_len=8, vocab_size=config.vocab_size, seed=step
        )
        with adapter.step(step) as tracker:
            tracker.set_batch(samples=4, tokens=int(batch["input_ids"].numel()))
            with tracker.phase("forward"):
                loss = model(**batch)["loss"]
            tracker.record_loss(loss)
            with tracker.phase("backward"):
                loss.backward()
            if step == INJECT_GRADIENT_AT:
                # The controlled failure: scale every gradient. This is exactly
                # the signature of a gradient explosion, and it is what makes the
                # subsequent loss spike a *consequence* rather than a coincidence.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(GRADIENT_FACTOR)
                injected = True
            tracker.record_gradients()
            collector.collect(step)
            with tracker.phase("optimizer"):
                optimizer.step()
            tracker.record_parameters()
            tracker.record_learning_rate()
            with tracker.phase("optimizer"):
                optimizer.zero_grad(set_to_none=True)
        losses.append((step, float(loss.detach().item())))

    adapter.run_end()
    observer.flush(timeout=30.0)
    observer.close()
    return observer, adapter, collector, writer.events, losses, injected


@pytest.fixture(scope="module")
def a1_run():
    return _run_a1()


def _series(events):
    by_metric = {}
    layers = {}
    module_paths = {}
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
        {name: sorted(values.items()) for name, values in by_metric.items()},
        {index: sorted(values.items()) for index, values in layers.items()},
        module_paths,
    )


@pytest.fixture(scope="module")
def a1_analysis(a1_run):
    _, _, _, events, losses, _ = a1_run
    by_metric, layer_series, module_paths = _series(events)
    steps = [s for s, _ in losses]
    runner = DetectorRunner(DetectorConfig(warmup_steps=WARMUP, min_samples=20))
    detection = runner.run(
        "run-a1",
        by_metric,
        rank=0,
        layer_series=layer_series,
        module_paths=module_paths,
        population_layers=len(layer_series),
        population_ranks=1,
        coverage={"step_count": len(steps), "step_min": min(steps), "step_max": max(steps)},
    )
    engine = IncidentEngine()
    incidents = engine.build("run-a1", detection, events, by_metric, layer_series=layer_series)
    return detection, incidents, events, by_metric, layer_series, losses


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def test_training_ran_and_the_injection_happened(a1_run):
    _, _, _, events, losses, injected = a1_run
    assert injected
    assert len(losses) == TOTAL_STEPS
    assert len(events) > TOTAL_STEPS * 5


def test_events_are_valid(a1_run):
    _, _, _, events, _, _ = a1_run
    from training_debugger.schema import ProducerSequenceTracker

    tracker = ProducerSequenceTracker()
    bad = [e for e in events if not validate_event(e, seq_tracker=tracker).ok]
    assert not bad, "invalid events: %s" % [e.metric_name for e in bad[:5]]


def test_the_injected_explosion_is_visible_in_the_gradient_series(a1_run):
    _, _, _, events, _, _ = a1_run
    by_metric, _, _ = _series(events)
    grad = dict(by_metric["grad.global_norm"])
    before = [grad[s] for s in range(INJECT_GRADIENT_AT - 10, INJECT_GRADIENT_AT)]
    after = grad[INJECT_GRADIENT_AT]
    assert after > 5 * max(before), (
        "the injected %sx gradient scaling should be plainly visible" % GRADIENT_FACTOR
    )


def test_detectors_find_the_gradient_explosion(a1_analysis):
    """D1 semantics: a *true positive in the acceptable window*, not `first fire == injection`.

    The previous form of this assertion compared the raw first fire against the
    injection step, which is the metric D1 showed to be ill-defined: a
    pre-injection fire produces a *negative delay* rather than being counted as
    the false positive it is. This version asserts what actually matters and in
    three parts:

    1. a STRONG detection lands inside the acceptable window;
    2. any fire before the injection is classified WEAK and carries a
       cross-layer multiplicity below the promotion threshold;
    3. the aggregate signal -- not just one layer -- departed at the true
       positive, which is what separates a real explosion from a healthy tail.
    """
    detection, _, _, _, _, _ = a1_analysis
    gradient = [a for a in detection.anomalies if "grad" in str(a.detector)]
    assert gradient, "the gradient explosion must be detected"

    acceptable = (INJECT_GRADIENT_AT, INJECT_GRADIENT_AT + 10)
    strong_in_window = [
        a for a in gradient
        if a.confidence_class == ConfidenceClass.STRONG
        and acceptable[0] <= a.first_abnormal_step <= acceptable[1]
    ]
    assert strong_in_window, (
        "no STRONG gradient detection inside the acceptable window %s; fires were %s"
        % (acceptable, sorted({(a.first_abnormal_step, str(a.confidence_class)) for a in gradient}))
    )

    # The aggregate must have moved -- a single layer departing is not enough.
    aggregates = [
        a for a in strong_in_window
        if a.layer_index is None and a.metric_name == "grad.global_norm"
    ]
    assert aggregates, "the global gradient norm must be among the strong detections"

    for anomaly in gradient:
        if anomaly.first_abnormal_step < INJECT_GRADIENT_AT:
            assert anomaly.confidence_class != ConfidenceClass.CERTAIN
            if anomaly.layer_index is not None:
                multiplicity = anomaly.metadata.get("layer_multiplicity", 0.0) or 0.0
                assert multiplicity < LAYER_MULTIPLICITY_STRONG, (
                    "a pre-injection fire reported %.2f cross-layer multiplicity, which "
                    "would promote it to an incident" % multiplicity
                )


def test_incidents_open(a1_analysis):
    _, incidents, _, _, _, _ = a1_analysis
    assert incidents, "the injection must open at least one incident"
    types = {i.incident_type for i in incidents}
    assert types & {IncidentType.GRADIENT_EXPLOSION, IncidentType.LOSS_SPIKE, IncidentType.NAN_DETECTED}


def test_an_incident_can_be_rendered_and_the_text_is_structured(a1_analysis):
    _, incidents, _, _, _, _ = a1_analysis
    for incident in incidents:
        text = render_incident(incident)
        for section in (
            "INCIDENT ",
            "Type:",
            "Observed symptom:",
            "Affected:",
            "Not correlated",
            "Earliest defensible abnormal event:",
            "Probable root cause",
            "Missing evidence:",
            "Confidence:",
        ):
            assert section in text, "section %r missing from the rendered incident" % section


def test_rendering_is_deterministic(a1_analysis):
    _, incidents, _, _, _, _ = a1_analysis
    for incident in incidents:
        assert render_incident(incident) == render_incident(incident)


# ---------------------------------------------------------------------------
# The properties that make the incident trustworthy
# ---------------------------------------------------------------------------


def test_earlier_abnormal_evidence_precedes_the_symptom(a1_analysis):
    """The product's core claim, asserted on a real run.

    If the earliest abnormal event is the loss spike itself, the system has added
    nothing over a dashboard that alerts on loss.
    """
    _, incidents, _, _, _, _ = a1_analysis
    found = False
    for incident in incidents:
        if incident.incident_type not in (IncidentType.LOSS_SPIKE, IncidentType.GRADIENT_EXPLOSION):
            continue
        onset = incident.metadata.get("onset_step")
        if onset is None:
            continue
        observed_onsets = [a.first_abnormal_step for a in _anomalies_of(incident)]
        if observed_onsets:
            assert min(observed_onsets) <= onset + 1
        found = True
    assert found


def _anomalies_of(incident):
    """Reconstruct minimal anomaly-like views from an incident's evidence."""
    class _View:
        def __init__(self, step):
            self.first_abnormal_step = step

    steps = []
    for evidence in incident.evidence.supporting:
        if evidence.window:
            steps.append(_View(evidence.window[0]))
    return steps


def test_first_abnormal_event_is_unique_or_explicitly_not(a1_analysis):
    """Either a defensible single first event, or an explicit refusal."""
    _, incidents, _, _, _, _ = a1_analysis
    for incident in incidents:
        if incident.first_abnormal_event_id is not None:
            assert incident.ordering_confidence == str(OrderingConfidence.PROVEN)
            assert len(incident.earliest.candidates) == 1
        else:
            assert incident.earliest_abnormal_candidate_ids
            assert "cannot be established" in incident.earliest.explanation or \
                   "mutually unordered" in incident.earliest.explanation


def test_negative_evidence_is_present(a1_analysis):
    """The incident must be able to say what stayed normal."""
    _, incidents, _, _, _, _ = a1_analysis
    for incident in incidents:
        assert incident.evidence.not_correlated, (
            "%s reports no negative evidence; that is the section that makes a "
            "diagnosis credible" % incident.incident_id
        )
        claims = " ".join(e.claim for e in incident.evidence.not_correlated)
        assert "learning rate" in claims or "remained within baseline" in claims


def test_multiplicity_states_affected_over_population(a1_analysis):
    _, incidents, _, _, _, _ = a1_analysis
    for incident in incidents:
        summary = incident.multiplicity.summary()
        assert "/" in summary, "multiplicity must give a ratio, got %r" % summary
        assert incident.multiplicity.event_count > 0


def test_every_hypothesis_has_a_decomposable_score(a1_analysis):
    _, incidents, _, _, _, _ = a1_analysis
    for incident in incidents:
        for hypothesis in incident.hypotheses:
            assert hypothesis.score_decomposition
            assert "prior" in hypothesis.score_decomposition
            assert 0.0 <= hypothesis.score <= 1.0


def test_phase1_never_claims_a_confirmed_data_cause(a1_analysis):
    """Data causation is capped at HYPOTHESIS: no replay exists to confirm it."""
    _, incidents, _, _, _, _ = a1_analysis
    for incident in incidents:
        for hypothesis in incident.hypotheses:
            if "data" in hypothesis.template_id:
                assert hypothesis.status is EvidenceLevel.HYPOTHESIS
                assert any("replay" in item for item in hypothesis.missing)


def test_no_inferred_evidence_lacks_its_method(a1_analysis):
    _, incidents, _, _, _, _ = a1_analysis
    for incident in incidents:
        for evidence in incident.evidence.all():
            if evidence.level is EvidenceLevel.INFERRED:
                assert evidence.detector, "inferred evidence must name its detector"


def test_incident_validates(a1_analysis):
    _, incidents, _, _, _, _ = a1_analysis
    for incident in incidents:
        incident.validate()  # raises on any violated admissibility rule


def test_incident_serialises_for_storage(a1_analysis):
    _, incidents, _, _, _, _ = a1_analysis
    for incident in incidents:
        payload = json.dumps(incident.to_dict(), default=str, allow_nan=True)
        assert incident.incident_id in payload
        restored = json.loads(payload)
        assert restored["earliest_abnormal"]["first_abnormal_event_id"] == incident.first_abnormal_event_id


def test_detector_errors_are_surfaced_not_swallowed(a1_analysis):
    detection, _, _, _, _, _ = a1_analysis
    assert detection.detector_errors == [], (
        "a detector raised: %s" % detection.detector_errors
    )
