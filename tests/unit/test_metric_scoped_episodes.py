"""W2: the signal key must identify the signal, not just the detector.

`numerics.finiteness` observes every metric in a run through one detector id. With a
detector-only signal key, a non-finite *gradient* at step 80, a non-finite *loss* at
step 81 and the non-finite *counter* all landed in ONE episode, which recorded
`metric_name = train.loss_nonfinite_count`.

The mechanism evidence -- "the gradient went non-finite first" -- was destroyed at the
episode boundary. That is why M1 produced no relation for three releases even though the
raw observation existed the whole time.
"""

from __future__ import annotations

from training_debugger.detectors.base import Anomaly, ConfidenceClass
from training_debugger.episodes import EpisodeEngine
from training_debugger.schema import EvidenceLevel, Severity


def anomaly(step, metric, *, layer=None, detector="numerics.finiteness",
            value=float("nan")):
    return Anomaly(
        detector=detector, detector_version="1.0", metric_name=metric,
        first_abnormal_step=step, observed_value=value, baseline=None, threshold=None,
        score=1.0, confidence=1.0, level=EvidenceLevel.OBSERVED,
        severity=Severity.ERROR, window=(step, step), rank=0, layer_index=layer,
        confidence_class=ConfidenceClass.CERTAIN,
        metadata={"gradient_band": None, "population_layers": 4},
    )


def test_a_nonfinite_gradient_and_a_nonfinite_loss_are_separate_episodes():
    engine = EpisodeEngine()
    engine.consume([anomaly(80, "grad.global_norm", value=float("inf"))], "r")
    engine.consume([anomaly(81, "train.loss", value=float("nan"))], "r")
    engine.finish("r", final_step=200)

    episodes = engine.all_episodes()
    metrics = sorted(e.metric_name for e in episodes)
    assert len(episodes) == 2, (
        "the gradient and the loss are different signals and must not share an episode"
    )
    assert metrics == ["grad.global_norm", "train.loss"]

    gradient = [e for e in episodes if e.metric_name == "grad.global_norm"][0]
    loss = [e for e in episodes if e.metric_name == "train.loss"][0]
    assert gradient.candidate_onset_step == 80
    assert loss.candidate_onset_step == 81, (
        "the mechanism ordering must survive into the episode layer"
    )


def test_the_counter_does_not_share_the_loss_episode():
    engine = EpisodeEngine()
    engine.consume([anomaly(81, "train.loss", value=float("nan"))], "r")
    engine.consume([anomaly(81, "train.loss_nonfinite_count", value=1.0)], "r")
    engine.finish("r", final_step=200)
    metrics = sorted(e.metric_name for e in engine.all_episodes())
    assert metrics == ["train.loss", "train.loss_nonfinite_count"]


def test_the_metric_identity_reaches_the_persisted_payload():
    """The evidence is useless if it is lost at the Parquet boundary."""
    engine = EpisodeEngine()
    engine.consume([anomaly(80, "grad.global_norm", value=float("inf"))], "r")
    engine.finish("r", final_step=200)
    payload = engine.all_episodes()[0].to_dict()
    assert payload["metric_name"] == "grad.global_norm"


def test_detectors_that_are_already_per_metric_are_not_split():
    """`grad.robust_zscore` must keep sharing an episode across layers.

    Aggregate-vs-layer confirmation depends on the per-layer observations and the
    aggregate being in the SAME episode, so the scoping must not be applied to it.
    """
    engine = EpisodeEngine()
    engine.consume([anomaly(80, "grad.layer_norm", layer=0,
                            detector="grad.robust_zscore")], "r")
    engine.consume([anomaly(80, "grad.layer_norm", layer=1,
                            detector="grad.robust_zscore")], "r")
    engine.consume([anomaly(80, "grad.global_norm", detector="grad.robust_zscore")], "r")
    engine.finish("r", final_step=200)
    assert len(engine.all_episodes()) == 1, (
        "gradient layers and their aggregate must remain one episode"
    )


def test_repeated_observations_at_the_same_metric_stay_idempotent():
    engine = EpisodeEngine()
    observation = anomaly(80, "grad.global_norm", value=float("inf"))
    engine.consume([observation], "r")
    engine.consume([observation], "r")
    engine.finish("r", final_step=200)
    episodes = engine.all_episodes()
    assert len(episodes) == 1
    assert episodes[0].signal_count == 1
