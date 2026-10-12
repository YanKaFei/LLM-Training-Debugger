"""The Phase 1 scenario corpus: A0-A4 as declarative data.

Each scenario states, up front, what it injects and what the detectors are
required to say about it -- including which detectors must **not** fire. That
last field is the one that matters most: a detector suite can trivially reach
100% recall by firing always, so the corpus pins precision too.

Scenarios are data, not code paths, so a golden-file comparison can be made over
the structured result instead of over prose (brief section 12 forbids prose
snapshots).

Determinism: every scenario carries a seed and the generators are pure functions
of it. Two runs of the same scenario produce byte-identical event streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Scenario", "SCENARIOS", "get_scenario"]


@dataclass
class Scenario:
    """One injected failure with its expected detection outcome."""

    name: str
    description: str
    seed: int
    steps: int = 300

    # -- injection (all optional; absent means "do not inject") -------------
    spike_at: Optional[int] = None
    spike_factor: float = 5.0
    grad_explode_at: Optional[int] = None
    grad_explode_factor: float = 25.0
    nan_at: Optional[int] = None
    inf_at: Optional[int] = None
    throughput_drop_at: Optional[int] = None
    throughput_drop_factor: float = 4.0
    layer_count: int = 6

    # -- expectations ------------------------------------------------------
    #: Detector ids that MUST fire.
    expected_detectors: List[str] = field(default_factory=list)
    #: Detector ids that must NOT fire anywhere in the run.
    forbidden_detectors: List[str] = field(default_factory=list)
    #: Inclusive step window in which the onset must land.
    expected_onset_window: Optional[Tuple[int, int]] = None
    #: Expected incident classification, if the scenario produces one.
    expected_incident_type: Optional[str] = None
    #: A specific layer index that must appear in the multiplicity, when known.
    expected_layer: Optional[int] = None

    def injection_summary(self) -> Dict[str, Any]:
        return {
            "spike_at": self.spike_at,
            "spike_factor": self.spike_factor,
            "grad_explode_at": self.grad_explode_at,
            "grad_explode_factor": self.grad_explode_factor,
            "nan_at": self.nan_at,
            "inf_at": self.inf_at,
            "throughput_drop_at": self.throughput_drop_at,
            "throughput_drop_factor": self.throughput_drop_factor,
            "layer_count": self.layer_count,
        }


# ---------------------------------------------------------------------------
# A0 -- healthy control
# ---------------------------------------------------------------------------

A0_HEALTHY = Scenario(
    name="A0_healthy",
    description=(
        "A healthy run with ordinary noise and no injection. Its purpose is to "
        "measure the false-positive rate: any detector firing here is a bug in "
        "the detector or in its warm-up policy."
    ),
    seed=1001,
    steps=400,
    expected_detectors=[],
    forbidden_detectors=[
        "loss.robust_zscore",
        "loss.trend_slope",
        "grad.robust_zscore",
        "numerics.finiteness",
        "throughput.regression",
    ],
)

# ---------------------------------------------------------------------------
# A1 -- loss spike (THE release gate)
# ---------------------------------------------------------------------------

A1_LOSS_SPIKE = Scenario(
    name="A1_loss_spike",
    description=(
        "A gradient anomaly at step 210 followed by a loss spike from step 220. "
        "The gap between the two is the point of the scenario: the loss spike is "
        "the *symptom* the user notices, and the gradient rise is the earlier "
        "abnormal evidence the product must surface."
    ),
    seed=2002,
    steps=320,
    grad_explode_at=210,
    grad_explode_factor=12.0,
    spike_at=220,
    spike_factor=4.5,
    expected_detectors=["loss.robust_zscore", "grad.robust_zscore"],
    forbidden_detectors=["numerics.finiteness"],
    expected_onset_window=(205, 225),
    expected_incident_type="LOSS_SPIKE",
)

A1B_LOSS_SPIKE_NO_GRADIENT = Scenario(
    name="A1b_loss_spike_without_gradient_precursor",
    description=(
        "A loss spike with no gradient precursor. The engine must still open an "
        "incident, and must NOT claim a gradient explosion that did not happen. "
        "This is the scenario that catches an engine which always blames the "
        "gradient because that is the story it prefers."
    ),
    seed=2003,
    steps=320,
    spike_at=220,
    spike_factor=4.5,
    expected_detectors=["loss.robust_zscore"],
    forbidden_detectors=["grad.robust_zscore", "numerics.finiteness"],
    expected_onset_window=(215, 225),
    expected_incident_type="LOSS_SPIKE",
)

# ---------------------------------------------------------------------------
# A2 -- gradient explosion
# ---------------------------------------------------------------------------

A2_GRADIENT_EXPLOSION = Scenario(
    name="A2_gradient_explosion",
    description=(
        "A single layer's gradient explodes at step 150 while the global norm "
        "stays comparatively modest. Tests per-layer attribution, which is the "
        "aggregation-masking confounder from Phase 0."
    ),
    seed=3003,
    steps=280,
    grad_explode_at=150,
    grad_explode_factor=30.0,
    layer_count=6,
    expected_detectors=["grad.robust_zscore"],
    forbidden_detectors=["numerics.finiteness"],
    expected_onset_window=(145, 160),
    expected_incident_type="GRADIENT_EXPLOSION",
)

# ---------------------------------------------------------------------------
# A3 -- NaN
# ---------------------------------------------------------------------------

A3_NAN = Scenario(
    name="A3_nan",
    description=(
        "The loss becomes NaN at step 180. This must be an OBSERVED finding with "
        "confidence 1.0, and it must not be reported as a mere statistical spike."
    ),
    seed=4004,
    steps=260,
    nan_at=180,
    expected_detectors=["numerics.finiteness"],
    expected_onset_window=(180, 180),
    expected_incident_type="NAN_DETECTED",
)

A3B_INF = Scenario(
    name="A3b_infinite_gradient",
    description="A gradient becomes infinite at step 170.",
    seed=4005,
    steps=260,
    inf_at=170,
    expected_detectors=["numerics.finiteness"],
    expected_onset_window=(170, 170),
    expected_incident_type="INF_DETECTED",
)

# ---------------------------------------------------------------------------
# A4 -- throughput regression
# ---------------------------------------------------------------------------

A4_THROUGHPUT = Scenario(
    name="A4_throughput_regression",
    description=(
        "Step time quadruples from step 160 onward. The detector must report the "
        "*symptom* and the onset, and must record that no root cause was "
        "identified -- Phase 1 does not attribute throughput drops to a GPU, a "
        "filesystem or a collective."
    ),
    seed=5005,
    steps=300,
    throughput_drop_at=160,
    throughput_drop_factor=4.0,
    expected_detectors=["throughput.regression"],
    forbidden_detectors=["loss.robust_zscore", "numerics.finiteness"],
    expected_onset_window=(155, 175),
    expected_incident_type="THROUGHPUT_REGRESSION",
)


SCENARIOS: List[Scenario] = [
    A0_HEALTHY,
    A1_LOSS_SPIKE,
    A1B_LOSS_SPIKE_NO_GRADIENT,
    A2_GRADIENT_EXPLOSION,
    A3_NAN,
    A3B_INF,
    A4_THROUGHPUT,
]

_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


def get_scenario(name: str) -> Scenario:
    if name not in _BY_NAME:
        raise KeyError("unknown scenario %r; known: %s" % (name, sorted(_BY_NAME)))
    return _BY_NAME[name]
