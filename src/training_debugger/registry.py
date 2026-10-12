"""E15: the detector registry.

A detector's existence and its trustworthiness are different facts, and Phase 1.2
showed the second one is easy to lose. This registry records, per detector and
version, whether it is ENABLED, EXPERIMENTAL or DISABLED, and why.

Two detectors are DISABLED here on measured evidence:

* ``loss.trend_slope`` v1 -- precision 0.000, recall 0.000 on the Phase 1.2
  holdout. It fires only before faults and never inside the acceptable window.
* ``throughput.regression`` v1 -- recall 0.250 with 9 pre-injection false positives.

They are not deleted. A disabled detector stays readable so historical incidents
that reference it still resolve, and so its failure remains measurable.

Capability honesty (brief section 16) is a field, not a promise: an anomaly type
that no enabled detector can reliably produce is marked ``UNSUPPORTED`` here rather
than left out and silently implied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .episodes.engine import EPISODE_RULES

__all__ = ["DetectorRegistryEntry", "DETECTOR_REGISTRY", "detector_registry", "write_registry"]

STATUS_ENABLED = "ENABLED"
STATUS_EXPERIMENTAL = "EXPERIMENTAL"
STATUS_DISABLED = "DISABLED"

#: R1: the four-state capability model, replacing the two-state
#: SUPPORTED/UNSUPPORTED that Phase 1.3 over-claimed.
#:
#:     IMPLEMENTED   code exists
#:     EXPERIMENTAL  works in some scenarios but fails the reliability gate
#:     VALIDATED     passes the frozen holdout quality gate
#:     UNSUPPORTED   unavailable, or deliberately disabled
#:
#: Phase 1.3 labelled GRADIENT_EXPLOSION and the numerics types SUPPORTED while
#: their measured episode precision was 0.181 and 0.314. Nothing in this build may
#: be called VALIDATED until a holdout that has never been looked at says so.
CAPABILITY_IMPLEMENTED = "IMPLEMENTED"
CAPABILITY_EXPERIMENTAL = "EXPERIMENTAL"
CAPABILITY_VALIDATED = "VALIDATED"
CAPABILITY_UNSUPPORTED = "UNSUPPORTED"

#: Per-capability quality gate a capability must pass on a FROZEN holdout before it
#: may be promoted to VALIDATED.
VALIDATION_GATE = {"episode_precision": 0.80, "episode_recall": 0.80}

ANOMALY_TYPE_CAPABILITY = {
    # Measured episode precision 0.314 (Phase 1.3). Recall is complete; precision
    # is limited by consequence manifestations raising their own episodes.
    "NAN_DETECTED": CAPABILITY_EXPERIMENTAL,
    "INF_DETECTED": CAPABILITY_EXPERIMENTAL,
    # Measured episode precision 0.181 (Phase 1.3).
    "GRADIENT_EXPLOSION": CAPABILITY_EXPERIMENTAL,
    # H2 noisy healthy loss alarmed in 8 of 14 runs; the brief's condition for
    # de-enabling was met. Kept EXPERIMENTAL and reported as a gate failure.
    "LOSS_SPIKE": CAPABILITY_EXPERIMENTAL,
    "LOSS_DIVERGENCE": CAPABILITY_UNSUPPORTED,
    "THROUGHPUT_REGRESSION": CAPABILITY_UNSUPPORTED,
}

#: Retained for callers that predate the four-state model.
SUPPORTED_ANOMALY_TYPES = ANOMALY_TYPE_CAPABILITY


@dataclass
class DetectorRegistryEntry:
    detector_id: str
    version: str
    status: str
    supported_anomaly_types: List[str] = field(default_factory=list)
    minimum_history: int = 0
    candidate_rule: str = ""
    confirmation_rule: str = ""
    recovery_rule: str = ""
    known_failure_modes: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    disabled_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "version": self.version,
            "status": self.status,
            "supported_anomaly_types": list(self.supported_anomaly_types),
            "minimum_history": self.minimum_history,
            "candidate_rule": self.candidate_rule,
            "confirmation_rule": self.confirmation_rule,
            "recovery_rule": self.recovery_rule,
            "known_failure_modes": list(self.known_failure_modes),
            "quality_metrics": dict(self.quality_metrics),
            "capability": {
                t: ANOMALY_TYPE_CAPABILITY.get(t, CAPABILITY_UNSUPPORTED)
                for t in self.supported_anomaly_types
            },
            "disabled_reason": self.disabled_reason,
        }


_REG = EPISODE_RULES

DETECTOR_REGISTRY: tuple = (
    DetectorRegistryEntry(
        detector_id="activation.health",
        version="1.0",
        status=STATUS_ENABLED,
        # Observational only. The states are threshold-free, so there is no
        # "vanishing"/"exploding" here -- those need a reference this project does not
        # have for activations, and inventing one is the mistake that has already cost
        # this repository two retractions.
        supported_anomaly_types=["ACTIVATION_DEAD", "ACTIVATION_NONFINITE",
                                 "ACTIVATION_SATURATED"],
        minimum_history=0,
        candidate_rule=(
            "act.zero_fraction >= 1.0 (every element exactly zero), "
            "act.nonfinite_count > 0 (a NaN or infinite element), or "
            "act.abs_max >= float32 max (saturation at the dtype boundary)"
        ),
        confirmation_rule=(
            "OBSERVED: confidence 1.0 on first observation. Every one of the three "
            "conditions is exact and needs no threshold, no window and no reference"
        ),
        recovery_rule="{} steps without a further finding".format(
            _REG["NAN_DETECTED"].recovery_steps
        ),
        known_failure_modes=[
            "Requires the opt-in ActivationProbe. With no probe attached no activation "
            "events exist and the detector never fires -- which is the default.",
            "Emitting five metrics per module per step measured +24.7 % overhead; "
            "edge-triggered emission (+3.3 %) is the default for that reason.",
            "A sampled observation cannot establish DEAD by absence: a sample of zeros "
            "does not prove the whole tensor is zero, so that case returns UNKNOWN "
            "rather than a claim.",
            "Compute cost is +2.7 % before emission, on a baseline that already misses "
            "the <3 % observer target.",
        ],
    ),
    DetectorRegistryEntry(
        detector_id="numerics.finiteness",
        version="1.0",
        status=STATUS_ENABLED,
        supported_anomaly_types=["NAN_DETECTED", "INF_DETECTED"],
        minimum_history=0,
        candidate_rule="any metric value is NaN or Inf",
        confirmation_rule=_REG["NAN_DETECTED"].persistence_required and
        "CERTAIN: confirms on first observation (no persistence required, because a "
        "measured non-finite value cannot be a statistical accident)",
        recovery_rule="{} steps without a non-finite observation".format(
            _REG["NAN_DETECTED"].recovery_steps
        ),
        known_failure_modes=[
            "A persistent non-finite condition previously produced one finding per "
            "step; now collapses to one episode by contiguity.",
            "A non-finite value in a metric that is not emitted (sampled out) is invisible.",
        ],
        quality_metrics={
            "phase1.2_holdout_precision": 0.478,
            "phase1.2_holdout_recall": 1.0,
            "phase1.2_post_fault_false_positives": 24,
            "note": "precision was 0.026 before the episode model and cooldown; "
                    "see E6_NUMERICS_REPORT.md for the episode-level figure",
        },
    ),
    DetectorRegistryEntry(
        detector_id="grad.robust_zscore",
        version="2.0",
        status=STATUS_ENABLED,
        supported_anomaly_types=["GRADIENT_EXPLOSION"],
        minimum_history=50,
        candidate_rule="value > min(median(window) * 3.0, quantile(window, 0.995) * 1.5), "
                       "window excluding warm-up",
        confirmation_rule=(
            "aggregate series departed, OR cross-layer multiplicity >= {:.0%}, OR peaking "
            ">= {:.0f}x, OR {} observations within {} steps".format(
                _REG["GRADIENT_EXPLOSION"].multiplicity_confirm,
                _REG["GRADIENT_EXPLOSION"].magnitude_override,
                _REG["GRADIENT_EXPLOSION"].persistence_required,
                _REG["GRADIENT_EXPLOSION"].persistence_window,
            )
        ),
        recovery_rule="{} steps without a breach".format(_REG["GRADIENT_EXPLOSION"].recovery_steps),
        known_failure_modes=[
            "Healthy heavy-tailed gradients: measured 1.25 FP/1k on the H1 class.",
            "Detection boundary between x2 (recall 0.75) and x3 (recall 1.00); not sharp.",
            "A lone layer excursion clears its own envelope easily; mitigated by WEAK "
            "classification but present in raw output.",
        ],
        quality_metrics={
            "phase1.2_holdout_precision": 0.063,
            "phase1.2_holdout_recall": 0.78,
            "envelope_quantile": 0.995,
            "envelope_factor": 1.5,
            "median_multiple": 3.0,
        },
    ),
    DetectorRegistryEntry(
        detector_id="loss.robust_zscore",
        version="1.0",
        status=STATUS_ENABLED,
        supported_anomaly_types=["LOSS_SPIKE"],
        minimum_history=50,
        candidate_rule="detrended robust z-score above spike_z, or value >= "
                       "relative_factor * window median",
        confirmation_rule="{} observations within {} steps, or peak ratio >= {:.0f}x".format(
            _REG["LOSS_SPIKE"].persistence_required,
            _REG["LOSS_SPIKE"].persistence_window,
            _REG["LOSS_SPIKE"].magnitude_override,
        ),
        recovery_rule="{} steps without a breach".format(_REG["LOSS_SPIKE"].recovery_steps),
        known_failure_modes=[
            "A real but benign loss spike (a large batch) is indistinguishable from a fault.",
            "Cannot distinguish a data-caused spike from an optimisation-caused one.",
            "Post-fault duplicates under sustained divergence.",
        ],
        quality_metrics={
            "phase1.2_holdout_precision": 0.700,
            "phase1.2_holdout_recall": 1.0,
        },
    ),
    DetectorRegistryEntry(
        detector_id="loss.trend_slope",
        version="1.0",
        status=STATUS_DISABLED,
        supported_anomaly_types=["LOSS_DIVERGENCE"],
        minimum_history=50,
        candidate_rule="OLS slope over a trailing window",
        confirmation_rule="n/a -- disabled",
        recovery_rule="n/a -- disabled",
        known_failure_modes=[
            "Fires only BEFORE faults: 16 pre-injection false positives.",
            "Never fires inside the acceptable window: recall 0.000 across a magnitude "
            "sweep of sustained divergence (1.006, 1.012, 1.03).",
            "A brief positive slope is not divergence; the criterion has no persistence "
            "requirement and no baseline normalisation.",
        ],
        quality_metrics={
            "phase1.2_holdout_precision": 0.0,
            "phase1.2_holdout_recall": 0.0,
            "phase1.2_pre_injection_false_positives": 16,
        },
        disabled_reason=(
            "Measured precision 0.000 and recall 0.000 on the Phase 1.2 holdout. It "
            "does not detect the fault it exists for and pollutes default system "
            "metrics. LOSS_DIVERGENCE is therefore declared UNSUPPORTED until a "
            "persistence-gated, baseline-normalised implementation is validated."
        ),
    ),
    DetectorRegistryEntry(
        detector_id="throughput.regression",
        version="1.0",
        status=STATUS_DISABLED,
        supported_anomaly_types=["THROUGHPUT_REGRESSION"],
        minimum_history=100,
        candidate_rule="step time above a trailing baseline by throughput_drop_ratio for "
                       "throughput_consecutive steps",
        confirmation_rule="n/a -- disabled",
        recovery_rule="n/a -- disabled",
        known_failure_modes=[
            "Recall 0.250 on the Phase 1.2 holdout.",
            "9 pre-injection false positives: it alarms during warm-up.",
            "Compares raw step time without normalising for tokens, batch size, sequence "
            "length or gradient accumulation.",
        ],
        quality_metrics={
            "phase1.2_holdout_precision": 0.100,
            "phase1.2_holdout_recall": 0.250,
            "phase1.2_pre_injection_false_positives": 9,
        },
        disabled_reason=(
            "Recall 0.250 with 9 pre-injection false positives. Not usable as a default "
            "incident source. THROUGHPUT_REGRESSION is declared UNSUPPORTED in this build."
        ),
    ),
)


def detector_registry() -> Dict[str, Any]:
    return {
        "registry_version": "phase1.3-1.0",
        "generated_by": "training_debugger.registry",
        "enabled_detector_ids": [
            entry.detector_id for entry in DETECTOR_REGISTRY if entry.status == STATUS_ENABLED
        ],
        "capability_states": [
            CAPABILITY_IMPLEMENTED, CAPABILITY_EXPERIMENTAL,
            CAPABILITY_VALIDATED, CAPABILITY_UNSUPPORTED,
        ],
        "anomaly_type_capability": dict(ANOMALY_TYPE_CAPABILITY),
        "validation_gate": dict(VALIDATION_GATE),
        "validated_capabilities": sorted(
            name for name, state in ANOMALY_TYPE_CAPABILITY.items()
            if state == CAPABILITY_VALIDATED
        ),
        "detectors": [entry.to_dict() for entry in DETECTOR_REGISTRY],
        "note": (
            "A DISABLED detector is not deleted: historical incidents that reference "
            "its detector_id must still resolve, and its measured failure must remain "
            "auditable. UNSUPPORTED means no enabled detector can reliably produce that "
            "anomaly type in this build."
        ),
    }


def write_registry(path: Any) -> None:
    Path(path).write_text(json.dumps(detector_registry(), indent=2), encoding="utf-8")
