"""Evaluation-only ground truth, built the same way by every corpus.

The corpora wrote `scenario` labels. The scorer reads `primary_fault_type` and
`acceptable_window`. Nothing bridged the two, and the bridge that existed -- an
enriched ground truth used by an ad-hoc command -- was never checked in. The result
was that this repository's quality figures could not be re-derived from this
repository.

This module is the bridge, so that the fields travel *with* the evidence instead of
being bolted on afterwards by a command nobody kept. Both corpora call it.

Two things are stated explicitly because they are choices, not measurements:

**The fault type is a fact about the injection.** `grad_large` is a gradient
excursion; calling it `GRADIENT_EXPLOSION` is a statement about what was done, not
about what was observed. There is nothing to infer.

**The acceptable window is a convention.** ``ACCEPTABLE_LATENCY_STEPS`` says how many
steps a detector may take after the injection and still be credited. It is an
evaluation criterion, not a property of the run, and it is the one number in this
file that could reasonably be set differently. It is therefore named, documented and
in one place, rather than repeated as a literal in two scripts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = [
    "ACCEPTABLE_LATENCY_STEPS", "SCENARIO_FAULT_TYPE", "fault_type_for",
    "acceptable_window", "evaluation_truth",
]

#: How many steps after the injection a detection may land and still be credited.
#:
#: This is an **evaluation convention**, not a measurement. It is the value the mixed
#: corpus already used, kept so that both corpora are scored by one criterion. A run
#: whose detector legitimately needs longer would be counted as a miss here, and that
#: is a property of the criterion.
ACCEPTABLE_LATENCY_STEPS = 12

#: Scenario label -> the anomaly type the injection is expected to produce.
#:
#: Covers both corpora's vocabularies. `FAULT_CLASS_TO_ANOMALY` in the library maps a
#: different set of labels (`F1_loss_spike`, `A1_single_huge_harmless_batch`, ...) and
#: does not intersect these, which is why the scorer could not resolve them.
SCENARIO_FAULT_TYPE: Dict[str, Optional[str]] = {
    # development corpus
    "healthy": None,
    "loss_spike": "LOSS_SPIKE",
    "grad_small": "GRADIENT_EXPLOSION",
    "grad_mid": "GRADIENT_EXPLOSION",
    "grad_large": "GRADIENT_EXPLOSION",
    "grad_extreme": "GRADIENT_EXPLOSION",
    "nan_loss": "NAN_DETECTED",
    # mixed-fault corpus
    "M1_gradient_then_nonfinite": "GRADIENT_EXPLOSION",
    "M2_standalone_nonfinite": "NAN_DETECTED",
    "M3_two_independent_faults": "GRADIENT_EXPLOSION",
    "M4_gradient_plus_loss_noise": "GRADIENT_EXPLOSION",
    "M5_healthy_heavy_tail": None,
    # adversarial corpus
    "A1_single_huge_harmless_batch": "GRADIENT_EXPLOSION",
    "A2_loss_spike_stable_gradients": "LOSS_SPIKE",
    "A3_gradient_spike_no_loss_consequence": "GRADIENT_EXPLOSION",
    "A4_nan_that_recovers": "NAN_DETECTED",
    "A5_inf_one_layer_only": "INF_DETECTED",
}


def fault_type_for(scenario: str) -> Optional[str]:
    """The expected anomaly type, or ``KeyError`` for a label nobody has classified.

    Deliberately loud. An unclassified label silently becomes a miss in the scorer,
    which is how a scoreboard reads 0.0 with no error -- a failure this project has
    recorded twice (D-015, D-018).
    """
    if scenario not in SCENARIO_FAULT_TYPE:
        raise KeyError(
            "scenario {!r} has no expected anomaly type. Add it to SCENARIO_FAULT_TYPE "
            "rather than letting the scorer resolve it to nothing -- an unresolved "
            "label is scored as a miss and produces a plausible wrong number."
            .format(scenario)
        )
    return SCENARIO_FAULT_TYPE[scenario]


def acceptable_window(injection_step: Optional[int]) -> Optional[list]:
    """``[injection, injection + ACCEPTABLE_LATENCY_STEPS]``, or None if not injected."""
    if injection_step is None:
        return None
    return [int(injection_step), int(injection_step) + ACCEPTABLE_LATENCY_STEPS]


def evaluation_truth(scenario: str, *, injected: bool,
                     injection_step: Optional[int] = None,
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The fields the scorer reads, for one run.

    The shape matters and is not obvious: the scorer reads ``acceptable_window`` and
    ``injection_step`` from the top level and ``primary_fault_type`` from a **nested**
    ``ground_truth`` object. Writing the fault type at the top level looks right and
    scores nothing, which is a mistake worth making once in a comment rather than
    again in a corpus.

    Returns an empty dict for a healthy run: it has no fault to match, and inventing
    a fault type for it would make an unlabelled run creditable as a true positive.
    """
    if not injected:
        return {}
    fault_type = fault_type_for(scenario)
    truth: Dict[str, Any] = {
        "acceptable_window": acceptable_window(injection_step),
        "ground_truth": {
            "primary_fault_type": fault_type,
            "allowed_secondary_manifestations": [],
        },
    }
    if extra:
        truth.update(extra)
    return truth
