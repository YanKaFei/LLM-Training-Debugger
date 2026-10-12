"""Hypothesis library and deterministic ranking.

Phase 0's rule (``ROOT_CAUSE_ARCHITECTURE.md`` section 4): the LLM never decides
what is true, and a confidence number without a visible decomposition is
unauditable. So hypotheses are:

* **templates in data**, not prompts -- reviewable in a diff;
* **scored by arithmetic** over evidence, never by a model;
* **fully decomposed** -- every factor that moved the score is stored, so a
  reviewer can recompute it by hand;
* **capped at HYPOTHESIS** for any causal claim Phase 1 cannot verify. In
  particular, data-causation is capped: without replay there is no way to confirm
  that a shard caused a spike, and Phase 0 forbids concluding it from proximity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..detectors.base import Anomaly
from ..schema.enums import EvidenceLevel, IncidentType
from ..schema.evidence import EvidenceBuckets
from ..schema.incident import Hypothesis

__all__ = ["HypothesisTemplate", "HYPOTHESIS_LIBRARY", "rank_hypotheses"]


@dataclass
class HypothesisTemplate:
    """One candidate cause with its discriminating conditions.

    ``prior`` is the base rate *before* any evidence. ``supports`` and ``refutes``
    are predicates over the incident's facts; each contributes a documented
    multiplicative factor, and every factor applied is recorded on the score.
    """

    template_id: str
    statement: str
    incident_types: Tuple[IncidentType, ...]
    prior: float
    #: fact-name -> multiplicative support factor (>1 raises the score)
    supports: Dict[str, float] = field(default_factory=dict)
    #: fact-name -> multiplicative penalty (0..1 lowers the score)
    refutes: Dict[str, float] = field(default_factory=dict)
    #: The highest level this hypothesis may claim in Phase 1.
    max_level: EvidenceLevel = EvidenceLevel.HYPOTHESIS
    #: What would be needed to promote it.
    missing: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


#: A deliberately small library. Every entry is justified by a Phase 0 taxonomy
#: entry, and every factor is a number a reviewer can argue with.
HYPOTHESIS_LIBRARY: Tuple[HypothesisTemplate, ...] = (
    HypothesisTemplate(
        template_id="gradient.explosion_precedes_loss",
        statement=(
            "A gradient explosion began before the loss deviated, so the loss "
            "change is a downstream consequence of unstable gradients."
        ),
        incident_types=(IncidentType.LOSS_SPIKE, IncidentType.LOSS_DIVERGENCE),
        prior=0.30,
        supports={"gradient_anomaly_precedes_symptom": 3.0, "gradient_anomaly_strong": 1.5},
        refutes={"gradient_norm_stable": 0.05, "lr_changed": 0.6},
        max_level=EvidenceLevel.INFERRED,
        missing=["per-layer attribution to identify which layer began first"],
        actions=[
            "inspect the per-layer gradient series around the onset",
            "compare against a healthy baseline run",
            "consider a lower peak learning rate or per-layer clipping",
        ],
    ),
    HypothesisTemplate(
        template_id="gradient.explosion_observed",
        statement=(
            "A gradient norm departed from its historical range without a "
            "non-finite value, which is the signature of an unstable step rather "
            "than of numerical breakage."
        ),
        incident_types=(
            IncidentType.GRADIENT_EXPLOSION,
            IncidentType.LOSS_SPIKE,
            IncidentType.LOSS_DIVERGENCE,
        ),
        prior=0.35,
        supports={"gradient_anomaly_strong": 3.0},
        refutes={"gradient_norm_stable": 0.05, "lr_changed": 0.7},
        max_level=EvidenceLevel.INFERRED,
        missing=["per-layer attribution to identify which layer began first"],
        actions=[
            "inspect the per-layer gradient series around the onset",
            "consider gradient clipping or a lower peak learning rate",
        ],
    ),
    HypothesisTemplate(
        template_id="optim.lr_change",
        statement="The learning rate changed at the incident, which alone can produce a loss jump.",
        incident_types=(IncidentType.LOSS_SPIKE, IncidentType.LOSS_DIVERGENCE, IncidentType.GRADIENT_EXPLOSION),
        prior=0.20,
        supports={"lr_changed": 4.0},
        refutes={"lr_stable": 0.05},
        max_level=EvidenceLevel.INFERRED,
        missing=["the scheduler configuration and the intended schedule"],
        actions=["verify the LR schedule against the run configuration", "if unintended, restore the previous LR"],
    ),
    HypothesisTemplate(
        template_id="numerics.nonfinite",
        statement=(
            "A non-finite value entered the optimisation, and the loss change is a "
            "consequence of it rather than a statistical fluctuation."
        ),
        incident_types=(
            IncidentType.LOSS_SPIKE, IncidentType.NAN_DETECTED,
            IncidentType.INF_DETECTED, IncidentType.GRADIENT_EXPLOSION,
        ),
        prior=0.25,
        supports={"nonfinite_observed": 10.0},
        refutes={"nonfinite_stable": 0.05},
        max_level=EvidenceLevel.OBSERVED,
        missing=[],
        actions=[
            "skip or roll back the affected step range",
            "inspect the last finite value of the loss and gradients",
            "if using mixed precision, inspect the loss-scale history",
        ],
    ),
    HypothesisTemplate(
        template_id="throughput.regression",
        statement=(
            "Throughput degraded at the incident, so the loss change coincides with "
            "a step-time change rather than with an optimisation instability."
        ),
        incident_types=(
            IncidentType.LOSS_SPIKE, IncidentType.THROUGHPUT_REGRESSION,
            IncidentType.LOSS_DIVERGENCE,
        ),
        prior=0.12,
        supports={"throughput_dropped": 3.0},
        refutes={"throughput_stable": 0.4},
        max_level=EvidenceLevel.INFERRED,
        missing=["a profiler trace to attribute the slowdown to compute, data or communication"],
        actions=["inspect the per-phase step-time breakdown", "check GPU telemetry for throttling"],
    ),
    HypothesisTemplate(
        template_id="data.outlier_batch",
        statement=(
            "An anomalous data batch or shard produced the observed change. Phase 1 "
            "cannot confirm this: no shard-to-step lineage is recorded and no replay "
            "exists, so this remains a hypothesis."
        ),
        incident_types=(
            IncidentType.LOSS_SPIKE, IncidentType.GRADIENT_EXPLOSION,
            IncidentType.NAN_DETECTED, IncidentType.LOSS_DIVERGENCE,
        ),
        prior=0.15,
        supports={},
        refutes={},
        # Hard cap. Even with strong circumstantial evidence this stays a
        # hypothesis until replay confirms it.
        max_level=EvidenceLevel.HYPOTHESIS,
        missing=[
            "dataset shard lineage for the affected step range",
            "a replay of the affected steps with the suspect shard excluded",
        ],
        actions=[
            "record the shard identity for the affected steps",
            "replay the affected step range before concluding anything about the data",
        ],
    ),
)


def _facts(
    anomalies: Sequence[Anomaly],
    buckets: EvidenceBuckets,
    symptom_step: Optional[int],
) -> Dict[str, Any]:
    """Derive the boolean facts the library's predicates are written against."""
    gradient = [a for a in anomalies if "grad" in str(a.detector)]
    loss = [a for a in anomalies if "loss" in str(a.detector)]
    finiteness = [a for a in anomalies if a.level is EvidenceLevel.OBSERVED]

    gradient_precedes = False
    if gradient and loss and symptom_step is not None:
        gradient_precedes = min(a.first_abnormal_step for a in gradient) < symptom_step

    stable_metrics = {e.metric_name for e in buckets.not_correlated}
    return {
        "gradient_anomaly_precedes_symptom": gradient_precedes,
        "gradient_anomaly_strong": bool(gradient),
        "gradient_norm_stable": "grad.global_norm" in stable_metrics,
        "lr_changed": "train.lr" in {e.metric_name for e in buckets.supporting if e.metric_name},
        "lr_stable": "train.lr" in stable_metrics,
        "nonfinite_observed": bool(finiteness),
        "nonfinite_stable": "train.loss_nonfinite_count" in stable_metrics
        or "grad.nonfinite_count" in stable_metrics,
        "throughput_dropped": any(
            e.metric_name in ("train.tokens_per_sec", "train.step_time")
            for e in buckets.supporting
        ),
        "throughput_stable": bool(
            {"train.tokens_per_sec", "train.step_time"} & stable_metrics
        ),
    }


def rank_hypotheses(
    incident_type: IncidentType,
    anomalies: Sequence[Anomaly],
    buckets: EvidenceBuckets,
    *,
    symptom_step: Optional[int],
) -> List[Hypothesis]:
    """Score every applicable template and return them ordered by score.

    The score is a product of the prior, every satisfied support factor, every
    triggered penalty, and a multiplicity penalty that stops a template from
    scoring highly merely because it is vague enough to fit everything.
    """
    facts = _facts(anomalies, buckets, symptom_step)
    applicable = [
        template for template in HYPOTHESIS_LIBRARY if incident_type in template.incident_types
    ]
    hypotheses: List[Hypothesis] = []

    for template in applicable:
        decomposition: Dict[str, float] = {"prior": template.prior}
        score = template.prior
        supporting: List[str] = []
        refuting: List[str] = []

        for fact, factor in sorted(template.supports.items()):
            if facts.get(fact):
                decomposition["support:" + fact] = factor
                score *= factor
                supporting.append(fact)
        for fact, penalty in sorted(template.refutes.items()):
            if facts.get(fact):
                decomposition["refute:" + fact] = penalty
                score *= penalty
                refuting.append(fact)

        # Multiplicity penalty: the more templates fit, the less any of them is
        # entitled to a high score. Without this a vague hypothesis wins simply
        # by being vague.
        multiplicity_penalty = 1.0 / max(1.0, len(applicable) ** 0.5)
        decomposition["multiplicity_penalty"] = multiplicity_penalty
        score *= multiplicity_penalty
        score = max(0.0, min(1.0, score))

        level = EvidenceLevel.HYPOTHESIS
        if facts.get("nonfinite_observed") and template.max_level is EvidenceLevel.OBSERVED:
            # The one promotion Phase 1 permits, because a finiteness finding is
            # measured rather than inferred.
            level = EvidenceLevel.OBSERVED
        elif template.max_level is EvidenceLevel.INFERRED and score >= 0.5:
            level = EvidenceLevel.INFERRED

        missing = list(template.missing)
        if level is not template.max_level:
            missing.append(
                "score {:.2f} below the promotion threshold for {}".format(score, template.max_level)
            )

        hypotheses.append(
            Hypothesis(
                template_id=template.template_id,
                statement=template.statement,
                score=score,
                status=level,
                score_decomposition=decomposition,
                supporting=supporting,
                refuting=refuting,
                missing=missing,
            )
        )

    hypotheses.sort(key=lambda h: (-h.score, h.template_id))
    return hypotheses
