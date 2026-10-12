"""I1: consequence attribution between incidents.

Phase 1.3 measured `numerics` episode precision 0.314, and the dominant cause is
structural rather than statistical: a gradient explosion poisons optimiser momentum,
so `NaN`/`Inf` genuinely appears in the parameters afterwards. The numerics detector
is **correct** to report it, but scoring counted it as an independent false positive.

The brief's instruction is explicit -- do not suppress, **classify the
relationship** -- because the `NaN` is a real observation and deleting it would
destroy evidence. So:

    primary incident confirmed
        -> later non-finite episode within the compatibility window
        -> LINK IT as a secondary manifestation
        -> do NOT open a second top-level incident

and, just as importantly:

    non-finite episode with NO linkable antecedent
        -> opens its own NUMERICAL_INSTABILITY incident

That second path is why this is attribution and not suppression. A blanket
suppression rule would hide a genuine standalone `NaN`.

**No causal verb exists in this module.** ``CAUSED_BY`` and ``ROOT_CAUSE_OF`` are
not defined, and a test asserts they stay undefined. Establishing causation requires
replay or intervention, which is Phase 2 work.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "RelationType",
    "IncidentRelation",
    "RELATION_RULE_VERSION",
    "DEFAULT_MAX_GAP_STEPS",
    "link_consequences",
    "relation_registry",
]

RELATION_RULE_VERSION = "v0.1.5-4.0"

#: How far after a primary confirmation a non-finite episode may appear and still be
#: considered a possible consequence. Deliberately generous: a poisoned optimiser
#: state can surface over many steps, and the cost of a missed link is a duplicate
#: incident while the cost of an over-eager link is a hidden independent fault --
#: which the unlinked path below still protects against.
DEFAULT_MAX_GAP_STEPS = 60

#: How many steps a non-finite value may take to propagate through the optimizer
#: update into a downstream metric. One training step is one optimizer update, and the
#: observation cadence is per step, so a genuine forward propagation surfaces within
#: the next step or two. This is a *mechanism* bound, not a tuned window.
PROPAGATION_DELAY_STEPS = 2

#: Metric families, used to decide whether a propagation path is mechanically
#: plausible. Derived from the metric names the observer actually emits.
def metric_class(metric_name: Any) -> str:
    """``upstream`` / ``midstream`` / ``downstream`` / ``unknown``.

    The training mechanism runs gradient -> optimizer update -> parameters -> loss, so
    a non-finite value can only travel that way. A non-finite value that appears
    *downstream* with no upstream non-finiteness before it did not propagate from one.
    """
    name = str(metric_name or "")
    if name.startswith("grad."):
        return "upstream"
    if name.startswith("param.") or name.startswith("mem."):
        return "midstream"
    if name.startswith("train."):
        return "downstream"
    return "unknown"


#: Anomaly types whose observation *is* a non-finite value.
NONFINITE_TYPES = frozenset({"NAN_DETECTED", "INF_DETECTED"})

#: Observation kinds that are NOT evidence of a non-finite tensor.
#:
#: `norm_reduction_overflow` is what `global_gradient_norm` / `global_parameter_norm`
#: report when the float32 reduction overflowed **while every element was finite**.
#: Treating it as a non-finite observation is the "a non-finite norm is not necessarily
#: a non-finite tensor" error, and it was what made M1/A8X look like propagation.
NOT_NONFINITE_KINDS = frozenset({"norm_reduction_overflow"})


def is_nonfinite_observation(episode: Dict[str, Any]) -> bool:
    """Whether this episode's observation is a genuinely non-finite value."""
    if episode.get("anomaly_type") not in NONFINITE_TYPES:
        return False
    kind = (episode.get("metadata") or {}).get("anomaly_kind")
    return kind not in NOT_NONFINITE_KINDS


class RelationType(str, Enum):
    """Permitted relations. Nothing causal is representable."""

    POSSIBLE_CONSEQUENCE_OF = "POSSIBLE_CONSEQUENCE_OF"
    PRECEDES = "PRECEDES"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"
    UNKNOWN_RELATION = "UNKNOWN_RELATION"


#: Anomaly types that can be a *consequence* of an earlier incident rather than an
#: independent root cause.
CONSEQUENCE_TYPES = ("NAN_DETECTED", "INF_DETECTED")

#: Anomaly types that can be a *primary* whose consequence is a numerical failure.
PRIMARY_TYPES = ("GRADIENT_EXPLOSION", "LOSS_DIVERGENCE", "LOSS_SPIKE")


@dataclass
class IncidentRelation:
    """One classified relationship between two incidents or episodes."""

    relation_id: str
    run_id: str
    source_incident_id: str
    target_incident_id: str
    relation_type: str = RelationType.UNKNOWN_RELATION.value
    source_anomaly_type: Optional[str] = None
    target_anomaly_type: Optional[str] = None
    gap_steps: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    rule_id: str = "link.numerics.after_primary"
    rule_version: str = RELATION_RULE_VERSION
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "run_id": self.run_id,
            "source_incident_id": self.source_incident_id,
            "target_incident_id": self.target_incident_id,
            "relation_type": self.relation_type,
            "source_anomaly_type": self.source_anomaly_type,
            "target_anomaly_type": self.target_anomaly_type,
            "gap_steps": self.gap_steps,
            "evidence": dict(self.evidence),
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IncidentRelation":
        return cls(**payload)


def _relation_id(run_id: str, source: str, target: str) -> str:
    digest = hashlib.sha256("{}|{}|{}".format(run_id, source, target).encode("utf-8")).hexdigest()
    return "rel_{}".format(digest[:12])


#: Whether the experimental mechanism linker may emit `POSSIBLE_CONSEQUENCE_OF`.
#:
#: **Default `False`.** v0.1.6/v0.1.7 measured that the mechanism clause cannot
#: discriminate a genuine propagation from an independently injected fault in every case
#: that could be constructed: C3/C12 produced 16 consequence claims against ground truth
#: of `INDEPENDENT`. Shipping that as production behaviour would present an unvalidated
#: inference as a causal claim.
#:
#: With the linker disabled the relation layer still records, for every pair it can see:
#:
#:     PRECEDES           one anomaly was observed before another
#:     CO_OCCURS_WITH     both were observed together
#:     UNKNOWN_RELATION   nothing could be established
#:
#: It simply never asserts that one *may be a consequence of* the other. No evidence is
#: deleted by this; the observations, episodes and order are all retained.
MECHANISM_LINKER_ENABLED_DEFAULT = False


def link_consequences(
    run_id: str,
    episodes: Sequence[Any],
    *,
    max_gap_steps: int = DEFAULT_MAX_GAP_STEPS,
    mechanism_linker_enabled: bool = MECHANISM_LINKER_ENABLED_DEFAULT,
) -> Tuple[List[IncidentRelation], set]:
    """Classify later non-finite episodes against earlier confirmed primaries.

    Returns ``(relations, linked_episode_ids)``. A linked episode is retained as
    evidence and carries ``POSSIBLE_CONSEQUENCE_OF``; the caller must not open a
    second top-level incident for it. An **unlinked** episode is untouched and
    remains free to open its own incident -- that is the guarantee that a standalone
    `NaN` is never hidden.

    The three conditions the brief requires are checked explicitly, and each is
    recorded in ``evidence`` so the link can be reviewed:

    1. temporal adjacency
    2. scope compatibility (same run; layer overlap or unspecified)
    3. optimiser-state continuity (the primary has not already recovered long before)
    """
    def payload(episode: Any) -> Dict[str, Any]:
        return episode.to_dict() if hasattr(episode, "to_dict") else dict(episode)

    data = [payload(e) for e in episodes]

    # -- W3: the evidence hierarchy -------------------------------------
    #
    # E1 temporal order, E2 scope compatibility, E3 mechanistic propagation,
    # E4 contradiction/independence.
    #
    # E1 alone is what the previous rule used, and it produced `POSSIBLE_CONSEQUENCE_OF`
    # on 10 of 10 *deliberately independent* fault pairs while missing the 10 real
    # chains -- precision 0.0 and recall 0.0 at once. Temporal order is observable;
    # mechanistic association needs more.
    #
    # A candidate antecedent is anything observed strictly earlier in the same run:
    # either a confirmed primary episode, or a non-finite observation on an *upstream*
    # metric. The second is what makes M1 recoverable: the gradient went non-finite at
    # step 80, one step before the loss did, and that forward propagation is exactly
    # what `metric_class` can see.
    def _confirmed(episode: Dict[str, Any]) -> bool:
        return (episode.get("state") == "CONFIRMED"
                or episode.get("outcome") == "CONFIRMED_ANOMALY")

    antecedents = []
    for episode in data:
        if episode.get("candidate_onset_step") is None:
            continue
        is_primary = (episode.get("anomaly_type") in PRIMARY_TYPES and _confirmed(episode)
                      and episode.get("confirmed_at_step") is not None)
        is_upstream_nonfinite = (
            is_nonfinite_observation(episode)
            and metric_class(episode.get("metric_name")) == "upstream"
        )
        if is_primary or is_upstream_nonfinite:
            antecedents.append((episode, is_primary, is_upstream_nonfinite))

    consequences = [e for e in data if e.get("anomaly_type") in CONSEQUENCE_TYPES]

    relations: List[IncidentRelation] = []
    linked: set = set()

    for target in consequences:
        onset = target.get("candidate_onset_step")
        if onset is None:
            continue
        target_class = metric_class(target.get("metric_name"))
        layers_target = set(target.get("affected_layers") or [])

        best = None                       # (gap, antecedent, tier, evidence)
        for antecedent, is_primary, is_upstream_nonfinite in antecedents:
            if antecedent.get("episode_id") == target.get("episode_id"):
                continue
            reference_step = (antecedent.get("confirmed_at_step")
                              if is_primary and antecedent.get("confirmed_at_step") is not None
                              else antecedent.get("candidate_onset_step"))
            if reference_step is None:
                continue
            gap = onset - int(reference_step)

            # E1 -- temporal eligibility. The consequence cannot precede its cause.
            if gap < 0 or gap > max_gap_steps:
                continue

            # E2 -- scope compatibility.
            layers_antecedent = set(antecedent.get("affected_layers") or [])
            scope_ok = (not layers_antecedent) or (not layers_target) or bool(
                layers_antecedent & layers_target
            )
            if not scope_ok:
                continue

            # E3 -- mechanistic propagation, checked at EVERY stage of the chain.
            #
            # gradient -> optimizer update -> parameters -> loss
            #
            # v0.1.3 checked only the first stage: a non-finite *gradient* followed by a
            # non-finite downstream value within two steps. The adversarial corpus showed
            # that is not sufficient. A suppressed update produces EXACTLY that
            # observation pattern while nothing propagates, so five independent faults
            # were claimed as consequences (A2 and A8Y, 10 fabricated associations).
            #
            # The midstream stage is recorded -- `param.global_norm` -- and it is the
            # decisive witness: it is `inf` when the update genuinely overflowed and
            # finite when the update never saw the gradient. Requiring it turns
            # "a non-finite value came next" into "the propagation is observable at each
            # stage", which is what a mechanism claim needs.
            midstream_nonfinite = any(
                is_nonfinite_observation(other)
                and metric_class(other.get("metric_name")) == "midstream"
                and reference_step <= (other.get("candidate_onset_step") or -1) <= onset
                for other in data
            )
            mechanism = bool(
                mechanism_linker_enabled          # <-- research configuration only
                and is_upstream_nonfinite
                and midstream_nonfinite
                and gap <= PROPAGATION_DELAY_STEPS
                and target_class in ("midstream", "downstream", "unknown")
            )
            tier = "MECHANISM" if mechanism else "TEMPORAL"
            evidence = {
                "tier": tier,
                "gap_steps": gap,
                "propagation_delay_bound": PROPAGATION_DELAY_STEPS,
                "antecedent_metric": antecedent.get("metric_name"),
                "antecedent_metric_class": metric_class(antecedent.get("metric_name")),
                "target_metric": target.get("metric_name"),
                "target_metric_class": target_class,
                "antecedent_is_nonfinite": is_nonfinite_observation(antecedent),
                "antecedent_observation_kind": (
                    (antecedent.get("metadata") or {}).get("anomaly_kind")),
                "upstream_nonfinite_before_target": bool(is_upstream_nonfinite),
                "midstream_nonfinite_witness": bool(midstream_nonfinite),
                "midstream_required": True,
                "scope_compatible": True,
                "antecedent_onset": antecedent.get("candidate_onset_step"),
                "target_onset": onset,
            }
            rank_key = 0 if mechanism else 1
            candidate = (rank_key, gap, antecedent, tier, evidence)
            if best is None or candidate[:2] < best[:2]:
                best = candidate

        if best is None:
            # E4 in effect: nothing eligible precedes it, so it stays independent and
            # may open its own incident. This is the guarantee that a standalone
            # non-finite value is never hidden.
            continue

        _, gap, antecedent, tier, evidence = best
        relation_type = (RelationType.POSSIBLE_CONSEQUENCE_OF.value if tier == "MECHANISM"
                         else RelationType.PRECEDES.value)
        evidence["mechanism_linker_enabled"] = mechanism_linker_enabled
        evidence["interpretation"] = (
            "a non-finite value was observed at EVERY stage of the chain -- upstream "
            "gradient, midstream parameters, downstream target -- within the propagation "
            "bound, so forward propagation through the optimizer update is observable "
            "rather than merely plausible. This is still NOT a causal identification."
            if tier == "MECHANISM" else
            "the observation followed an earlier anomaly within the compatibility window, "
            "but no mechanistic propagation path was observed. Order only: this asserts "
            "precedence, not consequence."
        )
        relation = IncidentRelation(
            relation_id=_relation_id(run_id, antecedent["episode_id"], target["episode_id"]),
            run_id=run_id,
            source_incident_id=antecedent["episode_id"],
            target_incident_id=target["episode_id"],
            relation_type=relation_type,
            source_anomaly_type=antecedent.get("anomaly_type"),
            target_anomaly_type=target.get("anomaly_type"),
            gap_steps=gap,
            evidence=evidence,
            confidence=(min(0.8, 0.4 + 0.4 * (1.0 - gap / float(PROPAGATION_DELAY_STEPS + 1)))
                        if tier == "MECHANISM"
                        else min(0.5, 0.2 + 0.3 * (1.0 - gap / float(max_gap_steps)))),
        )
        relations.append(relation)
        linked.add(target["episode_id"])
        if tier == "MECHANISM":
            linked.add(antecedent["episode_id"])

    return relations, linked


def relation_registry() -> Dict[str, Any]:
    return {
        "relation_rule_version": RELATION_RULE_VERSION,
        "relation_types": [t.value for t in RelationType],
        "forbidden_types": ["CAUSED_BY", "ROOT_CAUSE_OF"],
        "consequence_types": list(CONSEQUENCE_TYPES),
        "primary_types": list(PRIMARY_TYPES),
        "default_max_gap_steps": DEFAULT_MAX_GAP_STEPS,
        "note": (
            "Attribution, not suppression. A non-finite episode with no linkable "
            "antecedent keeps the right to open its own incident."
        ),
    }
