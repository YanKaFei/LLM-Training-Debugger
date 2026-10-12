"""B1: incident reconciliation.

Phase 1.9 measured **23 duplicate top-level incidents**. That number conflated three
genuinely different situations, and the brief requires them separated before anything
is changed:

```
A. two genuinely independent anomalies        -> must NOT be merged
B. one anomaly with a plausible consequence   -> may form one Incident Group
C. the same episode emitted twice              -> an idempotency defect, not a merge
```

Only **B** is reconciliation. **A** is wrongly merged if the association is weak, and
**C** is a software defect that grouping would paper over. Conflating them would let a
grouping rule "reduce duplicates" by merging faults that were never related.

**A group is not a root cause.** ``IncidentGroup`` says "these manifestations are
presented together because a relation was recorded"; it never says one caused the
other. ``has_proven_root_cause`` is therefore **always False** in this build, and the
only relation types reachable are the weak ones -- ``CAUSED_BY`` and ``ROOT_CAUSE_OF``
do not exist in the type system.

**No evidence is deleted.** Every episode id, relation and non-finite observation
survives; grouping changes presentation, never the record.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["IncidentGroup", "reconcile_incidents", "ReconciliationResult"]


#: Only a *consequence* claim may group a manifestation with a primary.
#:
#: v0.1.3 narrowed this. `PRECEDES` asserts order and nothing more, so grouping on it
#: would present two independently-classified episodes as one incident on the strength
#: of "one happened before the other" -- the same temporal-adjacency error the relation
#: rule was just repaired for, one layer up. Order is recorded and remains auditable;
#: it does not merge.
#:
#: `CO_OCCURS_WITH` and `UNKNOWN_RELATION` likewise leave the two independent: an
#: unclassified or merely concurrent association is not evidence of a shared root.
ATTACHING_RELATIONS = ("POSSIBLE_CONSEQUENCE_OF",)


@dataclass
class IncidentGroup:
    group_id: str
    run_id: str
    primary_episode_id: str
    primary_anomaly_type: Optional[str] = None
    secondary_episode_ids: List[str] = field(default_factory=list)
    secondary_anomaly_types: List[str] = field(default_factory=list)
    relation_ids: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    association_evidence: List[Dict[str, Any]] = field(default_factory=list)
    #: Always False before replay. A group is a presentation unit, not a proven cause.
    has_proven_root_cause: bool = False
    presentation_note: str = (
        "These manifestations are presented together because a temporal/scope relation "
        "was recorded. This grouping does NOT establish that one caused the other."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "run_id": self.run_id,
            "primary_episode_id": self.primary_episode_id,
            "primary_anomaly_type": self.primary_anomaly_type,
            "secondary_episode_ids": list(self.secondary_episode_ids),
            "secondary_anomaly_types": list(self.secondary_anomaly_types),
            "relation_ids": list(self.relation_ids),
            "relation_types": list(self.relation_types),
            "association_evidence": list(self.association_evidence),
            "has_proven_root_cause": self.has_proven_root_cause,
            "presentation_note": self.presentation_note,
            "member_count": 1 + len(self.secondary_episode_ids),
        }


@dataclass
class ReconciliationResult:
    run_id: str
    groups: List[IncidentGroup] = field(default_factory=list)
    independent_episode_ids: List[str] = field(default_factory=list)
    unresolved_relation_ids: List[str] = field(default_factory=list)
    duplicate_episode_ids: List[str] = field(default_factory=list)
    raw_episode_count: int = 0

    @property
    def raw_incident_count(self) -> int:
        return self.raw_episode_count

    @property
    def top_level_count(self) -> int:
        """Top-level incidents after grouping: one per group plus each independent."""
        return len(self.groups) + len(self.independent_episode_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "raw_episode_count": self.raw_episode_count,
            "raw_incident_count": self.raw_incident_count,
            "top_level_count": self.top_level_count,
            "groups": [g.to_dict() for g in self.groups],
            "independent_episode_ids": list(self.independent_episode_ids),
            "unresolved_relation_ids": list(self.unresolved_relation_ids),
            "duplicate_episode_ids": list(self.duplicate_episode_ids),
        }


def _group_id(run_id: str, primary: str, members: Sequence[str]) -> str:
    payload = "|".join([run_id, primary, ",".join(sorted(members))])
    return "grp_{}".format(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12])


def reconcile_incidents(
    run_id: str,
    episodes: Sequence[Any],
    relations: Sequence[Any] = (),
    *,
    confirmed_only: bool = True,
) -> ReconciliationResult:
    """Group related manifestations without merging independent ones.

    ``confirmed_only`` restricts grouping to episodes carrying confirmed evidence; a
    dismissed observation is neither a primary nor a manifestation.
    """
    def payload(item: Any) -> Dict[str, Any]:
        return item.to_dict() if hasattr(item, "to_dict") else dict(item)

    by_id: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    for raw in episodes:
        data = payload(raw)
        episode_id = data["episode_id"]
        if episode_id in by_id:
            # Situation C: the same episode presented twice. This is an idempotency
            # defect and must be reported as one, never absorbed by a grouping rule.
            duplicates.append(episode_id)
            continue
        by_id[episode_id] = data

    confirmed = {
        episode_id: data for episode_id, data in by_id.items()
        if (not confirmed_only)
        or data.get("state") == "CONFIRMED"
        or data.get("outcome") == "CONFIRMED_ANOMALY"
    }

    relation_list = [payload(r) for r in relations]
    attached: Dict[str, List[Dict[str, Any]]] = {}
    consuming: set = set()
    unresolved: List[str] = []

    for relation in relation_list:
        source = relation.get("source_incident_id")
        target = relation.get("target_incident_id")
        kind = relation.get("relation_type")
        if source not in confirmed or target not in confirmed:
            unresolved.append(relation.get("relation_id", ""))
            continue
        if kind not in ATTACHING_RELATIONS:
            # PRECEDES / CO_OCCURS_WITH / UNKNOWN_RELATION: recorded, but the two stay
            # independent. Mere order or co-occurrence does not establish association.
            unresolved.append(relation.get("relation_id", ""))
            continue
        attached.setdefault(source, []).append(relation)
        consuming.add(target)

    groups: List[IncidentGroup] = []
    for primary_id, links in sorted(attached.items()):
        members = [link["target_incident_id"] for link in links]
        group = IncidentGroup(
            group_id=_group_id(run_id, primary_id, members),
            run_id=run_id,
            primary_episode_id=primary_id,
            primary_anomaly_type=confirmed[primary_id].get("anomaly_type"),
            secondary_episode_ids=members,
            secondary_anomaly_types=[
                confirmed[m].get("anomaly_type") for m in members if m in confirmed
            ],
            relation_ids=[link.get("relation_id", "") for link in links],
            relation_types=[link.get("relation_type") for link in links],
            association_evidence=[link.get("evidence") or {} for link in links],
        )
        groups.append(group)

    grouped_primaries = {g.primary_episode_id for g in groups}
    independent = [
        episode_id for episode_id in sorted(confirmed)
        if episode_id not in consuming and episode_id not in grouped_primaries
    ]
    return ReconciliationResult(
        run_id=run_id,
        groups=groups,
        independent_episode_ids=independent,
        unresolved_relation_ids=[u for u in unresolved if u],
        duplicate_episode_ids=duplicates,
        raw_episode_count=len(by_id),
    )
