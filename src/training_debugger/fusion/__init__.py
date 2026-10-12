"""Deterministic evidence fusion.

Decides between CONFIRMED_INCIDENT, CANDIDATE_ONLY, INSUFFICIENT_EVIDENCE and
NO_ANOMALY using versioned explicit rules. Never learned, never a model.
"""

from __future__ import annotations

from .rules import (
    FUSION_RULE_VERSION,
    PROMOTABLE_STATES,
    PROMOTION_DISABLED_TYPES,
    PROMOTION_RULES,
    PromotionDisposition,
    ReasonCode,
    IncidentVerdict,
    PromotionRule,
    Verdict,
    fuse,
    is_promotable,
    promotion_enabled,
    fusion_rule_registry,
)

__all__ = [
    "Verdict",
    "ReasonCode",
    "PromotionDisposition",
    "IncidentVerdict",
    "fuse",
    "is_promotable",
    "PromotionRule",
    "PROMOTION_RULES",
    "PROMOTION_DISABLED_TYPES",
    "promotion_enabled",
    "PROMOTABLE_STATES",
    "FUSION_RULE_VERSION",
    "fusion_rule_registry",
]
