"""Anomaly episodes: the layer between a detector fire and an incident.

Cooldown reduces log volume; it does not define anomaly identity. Identity is
contiguity of observation, which is what makes a persistent `Inf` one episode
rather than one per step, and what makes `SUSPECT -> NORMAL` possible.
"""

from __future__ import annotations

from .engine import EPISODE_RULES, RULE_VERSION, EpisodeEngine, EpisodeRule
from .model import EPISODE_STATES, AnomalyEpisode, EpisodeOutcome, EpisodeState, episode_id_for

__all__ = [
    "AnomalyEpisode",
    "EpisodeState",
    "EpisodeOutcome",
    "EPISODE_STATES",
    "episode_id_for",
    "EpisodeEngine",
    "EpisodeRule",
    "EPISODE_RULES",
    "RULE_VERSION",
]
