"""Append-only incident persistence.

Parquet, not a database: Phase 0's trigger for Postgres was concurrent human
writers to the incident store, and that is still not the case. An incident's
history is a sequence of immutable part files, so a revision never overwrites the
state it superseded.
"""

from __future__ import annotations

from .analysis_state import (
    ANALYSIS_STATE_SCHEMA,
    AnalysisState,
    AnalysisStateStore,
    BaselineSnapshot,
    analysis_state_dir,
    capture_analysis_state,
)
from .episode_store import (
    EPISODE_RELATION_SCHEMA,
    EPISODE_REVISION_SCHEMA,
    EPISODE_SCHEMA,
    EpisodeSnapshot,
    EpisodeStore,
    episodes_dir,
)
from .incident_store import (
    EVIDENCE_SCHEMA,
    HYPOTHESIS_SCHEMA,
    INCIDENT_SCHEMA,
    RULE_VERSION,
    IncidentStore,
    PersistResult,
    incidents_dir,
)

__all__ = [
    "IncidentStore",
    "PersistResult",
    "INCIDENT_SCHEMA",
    "EVIDENCE_SCHEMA",
    "HYPOTHESIS_SCHEMA",
    "incidents_dir",
    "RULE_VERSION",
    "EpisodeStore",
    "EpisodeSnapshot",
    "EPISODE_SCHEMA",
    "EPISODE_REVISION_SCHEMA",
    "EPISODE_RELATION_SCHEMA",
    "episodes_dir",
    "AnalysisState",
    "AnalysisStateStore",
    "BaselineSnapshot",
    "capture_analysis_state",
    "ANALYSIS_STATE_SCHEMA",
    "analysis_state_dir",
]
