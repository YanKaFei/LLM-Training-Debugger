"""Observer SDK: the training process's only link to the debugger.

Public surface:

* :class:`TrainingObserver` -- emit, sample, validate, enqueue, never block
* :class:`ObserverConfig`   -- tunables in one place
* :class:`SamplingPolicy`   -- cadence owned by the observer, not the caller
* :class:`BoundedEventQueue`-- bounded, priority-aware, never blocking
* :class:`EventWriter`      -- the destination contract
"""

from __future__ import annotations

from .health import EmergencySink, ObserverStats
from .observer import ObserverConfig, TrainingObserver
from .queue import BoundedEventQueue, QueueStats
from .sampling import DEFAULT_POLICY_PHASE1, SamplingPolicy, SamplingRule
from .writer_protocol import (
    EventWriter,
    FailingWriter,
    JsonlWriter,
    MemoryWriter,
    MultiWriter,
    NullWriter,
)

__all__ = [
    "TrainingObserver",
    "ObserverConfig",
    "ObserverStats",
    "EmergencySink",
    "BoundedEventQueue",
    "QueueStats",
    "SamplingPolicy",
    "SamplingRule",
    "DEFAULT_POLICY_PHASE1",
    "EventWriter",
    "NullWriter",
    "MemoryWriter",
    "JsonlWriter",
    "MultiWriter",
    "FailingWriter",
]
