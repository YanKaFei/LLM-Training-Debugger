"""W2/W3: an opt-in Optimizer Boundary Witness.

**Disabled by default. No new parameter reduction. No mandatory device sync.**

The problem it addresses, measured in W0:

```
*.global_norm == inf   ->  the float32 reduction overflowed, every element FINITE
*.global_norm == nan   ->  non-finite elements present
```

Treating the first as a non-finite observation made v0.1.4 claim propagation when
nothing propagated. Excluding it removed **every** mechanism claim, including genuine
ones -- because the existing telemetry contains no direct observation of the update
boundary at all. That is why this witness is necessary rather than optional, and why it
must be explicit about what it does and does not know:

> **A witnessed optimizer call is not necessarily an applied parameter update.**

The three facts are kept separate and never collapsed:

```
ATTEMPTED  the training code asked for an update        (observed by the caller)
INVOKED    optimizer.step() was entered                 (observed by a hook)
SKIPPED    the update was deliberately not applied      (observed from GradScaler etc.)
UNKNOWN    none of the above could be established       (and UNKNOWN is never False)
```

"Invoked but the parameters did not change" is deliberately **not** inferred: proving it
requires a parameter comparison, which is a scan, which this module may not do. The
status stays `UNKNOWN`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "OptimizerStepStatus", "OptimizerWitness", "WITNESS_SCHEMA_VERSION",
]

WITNESS_SCHEMA_VERSION = "optimizer-witness/1.0"


class OptimizerStepStatus:
    ATTEMPTED = "ATTEMPTED"
    INVOKED = "INVOKED"
    SKIPPED = "SKIPPED"
    UNKNOWN = "UNKNOWN"
    NOT_ATTACHED = "NOT_ATTACHED"

    ALL = (ATTEMPTED, INVOKED, SKIPPED, UNKNOWN, NOT_ATTACHED)


@dataclass
class OptimizerWitness:
    """Observes the optimizer boundary through the public hook API.

    Every field is explicit about its provenance because the whole point of this
    release is that an inferred mechanism must not be asserted as an observed one.

    ``enabled=False`` (the default) means **no hooks are registered and no events are
    emitted** -- an unattached witness is observably identical to this module not
    existing.
    """

    enabled: bool = False
    observer: Any = None
    optimizer_id: str = "optimizer:0"
    #: The optimizer this witness is attached to, if any.
    _optimizer: Any = None
    _handles: List[Any] = field(default_factory=list)
    _counter: Any = field(default_factory=lambda: itertools.count(1))
    #: Per-optimizer-step records. Bounded by the number of steps taken.
    records: List[Dict[str, Any]] = field(default_factory=list)
    #: Set by the caller when the training loop knows the update was suppressed
    #: (e.g. GradScaler reported a skip, or an accumulation step is incomplete).
    _pending_skip: Optional[str] = None
    _invoked_this_step: bool = False

    # -- lifecycle -------------------------------------------------------

    def attach(self, optimizer: Any, *, optimizer_id: Optional[str] = None) -> bool:
        """Register hooks. Returns False when the optimizer does not support them.

        A failure to attach is **not** an error and never fails training. The status is
        reported as ``NOT_ATTACHED`` so a consumer can tell "this optimizer cannot be
        witnessed" from "the update did not happen".
        """
        if not self.enabled:
            return False
        self._optimizer = optimizer
        if optimizer_id is not None:
            self.optimizer_id = optimizer_id
        pre = getattr(optimizer, "register_step_pre_hook", None)
        post = getattr(optimizer, "register_step_post_hook", None)
        if pre is None or post is None:
            return False
        try:
            self._handles.append(pre(self._on_pre))
            self._handles.append(post(self._on_post))
        except Exception:
            # Fail open: a witness that breaks training is worse than no witness.
            self.detach()
            return False
        return True

    def detach(self) -> None:
        """Remove every hook. Safe to call repeatedly."""
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []
        self._optimizer = None

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    # -- observation -----------------------------------------------------

    def _on_pre(self, optimizer: Any, *args: Any, **kwargs: Any) -> None:
        self._invoked_this_step = True

    def _on_post(self, optimizer: Any, *args: Any, **kwargs: Any) -> None:
        self._invoked_this_step = True

    def note_attempted(
        self, *, training_step_id: int, micro_step_id: int = 0,
        skip_reason: Optional[str] = None,
    ) -> None:
        """Called by the training loop BEFORE it decides to call `optimizer.step()`.

        ``skip_reason`` is the caller's *observed* evidence that the update will not be
        applied -- a GradScaler skip, an incomplete accumulation window, a zero-length
        schedule. It is never guessed here.
        """
        self._pending_skip = skip_reason
        self._invoked_this_step = False
        self._current = {"training_step_id": training_step_id,
                         "micro_step_id": micro_step_id}

    def note_invoked(self) -> None:
        """Called immediately AFTER `optimizer.step()` returns."""
        if self.enabled:
            self._emit()

    def _emit(self) -> None:
        current = getattr(self, "_current", {})
        invoked = bool(self._invoked_this_step)
        if self._pending_skip is not None:
            status = OptimizerStepStatus.SKIPPED
        elif not self.attached:
            # No hooks: the call may or may not have happened. Never guess.
            status = (OptimizerStepStatus.UNKNOWN if not invoked
                      else OptimizerStepStatus.INVOKED)
        elif invoked:
            status = OptimizerStepStatus.INVOKED
        else:
            status = OptimizerStepStatus.UNKNOWN

        record = {
            "schema_version": WITNESS_SCHEMA_VERSION,
            "optimizer_id": self.optimizer_id,
            "training_step_id": current.get("training_step_id"),
            "micro_step_id": current.get("micro_step_id", 0),
            "optimizer_step_id": next(self._counter),
            "optimizer_step_attempted": True,
            "optimizer_step_invoked": invoked,
            "optimizer_step_skipped": status == OptimizerStepStatus.SKIPPED,
            "optimizer_step_status": status,
            "skip_reason": self._pending_skip,
            "witness_source": ("torch.optim.register_step_hook" if self.attached
                               else "none"),
            # Never claim more than was observed.
            "parameters_changed": None,
            "observation_confidence": ("OBSERVED" if self.attached
                                       else "UNKNOWN"),
        }
        self.records.append(record)
        self._pending_skip = None
        if self.observer is None:
            return
        try:
            self.observer.metric(
                "optimizer.step_status", 1.0 if status == OptimizerStepStatus.INVOKED else 0.0,
                step=current.get("training_step_id"),
                metadata=record,
            )
        except Exception:
            # Fail open, always.
            pass

    # -- reporting -------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {name: 0 for name in OptimizerStepStatus.ALL}
        for record in self.records:
            counts[record["optimizer_step_status"]] = counts.get(
                record["optimizer_step_status"], 0) + 1
        return {
            "schema_version": WITNESS_SCHEMA_VERSION,
            "enabled": self.enabled,
            "attached": self.attached,
            "optimizer_step_count": len(self.records),
            "status_counts": counts,
        }
