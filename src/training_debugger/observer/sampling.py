"""Sampling policy -- owned by the observer, not by the caller.

Phase 1 brief section 3.4 requires that sampling live here, for a reason worth
restating: when every callsite writes its own ``if step % N == 0``, the sampling
rate becomes an emergent property of the codebase. Nobody can answer "how often
is the layer gradient norm recorded?" without grepping, and the answer differs
per callsite. Centralising it makes the rate a declared, testable, per-metric
number.

Two hard rules:

**Determinism.** Sampling is ``step % every_n == 0``, never random. A random
sampler makes two runs of the same seed produce different event sets, which
would break the scenario regression corpus in ``tests/scenarios`` and make an
incident impossible to reproduce.

**P0 is never sampled.** An incident or an error is data the product exists to
capture; dropping one to save bandwidth inverts the priorities.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..schema.enums import EventPriority
from ..schema.event import TrainingEvent

__all__ = ["SamplingRule", "SamplingPolicy", "DEFAULT_POLICY_PHASE1"]


@dataclass
class SamplingRule:
    """One sampling rule.

    ``pattern`` is an ``fnmatch`` glob over the metric name (``*`` and ``?``),
    or ``"*"`` for a default. Rules are evaluated in order and the first match
    wins, so put specific rules before broad ones.
    """

    pattern: str
    every_n_steps: int = 1
    always_first_steps: int = 0
    reason: str = ""

    def __post_init__(self) -> None:
        if self.every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1, got %r" % (self.every_n_steps,))
        if self.always_first_steps < 0:
            raise ValueError("always_first_steps must be >= 0")

    def matches(self, metric_name: Optional[str]) -> bool:
        if metric_name is None:
            return self.pattern in ("*", "")
        return fnmatch.fnmatchcase(metric_name, self.pattern)


@dataclass
class SamplingPolicy:
    """Ordered rules plus the counters needed to audit the effective rate."""

    rules: List[SamplingRule] = field(default_factory=list)
    default_every_n_steps: int = 1
    default_always_first_steps: int = 0
    never_sample_priorities: Tuple[EventPriority, ...] = (EventPriority.P0,)

    # counters
    considered: int = 0
    emitted: int = 0
    sampled_out: int = 0
    sampled_out_by_metric: Dict[str, int] = field(default_factory=dict)

    # -- construction helpers ---------------------------------------------

    @classmethod
    def uniform(cls, every_n_steps: int = 1) -> "SamplingPolicy":
        return cls(rules=[], default_every_n_steps=every_n_steps)

    @classmethod
    def from_spec(cls, spec: Sequence[Tuple[str, int]]) -> "SamplingPolicy":
        """Build from ``[(pattern, every_n), ...]``."""
        return cls(rules=[SamplingRule(pattern=p, every_n_steps=n) for p, n in spec])

    # -- decision ----------------------------------------------------------

    def rule_for(self, metric_name: Optional[str]) -> Optional[SamplingRule]:
        for rule in self.rules:
            if rule.matches(metric_name):
                return rule
        return None

    def should_emit(self, event: TrainingEvent) -> bool:
        """Decide whether ``event`` is recorded. Pure apart from counters."""
        if event.priority in self.never_sample_priorities:
            self.considered += 1
            self.emitted += 1
            return True

        # Events not tied to a step cannot be cadence-sampled. They are rare
        # lifecycle events (RUN_START, ENV_SNAPSHOT) and are always kept.
        if event.step is None:
            self.considered += 1
            self.emitted += 1
            return True

        rule = self.rule_for(event.metric_name)
        if rule is not None:
            every_n = rule.every_n_steps
            warmup = rule.always_first_steps
        else:
            every_n = self.default_every_n_steps
            warmup = self.default_always_first_steps

        self.considered += 1
        if warmup and event.step < warmup:
            self.emitted += 1
            return True
        if every_n <= 1:
            self.emitted += 1
            return True
        keep = (event.step % every_n) == 0
        if keep:
            self.emitted += 1
        else:
            self.sampled_out += 1
            key = event.metric_name or "<none>"
            self.sampled_out_by_metric[key] = self.sampled_out_by_metric.get(key, 0) + 1
        return keep

    # -- audit -------------------------------------------------------------

    def clone(self) -> "SamplingPolicy":
        """Deep-enough copy: independent counters, independent rule list.

        ``DEFAULT_POLICY_PHASE1`` is a module-level template. Handing the same
        object to every observer would make the counters a shared mutable global
        and would let one run's tuning leak into another's, so observers always
        take a clone.
        """
        return SamplingPolicy(
            rules=[
                SamplingRule(
                    pattern=r.pattern,
                    every_n_steps=r.every_n_steps,
                    always_first_steps=r.always_first_steps,
                    reason=r.reason,
                )
                for r in self.rules
            ],
            default_every_n_steps=self.default_every_n_steps,
            default_always_first_steps=self.default_always_first_steps,
            never_sample_priorities=tuple(self.never_sample_priorities),
        )

    def effective_rate(self, metric_name: Optional[str]) -> float:
        """Fraction of events for a metric that survive sampling (0..1]."""
        rule = self.rule_for(metric_name)
        every_n = rule.every_n_steps if rule else self.default_every_n_steps
        return 1.0 / float(every_n)

    def summary(self) -> Dict[str, object]:
        return {
            "rules": [
                {
                    "pattern": r.pattern,
                    "every_n_steps": r.every_n_steps,
                    "always_first_steps": r.always_first_steps,
                    "reason": r.reason,
                }
                for r in self.rules
            ],
            "default_every_n_steps": self.default_every_n_steps,
            "considered": self.considered,
            "emitted": self.emitted,
            "sampled_out": self.sampled_out,
            "sampled_out_by_metric": dict(self.sampled_out_by_metric),
        }


#: Phase 1 default policy. Chosen to match the brief's example: loss every step,
#: gradient norm every second step, layer metrics every tenth, and a 50-step
#: warm-up at full resolution so an early spike is never missed.
DEFAULT_POLICY_PHASE1 = SamplingPolicy(
    rules=[
        SamplingRule("train.loss*", 1, always_first_steps=0, reason="loss is the primary symptom signal"),
        SamplingRule("train.lr", 1, reason="learning rate is cheap and is the first hypothesis to rule out"),
        SamplingRule("grad.global_norm", 2, always_first_steps=50, reason="global gradient norm, every 2nd step"),
        SamplingRule("grad.layer_norm", 10, always_first_steps=50, reason="per-layer gradient norm, every 10th step"),
        SamplingRule("param.*", 10, always_first_steps=50, reason="parameter/update norms are diagnostics"),
        SamplingRule("train.step_time", 1, reason="step time is needed for throughput regression"),
        SamplingRule("train.tokens_per_sec", 1, reason="throughput is a first-class incident signal"),
        SamplingRule("train.samples_per_sec", 2, reason="derivable from tokens/sec, sampled"),
        SamplingRule("train.forward_time_ms", 5, reason="phase timing is diagnostic only"),
        SamplingRule("train.backward_time_ms", 5, reason="phase timing is diagnostic only"),
        SamplingRule("train.optimizer_time_ms", 5, reason="phase timing is diagnostic only"),
        SamplingRule("mem.*", 10, reason="memory trend needs a coarse cadence only"),
    ],
    default_every_n_steps=1,
    default_always_first_steps=20,
)
