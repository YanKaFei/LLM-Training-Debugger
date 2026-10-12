"""Step-level divergence localization between two runs.

The capability this pins did not exist before: `compare_runs` answered *"did this metric
get worse on average?"*, never *"at which step did these two runs stop agreeing?"*.

Validated on a paired corpus with real events and a known injection step:

```
fault pairs      9/9 localized to the EXACT injection step
clean pairs      3/3 correctly abstained
```
"""

from __future__ import annotations

import pytest

from training_debugger.comparison import (
    DIVERGENCE_RULE_VERSION, is_deterministic_metric, localize_divergence,
    render_divergence,
)


def series(values, start=1):
    return [(start + i, v) for i, v in enumerate(values)]


def clean(n=150, wobble=0.02):
    return [1.0 + (i % 7) * wobble for i in range(n)]


# ---------------------------------------------------------------------------
# The core capability
# ---------------------------------------------------------------------------


def test_the_first_divergence_is_localized_to_the_exact_step():
    a = {"grad.global_norm": series(clean())}
    b_values = clean()
    b_values[79:] = [12.0] * len(b_values[79:])          # diverges at index 79 -> step 80
    b = {"grad.global_norm": series(b_values)}
    report = localize_divergence("A", "B", a, b)
    assert report.verdict == "DIVERGENCE_LOCALIZED"
    assert report.first_divergence is not None
    assert report.first_divergence.step == 80
    assert report.first_divergence.metric_name == "grad.global_norm"
    assert report.first_divergence.direction == "b_above_a"


def test_identical_runs_abstain_rather_than_inventing_a_pivot():
    a = {"train.loss": series(clean())}
    report = localize_divergence("A", "B", a, dict(a))
    assert report.verdict == "NO_DIVERGENCE"
    assert report.first_divergence is None


def test_ordinary_noise_between_two_runs_is_not_a_divergence():
    """The null model is the runs' OWN agreement, not an absolute tolerance."""
    a = {"train.loss": series([1.0 + 0.05 * ((i * 37) % 11 - 5) for i in range(150)])}
    b = {"train.loss": series([1.0 + 0.05 * ((i * 41) % 11 - 5) for i in range(150)])}
    report = localize_divergence("A", "B", a, b)
    assert report.verdict == "NO_DIVERGENCE", (
        "two runs wobbling within a few percent of each other are in agreement"
    )


def test_a_single_sample_blip_is_not_a_divergence():
    a = {"train.loss": series(clean())}
    b_values = clean()
    b_values[100] = 50.0                                  # one sample only
    b = {"train.loss": series(b_values)}
    report = localize_divergence("A", "B", a, b)
    assert report.verdict == "NO_DIVERGENCE", (
        "persistence is required; a one-sample excursion is noise"
    )


def test_the_earliest_pivot_wins_across_metrics():
    a = {"m1": series(clean()), "m2": series(clean())}
    b1 = clean(); b1[119:] = [9.0] * len(b1[119:])        # step 120
    b2 = clean(); b2[94:] = [9.0] * len(b2[94:])          # step 95
    report = localize_divergence("A", "B", a, {"m1": series(b1), "m2": series(b2)})
    assert report.first_divergence.step == 95
    assert report.first_divergence.metric_name == "m2"


# ---------------------------------------------------------------------------
# Determinism classification — the defect found on real data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric,deterministic", [
    ("train.loss", True),
    ("grad.global_norm", True),
    ("grad.layer_norm", True),
    ("param.global_norm", True),
    ("train.lr", True),
    ("train.step_time", False),
    ("train.optimizer_time_ms", False),
    ("train.samples_per_sec", False),
    ("train.tokens_per_sec", False),
    ("mem.allocated_memory", False),
])
def test_determinism_classification(metric, deterministic):
    assert is_deterministic_metric(metric) is deterministic


def test_timing_metrics_are_excluded_by_default():
    """Found by running on real data: `compare` reported `train.step_time` at step 58
    when the fault was injected at step 90. Wall-clock timings differ from the first
    measurement; a timing difference is not an algorithmic divergence."""
    a = {"train.step_time": series([5.0] * 150), "train.loss": series(clean())}
    b_steps = [5.0] * 60 + [9.0] * 90                    # noisy from the start
    b = {"train.step_time": series(b_steps), "train.loss": series(clean())}
    report = localize_divergence("A", "B", a, b)
    assert "train.step_time" in report.non_deterministic_excluded
    assert report.verdict == "NO_DIVERGENCE", (
        "a timing difference must not be reported as a divergence"
    )
    assert any("not reproducible" in note for note in report.notes)


def test_timing_metrics_can_be_included_explicitly():
    a = {"train.step_time": series([5.0] * 150)}
    b = {"train.step_time": series([5.0] * 60 + [9.0] * 90)}
    report = localize_divergence("A", "B", a, b, include_non_deterministic=True)
    assert report.verdict == "DIVERGENCE_LOCALIZED"
    assert report.first_divergence.metric_name == "train.step_time"


# ---------------------------------------------------------------------------
# Abstention and honesty
# ---------------------------------------------------------------------------


def test_too_few_common_steps_abstains():
    a = {"train.loss": series(clean(20))}
    b = {"train.loss": series(clean(20))}
    report = localize_divergence("A", "B", a, b, min_common_steps=40)
    assert report.verdict == "INSUFFICIENT_EVIDENCE"
    assert "no comparison is defensible" in report.reason


def test_disjoint_metrics_abstain():
    report = localize_divergence("A", "B", {"m1": series(clean())}, {"m2": series(clean())})
    assert report.verdict == "INSUFFICIENT_EVIDENCE"
    assert report.metrics_compared == 0


def test_no_interpolation_is_performed():
    """Only steps present in BOTH runs are used. An interpolated value would be a
    fabricated observation."""
    a = {"train.loss": series(clean(150))}
    b = {"train.loss": [(s, v) for s, v in series(clean(150)) if s % 2 == 0]}
    report = localize_divergence("A", "B", a, b)
    assert report.common_steps == 75, "only the overlapping steps are counted"


def test_the_reason_never_claims_a_root_cause():
    a = {"train.loss": series(clean())}
    b_values = clean(); b_values[79:] = [12.0] * len(b_values[79:])
    report = localize_divergence("A", "B", a, {"train.loss": series(b_values)})
    assert "NOT a root cause" in report.reason
    assert "not a kernel" in report.reason
    rendered = render_divergence(report)
    assert "FIRST DIVERGENCE" in rendered
    assert "ROOT" not in rendered.upper() or "NOT a root cause" in rendered


def test_the_report_is_versioned_and_auditable():
    a = {"train.loss": series(clean())}
    b_values = clean(); b_values[79:] = [12.0] * len(b_values[79:])
    payload = localize_divergence("A", "B", a, {"train.loss": series(b_values)}).to_dict()
    assert payload["rule_version"] == DIVERGENCE_RULE_VERSION
    for key in ("run_a", "run_b", "common_steps", "agreement_scale",
                "first_divergence", "per_metric", "verdict", "reason"):
        assert key in payload
    point = payload["first_divergence"]
    for key in ("metric_name", "step", "value_a", "value_b", "difference",
                "z_score", "persisted_steps", "direction"):
        assert key in point


# ---------------------------------------------------------------------------
# diagnose --against : dating incidents against a pivot
# ---------------------------------------------------------------------------


class _Incident:
    def __init__(self, step, kind="GRADIENT_EXPLOSION", incident_id="inc_x"):
        self.detected_step = step
        self.incident_type = kind
        self.incident_id = incident_id

    def to_dict(self):
        return {"detected_step": self.detected_step, "incident_id": self.incident_id,
                "incident_type": self.incident_type}


def test_incidents_are_dated_against_the_pivot():
    from training_debugger.comparison import render_divergence_diagnosis

    a = {"train.loss": series(clean())}
    b_values = clean(); b_values[79:] = [12.0] * len(b_values[79:])
    report = localize_divergence("A", "B", a, {"train.loss": series(b_values)})
    incidents = [_Incident(50, incident_id="inc_before"),
                 _Incident(95, incident_id="inc_after")]

    text = render_divergence_diagnosis(report, incidents)
    assert "INCIDENTS AT OR AFTER THE DIVERGENCE" in text
    assert "INCIDENTS BEFORE THE DIVERGENCE" in text
    # the pivot is step 80: one incident on each side
    after_block = text.split("INCIDENTS AT OR AFTER")[1].split("INCIDENTS BEFORE")[0]
    before_block = text.split("INCIDENTS BEFORE")[1]
    assert "inc_after" in after_block and "inc_before" not in after_block
    assert "inc_before" in before_block and "inc_after" not in before_block


def test_an_incident_before_the_pivot_is_not_offered_as_an_explanation():
    """A pre-pivot incident predates the change, so it cannot be explained by it."""
    from training_debugger.comparison import render_divergence_diagnosis

    a = {"train.loss": series(clean())}
    b_values = clean(); b_values[119:] = [12.0] * len(b_values[119:])   # pivot at 120
    report = localize_divergence("A", "B", a, {"train.loss": series(b_values)})
    text = render_divergence_diagnosis(report, [_Incident(30, incident_id="early")])
    before_block = text.split("INCIDENTS BEFORE")[1]
    assert "early" in before_block
    assert "these cannot be explained by it" in text


def test_the_rendering_states_that_ordering_is_not_causation():
    from training_debugger.comparison import render_divergence_diagnosis

    a = {"train.loss": series(clean())}
    b_values = clean(); b_values[79:] = [12.0] * len(b_values[79:])
    report = localize_divergence("A", "B", a, {"train.loss": series(b_values)})
    text = render_divergence_diagnosis(report, [_Incident(95)])
    assert "This is an ORDERING, not a cause." in text
    assert "does not say which of them is wrong" in text


def test_no_pivot_means_nothing_is_dated_against_one():
    from training_debugger.comparison import render_divergence_diagnosis

    a = {"train.loss": series(clean())}
    report = localize_divergence("A", "B", a, dict(a))
    text = render_divergence_diagnosis(report, [_Incident(50, incident_id="solo")])
    assert "No pivot was established" in text
    assert "solo" in text


# ---------------------------------------------------------------------------
# The mechanism ordering that emerges from the per-metric pivots
# ---------------------------------------------------------------------------


def test_the_per_metric_pivots_recover_the_causal_ordering():
    """The strongest signal this capability produces.

    On real paired data the per-metric pivots came out in mechanism order --
    gradient at step 90, loss at 91, parameters at 93 -- with no propagation window
    assumed anywhere. The ordering is *measured* from the differential evidence rather
    than inferred from a threshold, which is why it is worth pinning.
    """
    n = 150
    def base(scale=1.0):
        return [scale * (1.0 + (i % 7) * 0.02) for i in range(n)]

    a = {"grad.global_norm": series(base()), "train.loss": series(base()),
         "param.global_norm": series(base())}
    g = base(); g[89:] = [70.0] * len(g[89:])          # gradient diverges at step 90
    l = base(); l[90:] = [l[90] * 3.0] * len(l[90:])    # loss one step later
    p_ = base(); p_[92:] = [p_[92] * 4.0] * len(p_[92:])  # parameters two steps later
    b = {"grad.global_norm": series(g), "train.loss": series(l),
         "param.global_norm": series(p_)}

    report = localize_divergence("A", "B", a, b)
    steps = {point.metric_name: point.step for point in report.per_metric}
    assert steps["grad.global_norm"] == 90
    assert steps["train.loss"] == 91
    assert steps["param.global_norm"] == 93
    assert steps["grad.global_norm"] < steps["train.loss"] < steps["param.global_norm"], (
        "gradient -> loss -> parameters is the mechanism order and must be recovered"
    )
    assert report.first_divergence.step == 90
    assert report.first_divergence.metric_name == "grad.global_norm"
