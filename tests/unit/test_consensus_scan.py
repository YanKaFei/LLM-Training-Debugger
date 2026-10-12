"""Multi-run consensus: find the outlier in a batch.

Validated on real data with a like-for-like batch (five identical clean replicas plus one
faulty run, all at one seed):

```
LIKEFORLIKE_faulty   diverged 5/5   pivot 90   <- correctly named, 5/5 peer support
```

and on two negative controls, both of which it correctly refused:

```
6 healthy runs, different seeds          -> NO_SINGLE_OUTLIER
5 healthy + 1 faulty, but mixed seeds    -> NO_SINGLE_OUTLIER
```

The second control is the one that mattered. An earlier version reported an outlier on
six **healthy** runs, because independently sampled runs differ from each other
legitimately and it had no way to tell that from a real anomaly.
"""

from __future__ import annotations

import pytest

from training_debugger.comparison import CONSENSUS_RULE_VERSION, find_outlier, render_consensus


def base(wobble=0.02, n=150):
    return [1.0 + (i % 7) * wobble for i in range(n)]


def series(values):
    return list(enumerate(values, 1))


def faulty(step=90, value=12.0):
    values = base()
    values[step - 1:] = [value] * len(values[step - 1:])
    return values


def batch(**runs):
    return {name: {"train.loss": series(values)} for name, values in runs.items()}


# ---------------------------------------------------------------------------
# The capability
# ---------------------------------------------------------------------------


def test_the_one_different_run_is_named():
    runs = batch(r1=base(), r2=base(), r3=base(), r4=base(), r5=base(),
                 odd=faulty())
    report = find_outlier(runs)
    assert report.verdict == "OUTLIER_FOUND"
    assert report.pattern == "SINGLE_OUTLIER"
    assert report.outlier.run_id == "odd"
    assert report.outlier.divergence_count == 5
    assert int(report.outlier.median_pivot_step) == 90
    assert report.supporting_peers == 5, "every peer localises the same step"
    assert report.outlier.dominant_metric == "train.loss"


def test_a_consistent_batch_reports_no_outlier():
    runs = batch(r1=base(), r2=base(), r3=base(), r4=base())
    report = find_outlier(runs)
    assert report.verdict == "NO_OUTLIER"
    assert report.pattern == "NONE"
    assert report.outlier is None


def test_independently_varied_runs_are_refused():
    """The defect found on real data.

    Six healthy runs at three different seeds each diverge from their peers at scattered
    steps. A version keyed on pairwise agreement called one of them an outlier; it was
    not one. Scattered pivots mean accumulated variance, which has no single onset.
    """
    # Independently sampled runs drift apart in ways that are not ordered: no run is
    # "further" than the others, they simply differ. Distinct phases reproduce that --
    # pairs cross and re-cross, so divergences land at scattered steps.
    import math as _math

    def drifted(period, phase):
        return [1.0 + 0.6 * _math.sin(i / period + phase) for i in range(150)]

    runs = {
        "s1a": {"train.loss": series(drifted(9.0, 0.0))},
        "s1b": {"train.loss": series(drifted(11.0, 1.0))},
        "s2a": {"train.loss": series(drifted(13.0, 2.0))},
        "s2b": {"train.loss": series(drifted(15.0, 3.0))},
        "s3a": {"train.loss": series(drifted(17.0, 4.0))},
        "s3b": {"train.loss": series(drifted(19.0, 5.0))},
    }
    report = find_outlier(runs)
    # Either nothing diverged enough to rank, or divergences happened at scattered
    # steps and the batch was refused. What must NOT happen is a confident single
    # outlier: none of these runs is more different than the others.
    assert report.verdict in ("NO_OUTLIER", "NO_SINGLE_OUTLIER"), (
        "six mutually dissimilar but equally valid runs must not produce a named "
        "outlier; got {} for {}".format(
            report.verdict, report.outlier.run_id if report.outlier else None)
    )
    if report.verdict == "NO_SINGLE_OUTLIER":
        assert report.pattern == "DIFFUSE"
        assert "do not agree on WHEN" in report.reason


def test_two_runs_are_not_enough_for_a_consensus():
    report = find_outlier(batch(r1=base(), r2=base()))
    assert report.verdict in ("NO_OUTLIER", "INSUFFICIENT_EVIDENCE")


def test_a_single_run_cannot_be_scanned():
    report = find_outlier(batch(only=base()))
    assert report.verdict == "INSUFFICIENT_EVIDENCE"
    assert "at least two runs" in report.reason


def test_a_tie_is_reported_as_a_tie():
    """If two runs each diverge from everyone, there is no single answer."""
    runs = batch(r1=base(), r2=base(), r3=base(), r4=base(),
                 odd_a=faulty(), odd_b=faulty(value=9.0))
    report = find_outlier(runs)
    if report.outlier and report.outlier.divergence_count == report.ranking[1].divergence_count:
        assert "tie" in report.reason.lower()


# ---------------------------------------------------------------------------
# Honesty
# ---------------------------------------------------------------------------


def test_the_verdict_never_says_which_side_is_wrong():
    """A divergence between two runs is evidence about BOTH of them."""
    runs = batch(r1=base(), r2=base(), r3=base(), odd=faulty())
    report = find_outlier(runs)
    assert "NOT a faulty run" in report.reason
    assert "cannot say which side of a divergence moved" in report.reason


def test_the_rule_is_versioned_and_the_ranking_is_auditable():
    runs = batch(r1=base(), r2=base(), r3=base(), odd=faulty())
    payload = find_outlier(runs).to_dict()
    assert payload["rule_version"] == CONSENSUS_RULE_VERSION
    for key in ("runs", "pairs_compared", "ranking", "outlier", "supporting_peers",
                "pattern", "verdict", "reason"):
        assert key in payload
    assert payload["pairs_compared"] == 6, "four runs make six unordered pairs"
    for entry in payload["ranking"]:
        for key in ("run_id", "divergence_count", "peers", "divergence_rate",
                    "median_pivot_step", "pivot_steps", "dominant_metric"):
            assert key in entry


def test_every_pair_is_compared_once():
    runs = batch(r1=base(), r2=base(), r3=base(), r4=base(), r5=base())
    assert find_outlier(runs).pairs_compared == 10


def test_the_renderer_shows_the_ranking_and_the_verdict():
    runs = batch(r1=base(), r2=base(), r3=base(), odd=faulty())
    text = render_consensus(find_outlier(runs))
    assert "BATCH DIVERGENCE SCAN" in text
    assert "VERDICT: OUTLIER_FOUND" in text
    assert "odd" in text
