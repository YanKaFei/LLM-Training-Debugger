"""M0: the monotonicity reporter must not be able to hide a decrease.

Phase 1.8 reported ``monotonicity violations = 0`` beside a curve that visibly
decreased, because the reporter only flagged drops larger than an arbitrary 0.25.
These tests pin the corrected behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "experiments"))

from gradient_recalibration import monotonicity_report  # noqa: E402


def curve(pairs):
    return [{"magnitude": m, "p_confirmed": p} for m, p in pairs]


def test_the_phase_1_8_curve_is_not_reported_as_monotone():
    """The exact curve that was reported as having zero violations."""
    reported = curve([
        (1.5, 0.23), (2.0, 0.12), (2.5, 0.15), (3.0, 0.40), (4.0, 0.23),
        (5.0, 0.20), (7.0, 0.33), (10.0, 0.23), (12.0, 0.70), (20.0, 0.95),
        (25.0, 1.00), (40.0, 1.00),
    ])
    result = monotonicity_report(reported)
    assert result["monotone"] is False
    assert result["inversion_count"] == 4
    assert result["largest_inversion"] == pytest.approx(0.17)


def test_a_small_decrease_is_still_an_inversion():
    """Tolerance 0.0: a 0.01 drop must not be silently permitted."""
    result = monotonicity_report(curve([(1.0, 0.5), (2.0, 0.49)]))
    assert result["inversion_count"] == 1


def test_a_monotone_curve_reports_zero_inversions():
    result = monotonicity_report(curve([(1.0, 0.1), (2.0, 0.3), (3.0, 0.9)]))
    assert result["monotone"] is True
    assert result["inversion_count"] == 0
    assert result["rank_correlation"] == pytest.approx(1.0)


def test_rank_correlation_distinguishes_trend_from_noise():
    """Local inversions are reported, but the overall trend is a separate number."""
    result = monotonicity_report(curve([
        (1.0, 0.2), (2.0, 0.1), (3.0, 0.5), (4.0, 0.4), (5.0, 0.9),
    ]))
    assert result["inversion_count"] == 2
    assert result["rank_correlation"] >= 0.8, "the trend is strongly increasing"


def test_inversions_name_both_endpoints():
    result = monotonicity_report(curve([(1.0, 0.5), (2.0, 0.2)]))
    inversion = result["inversions"][0]
    assert inversion["from_magnitude"] == 1.0
    assert inversion["to_magnitude"] == 2.0
    assert inversion["drop"] == pytest.approx(0.3)
