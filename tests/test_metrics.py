"""Metrics on hand-computable inputs.

A calibration paper whose ECE had an off-by-one in the bin edges would still produce a
plausible-looking table, so the metrics are pinned to values that can be worked out on
paper.
"""
import numpy as np
import pytest

from promptcal.calibrate.metrics import (
    aurc,
    average_precision,
    band_ece,
    brier,
    ece,
    nll,
    risk_coverage,
)


def test_ece_perfect_is_zero():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0, 0, 1, 1])
    assert ece(p, y, bins=10) == pytest.approx(0.0, abs=1e-12)


def test_ece_constant_half_on_all_negatives():
    p = np.full(100, 0.5)
    y = np.zeros(100, dtype=int)
    assert ece(p, y, bins=10) == pytest.approx(0.5)


def test_ece_uses_bin_weights():
    # 90% of the rows at 0.95, all positive (gap 0.05); 10% at 0.05, all negative
    # (gap 0.05): ECE is the weighted mean of those two gaps
    p = np.r_[np.full(90, 0.95), np.full(10, 0.05)]
    y = np.r_[np.ones(90), np.zeros(10)].astype(int)
    assert ece(p, y, bins=10) == pytest.approx(0.05)


def test_brier_and_nll():
    assert brier(np.array([0.5]), np.array([1])) == pytest.approx(0.25)
    assert nll(np.array([0.5]), np.array([1])) == pytest.approx(np.log(2), rel=1e-6)
    # a confident mistake costs more than a hesitant one
    assert nll(np.array([0.1]), np.array([1])) > nll(np.array([0.4]), np.array([1]))


def test_band_ece_ignores_rows_outside_the_band():
    rng = np.random.RandomState(0)
    p = rng.uniform(0.3, 0.9, 200)
    y = (rng.uniform(size=200) < p).astype(int)
    base = band_ece(p, y)
    junk_x = np.full(5000, 0.01)
    junk_y = np.zeros(5000, dtype=int)
    assert band_ece(np.r_[p, junk_x], np.r_[y, junk_y]) == pytest.approx(base)


def test_band_ece_is_nan_when_the_band_is_empty():
    p = np.full(100, 0.05)
    y = np.zeros(100, dtype=int)
    assert np.isnan(band_ece(p, y))


def test_average_precision_bounds():
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    g = np.zeros(5, dtype=int)
    # three positives on top: every interpolated precision is 1
    assert average_precision(p, np.array([1, 1, 1, 0, 0]), g) == pytest.approx(1.0)
    # the same scores with the positives at the bottom: the 101-point interpolation can
    # do no better than the precision at full recall, which is 2/5
    assert average_precision(p, np.array([0, 0, 0, 1, 1]), g) == pytest.approx(0.4)


def test_average_precision_rewards_putting_positives_first():
    y = np.array([1, 1, 0, 0])
    g = np.zeros(4, dtype=int)
    best = average_precision(np.array([0.9, 0.8, 0.2, 0.1]), y, g)
    worse = average_precision(np.array([0.9, 0.2, 0.8, 0.1]), y, g)
    assert best == pytest.approx(1.0)
    assert worse < best


def test_average_precision_is_rank_invariant():
    """Any strictly monotone map leaves AP unchanged -- the property the paper leans on."""
    rng = np.random.RandomState(1)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < 0.3).astype(int)
    g = rng.randint(0, 20, size=500)
    base = average_precision(p, y, g)
    for mapped in (np.log(p + 1e-6), p ** 3, 1.0 / (1.0 + np.exp(-8 * (p - 0.5)))):
        assert average_precision(mapped, y, g) == pytest.approx(base, abs=1e-12)


def test_average_precision_ignores_groups_without_positives():
    p = np.array([0.9, 0.1, 0.8, 0.2])
    y = np.array([1, 0, 0, 0])
    g = np.array([0, 0, 1, 1])
    # group 1 contributes nothing, so AP is that of group 0 alone
    assert average_precision(p, y, g) == pytest.approx(1.0)


def test_aurc_prefers_the_better_ranking():
    rng = np.random.RandomState(2)
    y = (rng.uniform(size=400) < 0.3).astype(int)
    good = np.where(y == 1, 0.8, 0.2) + 0.1 * rng.uniform(size=400)
    bad = np.where(y == 1, 0.2, 0.8) + 0.1 * rng.uniform(size=400)
    assert aurc(good, y) < aurc(bad, y)


def test_risk_coverage_ends_at_full_coverage():
    p = np.array([0.9, 0.5, 0.1])
    y = np.array([1, 0, 0])
    cov, risk = risk_coverage(p, y, n_points=10)
    assert cov[-1] == pytest.approx(1.0)
    assert risk[0] == pytest.approx(0.0)      # the most confident box is correct
