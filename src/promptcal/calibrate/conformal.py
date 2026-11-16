"""Threshold selection with a false-discovery guarantee.

A deployment does not ship a calibrated probability, it ships a threshold.  The question a
user actually asks is "if I keep every box above this score, what fraction of what I keep
is wrong?" --- a false discovery rate, not an ECE.  Conformalised selection (Jin and
Candes, 2023) answers it from a labelled calibration sample, without assuming the scores
are calibrated.

The estimate is the conformal false-discovery proportion ``(1 + V(lambda)) / R(lambda)``,
where ``R`` is the number of kept boxes and ``V`` the number of kept negatives; the
admissible thresholds are those whose estimate is at most the target ``alpha``.
"""
import numpy as np


def pick_lambda(p, y, alpha):
    """Smallest threshold whose conformal FDP estimate is at most ``alpha``.

    The admissible set is a prefix in rank order, so the largest admissible set is the
    last admissible rank and hence the smallest admissible threshold --- which is the one
    that keeps the most boxes while still meeting the target.
    """
    o = np.argsort(-p)
    p_s, y_s = p[o], y[o]
    R = np.arange(1, len(p_s) + 1)
    V = np.cumsum(y_s == 0)
    ok = (1.0 + V) / np.maximum(R, 1) <= alpha
    if not ok.any():
        return float("inf")
    return float(p_s[int(np.where(ok)[0].max())])


def fdr(p, y, lam):
    """Empirical false discovery rate and number of kept boxes at threshold ``lam``."""
    keep = p >= lam
    if keep.sum() == 0:
        return float("nan"), 0
    return float(np.mean(y[keep] == 0)), int(keep.sum())
