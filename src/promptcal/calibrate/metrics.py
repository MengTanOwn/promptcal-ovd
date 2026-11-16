"""Evaluation metrics for detection scores.

Every metric here takes an array of ``(score, label)`` pairs.  ``band_ece`` is the
headline quantity of the paper, ``average_precision`` is what shows that a calibrator has
not touched the ranking, and ``aurc`` summarises the whole risk-coverage curve.
"""
import numpy as np


def ece(prob, y, bins=15):
    """Expected calibration error with equal-width bins on the probability scale."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    n = len(prob)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        tot += m.sum() / n * abs(y[m].mean() - prob[m].mean())
    return float(tot)


def brier(prob, y):
    return float(np.mean((prob - y) ** 2))


def nll(prob, y, eps=1e-7):
    p = np.clip(prob, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def band_ece(prob, y, lo=0.3, hi=0.9, bins=10):
    """ECE restricted to the band where retrieval actually happens.

    86% of detections sit below 0.1 and are almost all false positives, so a global
    ECE measures the junk band, not the operating point of the detector.
    """
    m = (prob >= lo) & (prob < hi)
    if m.sum() < 30:
        return float("nan")
    return ece(prob[m], y[m], bins)


def aurc(prob, y):
    """Area under the risk-coverage curve (lower is better)."""
    order = np.argsort(-prob)
    yy = y[order]
    cum_err = np.cumsum(1.0 - yy)
    k = np.arange(1, len(yy) + 1)
    risk = cum_err / k
    return float(np.mean(risk))


def risk_coverage(prob, y, n_points=200):
    """Risk (error rate on the accepted set) as a function of coverage."""
    order = np.argsort(-prob)
    yy = y[order]
    n = len(yy)
    ks = np.unique(np.linspace(1, n, n_points).astype(np.int64))
    risk = np.array([1.0 - yy[:k].mean() for k in ks])
    return ks / n, risk


def average_precision(prob, y, group):
    """Class-agnostic AP over grouped detections, matching COCO's 101-point rule."""
    aps, ws = [], []
    for g in np.unique(group):
        m = group == g
        if not m.any() or y[m].sum() == 0:
            continue
        p, t = prob[m], y[m]
        order = np.argsort(-p)
        t = t[order]
        tp = np.cumsum(t)
        fp = np.cumsum(1 - t)
        rec = tp / max(t.sum(), 1)
        prec = tp / np.maximum(tp + fp, 1e-9)
        grid = np.linspace(0, 1, 101)
        interp = np.array([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in grid])
        aps.append(interp.mean())
        ws.append(t.sum())
    return float(np.average(aps, weights=ws)) if aps else 0.0
