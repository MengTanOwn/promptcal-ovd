"""Scalar calibration baselines.

None of these methods sees the prompt.  They are the accuracy-preserving baselines of Guo
et al. (2017) plus isotonic regression, which is the standard non-parametric calibrator and
the only one here that is free to reorder a run.

They share one interface::

    fit(logits, X=None, y=None, group=None) -> self
    predict(logits, X=None, group=None) -> p

``logits`` are pre-sigmoid scores and ``group`` is a category id, used by
:class:`VectorScaling`, whose table is indexed by it.

The two fitted baselines are optimised single-threaded, for reproducibility rather than
speed: see :func:`single_threaded`.
"""
import contextlib

import numpy as np
import torch
import torch.nn as nn


@contextlib.contextmanager
def single_threaded():
    """Run a block on one CPU thread.

    LBFGS fits these models from a gradient summed over 1.5M rows.  In a threaded build
    the summation order depends on the schedule, so two identical runs differ in the low
    bits, the line search takes a slightly different path, and the fitted parameters ---
    and the fourth decimal of the band error --- move by up to 1e-3.  One thread makes the
    fit bitwise reproducible, and on data this size the cost is a second or two.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)



class Identity:
    name = "identity"

    def fit(self, logits, X=None, y=None, group=None):
        return self

    def predict(self, logits, X=None, group=None):
        return 1.0 / (1.0 + np.exp(-logits))


class TemperatureScaling:
    """One global temperature; the classic baseline (Guo et al., 2017)."""
    name = "temperature"

    def __init__(self):
        self.log_t = 0.0

    def fit(self, logits, X=None, y=None, group=None):
        t = torch.zeros(1, requires_grad=True)
        lg = torch.as_tensor(logits, dtype=torch.float32)
        yy = torch.as_tensor(y, dtype=torch.float32)
        opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce(lg / torch.exp(t), yy)
            loss.backward()
            return loss

        with single_threaded():
            opt.step(closure)
        self.log_t = float(t.detach())
        return self

    def predict(self, logits, X=None, group=None):
        return 1.0 / (1.0 + np.exp(-logits / np.exp(self.log_t)))


class VectorScaling:
    """Per-group temperature and bias: logit -> logit / exp(t_g) + b_g."""
    name = "vector"

    def __init__(self, min_count=200):
        self.min_count = min_count
        self.params = {}
        self.default = (0.0, 0.0)

    def fit(self, logits, X=None, y=None, group=None):
        keys, inv = np.unique(group, return_inverse=True)
        lg = torch.as_tensor(logits, dtype=torch.float32)
        yy = torch.as_tensor(y, dtype=torch.float32)
        inv_t = torch.as_tensor(inv, dtype=torch.long)
        counts = np.bincount(inv)
        t = torch.zeros(len(keys), requires_grad=True)
        b = torch.zeros(len(keys), requires_grad=True)
        opt = torch.optim.LBFGS([t, b], lr=0.1, max_iter=100)
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            z = lg / torch.exp(t)[inv_t] + b[inv_t]
            loss = bce(z, yy)
            loss.backward()
            return loss

        with single_threaded():
            opt.step(closure)
        self.default = (float(t.detach().mean()), 0.0)
        for i, k in enumerate(keys):
            if counts[i] >= self.min_count:
                self.params[int(k)] = (float(t.detach()[i]), float(b.detach()[i]))
        return self

    def predict(self, logits, X=None, group=None):
        t = np.full(len(logits), self.default[0], dtype=np.float32)
        b = np.zeros(len(logits), dtype=np.float32)
        for g, (tg, bg) in self.params.items():
            m = group == g
            t[m] = tg
            b[m] = bg
        return 1.0 / (1.0 + np.exp(-(logits / np.exp(t) + b)))


def _pava(y, w):
    """Pool-adjacent-violators: monotone non-decreasing fit of y with weights w."""
    y = y.astype(np.float64).copy()
    w = w.astype(np.float64).copy()
    n = len(y)
    lvl, wts, cnt = [], [], []
    for i in range(n):
        lvl.append(y[i])
        wts.append(w[i])
        cnt.append(1)
        while len(lvl) > 1 and lvl[-2] > lvl[-1]:
            v = (lvl[-1] * wts[-1] + lvl[-2] * wts[-2]) / (wts[-1] + wts[-2])
            c = cnt[-1] + cnt[-2]
            wt = wts[-1] + wts[-2]
            lvl = lvl[:-2] + [v]
            wts = wts[:-2] + [wt]
            cnt = cnt[:-2] + [c]
    out = np.empty(n)
    pos = 0
    for v, c in zip(lvl, cnt, strict=True):
        out[pos:pos + c] = v
        pos += c
    return out


class Isotonic:
    """Binned isotonic regression on the raw probability (non-parametric baseline)."""
    name = "isotonic"

    def __init__(self, bins=50):
        self.bins = bins
        self.edges = None
        self.vals = None

    def fit(self, logits, X=None, y=None, group=None):
        p = 1.0 / (1.0 + np.exp(-logits))
        qs = np.quantile(p, np.linspace(0, 1, self.bins + 1))
        qs = np.unique(qs)
        idx = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, len(qs) - 2)
        cnt = np.bincount(idx, minlength=len(qs) - 1).astype(np.float64)
        pos = np.bincount(idx, weights=y, minlength=len(qs) - 1)
        keep = cnt > 0
        rate = np.where(keep, pos / np.maximum(cnt, 1), 0.0)
        fitted = _pava(rate[keep], cnt[keep])
        self.edges = (qs[:-1] + qs[1:]) / 2.0
        self.edges = self.edges[keep]
        self.vals = fitted
        return self

    def predict(self, logits, X=None, group=None):
        p = 1.0 / (1.0 + np.exp(-logits))
        idx = np.clip(np.searchsorted(self.edges, p), 0, len(self.vals) - 1)
        return self.vals[idx]
