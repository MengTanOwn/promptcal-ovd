"""Phrase-conditioned calibration.

The detector sees a text query, but every calibrator in the standard toolbox sees
only a scalar.  When the query itself moves the score, a scalar map fitted on one
phrasing is the wrong map for another.  ``PhC`` therefore predicts the two parameters
of an affine-in-logit map from descriptors of the phrase itself:

    logit' = logit * exp(-log T(phi)) + b(phi),      p = sigmoid(logit')

``T > 0`` for every phrase, so the map is strictly increasing in the score within a
fixed phrasing: average precision is unchanged for any single-phrasing run, and the
only thing the calibrator can alter is where the scores sit between 0 and 1.

The descriptor is precomputed per *phrase*, not per detection: a corpus has a few
hundred distinct phrases and millions of detections, so projecting the 768-d text
embedding once per phrase keeps the per-detection cost at 33 floats instead of 768.
"""
import numpy as np
import torch
import torch.nn as nn

from ..prompts import N_VARIANT, variants

N_VAR = N_VARIANT


class PhraseTable:
    """(category, wording) -> 33-d descriptor, precomputed per phrase."""

    def __init__(self, key, emb, index, names, n_pca=24, ref=None):
        self.key, self.emb, self.index, self.names = key, emb, index, names
        self.n_pca = n_pca
        cats = sorted(names)
        self.cat_of = {c: i for i, c in enumerate(cats)}
        n = len(cats)
        self.row = np.zeros((n, N_VAR), dtype=np.int64)
        self.tok = np.zeros((n, N_VAR), dtype=np.float32)
        self.chars = np.zeros((n, N_VAR), dtype=np.float32)
        for c in cats:
            for v, s in enumerate(variants(names[c])):
                j = index[s]
                self.row[self.cat_of[c], v] = j
                self.tok[self.cat_of[c], v] = len(s.split())
                self.chars[self.cat_of[c], v] = len(s)
        self.bare = self.row[:, 0]
        self.n_cats = n

        mu = emb.mean(0, keepdims=True)
        if ref is not None:
            # Reuse the projection of the corpus the calibrator was fitted on, so that
            # deploying on a new vocabulary fits nothing at all.  The text encoder is
            # the same, so the basis is meaningful for the new phrases too.
            mu, W = ref.mu, ref.W
        else:
            _, _, Vt = np.linalg.svd(emb - mu, full_matrices=False)
            W = Vt[:n_pca].T.astype(np.float32)
        self.mu, self.W = mu, W.astype(np.float32)
        self.Z = ((emb - mu) @ self.W).astype(np.float32)          # (P, n_pca)
        self.nrm = (np.linalg.norm(emb, axis=1) / 20.0).astype(np.float32)
        a = emb[self.row]                                          # (n, 5, 768)
        b = emb[self.bare][:, None, :]
        self.cos = ((a * b).sum(-1) /
                    np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1),
                               1e-6)).astype(np.float32)

    def rows(self, cat, var):
        ci = np.fromiter((self.cat_of.get(int(c), 0) for c in cat), dtype=np.int64,
                         count=len(cat))
        v = np.asarray(var, dtype=np.int64)
        return ci, v, self.row[ci, v]

    def build(self, cat, var):
        ci, v, ri = self.rows(cat, var)
        onehot = np.zeros((len(ci), N_VAR), dtype=np.float32)
        onehot[np.arange(len(ci)), v] = 1.0
        return np.concatenate([
            self.Z[ri],
            np.log1p(self.tok[ci, v])[:, None],
            (self.chars[ci, v] / 10.0)[:, None],
            self.nrm[ri][:, None],
            self.cos[ci, v][:, None],
            onehot,
        ], axis=1)


class _Head(nn.Module):
    """Linear by default: the map from phrase descriptors to (log T, b) is smooth and
    low-dimensional, and a linear probe on a few hundred observed phrases is far less
    prone to overfit than anything deeper.  ``hidden > 0`` gives the MLP variant."""

    def __init__(self, d_in, hidden=0):
        super().__init__()
        if hidden > 0:
            self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
            d_out = hidden
        else:
            self.body = nn.Identity()
            d_out = d_in
        self.head = nn.Linear(d_out, 2)

    def forward(self, x):
        o = self.head(self.body(x))
        return o[:, 0].clamp(-3, 3), o[:, 1].clamp(-4, 4)


class PhC:
    """Phrase-conditioned affine calibration, fitted on cached logits."""

    name = "phc"

    def __init__(self, table, epochs=300, lr=3e-3, hidden=0, w_band=0.0,
                 band=(0.3, 0.9), seed=0, batch=131072, wd=1e-4, cat_bias=False,
                 w_band_norm=False):
        self.table, self.epochs, self.lr, self.hidden = table, epochs, lr, hidden
        self.w_band, self.band, self.seed = w_band, band, seed
        self.batch, self.wd, self.cat_bias = batch, wd, cat_bias
        # A run whose band holds fewer detections carries less total weight in the
        # objective, so its band is fitted less well exactly where the user reviews.
        # Normalising the band weight by the number of band rows in the run gives every
        # wording the same total pull on the band term.
        self.w_band_norm = w_band_norm
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = None
        self.emb_cat = None

    def fit(self, logit, cat, var, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = self.table.build(cat, var)
        z = torch.as_tensor(logit, dtype=torch.float32, device=self.dev)
        x = torch.as_tensor(X, device=self.dev)
        t = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.dev)
        w = torch.ones_like(z)
        if self.w_band > 0:
            p0 = torch.sigmoid(z)
            m = (p0 >= self.band[0]) & (p0 < self.band[1])
            wb = torch.full_like(z, float(self.w_band))
            if self.w_band_norm:
                vv = torch.as_tensor(np.asarray(var, dtype=np.int64), device=self.dev)
                cnt = torch.zeros(N_VAR, device=self.dev)
                cnt.scatter_add_(0, vv, m.to(z.dtype))
                mean_cnt = cnt[cnt > 0].mean()
                adj = torch.where(cnt > 0, mean_cnt / cnt.clamp(min=1), torch.ones_like(cnt))
                wb = wb * adj[vv]
            w = torch.where(m, wb, w)
        if self.net is None:
            self.net = _Head(X.shape[1], self.hidden).to(self.dev)
        if self.cat_bias and self.emb_cat is None:
            self.emb_cat = nn.Embedding(self.table.n_cats, 2).to(self.dev)
            nn.init.zeros_(self.emb_cat.weight)
        params = list(self.net.parameters())
        if self.emb_cat is not None:
            params += list(self.emb_cat.parameters())
        cat_i = torch.as_tensor(
            np.fromiter((self.table.cat_of.get(int(c), 0) for c in cat),
                        dtype=np.int64, count=len(cat)), device=self.dev)
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)
        n = len(z)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.dev)[: min(n, self.batch)]
            lt, b = self.net(x[perm])
            if self.emb_cat is not None:
                c = self.emb_cat(cat_i[perm])
                lt = lt + c[:, 0].clamp(-2, 2)
                b = b + c[:, 1].clamp(-3, 3)
            p = torch.sigmoid(z[perm] * torch.exp(-lt) + b)
            loss = (w[perm] * (p - t[perm]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def predict_logits(self, logit, cat, var, chunk=1000000):
        out = np.empty(len(logit), dtype=np.float64)
        for i in range(0, len(logit), chunk):
            sl = slice(i, min(i + chunk, len(logit)))
            X = self.table.build(cat[sl], var[sl])
            with torch.no_grad():
                lt, b = self.net(torch.as_tensor(X, device=self.dev))
                if self.emb_cat is not None:
                    ci = np.fromiter((self.table.cat_of.get(int(c), 0)
                                      for c in cat[sl]), dtype=np.int64,
                                     count=len(cat[sl]))
                    c = self.emb_cat(torch.as_tensor(ci, device=self.dev))
                    lt = lt + c[:, 0].clamp(-2, 2)
                    b = b + c[:, 1].clamp(-3, 3)
            out[sl] = logit[sl] * np.exp(-lt.cpu().numpy()) + b.cpu().numpy()
        return out

    def predict(self, logit, cat, var):
        z = np.clip(self.predict_logits(logit, cat, var), -40, 40)
        return 1.0 / (1.0 + np.exp(-z))


class VariantTemperature:
    """One temperature per wording: the natural scalar baseline for PhC."""

    name = "variant"

    def __init__(self, n_var=N_VAR):
        self.n_var = n_var
        self.log_t = np.zeros(n_var)

    def fit(self, logit, cat, var, y):
        lg = torch.as_tensor(logit, dtype=torch.float32)
        yy = torch.as_tensor(np.asarray(y, dtype=np.float32))
        vv = torch.as_tensor(np.asarray(var, dtype=np.int64))
        t = torch.zeros(self.n_var, requires_grad=True)
        opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce(lg / torch.exp(t[vv]), yy)
            loss.backward()
            return loss

        opt.step(closure)
        self.log_t = t.detach().numpy()
        return self

    def predict_logits(self, logit, cat, var):
        return logit / np.exp(self.log_t[np.asarray(var, dtype=np.int64)])

    def predict(self, logit, cat, var):
        z = np.clip(self.predict_logits(logit, cat, var), -40, 40)
        return 1.0 / (1.0 + np.exp(-z))
