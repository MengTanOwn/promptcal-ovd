"""Does normalising the band weight equalise the false-discovery rate across wordings?

This is the diagnostic behind one negative statement in the paper: the residual spread of
the attained FDR across wordings is *not* an artefact of one run carrying more band rows
than another.  Fitting with a band term normalised by the number of band rows in each run
changes the spread by a few thousandths, so the disagreement is in the ordering of the
runs, which is what Proposition 1 says no monotone map can repair.

Run it after ``promptcal calibration``; it needs only the feature shards.
"""
import argparse

import numpy as np

from ..calibrate.conformal import fdr, pick_lambda
from ..calibrate.phc import PhC, PhraseTable
from . import common as C

DEFAULT_ALPHA = 0.30

CANDIDATES = {
    "plain": dict(),
    "band": dict(w_band=10.0),
    "band+norm": dict(w_band=10.0, w_band_norm=True),
    "band+norm5": dict(w_band=5.0, w_band_norm=True),
    "norm only": dict(w_band=1.0, w_band_norm=True),
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="promptcal diagnose-band",
        description="Per-wording FDR at a pooled threshold, with and without band weighting.")
    ap.add_argument("--bench", default="coco", choices=["coco", "lvis"])
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    feat, (key, emb, index, names) = C.load(a.bench)
    table = PhraseTable(key, emb, index, names)
    var, cat, y, lg = feat["phrase"], feat["cat_id"], feat["label"], feat["logit"]
    tr_m, te_m = C.load_split(feat)
    tr, test = np.where(tr_m)[0], np.where(te_m)[0]
    print(f"{len(tr)} fit rows, {len(test)} test rows, alpha {a.alpha:g}")

    for name, kwargs in CANDIDATES.items():
        model = PhC(table, epochs=a.epochs, lr=a.lr, seed=a.seed, **kwargs)
        model.fit(lg[tr], cat[tr], var[tr], y[tr])
        p = model.predict(lg, cat, var)
        lam = pick_lambda(p[tr], y[tr], a.alpha)
        tot, kept = fdr(p[test], y[test], lam)
        per = {}
        for v, nm in enumerate(C.VAR):
            k = test[var[test] == v]
            per[nm] = fdr(p[k], y[k], lam)
        vals = np.array([per[nm][0] for nm in C.VAR])
        print(f"{name:<12} pooled {tot:.3f}  lambda {lam:.3f}  kept {kept:6d} | "
              + " ".join(f"{nm[:3]} {per[nm][0]:.3f}" for nm in C.VAR)
              + f" | spread {vals.max() - vals.min():.3f}  "
                f"worst deviation {np.abs(vals - a.alpha).max():.3f}", flush=True)


if __name__ == "__main__":
    main()
