"""Selective detection with a false-discovery guarantee.

A deployment does not ship a calibrated probability, it ships a threshold.  The question
a user actually asks is "if I keep every box above this score, what fraction of what I
keep is wrong?" -- a false discovery rate, not an ECE.

Thresholds are chosen by conformalised selection on a labelled calibration half: the
smallest lambda whose conformal estimate of the false discovery proportion,
``(1 + V(lambda)) / max(R(lambda), 1)``, is at most the target alpha.

Four ways to end up with a threshold:

``pooled``      one lambda from the calibration detections of all wordings,
``worst``       one lambda that satisfies the constraint inside *every* wording
                separately, i.e. the largest of the per-wording lambdas; this is the
                only one of the four that is a single number a tool can ship,
``per-wording`` one lambda per wording, each from that wording's calibration rows,
``bare-only``   one lambda calibrated on the bare name and shipped everywhere.

The point of conditioning is not that the pooled lambda can be rescued -- it cannot,
because Proposition 1 forbids a monotone map from changing the order of a run, and the
false discovery rate at a shared threshold is set by that order.  What conditioning buys
is *coverage under the worst-case protocol*: a score on which the five wordings agree
more closely has a less binding worst-case constraint, so the same guarantee costs fewer
boxes.
"""
import argparse
import json
import os

import numpy as np

from .. import config
from ..calibrate.baselines import TemperatureScaling, VectorScaling
from ..calibrate.conformal import fdr, pick_lambda
from ..calibrate.phc import PhC, PhraseTable
from . import common as C

ALPHAS = (0.05, 0.10, 0.20, 0.30)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="promptcal conformal",
        description="Thresholds with a false-discovery guarantee, pooled and per wording.")
    ap.add_argument("--bench", default="coco", choices=["coco", "lvis"])
    ap.add_argument("--tag", default=None, help="shard prefix (default: <bench>_gd)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=1500000)
    ap.add_argument("--alphas", default=None,
                    help="comma-separated targets (default: 0.05,0.10,0.20,0.30)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: $PROMPTCAL_RESULTS/p3_conformal.json)")
    a = ap.parse_args(argv)
    out_path = a.out or str(config.results_dir() / "p3_conformal.json")
    max_train = a.max_train if a.max_train > 0 else 10 ** 12
    alphas = ALPHAS
    if a.alphas:
        alphas = tuple(float(x) for x in a.alphas.split(","))

    feat, (key, emb, index, names) = C.load(a.bench, a.tag)
    table = PhraseTable(key, emb, index, names)
    var, cat, y = feat["phrase"], feat["cat_id"], feat["label"]
    tr_m, te_m = C.load_split(feat)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    rng = np.random.RandomState(a.seed)
    tr_fit = tr if len(tr) <= max_train else rng.choice(tr, max_train, replace=False)
    lg = feat["logit"]

    scores = {"identity": feat["score"].astype(np.float64)}
    ts = TemperatureScaling().fit(lg[tr_fit], None, y[tr_fit], None)
    scores["temperature"] = ts.predict(lg, None, cat)
    vs = VectorScaling().fit(lg[tr_fit], None, y[tr_fit], cat[tr_fit])
    scores["vector"] = vs.predict(lg, None, cat)
    phc = PhC(table, epochs=a.epochs, lr=a.lr, seed=a.seed, w_band=10.0,
              w_band_norm=True).fit(lg[tr_fit], cat[tr_fit], var[tr_fit], y[tr_fit])
    scores["phc"] = phc.predict(lg, cat, var)

    out = {"bench": a.bench, "alphas": list(alphas),
           "pooled": {}, "pooled_per_run": {}, "per_wording": {}, "bare_only": {},
           "worst": {}}

    for name, p in scores.items():
        row, per = {}, {}
        for alpha in alphas:
            lam = pick_lambda(p[tr_fit], y[tr_fit], alpha)
            f, n = fdr(p[te], y[te], lam)
            row[f"{alpha:.2f}"] = {"lambda": lam, "fdr": f, "kept": n}
            pr = {}
            for v, nm in enumerate(C.VAR):
                m = te[var[te] == v]
                fv, nv = fdr(p[m], y[m], lam)
                pr[nm] = {"fdr": fv, "kept": nv}
            per[f"{alpha:.2f}"] = pr
        out["pooled"][name] = row
        out["pooled_per_run"][name] = per
        worst = max(abs(per[f"{alphas[-1]:.2f}"][nm]["fdr"] - alphas[-1]) for nm in C.VAR)
        print(f"{name:<10} pooled: " + "  ".join(
            f"a={al}: {row[f'{al:.2f}']['fdr']:.3f}/{row[f'{al:.2f}']['kept']}"
            for al in alphas), flush=True)
        print(f"{'':<10} per wording at a={alphas[-1]:.2f}: " + "  ".join(
            f"{nm} {per[f'{alphas[-1]:.2f}'][nm]['fdr']:.3f}" for nm in C.VAR),
            f"(worst dev {worst:.3f})", flush=True)

        pw = {}
        for v, nm in enumerate(C.VAR):
            mtr = tr_fit[var[tr_fit] == v]
            mte = te[var[te] == v]
            if len(mtr) == 0 or len(mte) == 0:
                continue
            r = {}
            for alpha in alphas:
                lam = pick_lambda(p[mtr], y[mtr], alpha)
                f, n = fdr(p[mte], y[mte], lam)
                r[f"{alpha:.2f}"] = {"lambda": lam, "fdr": f, "kept": n}
            pw[nm] = r
        out["per_wording"][name] = pw

        # One number a tool can actually ship: the strictest of the per-wording lambdas.
        # FDR falls as lambda rises, so requiring the constraint inside every wording
        # means taking the *largest* of the five.  Taking the max over strata also means
        # the simultaneous statement needs no multiplicity correction.
        wr = {}
        for alpha in alphas:
            lam = float(max(pw[nm][f"{alpha:.2f}"]["lambda"] for nm in pw))
            pr = {}
            for v, nm in enumerate(C.VAR):
                m = te[var[te] == v]
                fv, nv = fdr(p[m], y[m], lam)
                pr[nm] = {"fdr": fv, "kept": nv}
            keep = p[te] >= lam
            wr[f"{alpha:.2f}"] = {
                "lambda": lam, "per_run": pr,
                "kept": int(sum(pr[nm]["kept"] for nm in pr)),
                "fdr": float(np.mean(y[te][keep] == 0)) if keep.any() else float("nan"),
            }
        out["worst"][name] = wr
        w = wr[f"{alphas[-1]:.2f}"]
        print(f"{'':<10} worst-case lambda at a={alphas[-1]:.2f}: {w['lambda']:.4f} "
              f"kept {w['kept']:6d}  per wording "
              + " ".join(f"{nm} {w['per_run'][nm]['fdr']:.3f}" for nm in C.VAR),
              flush=True)

        mtr = tr_fit[var[tr_fit] == 0]
        r = {}
        for alpha in alphas:
            lam = pick_lambda(p[mtr], y[mtr], alpha)
            pr = {}
            for v, nm in enumerate(C.VAR):
                m = te[var[te] == v]
                fv, nv = fdr(p[m], y[m], lam)
                pr[nm] = {"fdr": fv, "kept": nv}
            r[f"{alpha:.2f}"] = {"lambda": lam, "per_run": pr}
        out["bare_only"][name] = r
        print(f"{'':<10} bare-only lambda, per wording at a={alphas[-1]:.2f}: "
              + "  ".join(f"{nm} {r[f'{alphas[-1]:.2f}']['per_run'][nm]['fdr']:.3f}"
                          for nm in C.VAR), flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("->", out_path)


if __name__ == "__main__":
    main()
