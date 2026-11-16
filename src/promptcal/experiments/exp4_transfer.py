"""Does the calibrator survive a new vocabulary?

Fit on COCO -- five wordings, the 80 categories the user of an annotation tool can
enumerate -- and deploy on LVIS v1.0 minival, whose 1,203 categories were never seen.
Nothing is refitted: not the head, not the projection of the phrase descriptor, not a
per-category table.  This is the protocol that separates a calibrator which has learned
something about *phrases* from one which has memorised *categories*: vector scaling keeps
a table indexed by category and has nothing to look up here, while PhC reads the text
embedding the detector already produced for the new phrase.

The band is still fixed by the uncalibrated score of each run, which needs no labels and
is therefore available at deployment.
"""
import argparse
import json
import os

import numpy as np

from .. import config
from ..calibrate.baselines import Isotonic, TemperatureScaling, VectorScaling
from ..calibrate.phc import PhC, PhraseTable, VariantTemperature
from . import common as C

KEYS = ("ece", "band_ece", "band_brier", "brier", "nll", "ap", "aurc")


def fit_one(name, obj, feat, idx):
    if obj is None:
        return None
    lg, cat, var, y = (feat["logit"][idx], feat["cat_id"][idx],
                       feat["phrase"][idx], feat["label"][idx])
    if name.startswith("phc") or name == "variant":
        return obj.fit(lg, cat, var, y)
    return obj.fit(lg, None, y, cat)


def predict_one(name, obj, feat, idx, table=None):
    lg, cat, var = feat["logit"][idx], feat["cat_id"][idx], feat["phrase"][idx]
    if obj is None:
        return feat["score"][idx].astype(np.float64)
    if name.startswith("phc"):
        if table is not None:
            obj.table = table           # the descriptor basis stays the COCO one
        return obj.predict(lg, cat, var)
    if name == "variant":
        return obj.predict(lg, cat, var)
    return obj.predict(lg, None, cat)


def eval_runs(name, obj, feat, idx, table=None):
    p = predict_one(name, obj, feat, idx, table)
    y, cat, var = feat["label"][idx], feat["cat_id"][idx], feat["phrase"][idx]
    sc = feat["score"][idx].astype(np.float64)
    per = {}
    for v, nm in enumerate(C.VAR):
        m = var == v
        if m.sum() == 0:
            continue
        per[nm] = C.evaluate(p[m], y[m], cat[m], C.band_mask(sc[m]))
    agg = {k: float(np.mean([per[nm][k] for nm in per])) for k in KEYS}
    agg["band_n"] = int(sum(per[nm]["band_n"] for nm in per))
    return agg, per


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="promptcal transfer",
        description="Fit on one vocabulary, deploy on another, refit nothing.")
    ap.add_argument("--src", default="coco", choices=["coco", "lvis"])
    ap.add_argument("--dst", default="lvis", choices=["coco", "lvis"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--w-band", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=1500000)
    ap.add_argument("--src-dir", default=None, help="feature dir for --src")
    ap.add_argument("--dst-dir", default=None, help="feature dir for --dst")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: $PROMPTCAL_RESULTS/p4_transfer.json)")
    a = ap.parse_args(argv)
    out_path = a.out or str(config.results_dir() / "p4_transfer.json")
    max_train = a.max_train if a.max_train > 0 else 10 ** 12

    src, (skey, semb, sindex, snames) = C.load(a.src, feat_dir=a.src_dir)
    st = PhraseTable(skey, semb, sindex, snames)
    tr_m, _ = C.load_split(src)
    tr = np.where(tr_m)[0]
    rng = np.random.RandomState(a.seed)
    tr_fit = tr if len(tr) <= max_train else rng.choice(tr, max_train, replace=False)
    print(f"fit on {a.src}: {len(tr_fit)} rows, {st.n_cats} categories", flush=True)

    dst, (dkey, demb, dindex, dnames) = C.load(a.dst, feat_dir=a.dst_dir)
    dt = PhraseTable(dkey, demb, dindex, dnames, ref=st)
    print(f"deploy on {a.dst}: {len(dst['label'])} rows, {dt.n_cats} categories, "
          f"{len(np.unique(dst['img_id']))} images", flush=True)

    specs = {
        "identity": None,
        "temperature": TemperatureScaling(),
        "vector": VectorScaling(),
        "isotonic": Isotonic(),
        "variant": VariantTemperature(),
        "phc": PhC(st, epochs=a.epochs, lr=a.lr, seed=a.seed),
        "phc_band": PhC(st, epochs=a.epochs, lr=a.lr, seed=a.seed, w_band=a.w_band),
    }

    idx = np.arange(len(dst["label"]))
    out = {"src": a.src, "dst": a.dst, "n": int(len(idx)),
           "n_images": int(len(np.unique(dst["img_id"]))), "n_cats": int(dt.n_cats),
           "fit_rows": int(len(tr_fit)), "methods": {}}

    # A per-category table is indexed by category id, and the two vocabularies share no
    # category although their integer ids overlap by accident. Emptying the table after
    # fitting is the faithful implementation of "deploy on a vocabulary you have not
    # seen": what remains is the global fallback the calibrator would use for an unseen
    # label, which is plain temperature scaling.
    for name, obj in specs.items():
        obj = fit_one(name, obj, src, tr_fit)
        if name == "vector":
            out["vector_fitted"] = int(len(obj.params))
            obj.params = {}
        agg, per = eval_runs(name, obj, dst, idx, table=dt if name.startswith("phc") else None)
        out["methods"][name] = agg
        out["methods"][name]["per_wording"] = per
        print(f"{name:<12} ECE {agg['ece']:.4f}  band {agg['band_ece']:.4f}  "
              f"bandB {agg['band_brier']:.4f}  Brier {agg['brier']:.4f}  "
              f"AP {agg['ap']:.4f}  | band ECE: "
              + " ".join(f"{per[nm]['band_ece']:.3f}" for nm in C.VAR), flush=True)

    if "vector" in out["methods"]:
        print(f"vector scaling: {out['vector_fitted']} fitted categories, "
              f"0 of them present in {a.dst}", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("->", out_path)


if __name__ == "__main__":
    main()
