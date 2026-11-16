"""Does conditioning on the phrase beat conditioning on the score?

A run is (images, one phrasing): the detector is queried five times on the same
images with different wording, and each query is a deployment in its own right.
Every metric is therefore computed inside a run, on that run's own operating band
(fixed by the *uncalibrated* score of that run), and averaged over the five runs.

Two fit protocols:

``mixed``     fitted on all five phrasings (the calibrator knows all wordings),
``bare-only`` fitted on the bare name alone and deployed on the other four, which is
              what happens when a threshold is tuned once and then shipped.
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
held = ("image", "plural")


def methods(table, a):
    return {
        "identity": None,
        "temperature": TemperatureScaling(),
        "vector": VectorScaling(),
        "isotonic": Isotonic(),
        "variant": VariantTemperature(),
        "phc": PhC(table, epochs=a.epochs, lr=a.lr, seed=a.seed),
        "phc_band": PhC(table, epochs=a.epochs, lr=a.lr, seed=a.seed,
                        w_band=a.w_band),
        "phc_cat": PhC(table, epochs=a.epochs, lr=a.lr, seed=a.seed,
                       w_band=a.w_band, cat_bias=True),
    }


def fit(name, obj, feat, idx):
    if obj is None:
        return None
    lg, cat, var, y = (feat["logit"][idx], feat["cat_id"][idx],
                       feat["phrase"][idx], feat["label"][idx])
    if name.startswith("phc") or name == "variant":
        return obj.fit(lg, cat, var, y)
    return obj.fit(lg, None, y, cat)


def predict(name, obj, feat, idx):
    lg, cat, var = feat["logit"][idx], feat["cat_id"][idx], feat["phrase"][idx]
    if obj is None:
        return feat["score"][idx].astype(np.float64)
    if name.startswith("phc") or name == "variant":
        return obj.predict(lg, cat, var)
    return obj.predict(lg, None, cat)


def eval_runs(name, obj, feat, idx):
    """Evaluate inside each phrasing run and average over runs."""
    p = predict(name, obj, feat, idx)
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
        prog="promptcal calibration",
        description="Does conditioning on the phrase beat conditioning on the score?")
    ap.add_argument("--bench", default="coco", choices=["coco", "lvis"])
    ap.add_argument("--tag", default=None, help="shard prefix (default: <bench>_gd)")
    ap.add_argument("--epochs", type=int, default=300, help="PhC fitting epochs")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--w-band", type=float, default=10.0,
                    help="Brier weight inside the band (0 disables it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=1500000,
                    help="cap on fitting rows (0 keeps all)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: $PROMPTCAL_RESULTS/p2_calibration.json)")
    a = ap.parse_args(argv)
    out_path = a.out or str(config.results_dir() / "p2_calibration.json")
    max_train = a.max_train if a.max_train > 0 else 10 ** 12

    feat, (key, emb, index, names) = C.load(a.bench, a.tag)
    table = PhraseTable(key, emb, index, names)
    var = feat["phrase"]
    tr_m, te_m = C.load_split(feat)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    rng = np.random.RandomState(a.seed)
    tr_fit = tr if len(tr) <= max_train else rng.choice(tr, max_train, replace=False)
    print(f"{a.bench}: train {len(tr_fit)}/{len(tr)}  test {len(te)}", flush=True)

    out = {"bench": a.bench, "n_test": int(len(te)), "fit_rows": int(len(tr_fit)),
           "n_phrases": int(len(key)), "mixed": {}, "per_variant": {}, "heldout": {}}

    for name, proto in methods(table, a).items():
        obj = fit(name, proto, feat, tr_fit)
        agg, per = eval_runs(name, obj, feat, te)
        out["mixed"][name] = agg
        out["per_variant"][name] = per
        print(f"{name:<12} ECE {agg['ece']:.4f}  band {agg['band_ece']:.4f}  "
              f"bandB {agg['band_brier']:.4f}  Brier {agg['brier']:.4f}  "
              f"AP {agg['ap']:.4f}  AURC {agg['aurc']:.4f}  | band ECE: "
              + " ".join(f"{per[nm]['band_ece']:.3f}" for nm in C.VAR), flush=True)

    # fitted on three phrasings, deployed on the two it never saw: the only way a
    # phrasing-aware calibrator can help a deployment that cannot enumerate wording
    seen = (0, 1, 2)
    tr_h = tr_fit[np.isin(var[tr_fit], seen)]
    for name, proto in methods(table, a).items():
        obj = fit(name, proto, feat, tr_h)
        if name == "variant":
            obs = np.array([5.0] * 5)
            obs[list(seen)] = obj.log_t[list(seen)]
            obj.log_t = np.array([obj.log_t[i] if i in seen else obs[list(seen)].mean()
                                  for i in range(5)])
        per = {}
        for nm in held:
            v = C.VAR.index(nm)
            m = te[var[te] == v]
            if len(m) == 0:
                continue
            per[nm] = eval_runs(name, obj, feat, m)[1][nm]
        out["heldout"][name] = per
        print(f"[held-out] {name:<12} " + " ".join(
            f"{nm}: band {per[nm]['band_ece']:.3f} Brier {per[nm]['brier']:.4f}"
            for nm in per), flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("->", out_path)


if __name__ == "__main__":
    main()
