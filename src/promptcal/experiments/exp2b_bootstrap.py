"""How much of the band-error drop survives a resampling of images?

Every number in Section 5 is computed on millions of detections, but they come from a few
thousand images and the detections of one image are strongly correlated: the effective
sample size is the number of images, not the number of rows.  This script puts an interval
on the two claims that matter most -- that PhC lowers the band error, and that it lowers it
by more than a scalar map -- by resampling *images* with replacement and recomputing the
statistic inside each wording run.

The band is fixed once, by the uncalibrated score of each run, exactly as in p2; it does not
move under the bootstrap, so the interval describes the sampling variability of the error on
a fixed set of detections rather than of the band itself.
"""
import argparse
import json
import os

import numpy as np

from .. import config
from ..calibrate.baselines import VectorScaling
from ..calibrate.phc import PhC, PhraseTable
from . import common as C

METHODS = ("identity", "phc_band", "vector")


def ece_fast(p, y, bins=10):
    """Same quantity as calibrators.ece, without the per-bin boolean masks."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    cnt = np.bincount(idx, minlength=bins).astype(np.float64)
    sy = np.bincount(idx, weights=y, minlength=bins)
    sp = np.bincount(idx, weights=p, minlength=bins)
    nz = cnt > 0
    n = float(len(p))
    return float(np.sum(cnt[nz] / n * np.abs(sy[nz] / cnt[nz] - sp[nz] / cnt[nz])))


def image_of_row(img_sorted, imgs):
    """Position of each row's image in the sorted array of unique images."""
    return np.searchsorted(imgs, img_sorted)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="promptcal bootstrap",
        description="Image-level bootstrap interval on the band error of each calibrator.")
    ap.add_argument("--bench", default="coco", choices=["coco", "lvis"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--w-band", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=1500000)
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: $PROMPTCAL_RESULTS/p2b_ci.json)")
    a = ap.parse_args(argv)
    out_path = a.out or str(config.results_dir() / "p2b_ci.json")
    max_train = a.max_train if a.max_train > 0 else 10 ** 12

    feat, (key, emb, index, names) = C.load(a.bench)
    table = PhraseTable(key, emb, index, names)
    var = feat["phrase"]
    tr_m, te_m = C.load_split(feat)
    tr = np.where(tr_m)[0]
    te = np.where(te_m)[0]
    rng = np.random.RandomState(a.seed)
    tr_fit = tr if len(tr) <= max_train else rng.choice(tr, max_train, replace=False)

    specs = {
        "identity": None,
        "phc_band": PhC(table, epochs=a.epochs, lr=a.lr, seed=a.seed, w_band=a.w_band),
        "vector": VectorScaling(),
    }

    # one band-row layout per wording, shared by all methods so that the resamples are
    # paired: the same images, hence the same detections, for every method
    layout = {}
    te_var = var[te]
    te_img = feat["img_id"][te]
    te_y = feat["label"][te]
    sc_all = feat["score"][te].astype(np.float64)
    for v, nm in enumerate(C.VAR):
        m = (te_var == v) & C.band_mask(sc_all)
        rows = np.where(m)[0]                  # positions inside the test split
        img = te_img[rows]
        order = np.argsort(img, kind="stable")
        rows, img = rows[order], img[order]
        imgs = np.unique(img)
        layout[nm] = {"rows": rows, "img": img, "imgs": imgs,
                      "block": image_of_row(img, imgs)}

    boot_imgs = {nm: np.random.RandomState(1000 + i).randint(
        0, len(layout[nm]["imgs"]), size=(a.boot, len(layout[nm]["imgs"])))
        for i, nm in enumerate(C.VAR)}

    prob, point, pw_point = {}, {}, {}
    for name, obj in specs.items():
        if obj is not None:
            lg, cat, vv, y = (feat["logit"][tr_fit], feat["cat_id"][tr_fit],
                              var[tr_fit], feat["label"][tr_fit])
            obj = obj.fit(lg, cat, vv, y) if name.startswith("phc") else obj.fit(lg, None, y, cat)
        if obj is None:
            prob[name] = feat["score"][te].astype(np.float64)
        elif name.startswith("phc"):
            prob[name] = obj.predict(feat["logit"][te], feat["cat_id"][te], var[te])
        else:
            prob[name] = obj.predict(feat["logit"][te], None, feat["cat_id"][te])
        pw_point[name] = {nm: ece_fast(prob[name][layout[nm]["rows"]], te_y[layout[nm]["rows"]])
                          for nm in layout}
        point[name] = float(np.mean(list(pw_point[name].values())))
        print(f"{name:<10} band ECE {point[name]:.4f}", flush=True)

    out = {"bench": a.bench, "boot": a.boot, "methods": {}}
    per_w = {nm: {k: [] for k in specs} for nm in layout}
    agg = {k: [] for k in specs}
    for nm in layout:
        L = layout[nm]
        y_b = te_y[L["rows"]].astype(np.float64)
        for b in range(a.boot):
            c = np.bincount(boot_imgs[nm][b], minlength=len(L["imgs"])).astype(np.int64)
            take = np.repeat(np.arange(len(L["rows"])), c[L["block"]])
            yy = y_b[take]
            for k in specs:
                e = ece_fast(prob[k][L["rows"]][take], yy)
                per_w[nm][k].append(e)
                agg[k].append(e)
    # agg above stacked wordings; average the five per wording before averaging over runs
    n = a.boot
    for k in specs:
        stacked = np.array(agg[k]).reshape(len(layout), n).T     # (boot, wording)
        means = stacked.mean(axis=1)
        # the interval comes from the bootstrap; the point estimate is the statistic on the
        # full sample, which is not the mean of the bootstrap draws (ECE is nonlinear, and
        # the resamples carry a small upward bias of about 0.002)
        out["methods"][k] = {
            "band_ece": point[k],
            "boot_mean": float(np.mean([np.mean(per_w[nm][k]) for nm in layout])),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            # basic bootstrap: reflect the percentile interval about the point estimate,
            # which removes the upward bias that a non-negative statistic such as ECE picks
            # up from the resampling noise
            "lo_basic": float(2 * point[k] - np.percentile(means, 97.5)),
            "hi_basic": float(2 * point[k] - np.percentile(means, 2.5)),
            "per_wording": {nm: [pw_point[k][nm],
                                 float(np.percentile(per_w[nm][k], 2.5)),
                                 float(np.percentile(per_w[nm][k], 97.5))]
                            for nm in layout},
        }

    # the paired difference: same images in every resample, so subtract per resample
    for k in ("phc_band", "vector"):
        if k in specs and "identity" in specs:
            d = np.array(agg["identity"]).reshape(len(layout), n).T.mean(axis=1) - \
                np.array(agg[k]).reshape(len(layout), n).T.mean(axis=1)
            out[f"diff_identity_{k}"] = {
                "mean": float(point["identity"] - point[k]),
                "boot_mean": float(d.mean()),
                "lo": float(np.percentile(d, 2.5)),
                "hi": float(np.percentile(d, 97.5)),
                "frac_positive": float((d > 0).mean()),
            }

    for k, v in out["methods"].items():
        print(f"{k:<10} {v['band_ece']:.4f}  pct [{v['lo']:.4f}, {v['hi']:.4f}]  "
              f"basic [{v['lo_basic']:.4f}, {v['hi_basic']:.4f}]")
    for k in ("diff_identity_phc_band", "diff_identity_vector"):
        if k in out:
            d = out[k]
            print(f"{k:<24} {d['mean']:+.4f}  [{d['lo']:+.4f}, {d['hi']:+.4f}]  "
                  f"P(>0)={d['frac_positive']:.3f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("->", out_path)


if __name__ == "__main__":
    main()
