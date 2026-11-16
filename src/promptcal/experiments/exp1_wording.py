"""How much of the score is the phrasing?

For every (image, category) the detector is queried five times with the same
referent and different wording.  This script quantifies what changes: how many
boxes clear a threshold, how often those boxes are right, and how far the score of
the *same* object moves.
"""
import argparse
import json
import os

import numpy as np

from .. import config
from ..calibrate import metrics as K
from ..data import features as L
from . import common as C

VAR = C.VAR


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="promptcal wording",
        description="How far the score of the same object moves across five wordings.")
    ap.add_argument("--bench", default="coco", choices=["coco", "lvis"])
    ap.add_argument("--tag", default=None, help="shard prefix (default: <bench>_gd)")
    ap.add_argument("--feat", default=None,
                    help="feature directory (default: $PROMPTCAL_FEATURES)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: $PROMPTCAL_RESULTS/p1_describe.json)")
    a = ap.parse_args(argv)
    tag = a.tag or f"{a.bench}_gd"
    out_path = a.out or str(config.results_dir() / "p1_describe.json")

    feat = L.attach_labels(L.load_shards(a.feat, tag), a.bench)
    y, sc, ph = feat["label"], feat["score"], feat["phrase"]
    img, cat = feat["img_id"], feat["cat_id"]
    print(f"{tag}: {len(y)} detections, {len(np.unique(img))} images, "
          f"positive rate {y.mean():.4f}", flush=True)

    out = {"tag": tag, "bench": a.bench, "n": int(len(y)),
           "n_images": int(len(np.unique(img))), "pos_rate": float(y.mean()),
           "variants": {}, "spread": {}}
    for v in range(5):
        m = ph == v
        out["variants"][VAR[v]] = {
            "n": int(m.sum()),
            "per_image": float(m.sum() / len(np.unique(img))),
            "pos_rate": float(y[m].mean()),
            "mean_score": float(sc[m].mean()),
            "frac_gt_0_3": float((sc[m] >= 0.3).mean()),
            "frac_gt_0_5": float((sc[m] >= 0.5).mean()),
            "ap": float(K.average_precision(sc[m], y[m], cat[m])),
            "ece": float(K.ece(sc[m], y[m])),
        }
        d = out["variants"][VAR[v]]
        print(f"{VAR[v]:8s} n={d['n']:8d} {d['per_image']:6.1f}/img  pos {d['pos_rate']:.4f} "
              f"mean {d['mean_score']:.4f}  >0.3 {d['frac_gt_0_3']:.4f}  AP {d['ap']:.4f}  "
              f"ECE {d['ece']:.4f}", flush=True)

    # same object, five phrasings: the box the detector is most confident about
    top = {}
    order = np.lexsort((img, ph))
    for i in order:
        k = (img[i], cat[i], ph[i])
        if k not in top or sc[i] > sc[top[k]]:
            top[k] = i
    grp = {}
    for (i_, c_, v_), i in top.items():
        grp.setdefault((i_, c_), {})[v_] = i
    full = [g for g in grp.values() if len(g) == 5]
    S = np.array([[sc[g[v]] for v in range(5)] for g in full])
    Y = np.array([[y[g[v]] for v in range(5)] for g in full])
    spread = S.max(1) - S.min(1)
    agre = (Y.max(1) == Y.min(1))
    h, e = np.histogram(spread, bins=40, range=(0.0, 1.0))
    out["spread"] = {
        "n_groups": int(len(S)),
        "mean_top_score": S.mean(0).tolist(),
        "mean_spread": float(spread.mean()),
        "median_spread": float(np.median(spread)),
        "frac_spread_gt_0_1": float((spread > 0.1).mean()),
        "frac_spread_gt_0_2": float((spread > 0.2).mean()),
        "label_agreement": float(agre.mean()),
        # kept so Fig. 1(b) plots exactly the numbers quoted in the text
        "hist_counts": h.tolist(), "hist_edges": e.tolist(),
        "mean_range_in_pos": float(spread[Y[:, 0] == 1].mean()) if (Y[:, 0] == 1).any() else None,
    }
    s = out["spread"]
    print(f"\n{len(S)} (image, category) groups with all five phrasings")
    print("  mean top score per phrasing:", np.round(s["mean_top_score"], 4).tolist())
    print(f"  mean spread {s['mean_spread']:.4f}  median {s['median_spread']:.4f}")
    print(f"  spread > 0.1 in {s['frac_spread_gt_0_1']:.3f} of groups; "
          f"> 0.2 in {s['frac_spread_gt_0_2']:.3f}")
    print(f"  labels agree across all five in {s['label_agreement']:.3f} of groups")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("->", out_path)


if __name__ == "__main__":
    main()
