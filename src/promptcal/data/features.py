"""Feature shards: loading, label attachment, phrase tables.

A *shard* is what :mod:`promptcal.extract` writes: one ``.npz`` per few hundred images,
holding one row per detection under one wording.  Concatenating the shards gives a flat
table with the columns

``img_id`` ``cat_id`` ``phrase`` ``box`` ``img_wh`` ``score`` ``logit`` ``label``

where ``phrase`` is an index into :data:`promptcal.prompts.VARIANT_NAMES` and ``label``
is 1 if the box matches a ground-truth instance of the prompted category at IoU 0.5.
"""
import glob
import os

import numpy as np

from .. import config
from .gt import coco_gt, greedy_match, lvis_minival


def load_shards(feat_dir=None, tag="coco_gd"):
    """Concatenate every ``{tag}_part*.npz`` in a directory into one dict of arrays."""
    feat_dir = feat_dir or config.features_dir()
    files = sorted(glob.glob(os.path.join(feat_dir, f"{tag}_part*.npz")))
    if not files:
        raise SystemExit(
            f"no shards matching {tag}_part*.npz in {feat_dir}\n"
            f"run `promptcal extract --bench coco` first, or set PROMPTCAL_FEATURES")
    ds = [np.load(f) for f in files]
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0].files}


def attach_labels(feat, bench="coco"):
    """Add ``label`` and four box descriptors to a feature dict, in place.

    Matching is greedy and one-to-one *within one run*.  A run is a single phrasing, so
    the grouping key is ``(image, category, phrasing)``: the five phrasings are five
    independent queries, and sharing one ground-truth instance between them would score
    four of the five as wrong by construction --- a pooled match would manufacture false
    negatives that have nothing to do with the detector.
    """
    if bench == "coco":
        by_img, _, _ = coco_gt()
    else:
        by_img, _, _, _ = lvis_minival()
    box = feat["box"].astype(np.float64)
    sc = feat["score"].astype(np.float64)
    img_id, cat_id, ph = feat["img_id"], feat["cat_id"], feat["phrase"]
    n = len(sc)
    label = np.zeros(n, dtype=np.float32)

    order = np.lexsort((-sc, ph, cat_id, img_id))
    iid_s, cat_s, ph_s = img_id[order], cat_id[order], ph[order]
    starts = np.r_[0, np.where((iid_s[1:] != iid_s[:-1]) |
                              (cat_s[1:] != cat_s[:-1]) |
                              (ph_s[1:] != ph_s[:-1]))[0] + 1]
    starts = np.r_[starts, n]
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        idx = order[a:b]
        gts = by_img.get(int(iid_s[a]), {}).get(int(cat_s[a]), [])
        label[idx] = greedy_match(box[idx], gts, 0.5)
    feat["label"] = label

    wh = feat["img_wh"].astype(np.float64)
    bw = (box[:, 2] - box[:, 0]) / np.maximum(wh[:, 0], 1)
    bh = (box[:, 3] - box[:, 1]) / np.maximum(wh[:, 1], 1)
    feat["area"] = (bw * bh).astype(np.float32)
    feat["aspect"] = (bw / np.maximum(bh, 1e-6)).astype(np.float32)
    feat["cx"] = ((box[:, 0] + box[:, 2]) / 2 / np.maximum(wh[:, 0], 1)).astype(np.float32)
    feat["cy"] = ((box[:, 1] + box[:, 3]) / 2 / np.maximum(wh[:, 1], 1)).astype(np.float32)
    return feat


def split_by_image(feat, frac=0.5, seed=0):
    """Two image-disjoint masks.

    The split has to be by image: one image contributes roughly 1,400 correlated queries,
    so a row-wise split would put near-duplicates of the same object on both sides and
    make every test number optimistic.
    """
    imgs = np.unique(feat["img_id"])
    rng = np.random.RandomState(seed)
    rng.shuffle(imgs)
    take = set(imgs[:int(len(imgs) * frac)].tolist())
    m = np.array([i in take for i in feat["img_id"]])
    return m, ~m


def load_phrases(bench="coco", feat_dir=None):
    """Phrase string -> pooled text embedding, plus the category names of the bench."""
    feat_dir = feat_dir or config.features_dir()
    if bench == "coco":
        _, _, names = coco_gt()
    else:
        _, _, names, _ = lvis_minival()
    path = os.path.join(feat_dir, f"{bench}_gd_phrases.npz")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}\nrun `promptcal extract --bench {bench}` first")
    z = np.load(path, allow_pickle=True)
    key = [str(k) for k in z["key"]]
    emb = z["emb"].astype(np.float32)
    index = {k: i for i, k in enumerate(key)}
    return key, emb, index, names
