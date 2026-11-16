"""Shared loading, splitting and evaluation for the experiments.

The one convention worth stating twice: a *run* is one wording over the whole corpus, and
every metric is computed inside a run and then averaged over runs.  Pooling the five
wordings into one array would put the same ground-truth instance into five different
queries, four of which the matcher would then score as false positives; the resulting
number would be an artefact of the protocol, not a property of the detector.
"""
import numpy as np

from ..calibrate import metrics as K
from ..data import features as F
from ..prompts import VARIANT_NAMES

#: The five wordings, in the canonical order (see :mod:`promptcal.prompts`).
VAR = VARIANT_NAMES


def load(bench="coco", tag=None, feat_dir=None):
    """Feature shards with labels attached, plus the phrase table of the bench.

    Returns ``(feat, (keys, embeddings, index, category_names))``.
    """
    tag = tag or f"{bench}_gd"
    feat = F.load_shards(feat_dir, tag)
    feat = F.attach_labels(feat, bench)
    key, emb, index, names = F.load_phrases(bench, feat_dir)
    return feat, (key, emb, index, names)


def evaluate(prob, y, cat, band_mask=None):
    """Every metric of the paper for one array of (score, label) pairs."""
    if band_mask is None:
        band_mask = np.ones(len(y), dtype=bool)
    nb = int(band_mask.sum())
    return {
        "n": int(len(y)),
        "band_n": nb,
        "pos_rate": float(y.mean()),
        "ece": float(K.ece(prob, y)),
        "band_ece": float(K.ece(prob[band_mask], y[band_mask], 10)) if nb > 30
        else float("nan"),
        "brier": float(K.brier(prob, y)),
        "band_brier": float(np.mean((prob[band_mask] - y[band_mask]) ** 2)) if nb > 30
        else float("nan"),
        "nll": float(K.nll(prob, y)),
        "ap": float(K.average_precision(prob, y, cat)),
        "aurc": float(K.aurc(prob, y)),
    }


def split_by_image(feat, frac=0.5, seed=0):
    """Fit/test masks, split by image (see :func:`promptcal.data.features.split_by_image`)."""
    return F.split_by_image(feat, frac=frac, seed=seed)


# kept under the name the experiment modules use
load_split = split_by_image


def band_mask(score, lo=0.3, hi=0.9):
    """The operating band: the scores a reviewer of a detection UI actually looks at.

    Fixed by the *uncalibrated* score of a run, so it is available at deployment time
    (a score is) and cannot be moved by the calibrator being evaluated.
    """
    return (score >= lo) & (score < hi)


def image_bootstrap(idx, img, stat, n_boot=500, seed=0):
    """Cluster bootstrap over images; ``stat`` maps a row-index array to a float.

    Rows within an image are strongly correlated, so resampling rows understates the
    interval by a wide margin.  Resampling images is the unit the corpus was drawn in.
    """
    rng = np.random.RandomState(seed)
    imgs = np.unique(img)
    buckets = {i: np.where(img == i)[0] for i in imgs}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(imgs, size=len(imgs), replace=True)
        rows = np.concatenate([buckets[i] for i in pick])
        out.append(stat(rows))
    return np.array(out, dtype=np.float64)
