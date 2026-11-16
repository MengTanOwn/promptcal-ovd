"""Ground-truth access and box matching.

Two datasets are used, and each plays a different role:

``coco_gt``       COCO val2017, 80 categories.  The vocabulary the calibrator is fitted
                  on, and the corpus the wording study is measured on.
``lvis_minival``  LVIS v1.0 minival, 1,203 categories.  The vocabulary the calibrator is
                  deployed on without refitting.  Its images *are* COCO 2017 JPEGs, so
                  one image tree serves both.

Nothing here depends on the detector, which is why this module can be imported by the
analysis without loading torch.
"""
import json
import os

import numpy as np

from .. import config


def iou_matrix(a, b):
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def greedy_match(dets, gts, thr=0.5):
    """Greedy one-to-one assignment; ``dets`` must already be sorted by score desc.

    Greedy rather than Hungarian because that is what the detector's own AP
    implementation does: the label of a box should agree with the metric the paper
    reports.  One ground-truth instance can be claimed by at most one detection *per
    run*, and a run is a single wording (see :mod:`promptcal.data.features`).
    """
    if len(dets) == 0 or len(gts) == 0:
        return np.zeros(len(dets), dtype=bool)
    ious = iou_matrix(dets, gts)
    taken = np.zeros(len(gts), dtype=bool)
    hit = np.zeros(len(dets), dtype=bool)
    for i in range(len(dets)):
        row = np.where(taken, -1.0, ious[i])
        j = int(np.argmax(row))
        if row[j] >= thr:
            taken[j] = True
            hit[i] = True
    return hit


def coco_gt(json_path=None, img_dir=None):
    """COCO val2017: ``(by_image, images, category_names)``.

    ``by_image[image_id][category_id]`` is a list of xyxy boxes and ``images`` maps an
    image id to a file path.
    """
    json_path = json_path or config.coco_annotations()
    img_dir = img_dir or config.coco_images()
    with open(json_path, encoding="utf-8") as fh:
        d = json.load(fh)
    by_img = {}
    for a in d["annotations"]:
        if a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], {}).setdefault(a["category_id"], []).append(
            [x, y, x + w, y + h])
    imgs = {im["id"]: os.path.join(img_dir, im["file_name"]) for im in d["images"]}
    names = {c["id"]: c["name"] for c in d["categories"]}
    return by_img, imgs, names


def lvis_minival(json_path=None, img_dir=None):
    """LVIS v1.0 minival: ``(by_image, images, category_names, synonyms)``.

    The minival split ships with COCO image ids and file names, so ``img_dir`` is the
    COCO ``val2017`` directory unless told otherwise.
    """
    json_path = json_path or config.lvis_annotations()
    img_dir = img_dir or config.coco_images()
    with open(json_path, encoding="utf-8") as fh:
        d = json.load(fh)
    by_img = {}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], {}).setdefault(a["category_id"], []).append(
            [x, y, x + w, y + h])
    imgs = {im["id"]: os.path.join(img_dir, os.path.basename(im["file_name"]))
            for im in d["images"]}
    names = {c["id"]: c["name"] for c in d["categories"]}
    syn = {c["id"]: (c.get("synonyms") or []) for c in d["categories"]}
    return by_img, imgs, names, syn
