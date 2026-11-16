"""``promptcal extract`` and ``promptcal embed``: the only GPU stage of the pipeline.

``extract`` walks the images of a bench, queries the detector five times per image (one
query per wording) and writes one ``.npz`` shard every ``--part-size`` images.  A row is
a detection, not a matched object: the analysis decides what is a true positive, so the
dump stays independent of the evaluation protocol.

``embed`` writes the pooled text embedding of every phrase.  The extractor writes the
same file at the end of a run; the separate entry point exists so that the analysis can
start while a long extraction is still going, and so that a new corpus can be added
without re-running the detector.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from .. import config
from ..data import coco_gt, lvis_minival
from ..prompts import VARIANT_NAMES
from . import grounder as E


def _dataset(bench):
    return coco_gt() if bench == "coco" else lvis_minival()


def _add_common(ap):
    ap.add_argument("--bench", default="coco", choices=["coco", "lvis"])
    ap.add_argument("--out-dir", default=None,
                    help="where to write the shards (default: $PROMPTCAL_FEATURES)")
    ap.add_argument("--device", default="cuda")
    return ap


def main_extract(argv=None):
    ap = argparse.ArgumentParser(
        prog="promptcal extract",
        description="Run Grounding DINO under five wordings and dump per-detection rows.")
    _add_common(ap)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images")
    ap.add_argument("--thr", type=float, default=0.01,
                    help="keep detections with score >= thr (default: 0.01)")
    ap.add_argument("--cap", type=int, default=80,
                    help="max detections per (category, wording) per image")
    ap.add_argument("--part-size", type=int, default=200, help="images per shard")
    ap.add_argument("--seed", type=int, default=0, help="image order")
    ap.add_argument("--force", action="store_true", help="overwrite existing shards")
    a = ap.parse_args(argv)

    by_img, imgs, names, *_ = _dataset(a.bench)
    rng = np.random.RandomState(a.seed)
    ids = sorted(imgs)
    rng.shuffle(ids)
    if a.limit:
        ids = ids[:a.limit]
    out_dir = a.out_dir or str(config.features_dir())
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.startswith(f"{a.bench}_gd_part")]
    if existing and not a.force:
        raise SystemExit(
            f"{len(existing)} shards already in {out_dir}; pass --force to overwrite "
            f"or --out-dir to write elsewhere")
    print(f"{a.bench}: {len(ids)} images, {len(names)} categories", flush=True)
    print(config.describe(), flush=True)

    proc, model = E.load_model(a.device)
    shard, part = [], 0
    t0, done = time.time(), 0
    for n, iid in enumerate(ids, 1):
        img = cv2.imread(imgs[iid])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cats = sorted(by_img.get(iid, {}))
        if not cats:
            continue
        boxes, scs, enc = E.run_image(model, proc, img, E.variant_phrases(names, cats),
                                      a.device)
        h, w = img.shape[:2]
        for v in range(len(VARIANT_NAMES)):
            for k, c in enumerate(cats):
                s = scs[v][k]
                keep = np.where(s >= a.thr)[0]
                if len(keep) == 0:
                    continue
                if len(keep) > a.cap:
                    keep = keep[np.argsort(-s[keep])[:a.cap]]
                shard.append(dict(
                    img_id=np.full(len(keep), iid, np.int32),
                    cat_id=np.full(len(keep), c, np.int32),
                    phrase=np.full(len(keep), v, np.int8),
                    box=boxes[v][keep].astype(np.float32),
                    img_wh=np.tile(np.array([w, h], np.float32), (len(keep), 1)),
                    score=s[keep].astype(np.float32),
                ))
        done += 1
        if done % 100 == 0 or n == len(ids):
            el = time.time() - t0
            print(f"[{a.bench}] {done}/{len(ids)} imgs  {el / done:.3f}s/img  "
                  f"eta {(len(ids) - done) * el / done / 60:.1f} min", flush=True)
        if done % a.part_size == 0 or n == len(ids):
            if not shard:
                continue
            cat = {k: np.concatenate([d[k] for d in shard]) for k in shard[0]}
            p = np.clip(cat["score"].astype(np.float64), 1e-7, 1 - 1e-7)
            cat["logit"] = np.log(p / (1 - p)).astype(np.float32)
            fn = os.path.join(out_dir, f"{a.bench}_gd_part{part:03d}.npz")
            np.savez(fn, **cat)
            print(f"  shard {part} -> {fn}  {len(cat['score'])} dets", flush=True)
            part += 1
            shard = []

    keys = E.phrase_keys(names, sorted(names))
    emb = E.embed_phrases(model, proc, keys, a.device)
    np.savez(os.path.join(out_dir, f"{a.bench}_gd_phrases.npz"),
             key=np.array(keys), emb=emb)
    meta = dict(bench=a.bench, model=E.MODEL, thr=a.thr, cap=a.cap,
                n_images=len(ids), shards=part, seed=a.seed)
    with open(os.path.join(out_dir, f"{a.bench}_gd_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("done", meta, flush=True)


def main_embed(argv=None):
    ap = argparse.ArgumentParser(
        prog="promptcal embed",
        description="Write the pooled text embedding of every phrase of a bench.")
    _add_common(ap)
    ap.add_argument("--bench", default="coco", choices=["coco", "lvis"])
    a = ap.parse_args(argv)

    _, _, names, *_ = _dataset(a.bench)
    keys = E.phrase_keys(names, sorted(names))
    out_dir = a.out_dir or str(config.features_dir())
    os.makedirs(out_dir, exist_ok=True)
    proc, model = E.load_model(a.device)
    emb = E.embed_phrases(model, proc, keys, a.device)
    path = os.path.join(out_dir, f"{a.bench}_gd_phrases.npz")
    np.savez(path, key=np.array(keys), emb=emb)
    print(f"{len(keys)} phrases -> {path}")


if __name__ == "__main__":
    sys.exit(main_extract())
