# Data

Two corpora, one image tree.  Nothing in this repository is trained, so the data are only
needed at inference time and for labelling the detections.

## What to download

| what | file | where |
|---|---|---|
| COCO val2017 images (5,000 JPEGs) | `val2017.zip` | <https://cocodataset.org/#download> |
| COCO val2017 annotations | `annotations_trainval2017.zip` | same page |
| LVIS v1.0 minival annotations | `lvis_v1_minival_inserted_image_name.json` | <https://www.lvisdataset.org/dataset>, or the copy in the Detic/Detectron2 model zoo |

The LVIS minival split is a subset of the COCO 2017 images, and the file above carries the
COCO file names already inserted, so a single `val2017/` directory serves both experiments.
No LVIS images need to be downloaded separately.

## Where to put them

```
data/
├── coco/
│   ├── annotations/instances_val2017.json
│   └── val2017/
│       ├── 000000000139.jpg
│       └── ...                              # 5,000 files
└── lvis/
    └── annotations/lvis_v1_minival_inserted_image_name.json
```

```bash
export PROMPTCAL_DATA=/absolute/path/to/data      # default: ./data
```

| variable | default | what it points at |
|---|---|---|
| `PROMPTCAL_DATA` | `./data` | root of the tree above |
| `PROMPTCAL_COCO_ROOT` | `$PROMPTCAL_DATA/coco` | overrides the COCO subtree only |
| `PROMPTCAL_LVIS_ROOT` | `$PROMPTCAL_DATA/lvis` | overrides the LVIS subtree only |
| `PROMPTCAL_FEATURES` | `./features` | where the extractor writes its shards |
| `PROMPTCAL_RESULTS` | `./results` | where the experiments write their JSON |
| `PROMPTCAL_PAPER` | `./paper` | where the macros, tables and figures go |

`coco`, `COCO` and `Coco` are all accepted as directory names, and the two roots may live
on different disks (not unusual when one of them is on a shared volume).

## What the extractor writes

`promptcal extract --bench coco` writes, into `$PROMPTCAL_FEATURES`:

```
coco_gd_part000.npz ... coco_gd_part019.npz    # ~285k detection rows each
coco_gd_phrases.npz                            # key (P,) str, emb (P,768) float16
coco_gd_meta.json                              # model, thresholds, shard count, seed
```

A shard holds one row per detection:

| column | dtype | meaning |
|---|---|---|
| `img_id` | int32 | COCO image id |
| `cat_id` | int32 | category id of the *prompted* category |
| `phrase` | int8 | index into `promptcal.prompts.VARIANT_NAMES` |
| `box` | float32 (N,4) | xyxy in pixels |
| `img_wh` | float32 (N,2) | image size, so the box can be normalised later |
| `score` | float32 | the detector's score for that phrase |
| `logit` | float32 | `logit(score)`, cached so no calibrator has to recompute it |

Labels are **not** stored: `promptcal.data.features.attach_labels` recomputes them from the
annotations and the boxes, so the same shard can be re-labelled (a different IoU, a
different matcher) without re-running the detector.  That is also why the shards are only
useful together with the annotation JSONs of the matching corpus.

Size: about 490 MB for both corpora, 5.7 M + 6.1 M rows.

## Thresholds used in the paper

`--thr 0.01 --cap 80`: keep every detection above 0.01, at most 80 boxes per
(category, wording) per image.  The cap keeps the files bounded — without it a single
image with a large object produces thousands of near-duplicate boxes — and 80 is far above
the number of objects COCO puts in one image.

Because the band of interest is `[0.3, 0.9)`, almost all retained rows are irrelevant to
the headline metric; they are kept because a calibrator has to see the low tail to learn
the temperature, and because AP is computed over the whole range.

## Licensing

COCO annotations are CC BY 4.0; the images are subject to the Flickr terms collected in the
COCO release.  LVIS annotations are CC BY 4.0.  Neither is redistributed here — only the
code that reads them.
