# Reproducing the paper

Everything below runs from the repository root.  `PROMPTCAL_DATA` must point at the data
tree described in [data.md](data.md); the other variables have sensible defaults.

```bash
export PROMPTCAL_DATA=/path/to/data
pip install -e ".[extract,figures,dev]"
```

## The quick path: manuscript from the committed results

The JSON of every experiment in the paper is committed under `results/`, so the paper can
be rebuilt without a GPU and without touching the detector:

```bash
promptcal macros     # nums.tex + tables/tab_*.tex
promptcal figures    # fig/fig1_wording.pdf, fig2_calib.pdf, fig3_fdr.pdf
make paper           # pdflatex x2 + bibtex, needs paper/sn-jnl.cls
```

Both commands read `results/*.json`; a missing file makes the corresponding macros print
`--` instead of failing, which is how the manuscript can be compiled while an experiment is
still running.

## The full path

| step | command | machine | time here |
|---|---|---|---|
| 1 | `promptcal extract --bench coco` | 1x RTX 3090 | ~55 min |
| 2 | `promptcal wording` | CPU | 20 s |
| 3 | `promptcal calibration` | 1 GPU (any) | 45 s |
| 4 | `promptcal bootstrap` | 1 GPU | 22 s |
| 5 | `promptcal conformal` | 1 GPU | 27 s |
| 6 | `promptcal extract --bench lvis` | 1x RTX 3090 | ~65 min |
| 7 | `promptcal transfer` | 1 GPU | 75 s |
| 8 | `promptcal macros && promptcal figures` | CPU | 15 s |

`scripts/reproduce_all.sh` runs exactly this sequence and stops at the first failure.
Steps 3-5 need no LVIS data; step 7 needs both shard sets.

### What each stage prints

**`promptcal wording`** — one line per wording, then the spread statistics:

```
coco_gd: 5693221 detections, 4952 images, positive rate 0.0287
bare     n= 1063949  214.9/img  pos 0.0323 mean 0.0747  >0.3 0.0429  AP 0.7770  ECE 0.0500
article  n= 1155657  233.4/img  pos 0.0282 mean 0.0742  >0.3 0.0204  AP 0.7430  ECE 0.0602
photo    n= 1170394  236.3/img  pos 0.0264 mean 0.0812  >0.3 0.0193  AP 0.4622  ECE 0.0596
image    n= 1169850  236.2/img  pos 0.0269 mean 0.0731  >0.3 0.0217  AP 0.5020  ECE 0.0512
plural   n= 1133371  228.9/img  pos 0.0300 mean 0.0652  >0.3 0.0261  AP 0.7217  ECE 0.0474

14631 (image, category) groups with all five phrasings
  mean top score per phrasing: [0.6577, 0.5681, 0.3957, 0.436, 0.5402]
  mean spread 0.3276  median 0.3351
  spread > 0.1 in 0.938 of groups; > 0.2 in 0.766
  labels agree across all five in 0.748 of groups
```

These are the numbers of Table 1 and Fig. 1.  Note `n=4952` images, not 5,000: 48 images
of the val2017 split carry no annotation for any prompted category and are skipped, since
without annotations a detection there cannot be scored.

**`promptcal calibration`** — eight calibrators on the held-out half of COCO:

```
coco: train 1500000/2836798  test 2856423
identity     ECE 0.0536  band 0.1800  bandB 0.2287  Brier 0.0199  AP 0.6395  ...
temperature  ECE 0.0087  band 0.2123  bandB 0.2421  Brier 0.0173  AP 0.6395  ...
vector       ECE 0.0051  band 0.0812  bandB 0.1849  Brier 0.0158  AP 0.6395  ...
isotonic     ECE 0.0076  band 0.1246  bandB 0.2428  Brier 0.0180  AP 0.5236  ...
variant      ECE 0.0079  band 0.2108  bandB 0.2418  Brier 0.0172  AP 0.6395  ...
phc          ECE 0.0040  band 0.0597  bandB 0.1817  Brier 0.0157  AP 0.6395  ...
phc_band     ECE 0.0068  band 0.0324  bandB 0.1754  Brier 0.0157  AP 0.6395  ...
phc_cat      ECE 0.0065  band 0.0322  bandB 0.1729  Brier 0.0155  AP 0.6395  ...
```

then the held-out wordings (`--` fitted on bare/article/photo, deployed on image/plural).
This is Table 2, Table 3 and Fig. 2.  Two things to check when re-running: every monotone
method reports the same AP (0.6395) — that is Proposition 1 — and `isotonic`, which is free
to reorder, does not.

**`promptcal conformal`** — per wording at four targets, three protocols.  The line to look
at is the last one per method, the worst-case threshold at `alpha = 0.30`; this is Table 4
and Fig. 3.

**`promptcal diagnose-band`** is not part of the pipeline: it reports the per-wording FDR
at a pooled threshold with and without the band weighting, and is the evidence behind one
sentence in the discussion (that the residual disagreement is not a weighting artefact).
It takes about a minute and writes nothing.

**`promptcal transfer`** — COCO to LVIS, 6.06M detections over 4,752 annotated images and
1,203 unseen categories.  Expect band ECE 0.198 (uncalibrated) and 0.084 (PhC), AP 0.577 for
both, and `vector` at 0.277 — worse than doing nothing, because none of its 80 category rows
exists in LVIS.

Runtimes above are from one RTX 3090 with torch 2.14.  The extraction is the only stage that
needs the GPU for more than a minute.

## Determinism

- **Extraction** is deterministic given `--seed` (it fixes the image order), the same
  `transformers` version, and the same GPU; a different GPU model can change the last bits
  of the logits.
- **PhC** is seeded (`torch.manual_seed`, `np.random.seed` in `PhC.fit`), so a re-run on the
  same device gives the same parameters.
- **The two LBFGS-fitted baselines** (temperature, vector scaling) are fitted
  single-threaded (`promptcal.calibrate.baselines.single_threaded`) because a threaded
  reduction sums the 1.5M-row gradient in a schedule-dependent order; without that, two
  identical runs differ in the fourth decimal of the band error.  With it, the fits are
  bitwise reproducible on one machine.
- **The bootstrap** uses fixed per-wording seeds derived from the wording index, so its
  intervals are reproducible too.

## Where the numbers land

| file | written by |
|---|---|
| `results/p1_describe.json` | `promptcal wording` |
| `results/p2_calibration.json` | `promptcal calibration` |
| `results/p2b_ci.json` | `promptcal bootstrap` |
| `results/p3_conformal.json` | `promptcal conformal` |
| `results/p4_transfer.json` | `promptcal transfer` |
| `paper/nums.tex`, `paper/tables/*.tex` | `promptcal macros` |
| `paper/fig/*.pdf` | `promptcal figures` |

`paper/nums.tex` is the only place a number appears; the sections refer to `\MACRO{}` and
never to a literal.  If a re-run moves a number, the manuscript moves with it — except for
the prose, which quotes some values in words ("a factor of 5.6").  Those are also macros.

## Common snags

**`no shards matching coco_gd_part*.npz`** — `PROMPTCAL_FEATURES` points somewhere else;
run `promptcal --help` after `python -c "from promptcal import config; print(config.describe())"`
to see the resolved paths.

**`missing .../instances_val2017.json`** — `PROMPTCAL_DATA` is unset or wrong.  The
annotation files are needed by the analysis, not only by the extractor, because labels are
recomputed from them.

**CUDA out of memory during `extract`** — the five wordings are already run one at a time
(see `docs/design.md`); if it still does not fit, lower `--cap`, which bounds the number of
boxes kept per (category, wording) per image.

**`File 'sn-jnl.cls' not found`** — the class is not redistributed here.  See
`paper/README.md`.
