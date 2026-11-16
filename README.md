# Naming the object changes the answer

**Calibration and risk control for open-vocabulary detectors.**

Reference code for the paper *Naming the object changes the answer: calibration and risk
control for open-vocabulary detectors*.  The paper asks what an open-vocabulary detector's
score actually means once the label is a sentence typed at inference time, and what a
calibrator can and cannot repair.

The short answer:

- The same object, scored five ways ("dog", "a dog", "a photo of a dog", "an image of a
  dog", "dogs"), moves by **0.33** on average, on a scale from 0 to 1.  AP moves from
  **0.777** to **0.462** depending only on the wording.
- Calibration repairs the **scale**: band ECE falls from **0.180** to **0.032**, a factor
  of 5.6, with AP unchanged.
- It cannot repair the **order**.  A threshold chosen once and shipped to every wording
  keeps 1.4x its target false-discovery rate, and *no* monotone calibrator can fix that —
  Proposition 1 forbids it.  What does work is requiring the guarantee inside every
  wording, where calibration buys 11% more coverage at the same risk.

Everything below is the code that produced those numbers, from the detector dump to the
LaTeX macros in the manuscript.

## What is in the box

| stage | command | what it does |
|---|---|---|
| extract | `promptcal extract` | Grounding DINO under five wordings, one row per detection |
| wording | `promptcal wording` | how far the score of the same object moves |
| calibration | `promptcal calibration` | eight calibrators, all wordings and held-out ones |
| bootstrap | `promptcal bootstrap` | image-level intervals on the band error |
| conformal | `promptcal conformal` | thresholds with a false-discovery guarantee |
| transfer | `promptcal transfer` | fit on COCO, deploy on LVIS, refit nothing |
| macros, figures | `promptcal macros`, `promptcal figures` | the numbers and plots of the paper |

A *run* is one wording over the whole corpus.  Every metric is computed inside a run and
then averaged over runs; pooling the five wordings would put one ground-truth instance
into five queries and manufacture false negatives.  That single convention explains most
of the numbers in the paper, and it is implemented in `promptcal/experiments/common.py`.

## Install

```bash
git clone https://github.com/your-org/promptcal-ovd.git
cd promptcal-ovd
pip install -e .                 # analysis only: numpy + torch
pip install -e ".[extract,figures,dev]"   # + transformers, OpenCV, matplotlib, pytest
```

Python 3.10 or newer.  The reference run used Python 3.14, torch 2.14, transformers 5.19,
numpy 2.5, on one RTX 3090 (24 GB).  The analysis stages run on CPU and need no GPU; only
`extract` and `embed` touch the detector.

## Data

The pipeline needs COCO val2017 and the LVIS v1.0 minival annotations.  The images are the
same in both cases, so one image tree serves the two experiments.

```
data/
├── coco/
│   ├── annotations/instances_val2017.json
│   └── val2017/*.jpg
└── lvis/
    └── annotations/lvis_v1_minival_inserted_image_name.json
```

Set `PROMPTCAL_DATA` if the tree lives elsewhere, or `PROMPTCAL_COCO_ROOT` /
`PROMPTCAL_LVIS_ROOT` if the two datasets are on different disks.  See
[`docs/data.md`](docs/data.md) for the download links and for what the extractor writes.

## Reproduce the paper

```bash
export PROMPTCAL_DATA=/path/to/data

promptcal extract --bench coco          # 5.69M detections, ~55 min on a 3090
promptcal wording                      # -> results/p1_describe.json
promptcal calibration                  # -> results/p2_calibration.json   (~45 s)
promptcal bootstrap                    # -> results/p2b_ci.json           (~25 s)
promptcal conformal                    # -> results/p3_conformal.json     (~30 s)
promptcal extract --bench lvis          # 6.06M detections, ~65 min
promptcal transfer                     # -> results/p4_transfer.json      (~75 s)
promptcal macros && promptcal figures   # -> paper/nums.tex, paper/tables, paper/fig
```

The result JSONs of the published runs are committed under `results/`, so the last two
commands work without re-running anything: clone, run `promptcal macros`, compile the
paper, and you get the numbers in the PDF.  Re-running an experiment overwrites its JSON
and the manuscript follows.

`docs/reproduction.md` gives the expected console output of each stage, the runtime, and
the tolerances to expect on a different GPU or BLAS.

## Using PhC on your own detector

PhC is post-hoc: it needs the scores, not the model.  Cache one row per detection with

- the pre-sigmoid `logit` and the prompt string that produced it,
- a category id,
- the wording index (0-4, or your own), and
- a label (whether the box matches a ground-truth instance of that category at IoU 0.5).

Then:

```python
from promptcal.calibrate import PhC, PhraseTable

# keys/emb/index: the distinct prompt strings and their pooled text embeddings
# (promptcal.extract.embed_phrases writes these; any 768-d text encoder works)
table = PhraseTable(keys, emb, index, cat_names)

cal = PhC(table, epochs=300, lr=3e-3, w_band=10.0).fit(logit, cat_id, variant, label)
p = cal.predict(logit_test, cat_id_test, variant_test)
```

Two properties are worth knowing before you deploy it:

1. **It cannot change average precision.**  PhC applies `logit' = logit * exp(-l(phi)) +
   b(phi)` with `exp(-l) > 0`, which is strictly increasing within a wording, so the
   ordering of a run — and hence its AP — is untouched.  That is a feature for a
   deployment that already ships a threshold, and a limitation for one that does not.
2. **It carries no per-category table.**  Its parameters are predicted from the phrase
   itself, so it survives a vocabulary it was never fitted on (section 5.6 of the paper).
   Vector scaling does not: on LVIS every lookup misses and the result is *worse* than not
   calibrating at all.

Fitting on cached logits takes seconds; the descriptor costs 33 floats per phrase, computed
once per phrase rather than once per detection.

## Layout

```
src/promptcal/
├── prompts.py          the five wordings and the surface-form rules
├── config.py           PROMPTCAL_DATA / _FEATURES / _RESULTS / _PAPER
├── data/               annotations, box matching, label attachment, shards
├── extract/            Grounding DINO: one forward per wording, phrase embeddings
├── calibrate/          metrics, scalar baselines, conformal selection, PhC
├── experiments/        the five studies, one module each
├── report/             macros and figures for the manuscript
└── cli.py              `promptcal <command>`
paper/                  LaTeX source of the manuscript
results/                the JSON of every experiment reported in the paper
tests/                  unit and property tests (no data, no GPU needed)
docs/                   data, reproduction, and design notes
```

`docs/design.md` states the two propositions the code is built around and points at the
test that checks each one.

## Tests

```bash
pip install -e ".[dev]"
pytest              # a few seconds, CPU only, no data, no GPU
```

The tests pin the parts a refactor could silently break: the five wordings stay distinct
and correctly inflected, ECE and AP agree with hand-computed values, PhC is monotone
within a wording and leaves AP bit-identical, and the conformal threshold meets its target
on a synthetic calibration set.

## Citation

```bibtex
@article{tan2026naming,
  title   = {Naming the object changes the answer: calibration and risk control for
             open-vocabulary detectors},
  author  = {Tan, Meng and Zhou, Shengkai and Lin, Ling},
  journal = {Applied Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

`CITATION.cff` carries the same information for GitHub's "Cite this repository" button.

If you are the one publishing this repository, `docs/release-checklist.md` lists what to
change first (the `your-org` placeholder, the tag, the DOI).

## License and third-party components

MIT, see [`LICENSE`](LICENSE).  The code calls, but does not vendor, the following:

- **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`), Apache-2.0, via
  `transformers`.  The weights are downloaded at first use and are not redistributed here.
- **COCO val2017** — annotations CC BY 4.0, images per the COCO terms.
- **LVIS v1.0** — annotations CC BY 4.0; its images are the COCO 2017 JPEGs.

The manuscript in `paper/` uses the Springer `sn-jnl` class, which is *not* redistributed
here; download the Springer Nature LaTeX template to compile it (see `paper/README.md`).
