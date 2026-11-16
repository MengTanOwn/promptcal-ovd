# Design notes

Three ideas do all the work in this repository.  Each one is short to state, easy to get
wrong in code, and checkable by a test.

## 1. A run is one wording

The corpus is a set of images and a vocabulary; a *run* is that corpus queried with one
wording.  Five wordings give five runs over the same images.

```
image ─┬─ "dog"               ─> scores ─┐
       ├─ "a dog"             ─> scores ─┤
       ├─ "a photo of a dog"  ─> scores ─┼─ each column is a deployment:
       ├─ "an image of a dog" ─> scores ─┤   its own ranking, its own threshold
       └─ "dogs"              ─> scores ─┘
```

Consequences, all of them implemented in `promptcal/experiments/common.py`:

- **Matching is per run.** The label of a box is decided by a greedy one-to-one assignment
  against the ground-truth instances of that category *within that wording*
  (`attach_labels` sorts by `(image, category, wording)` and matches inside each cell).  A
  single ground-truth dog cannot be a true positive in five queries at once: that would
  score four runs as wrong by construction.
- **Metrics are per run.** ECE, band ECE, AP and AURC are computed inside a run and then
  averaged over the five.  The paper's AP of 0.6395 is the mean of five per-run APs.
- **The band is per run, and frozen.** `[0.3, 0.9)` is applied to the *uncalibrated* score
  of that run, before any calibrator is fitted.  Fixing it after calibration would let a
  calibrator improve its own metric by moving rows out of the band it is scored on.

## 2. Proposition 1: a monotone map cannot change AP

PhC applies

```
logit' = logit * exp(-l(phi)) + b(phi),      p = sigmoid(logit')
```

where `phi` is the phrase descriptor and `l, b` depend only on `(category, wording)`,
never on the individual box.  Inside a run, `phi` is constant per category, so the map is
`logit -> a * logit + b` with `a = exp(-l) > 0`: strictly increasing.  A strictly increasing
map preserves every within-category ranking, and AP depends on nothing else.  So

> **AP(calibrated) = AP(uncalibrated), exactly.**

`tests/test_calibrate.py::test_phc_leaves_average_precision_untouched` checks this to 1e-12,
and the paper's Table 2 shows all six monotone methods reporting the identical 0.6395.
Isotonic regression is the exception, and it is deliberately included: it can reorder a run,
and it does, dropping AP to 0.524.

This is why PhC is a *scale* repair and not a *ranking* repair.  It is the right tool when a
deployment already ships a threshold that was tuned on the calibrated scale; it is the wrong
tool if the ranking itself is what needs fixing.

## 3. Selection: pooled, worst-case, per wording

A user asks "if I keep everything above `s`, what fraction of what I keep is wrong?".
Conformalised selection answers it from a labelled calibration half:

```
lambda_hat(alpha) = min { lambda : (1 + V(lambda)) / max(R(lambda), 1) <= alpha }
```

with `R` kept boxes and `V` kept negatives.  Four ways to ship a number:

| protocol | lambda | what it promises |
|---|---|---|
| `pooled` | one, from all wordings' calibration rows | nothing per wording |
| `per_wording` | five, one each | each wording separately, but needs five thresholds |
| `bare_only` | one, from the bare name | nothing for the other four |
| `worst` | `max` of the five per-wording lambdas | every wording simultaneously |

`worst` is the interesting one.  Because FDR falls as the threshold rises, requiring the
constraint inside *every* wording means taking the largest of the five per-wording lambdas,
and — being a maximum over strata — the simultaneous statement needs no multiplicity
correction.

The negative result the paper reports is that `pooled` misses its target badly (0.417
attained against `alpha = 0.30`), while `worst` attains 0.310 and 0.294 for the uncalibrated
and PhC scores respectively.  Calibration matters here not because it fixes the pooled
threshold (Proposition 1 forbids it) but because a score on which the five runs agree more
closely has a less binding worst case, which buys 11% more coverage at the same risk.

## 4. The descriptor costs 33 floats per phrase, not per detection

```
phi(phrase) = [ 24 PCA dims of the pooled 768-d text embedding
              , log(1 + #tokens)
              , #characters / 10
              , ||embedding|| / 20
              , cosine(embedding, embedding of the bare name)
              , one-hot over the five wordings ]            = 33 dims
```

The 768-d embedding is projected once per *distinct phrase* (a few thousand per corpus),
so the per-detection cost of the calibrator is 33 floats and a 33x2 linear head — 68
parameters in total.  On a corpus with 5.7 M detections that is the difference between a
few seconds and a few hours.

The PCA basis is fitted once, on the *source* vocabulary, and reused unchanged when
deploying on a new one (`PhraseTable(..., ref=source_table)`).  That is what makes
`promptcal transfer` a transfer in the strict sense: no refit of the head, of the
projection, or of a per-category table.

## 5. Why the code carries a `phc_cat` variant it does not recommend

`PhC(..., cat_bias=True)` adds a per-category `(dl, db)` embedding — 160 extra parameters on
COCO.  It scores marginally better in-domain (band ECE 0.032) and it is included because a
reviewer will ask whether the phrase descriptor is doing anything a category table could
not.  It is not the recommended configuration: a table indexed by a label cannot be shipped
to a deployment that adds labels, which is precisely what the LVIS experiment shows.
