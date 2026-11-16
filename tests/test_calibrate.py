"""The calibrators: monotonicity, AP invariance, and the conformal guarantee.

Two of these are statements from the paper rather than implementation details:

* Proposition 1 -- PhC is strictly increasing inside a (category, wording) cell, so the
  average precision of a run cannot change.  The test compares AP bit for bit against the
  uncalibrated scores.
* Isotonic regression is the counter-example: it is free to reorder a run, and it does,
  which is why it loses AP in Table 2.

The fixtures are synthetic and tiny: the point is the property, not the dataset.
"""
import numpy as np
import pytest

from promptcal.calibrate.baselines import Isotonic, TemperatureScaling, VectorScaling
from promptcal.calibrate.conformal import fdr, pick_lambda
from promptcal.calibrate.metrics import average_precision
from promptcal.calibrate.phc import PhC, PhraseTable, VariantTemperature
from promptcal.prompts import N_VARIANT, variants

NAMES = {1: "dog", 2: "cat", 3: "person", 4: "apple", 5: "sheep"}


def make_table(dim=32, n_pca=8, seed=0):
    rng = np.random.RandomState(seed)
    keys = [s for c in sorted(NAMES) for s in variants(NAMES[c])]
    emb = rng.normal(size=(len(keys), dim)).astype(np.float32)
    index = {k: i for i, k in enumerate(keys)}
    return PhraseTable(keys, emb, index, NAMES, n_pca=n_pca)


def make_data(n=6000, seed=0):
    """Scores whose meaning depends on the wording, which is the situation under study."""
    rng = np.random.RandomState(seed)
    cat = rng.choice(sorted(NAMES), size=n).astype(np.int64)
    var = rng.randint(0, N_VARIANT, size=n).astype(np.int64)
    offset = np.array([0.0, -0.2, -1.4, -1.2, -0.3])          # "photo" is much worse
    logit = rng.normal(0, 2.0, size=n) + offset[var]
    # true probability: a squashed, wording-shifted version of the same score
    truth = 1.0 / (1.0 + np.exp(-(0.45 * logit - 0.9)))
    y = (rng.uniform(size=n) < truth).astype(np.float32)
    return dict(logit=logit.astype(np.float32), cat=cat, var=var, y=y)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_temperature_scaling_is_monotone_and_improves_nll():
    d = make_data()
    before = -np.mean(d["y"] * np.log(sigmoid(d["logit"]) + 1e-12)
                      + (1 - d["y"]) * np.log(1 - sigmoid(d["logit"]) + 1e-12))
    ts = TemperatureScaling().fit(d["logit"], None, d["y"], None)
    p = ts.predict(d["logit"], None, d["cat"])
    after = -np.mean(d["y"] * np.log(p + 1e-12) + (1 - d["y"]) * np.log(1 - p + 1e-12))
    assert after < before
    # strictly increasing in the score: check on float64 inputs, because the scores of
    # the synthetic data are float32 and ties there are rounding, not ties in the map
    z = np.linspace(-9, 9, 257)
    pz = ts.predict(z, None, None)
    assert np.all(np.diff(pz) > 0)
    order = np.argsort(d["logit"])
    assert np.all(np.diff(p[order]) >= 0)


def test_vector_scaling_keys_its_table_on_the_category():
    d = make_data()
    vs = VectorScaling().fit(d["logit"], None, d["y"], d["cat"])
    assert set(vs.params) <= set(NAMES)
    p = vs.predict(d["logit"], None, d["cat"])
    assert p.shape == d["logit"].shape
    assert np.all((p > 0) & (p < 1))
    # monotone in the score inside a category
    m = d["cat"] == 1
    order = np.argsort(d["logit"][m])
    assert np.all(np.diff(p[m][order]) >= 0)


def test_isotonic_is_non_decreasing_but_not_necessarily_strict():
    d = make_data()
    iso = Isotonic(bins=32).fit(d["logit"], None, d["y"], None)
    p = iso.predict(d["logit"], None, d["cat"])
    order = np.argsort(d["logit"])
    assert np.all(np.diff(p[order]) >= 0)
    assert len(np.unique(p)) < len(p)      # piecewise constant by construction


def test_variant_temperature_has_one_parameter_per_wording():
    d = make_data()
    vt = VariantTemperature().fit(d["logit"], d["cat"], d["var"], d["y"])
    assert vt.log_t.shape == (N_VARIANT,)
    p = vt.predict(d["logit"], d["cat"], d["var"])
    assert np.all((p > 0) & (p < 1))


@pytest.fixture(scope="module")
def fitted():
    table = make_table()
    d = make_data()
    cal = PhC(table, epochs=120, lr=5e-3, seed=0, w_band=10.0).fit(
        d["logit"], d["cat"], d["var"], d["y"])
    return table, d, cal


def test_phc_is_strictly_monotone_inside_a_cell(fitted):
    _, d, cal = fitted
    for cat in sorted(NAMES):
        for var in range(N_VARIANT):
            z = np.linspace(-8, 8, 64)
            c = np.full_like(z, cat, dtype=np.int64)
            v = np.full_like(z, var, dtype=np.int64)
            p = cal.predict(z.astype(np.float32), c, v)
            assert np.all(np.diff(p) > 0), (cat, var)


def run_ap(p, d):
    """AP inside each wording run, averaged over runs -- the paper's protocol.

    Pooling the five wordings into one array is *not* the same quantity: within a wording
    the transform is constant per category, so the order is untouched; across wordings it
    is not, and the pooled number would move by about 0.001 for reasons that have nothing
    to do with the calibrator.
    """
    vals = []
    for v in range(N_VARIANT):
        m = d["var"] == v
        vals.append(average_precision(p[m], d["y"][m], d["cat"][m]))
    return float(np.mean(vals))


def test_phc_leaves_average_precision_untouched(fitted):
    """Proposition 1, measured: the same AP to machine precision."""
    _, d, cal = fitted
    base = sigmoid(d["logit"]).astype(np.float64)
    p = cal.predict(d["logit"], d["cat"], d["var"])
    assert run_ap(base, d) == pytest.approx(run_ap(p, d), abs=1e-12)


def test_pooling_wordings_would_have_broken_it(fitted):
    """Why the evaluation is per run: the same map does move a pooled ranking."""
    _, d, cal = fitted
    base = sigmoid(d["logit"]).astype(np.float64)
    p = cal.predict(d["logit"], d["cat"], d["var"])
    pooled_base = average_precision(base, d["y"], d["cat"])
    pooled_phc = average_precision(p, d["y"], d["cat"])
    assert pooled_phc != pytest.approx(pooled_base, abs=1e-9)


def test_phc_with_a_band_term_still_leaves_ap_untouched(fitted):
    table, d, _ = fitted
    cal = PhC(table, epochs=60, lr=5e-3, seed=1, w_band=10.0, w_band_norm=True)
    cal.fit(d["logit"], d["cat"], d["var"], d["y"])
    base = sigmoid(d["logit"]).astype(np.float64)
    p = cal.predict(d["logit"], d["cat"], d["var"])
    assert run_ap(base, d) == pytest.approx(run_ap(p, d), abs=1e-12)


def test_phc_uses_the_reference_projection_when_deploying(fitted):
    """Deploying on a new vocabulary must refit nothing, including the PCA basis."""
    table, _, _ = fitted
    other = {10: "table", 11: "chair", 12: "bottle"}
    keys = [s for c in sorted(other) for s in variants(other[c])]
    emb = np.random.RandomState(3).normal(size=(len(keys), 32)).astype(np.float32)
    index = {k: i for i, k in enumerate(keys)}
    new = PhraseTable(keys, emb, index, other, n_pca=8, ref=table)
    assert np.allclose(new.mu, table.mu)
    assert np.allclose(new.W, table.W)
    assert new.Z.shape == (len(keys), 8)


def test_phrase_descriptor_has_the_documented_width(fitted):
    table, _, _ = fitted
    X = table.build(np.array([1, 2, 3]), np.array([0, 2, 4]))
    # 8 PCA dims + log token count + character length + embedding norm + cosine to the
    # bare name + 5-way one-hot of the wording (24 + 4 + 5 = 33 in the paper)
    assert X.shape == (3, 8 + 4 + N_VARIANT)


def test_conformal_threshold_meets_its_target_in_sample():
    rng = np.random.RandomState(7)
    n = 20000
    p = rng.uniform(size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    for alpha in (0.05, 0.1, 0.2):
        lam = pick_lambda(p, y, alpha)
        got, kept = fdr(p, y, lam)
        assert kept > 0
        # the conformal estimate is conservative by (1 + V)/R, so allow a small slack
        assert got <= alpha + 0.01


def test_conformal_prefers_the_largest_admissible_set():
    # all scores below 1 with no negatives: every threshold down to the minimum is
    # admissible, and the estimate picks the *smallest* one
    p = np.array([0.9, 0.5, 0.2, 0.1])
    y = np.array([1, 1, 1, 1])
    assert pick_lambda(p, y, 0.3) == pytest.approx(0.1)


def test_fdr_reports_nan_when_nothing_is_kept():
    got, kept = fdr(np.array([0.1, 0.2]), np.array([1, 0]), 0.99)
    assert kept == 0 and np.isnan(got)
