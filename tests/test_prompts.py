"""The five wordings and the surface-form rules behind them.

These tests exist because the entire experiment rests on the five strings being what a
user would actually type, and on them being *distinct*: a bug that made "a photo of a dog"
and "an image of a dog" collapse to the same string would quietly turn a five-run study
into a four-run one.
"""
from promptcal.prompts import INVARIANT, N_VARIANT, VARIANT_NAMES, pluralise, variants


def test_variant_names_are_the_documented_five():
    assert VARIANT_NAMES == ("bare", "article", "photo", "image", "plural")
    assert N_VARIANT == 5


def test_dog_in_all_five_forms():
    assert variants("dog") == ["dog", "a dog", "a photo of a dog",
                               "an image of a dog", "dogs"]


def test_article_follows_the_initial_vowel():
    assert variants("apple")[1] == "an apple"
    assert variants("apple")[2] == "a photo of an apple"
    assert variants("apple")[3] == "an image of an apple"


def test_irregular_plurals():
    assert pluralise("person") == "people"
    assert pluralise("child") == "children"
    assert pluralise("toothbrush") == "toothbrushes"
    assert pluralise("knife") == "knives"


def test_uncountable_names_are_not_inflected():
    for name in ("sheep", "scissors", "broccoli"):
        assert name in INVARIANT
        assert pluralise(name) == name


def test_plural_inflects_the_head_noun_of_a_compound():
    assert pluralise("teddy bear") == "teddy bears"
    assert pluralise("traffic light") == "traffic light"   # mass-noun head


def test_lvis_underscores_become_spaces():
    # a user typing a query does not type "hot_dog"
    assert variants("hot_dog")[0] == "hot dog"
    assert variants("hot_dog")[2] == "a photo of a hot dog"


def test_the_five_forms_differ_for_every_countable_name_in_a_sample():
    names = ["dog", "cat", "person", "apple", "teddy bear", "wine glass", "hot dog",
             "knife", "bus", "box"]
    for name in names:
        forms = variants(name)
        assert len(forms) == 5
        assert len(set(forms)) == 5, (name, forms)


def test_an_invariant_noun_makes_bare_and_plural_the_same_query():
    """A known and intended collapse: "sheep" is its own plural.

    The five *wordings* stay distinct as hypotheses about how a user phrases a query, but
    for an uncountable or invariant noun two of them are the same string, and the detector
    returns exactly the same scores for both.  The analysis carries them as two runs
    anyway, which makes those two runs identical for that category.
    """
    forms = variants("sheep")
    assert forms[0] == forms[-1] == "sheep"
    assert len(set(forms)) == 4
