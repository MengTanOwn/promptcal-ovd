"""The five phrasings used as prompts, and the surface-form helpers they need.

The point of the study is that a user types *one* of these and the detector sees a
different text query in each case, while the thing being detected is identical.
"""

IRREGULAR = {
    "person": "people", "man": "men", "woman": "women", "child": "children",
    "mouse": "mice", "sheep": "sheep", "fish": "fish", "deer": "deer",
    "airplane": "airplanes", "scissors": "scissors", "skis": "skis",
    "toothbrush": "toothbrushes", "foot": "feet", "goose": "geese",
    "ox": "oxen", "knife": "knives", "loaf": "loaves",
}

# names that are already plural or uncountable, and must not be inflected
INVARIANT = {
    "scissors", "skis", "sheep", "fish", "deer", "moose", "shrimp", "swine",
    "clothes", "glasses", "sunglasses", "goggles", "pants", "trousers",
    "shorts", "jeans", "tongs", "pliers", "binoculars", "earphones",
    "headphones", "asparagus", "broccoli", "rice", "lettuce", "corn",
    "luggage", "baggage", "furniture", "equipment", "traffic light",
}

# names containing a mass/qualifier head that should stay singular
NO_PLURAL_PREFIX = ("traffic ", "tennis ", "baseball ", "basketball ", "fire ",
                    "stop ", "parking ")

#: The canonical order.  Every array in the pipeline indexes wordings with it, so a
#: shard written by one version of the code can be read by another.
VARIANT_NAMES = ("bare", "article", "photo", "image", "plural")
N_VARIANT = len(VARIANT_NAMES)


def _article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def pluralise(name):
    low = name.lower()
    if low in INVARIANT or low in IRREGULAR:
        return IRREGULAR.get(low, name)
    if low.startswith(NO_PLURAL_PREFIX):
        return name
    if len(name.split()) > 1:
        head, _, tail = name.rpartition(" ")
        return head + " " + pluralise(tail)
    if low.endswith(("s", "x", "z", "ch", "sh")):
        return name + "es"
    if low.endswith("y") and low[-2:-1] not in "aeiou":
        return name[:-1] + "ies"
    if low.endswith("f"):
        return name[:-1] + "ves"
    return name + "s"


def variants(name):
    """The five prompt strings, in the fixed order of ``VARIANT_NAMES``.

    LVIS spells categories with underscores; a user typing a query would not, so the
    prompt text is the human-readable form.  Both the extractor and the analysis go
    through this function, so the phrase keys always agree.
    """
    name = name.replace("_", " ")
    a = _article(name)
    return [name, f"{a} {name}", f"a photo of {a} {name}",
            f"an image of {a} {name}", pluralise(name)]
