"""Detector extraction: per-detection records under five wordings."""
from .grounder import (
                       MODEL,
                       embed_phrases,
                       load_model,
                       phrase_keys,
                       run_image,
                       spans_for,
                       variant_phrases,
)

__all__ = ["MODEL", "load_model", "run_image", "embed_phrases", "phrase_keys",
           "spans_for", "variant_phrases"]
