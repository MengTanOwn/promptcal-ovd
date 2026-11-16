"""Scoring, calibration and selection.

``metrics``     ECE, band ECE, Brier, NLL, AP, AURC.
``baselines``   temperature scaling, vector scaling, isotonic regression.
``conformal``   threshold selection with a false-discovery guarantee.
``phc``         phrase-conditioned calibration, the method of the paper.
"""
from .baselines import Isotonic, TemperatureScaling, VectorScaling
from .conformal import fdr, pick_lambda
from .phc import PhC, PhraseTable, VariantTemperature

__all__ = ["Isotonic", "TemperatureScaling", "VectorScaling", "PhC", "PhraseTable",
           "VariantTemperature", "pick_lambda", "fdr"]
