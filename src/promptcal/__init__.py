"""Prompt-conditioned calibration for open-vocabulary detectors.

The package is organised around the four stages of the pipeline:

``promptcal.extract``     run a detector under several phrasings, dump per-detection
                          records (score, logit, box, category, wording).
``promptcal.data``        annotations, box matching, label attachment, shard loading.
``promptcal.calibrate``   metrics, the scalar baselines, and PhC.
``promptcal.experiments`` the five studies reported in the paper.

``promptcal.prompts`` holds the five wordings themselves, and ``promptcal.config``
resolves the three directories (data, features, results) from the environment.

The central result is a proposition rather than a number: PhC is strictly monotone
within a wording, so it cannot change average precision.  Anything that does change
the ranking is therefore a different method, and is reported as such.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
