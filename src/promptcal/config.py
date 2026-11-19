"""Where the data, the features and the results live.

Three environment variables, all optional, keep the code free of absolute paths:

===========================  ==========================================  =============
variable                     meaning                                     default
===========================  ==========================================  =============
``PROMPTCAL_DATA``           root holding annotations and images         ``./data``
``PROMPTCAL_FEATURES``       where the extractor writes its shards       ``./features``
``PROMPTCAL_RESULTS``        where the experiments write their JSON      ``./results``
``PROMPTCAL_PAPER``          where ``macros``/``figures`` write their
                             output (``nums.tex``, ``tables/``, ``fig/``) ``./paper``
===========================  ==========================================  =============

Within the data root, the COCO and LVIS subtrees are looked up under both ``coco`` and
``COCO`` (and likewise for LVIS), because both spellings are in circulation.  An explicit
``PROMPTCAL_COCO_ROOT`` / ``PROMPTCAL_LVIS_ROOT`` overrides the lookup entirely, which is
what to use when the two datasets live on different disks.

All functions resolve at call time rather than at import time, so a test can point the
pipeline at a temporary directory by setting the variable and calling the function.
"""
import os
from pathlib import Path

# src/promptcal/config.py -> repository root
ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def data_root():
    return _path_from_env("PROMPTCAL_DATA", ROOT / "data")


def features_dir():
    return _path_from_env("PROMPTCAL_FEATURES", ROOT / "features")


def results_dir():
    return _path_from_env("PROMPTCAL_RESULTS", ROOT / "results")


def paper_dir():
    return _path_from_env("PROMPTCAL_PAPER", ROOT / "paper")


def _subtree(env_var, name):
    """``env_var`` if set, else ``<data>/<name>``, preferring whatever exists."""
    explicit = os.environ.get(env_var)
    if explicit:
        return Path(explicit).expanduser()
    root = data_root()
    for candidate in (root / name.lower(), root / name, root / name.upper()):
        if candidate.is_dir():
            return candidate
    return root / name.lower()


def coco_root():
    return _subtree("PROMPTCAL_COCO_ROOT", "coco")


def lvis_root():
    return _subtree("PROMPTCAL_LVIS_ROOT", "lvis")


def coco_annotations():
    return coco_root() / "annotations" / "instances_val2017.json"


def coco_images():
    return coco_root() / "val2017"


def lvis_annotations():
    """LVIS v1.0 minival, with the COCO file names inserted (the official split file)."""
    return lvis_root() / "annotations" / "lvis_v1_minival_inserted_image_name.json"


def describe():
    """One-line-per-path summary, printed by the experiments at start-up."""
    return "\n".join([
        f"data      {data_root()}",
        f"coco      {coco_root()}",
        f"lvis      {lvis_root()}",
        f"features  {features_dir()}",
        f"results   {results_dir()}",
    ])
