"""The ``promptcal`` command line.

Each subcommand is a module with a ``main(argv)`` of its own, so it can also be run
directly (``python -m promptcal.experiments.exp2_calibration``).  This module only
dispatches, so that the documented reproduction recipe is one program with one help text.
"""
import importlib
import sys

from . import __version__

#: name -> (module, attribute, one-line description)
COMMANDS = {
    "extract": ("promptcal.extract.cli", "main_extract",
                "run Grounding DINO under the five wordings"),
    "embed": ("promptcal.extract.cli", "main_embed",
              "write the text embedding of every phrase"),
    "wording": ("promptcal.experiments.exp1_wording", "main",
                "how far the score moves across wordings"),
    "calibration": ("promptcal.experiments.exp2_calibration", "main",
                    "calibrators on COCO, all wordings and held-out ones"),
    "bootstrap": ("promptcal.experiments.exp2b_bootstrap", "main",
                  "image-level intervals on the band error"),
    "conformal": ("promptcal.experiments.exp3_conformal", "main",
                  "thresholds with a false-discovery guarantee"),
    "transfer": ("promptcal.experiments.exp4_transfer", "main",
                 "fit on COCO, deploy on LVIS"),
    "diagnose-band": ("promptcal.experiments.diagnose_band_weight", "main",
                      "does band weighting equalise the FDR across wordings?"),
    "macros": ("promptcal.report.macros", "main",
               "write the paper's macros and tables"),
    "figures": ("promptcal.report.figures", "main",
                "draw the paper's figures"),
}

PIPELINE = ("extract", "wording", "calibration", "bootstrap", "conformal", "transfer",
            "macros", "figures")


def usage():
    width = max(len(k) for k in COMMANDS) + 2
    lines = [
        f"promptcal {__version__} -- prompt-conditioned calibration for "
        f"open-vocabulary detectors",
        "",
        "usage: promptcal <command> [options]",
        "",
    ]
    lines += [f"  {name:<{width}}{COMMANDS[name][2]}" for name in COMMANDS]
    lines += [
        "",
        "Run `promptcal <command> --help` for the options of one command.",
        "",
        "The pipeline, in order: " + " -> ".join(PIPELINE) + ".",
        "Every command writes a JSON into $PROMPTCAL_RESULTS (default ./results); the",
        "last two turn those JSONs into the manuscript's macros and figures.",
    ]
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command {name!r}\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2
    module, attr = COMMANDS[name][:2]
    func = getattr(importlib.import_module(module), attr)
    return func(rest) or 0


if __name__ == "__main__":
    sys.exit(main())
