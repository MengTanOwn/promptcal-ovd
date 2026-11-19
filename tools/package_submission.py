"""Build the submission package: every file the manuscript needs, nothing else.

Maintainer utility, not part of the pipeline: it is how the zip uploaded to the journal is
assembled from ``paper/``.  The zip is built from an explicit list rather than a glob so
that a stray local figure, a ``.bbl`` from an older bibliography or a stray results dump
cannot end up in the upload.

It also recompiles the manuscript, so it needs ``pdflatex`` and ``bibtex`` on PATH and the
Springer class files in ``paper/``.

The manuscript source is deliberately not part of the released repository, so this script
only runs in the authors' working copy, where ``paper/`` exists.  It is kept here because
it documents exactly which files the journal receives.
"""
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from promptcal import config  # noqa: E402

PAPER = str(config.paper_dir())
OUT = os.path.join(os.path.dirname(HERE), "submission")
STEM = "promptcal_ovd_submission"

TEX = ["main.tex", "sec1_intro.tex", "sec2_related.tex", "sec3_task.tex",
       "sec4_method.tex", "sec5_experiments.tex", "sec6_discussion.tex",
       "nums.tex", "references.bib", "sn-jnl.cls", "sn-basic.bst"]
TABLES = ["tab_setup.tex", "tab_main.tex", "tab_heldout.tex", "tab_fdr.tex",
          "tab_transfer.tex"]
FIGS = ["fig1_wording.pdf", "fig2_calib.pdf", "fig3_fdr.pdf"]


def main():
    missing = []
    for f in TEX:
        if not os.path.exists(os.path.join(PAPER, f)):
            missing.append(f)
    for f in TABLES:
        if not os.path.exists(os.path.join(PAPER, "tables", f)):
            missing.append("tables/" + f)
    for f in FIGS:
        if not os.path.exists(os.path.join(PAPER, "fig", f)):
            missing.append("fig/" + f)
    if missing:
        print("MISSING (run `promptcal macros` and `promptcal figures` first):")
        for m in missing:
            print("   ", m)
        if not os.environ.get("PCC_FORCE"):
            sys.exit(1)

    # a clean compile, twice, so the bbl and the cross-references are current
    tex = os.environ.get("PCC_PDFLATEX", "pdflatex")
    for cmd in ([tex, "-interaction=nonstopmode", "main.tex"],
                ["bibtex", "main"],
                [tex, "-interaction=nonstopmode", "main.tex"],
                [tex, "-interaction=nonstopmode", "main.tex"]):
        r = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)
        if cmd[0] == tex and r.returncode != 0:
            print("compile failed:", " ".join(cmd))
            print(r.stdout[-2000:])
            sys.exit(1)

    os.makedirs(OUT, exist_ok=True)
    zpath = os.path.join(OUT, STEM + ".zip")
    if os.path.exists(zpath):
        os.remove(zpath)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in TEX:
            z.write(os.path.join(PAPER, f), f)
        for f in TABLES:
            z.write(os.path.join(PAPER, "tables", f), "tables/" + f)
        for f in FIGS:
            z.write(os.path.join(PAPER, "fig", f), "fig/" + f)
        for f in ("main.bbl", "main.pdf"):
            p = os.path.join(PAPER, f)
            if os.path.exists(p):
                z.write(p, f)

    n = len(zipfile.ZipFile(zpath).namelist())
    mb = os.path.getsize(zpath) / 1e6
    print(f"{n} files, {mb:.2f} MB -> {zpath}")
    # a copy of the PDF next to the zip, easier to open than digging inside it
    shutil.copy(os.path.join(PAPER, "main.pdf"), os.path.join(OUT, STEM + ".pdf"))


if __name__ == "__main__":
    main()
