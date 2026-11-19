"""Build the submission package for the Journal of Real-Time Image Processing.

The journal asks for the LaTeX source in a **single directory** -- no ``fig/`` or ``tables/``
sub-directories, because its submission system unpacks the archive and compiles it flat.  This
script rebuilds the manuscript, copies the source into such a directory with the include paths
flattened, compiles that copy to prove it stands on its own, and writes the archive.

Usage::

    python tools/package_jrtip.py              # rebuild and package
    python tools/package_jrtip.py --no-build   # package the PDF as it is

It needs ``pdflatex`` and ``bibtex`` on PATH, and it only runs in the authors' working copy,
where ``paper/`` exists: the manuscript source is not part of the released repository.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(HERE, "paper")
OUT = os.path.join(HERE, "submission_jrtip")
FLAT = os.path.join(OUT, "latex")

TEX_FILES = ["main.tex", "sec1_intro.tex", "sec2_related.tex", "sec3_task.tex",
             "sec4_method.tex", "sec5_experiments.tex", "sec6_discussion.tex",
             "nums.tex", "references.bib", "sn-jnl.cls", "sn-basic.bst"]


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)


def latex(build_dir):
    """Three passes plus bibtex; returns (ok, pages, notes)."""
    for _ in range(1):
        run("pdflatex -interaction=nonstopmode main.tex", build_dir)
    run("bibtex main", build_dir)
    run("pdflatex -interaction=nonstopmode main.tex", build_dir)
    res = run("pdflatex -interaction=nonstopmode main.tex", build_dir)
    log = res.stdout + res.stderr
    errors = log.count("\n! ")
    overfull = log.count("Overfull")
    undefined = len(re.findall(r"undefined", log, re.I))
    pages = 0
    m = re.findall(r"\((\d+) pages", log)
    if m:
        pages = int(m[-1])
    return errors == 0, pages, dict(errors=errors, overfull=overfull, undefined=undefined)


def included_figures():
    """The figures the manuscript actually includes, in source order."""
    names = []
    for path in sorted(glob.glob(os.path.join(PAPER, "sec*.tex"))):
        text = open(path, encoding="utf-8").read()
        for ref in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text):
            stem = os.path.basename(ref)
            if stem not in names:
                names.append(stem)
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-build", action="store_true",
                    help="do not recompile the manuscript first")
    args = ap.parse_args(argv)

    if not os.path.isdir(PAPER):
        sys.exit("paper/ not found: this script only runs in the authors' working copy")
    if not args.no_build:
        ok, pages, notes = latex(PAPER)
        print(f"manuscript: {pages} pages, {notes}")
        if not ok:
            sys.exit("manuscript does not compile")

    if os.path.isdir(FLAT):
        shutil.rmtree(FLAT)
    os.makedirs(FLAT)

    for name in TEX_FILES:
        shutil.copy(os.path.join(PAPER, name), os.path.join(FLAT, name))
    for path in glob.glob(os.path.join(PAPER, "tables", "*.tex")):
        shutil.copy(path, os.path.join(FLAT, os.path.basename(path)))
    figures = included_figures()
    for name in figures:
        shutil.copy(os.path.join(PAPER, "fig", name), os.path.join(FLAT, name))

    for name in os.listdir(FLAT):
        if name.endswith(".tex"):
            path = os.path.join(FLAT, name)
            text = open(path, encoding="utf-8").read()
            text = text.replace(r"\input{tables/", r"\input{")
            text = text.replace(r"\IfFileExists{tables/", r"\IfFileExists{")
            text = text.replace("{fig/", "{")
            open(path, "w", encoding="utf-8").write(text)

    leftovers = [n for n in os.listdir(FLAT)
                 if re.search(r"\b(tables|fig)/", open(os.path.join(FLAT, n), encoding="utf-8",
                                                       errors="ignore").read())
                 and n.endswith(".tex")]
    if leftovers:
        sys.exit(f"sub-directory paths survive in: {leftovers}")

    ok, pages, notes = latex(FLAT)
    print(f"flat copy:  {pages} pages, {notes}")
    if not ok:
        sys.exit("the flat copy does not compile")

    pdf_dst = os.path.join(OUT, "promptcal_ovd_jrtip.pdf")
    shutil.copy(os.path.join(FLAT, "main.pdf"), pdf_dst)

    for pattern in ("*.aux", "*.log", "*.out", "*.blg"):
        for path in glob.glob(os.path.join(FLAT, pattern)):
            os.remove(path)

    zip_path = os.path.join(OUT, "jrtip_latex_source.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(FLAT)):
            z.write(os.path.join(FLAT, name), name)

    size = os.path.getsize(zip_path) / 1e6
    print(f"archive: {zip_path} ({len(os.listdir(FLAT))} files, {size:.2f} MB)")
    print(f"pdf:     {pdf_dst}")
    if pages > 12:
        print(f"[warn] {pages} pages: the journal returns anything over 12 without review")


if __name__ == "__main__":
    main()
