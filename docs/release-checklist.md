# Release checklist

Checks for a public release of this repository.  They were carried out for v1.0.0, the
release that accompanies the submission, and must be repeated for every later version.

1. **Keep the owner in sync.**  The URL `https://github.com/MengTanOwn/promptcal-ovd`
   appears in `README.md`, `CITATION.cff`, `pyproject.toml` (`[project.urls]`) and in the
   code-availability statement of the manuscript (`paper/main.tex`).  If the repository
   moves, update all of them, then check that `pip install -e .` still resolves the URLs.
2. **Decide about the manuscript.**  `paper/` is included; the Springer class files are
   not, by design (see `paper/README.md`).  If the journal's policy forbids posting the
   source of a manuscript under review, drop `paper/` and keep `results/` — the macros and
   tables can be rebuilt from it.
3. **Check the licensing of the third-party material you may add.**  The code is MIT; the
   detector weights (Apache-2.0) and the datasets (CC BY 4.0) are not redistributed here and
   should not be committed.
4. **Tag a version that matches the submission.**  `git tag -a v1.0.0 -m "paper submission"`
   and, if the DOI is minted through Zenodo, upload the archived release and add the DOI to
   `README.md` and `CITATION.cff`.
5. **Do not commit** `features/`, `data/`, `submission/` or `results/log_*.txt`; they are
   excluded by `.gitignore` and are either large or local to a machine.
6. **Run the checks the badge will run**: `pytest` and `ruff check src tests` (both are in
   `.github/workflows/tests.yml`).

After the paper is accepted, add the volume, pages and DOI to the `preferred-citation`
block of `CITATION.cff` and drop the "under review" wording from `README.md`.
