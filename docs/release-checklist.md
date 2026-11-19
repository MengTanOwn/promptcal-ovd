# Release checklist

Checks for a public release of this repository.  They were carried out for v1.0.0, the
release that accompanies the submission, and must be repeated for every later version.

1. **Keep the owner in sync.**  The URL `https://github.com/MengTanOwn/promptcal-ovd`
   appears in `README.md`, `CITATION.cff`, `pyproject.toml` (`[project.urls]`) and in the
   code-availability statement of the manuscript.  The last one lives in the manuscript
   source, which is kept outside this repository, so it has to be edited there.  If the
   repository moves, update all four, then check that `pip install -e .` still resolves the
   URLs.
2. **Keep the manuscript out.**  `paper/` is not committed: it is ignored wholesale by
   `.gitignore`, and this release ships `results/` only.  What ships is what regenerates the
   manuscript, not the manuscript.  If you decide to publish the source, check the venue's
   policy on manuscripts under review first, and remember that the Springer `sn-jnl` class
   files must still be excluded.
3. **Check the licensing of the third-party material you may add.**  The code is MIT; the
   detector weights (Apache-2.0) and the datasets (CC BY 4.0) are not redistributed here and
   should not be committed.
4. **Tag a version that matches the submission.**  `git tag -a v1.0.0 -m "paper submission"`
   and, if the DOI is minted through Zenodo, upload the archived release and add the DOI to
   `README.md` and `CITATION.cff`.
5. **Do not commit** `paper/`, `features/`, `data/`, `submission/` or `results/log_*.txt`;
   they are excluded by `.gitignore` and are either large, local to a machine, or reserved
   for the submission.
6. **Run the checks the badge will run**: `pytest` and `ruff check src tests` (both are in
   `.github/workflows/tests.yml`).

After the paper is accepted, add the volume, pages and DOI to the `preferred-citation`
block of `CITATION.cff` and drop the "under review" wording from `README.md`.
