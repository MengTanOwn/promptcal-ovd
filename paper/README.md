# The manuscript

`main.tex` is the compiled paper; every number in it is a macro from `nums.tex`, and every
table is a generated file under `tables/`.  Both are written by

```bash
promptcal macros      # nums.tex + tables/tab_*.tex, from the JSON in results/
promptcal figures     # fig/fig*.pdf
```

The figures are committed as PDFs so the paper builds from a fresh clone without
matplotlib.  The class files are **not** committed.

## Building

The paper uses the Springer Nature LaTeX class `sn-jnl` with the `sn-basic` bibliography
style.  Download the *Springer Nature LaTeX template* from the journal's submission page
(or from Overleaf's template gallery, "Springer Nature"), and copy `sn-jnl.cls` and
`sn-basic.bst` into this directory.  Then

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

or `make paper` from the repository root, which also regenerates the macros and figures
first.

## Files

| file | content |
|---|---|
| `main.tex` | preamble, title, abstract, author block, `\input` of the sections |
| `sec1_intro.tex` | the three findings, the contributions |
| `sec2_related.tex` | open-vocabulary detection, what a detection score is, calibration, conformal selection, prompt sensitivity |
| `sec3_task.tex` | runs, the band, the metrics, the protocol |
| `sec4_method.tex` | PhC, Proposition 1, the descriptor, the four selection protocols |
| `sec5_experiments.tex` | setup, wording, calibration, selection, held-out wordings, transfer |
| `sec6_discussion.tex` | what a calibrator can and cannot repair, limitations, recommendations |
| `nums.tex` | generated: one macro per number quoted in the text |
| `tables/` | generated: `tab_setup`, `tab_main`, `tab_heldout`, `tab_fdr`, `tab_transfer` |
| `fig/` | generated: `fig1_wording`, `fig2_calib`, `fig3_fdr` |

Do not edit `nums.tex`, `tables/` or `fig/` by hand: they are outputs, and the next
`promptcal macros` overwrites them.
