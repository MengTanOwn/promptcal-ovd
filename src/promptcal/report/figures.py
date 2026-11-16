"""Figures for the paper, drawn at print width.

A figure drawn 12 in wide and included with ``width=\\textwidth`` is scaled to 43%, which
turns a 9 pt tick label into a 4 pt one.  Everything here is drawn at the width it will be
printed at (5.2 in for this journal), so the point sizes in this file are the point sizes
on the page.  Fonts are embedded as Type 42 so the PDF is accepted by the typesetter.

Each figure reads one of the result JSONs written by :mod:`promptcal.experiments`, so a
re-run of the experiments redraws the figures with the new numbers.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import config
from ..prompts import VARIANT_NAMES

FIG = str(config.paper_dir() / "fig")
RES = str(config.results_dir())
VAR = VARIANT_NAMES
W = 5.2                      # print width in inches
COL = {"identity": "#3d3d3d", "temperature": "#c0392b", "vector": "#e08a1e",
       "isotonic": "#8e44ad", "variant": "#2f6fb5", "phc": "#12876f",
       "phc_band": "#26b39a"}
LBL = {"identity": "uncalibrated", "temperature": "temperature", "vector": "vector",
       "isotonic": "isotonic", "variant": "per-wording T", "phc": "PhC",
       "phc_band": "PhC (band-weighted)"}
NICE = ["bare", "article", "photo", "image", "plural"]


def _style():
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _load(stem):
    """Same fallback as report.py: full run first, then the reduced runs."""
    cands = [stem, f"_partial_{stem}", f"_partial_{stem.split('_')[0]}",
             f"_smoke_{stem}"]
    for c in cands:
        p = os.path.join(RES, c + ".json")
        if os.path.exists(p):
            if c != stem:
                print(f"  [warn] {stem}: reading {c}.json")
            return json.load(open(p))
    raise FileNotFoundError(stem)


def fig_wording(d):
    fig, ax = plt.subplots(1, 3, figsize=(W, 1.72))
    x = np.arange(5)

    ms = d["spread"]["mean_top_score"]
    ax[0].bar(x, ms, color="#3d3d3d", width=0.66)
    for i, y in enumerate(ms):
        ax[0].text(i, y + 0.015, f"{y:.2f}", ha="center", va="bottom", fontsize=5.6)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(NICE, rotation=25, ha="right")
    ax[0].set_ylabel("mean score, top box")
    ax[0].set_ylim(0, 0.82)
    ax[0].set_title("(a) same object, five wordings", loc="left")

    sp = d["spread"]
    h = np.asarray(sp["hist_counts"], float)
    e = np.asarray(sp["hist_edges"], float)
    ax[1].bar(e[:-1], h, width=np.diff(e), align="edge", color="#3d3d3d",
              alpha=0.85, linewidth=0)
    ax[1].axvline(sp["mean_spread"], color="#c0392b", lw=0.9)
    ax[1].set_xlabel("range of the five scores")
    ax[1].set_ylabel("objects")
    ax[1].set_title(f"(b) mean range {sp['mean_spread']:.2f}", loc="left")

    fr = [d["variants"][n]["frac_gt_0_5"] * d["variants"][n]["per_image"] for n in VAR]
    ax[2].bar(x, fr, color="#3d3d3d", width=0.66)
    for i, y in enumerate(fr):
        ax[2].text(i, y + 0.06, f"{y:.1f}", ha="center", va="bottom", fontsize=5.6)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(NICE, rotation=25, ha="right")
    ax[2].set_ylabel("boxes kept per image")
    ax[2].set_ylim(0, max(fr) * 1.16)
    ax[2].set_title(r"(c) queue at $s\geq0.5$", loc="left")

    for a in ax:
        _clean(a)
    fig.tight_layout(pad=0.25, w_pad=0.6)
    _save(fig, "fig1_wording")


def fig_calib(d):
    meth = ["identity", "temperature", "vector", "isotonic", "variant", "phc", "phc_band"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.95))
    x = np.arange(len(VAR))
    w = 0.115
    off = (len(meth) - 1) / 2
    for j, key in enumerate(("band_ece", "band_brier")):
        for k, m in enumerate(meth):
            v = [d["per_variant"][m][n][key] for n in VAR]
            ax[j].bar(x + (k - off) * w, v, width=w, color=COL[m], edgecolor="none",
                      label=LBL[m])
        ax[j].set_xticks(x)
        ax[j].set_xticklabels(NICE, rotation=22, ha="right")
        _clean(ax[j])
    ax[0].set_ylabel("band ECE")
    ax[0].set_title("(a) error in the operating band", loc="left")
    ax[1].set_ylabel("band Brier")
    ax[1].set_title("(b) proper score in the band", loc="left")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, handlelength=1.1,
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(pad=0.25, w_pad=0.8, rect=(0, 0.16, 1, 1))
    _save(fig, "fig2_calib")


def fig_fdr(d, alpha=0.30):
    """One shipped threshold versus one threshold per wording."""
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.85))
    key = f"{alpha:.2f}"
    meth = ("identity", "vector", "phc")
    n = len(meth)
    w = 0.78 / n
    x = np.arange(len(VAR))
    for k, m in enumerate(meth):
        v = [d["pooled_per_run"][m][key][nm]["fdr"] for nm in VAR]
        ax[0].bar(x + (k - (n - 1) / 2) * w, v, width=w, color=COL[m], edgecolor="none",
                  label=LBL[m])
    ax[0].axhline(alpha, color="#c0392b", lw=0.9, ls="--")
    ax[0].text(len(VAR) - 0.62, alpha + 0.012, rf"target {alpha:.2f}", ha="right",
               fontsize=6, color="#c0392b")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(NICE, rotation=22, ha="right")
    ax[0].set_ylabel("attained FDR")
    ax[0].set_ylim(0, 0.5)
    ax[0].set_title("(a) one threshold, five wordings", loc="left")
    ax[0].legend(frameon=False, loc="upper left", handlelength=1.0,
                 handletextpad=0.4, labelspacing=0.25)

    al = [float(a) for a in d["alphas"]]
    ax[1].plot([0, al[-1] * 1.08], [0, al[-1] * 1.08], color="#999999", lw=0.8, ls=":",
               label="target")
    for m, mk in (("identity", "o"), ("phc", "^")):
        y = [max(d["pooled_per_run"][m][f"{a:.2f}"][nm]["fdr"] for nm in VAR) for a in al]
        ax[1].plot(al, y, marker=mk, color=COL[m], ms=3.0,
                   label=rf"{LBL[m]}, one $\lambda$")
        y2 = [max(d["per_wording"][m][nm][f"{a:.2f}"]["fdr"] for nm in VAR) for a in al]
        ax[1].plot(al, y2, marker=mk, color=COL[m], ms=2.4, ls=":", lw=0.9,
                   label=rf"{LBL[m]}, per-wording $\lambda$")
    ax[1].set_xlabel(r"target $\alpha$")
    ax[1].set_ylabel("FDR, worst wording")
    ax[1].set_title("(b) what the guarantee must promise", loc="left")
    ax[1].legend(frameon=False, loc="upper left", handlelength=1.2,
                 handletextpad=0.4, labelspacing=0.2)
    for a in ax:
        _clean(a)
    fig.tight_layout(pad=0.25, w_pad=0.8)
    _save(fig, "fig3_fdr")


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), dpi=400)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="promptcal figures",
        description="Draw the three figures of the paper from the result JSONs.")
    ap.add_argument("--results-dir", default=None,
                    help="where the JSONs are (default: $PROMPTCAL_RESULTS)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the PDFs (default: $PROMPTCAL_PAPER/fig)")
    a = ap.parse_args(argv)
    global FIG, RES
    FIG = a.out_dir or FIG
    RES = a.results_dir or RES
    os.makedirs(FIG, exist_ok=True)
    _style()
    fig_wording(_load("p1_describe"))
    fig_calib(_load("p2_calibration"))
    fig_fdr(_load("p3_conformal"))
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
