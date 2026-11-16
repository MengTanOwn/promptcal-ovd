"""Turn the result JSONs into the paper's macros and tables.

Everything the manuscript quotes numerically is generated here, so a re-run of the
experiments updates the paper instead of contradicting it.  The macros land in
``nums.tex`` and the tables in ``tables/tab_*.tex``; the manuscript only ever refers to
``\\NAME``, never to a literal number.

A missing result file is not an error: the corresponding macros are written as ``--``,
which is how the paper can be compiled while a long experiment is still running.
"""
import argparse
import json
import os

import numpy as np

from .. import config
from ..prompts import VARIANT_NAMES

RES = str(config.results_dir())
PAPER = str(config.paper_dir())
VAR = VARIANT_NAMES
NICE = {"bare": "bare", "article": "article", "photo": "photo", "image": "image",
        "plural": "plural"}
METH = ("identity", "temperature", "vector", "isotonic", "variant", "phc",
        "phc_band", "phc_cat")
LAB = {"identity": "uncalibrated", "temperature": "temperature scaling",
       "vector": "vector scaling", "isotonic": "isotonic regression",
       "variant": "per-wording temperature", "phc": "PhC (phrase, linear)",
       "phc_band": "PhC (phrase, linear, band-weighted)",
       "phc_cat": "PhC+category"}


# Results are written as ``<stem>.json`` by the full runs and as
# ``_partial_<stem>.json`` / ``_smoke_<stem>.json`` by the reduced runs used while
# the extraction is still going. ``PCC_TAG`` pins the prefix explicitly so a smoke
# run never silently overwrites the numbers of a full one.
def load(name):
    stem = name.replace(".json", "")
    short = stem.split("_")[0]          # p1_describe -> p1
    cands = []
    tag = os.environ.get("PCC_TAG", "")
    if tag:
        cands.append(f"{tag}{stem}")
    cands += [stem, f"_partial_{stem}", f"_partial_{short}", f"_smoke_{stem}"]
    for c in cands:
        p = os.path.join(RES, c + ".json")
        if os.path.exists(p):
            if c != stem:
                print(f"  [warn] {stem}: reading {c}.json (reduced run)")
            return json.load(open(p, encoding="utf-8"))
    return None


def _write(name, lines):
    """Write one generated file next to the manuscript."""
    path = os.path.join(PAPER, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def f(x, n=4):
    return f"{x:.{n}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="promptcal macros",
        description="Write nums.tex and the tables from the result JSONs.")
    ap.add_argument("--results-dir", default=None,
                    help="where the JSONs are (default: $PROMPTCAL_RESULTS)")
    ap.add_argument("--paper-dir", default=None,
                    help="where nums.tex and tables/ live (default: $PROMPTCAL_PAPER)")
    a = ap.parse_args(argv)
    global RES, PAPER
    RES = a.results_dir or RES
    PAPER = a.paper_dir or PAPER

    p1 = load("p1_describe.json")
    p2 = load("p2_calibration.json")
    p3 = load("p3_conformal.json")
    p4 = load("p4_transfer.json")
    p2b = load("p2b_ci.json")
    os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)

    n = {}
    if p1:
        sp = p1["spread"]
        n["SPREADPTS"] = f(sp["mean_spread"], 3)
        n["SPREADFRAC"] = f"{100 * sp['frac_spread_gt_0_1']:.1f}\\%"
        n["SPREADMED"] = f(sp["median_spread"], 3)
        n["AGREE"] = f"{100 * sp['label_agreement']:.1f}\\%"
        n["NGROUPS"] = f"{sp['n_groups']:,}".replace(",", "{,}")
        # boxes that survive a fixed cut of 0.5, per image: the queue a user sees
        fr = [p1["variants"][v]["frac_gt_0_5"] * p1["variants"][v]["per_image"]
              for v in VAR]
        n["BIGRATIO"] = f"{max(fr) / max(min(fr), 1e-9):.0f}"
        n["NDETS"] = f"{p1['n']:,}".replace(",", "{,}")
        n["NIMAGES"] = f"{p1['n_images']:,}".replace(",", "{,}")
        n["DETPERIMG"] = f"{p1['n'] / p1['n_images']:.0f}"
        n["BANDFRAC"] = f"{100 * np.mean([p1['variants'][v]['frac_gt_0_3'] for v in VAR]):.1f}\\%"
        n["BANDPOS"] = f"{100 * np.mean([p1['variants'][v]['pos_rate'] for v in VAR]):.1f}\\%"
        n["APMIN"] = f(min(p1["variants"][v]["ap"] for v in VAR), 3)
        n["APMAX"] = f(max(p1["variants"][v]["ap"] for v in VAR), 3)
        n["APGAP"] = f(max(p1["variants"][v]["ap"] for v in VAR)
                       - min(p1["variants"][v]["ap"] for v in VAR), 3)
        for i, tag in enumerate(("BARE", "ART", "PHOTO", "IMG", "PLUR")):
            n[f"SC{tag}"] = f(sp["mean_top_score"][i], 3)
        if sp.get("mean_range_in_pos") is not None:
            n["SPREADPOS"] = f(sp["mean_range_in_pos"], 3)

    if p2:
        m = p2["mixed"]
        if "n_phrases" in p2:
            n["NPHRS"] = f"{p2['n_phrases']:,}".replace(",", "{,}")
        if p2.get("fit_rows"):
            n["FITROWS"] = f"{p2['fit_rows']:,}".replace(",", "{,}")
        n["BANDID"] = f(m["identity"]["band_ece"], 3)
        n["BANDBSID"] = f(m["identity"]["band_brier"], 3)
        # the headline method carries no per-category table, so the number quoted in the
        # abstract is the one the transfer experiment can reproduce
        best = ("phc_band" if "phc_band" in m
                else ("phc_cat" if "phc_cat" in m else "phc"))
        n["BANDPHC"] = f(m[best]["band_ece"], 3)
        n["BANDBS"] = f(m[best]["band_brier"], 3)
        n["BANDPHCNAME"] = LAB[best]
        if "phc_cat" in m:
            n["BANDCAT"] = f(m["phc_cat"]["band_ece"], 3)
            # four decimals: the category table changes the third one, and the claim is
            # precisely that this difference is not measurable
            n["BANDCATF"] = f(m["phc_cat"]["band_ece"], 4)
            n["BANDPHCF"] = f(m[best]["band_ece"], 4)
        n["BANDTEMP"] = f(m["temperature"]["band_ece"], 3)
        n["BANDVEC"] = f(m["vector"]["band_ece"], 3)
        n["BANDISO"] = f(m["isotonic"]["band_ece"], 3)
        n["APISO"] = f(m["isotonic"]["ap"], 3)
        n["APID"] = f(m["identity"]["ap"], 3)
        n["BANDIDN"] = f(m["identity"]["band_ece"], 3)
        n["FACTOR"] = f"{m['identity']['band_ece'] / max(m[best]['band_ece'], 1e-9):.1f}"
        h = p2.get("heldout", {})
        if h:
            n["HELDPHC"] = f(h["phc_cat"]["image"]["band_ece"], 3) if "phc_cat" in h \
                else f(h["phc"]["image"]["band_ece"], 3)
            n["HELDID"] = f(h["identity"]["image"]["band_ece"], 3)
            n["HELDPLURAL"] = f((h["phc_cat"] if "phc_cat" in h else h["phc"])["plural"]["band_ece"], 3)
            n["HELDPLURALID"] = f(h["identity"]["plural"]["band_ece"], 3)
            n["HELDVEC"] = f(h["vector"]["plural"]["band_ece"], 3)
        pv = p2.get("per_variant", {})
        if "identity" in pv and "bare" in pv["identity"]:
            n["COCBARE"] = f(pv["identity"]["bare"]["band_ece"], 3)
            n["HELDVECIMG"] = f(h["vector"]["image"]["band_ece"], 3)
            n["HELDBAND"] = f(h["phc_band"]["image"]["band_ece"], 3) if "phc_band" in h else "--"
            n["HELDBANDPL"] = f(h["phc_band"]["plural"]["band_ece"], 3) if "phc_band" in h else "--"
            n["HELDISO"] = f(h["isotonic"]["plural"]["band_ece"], 3)
            n["HELDISOAP"] = f(h["isotonic"]["plural"]["ap"], 3)

    if p2b:
        # image-level bootstrap of the band error: the rows are strongly correlated within
        # an image, so the interval has to be drawn over images, not over detections
        d = p2b["methods"]

        def band(k):
            return f"{d[k]['lo_basic']:.3f}--{d[k]['hi_basic']:.3f}"

        n["BANDIDCI"] = band("identity")
        n["BANDPHCCI"] = band("phc_band")
        n["BANDVECCI"] = band("vector")
        n["BOOTN"] = str(p2b["boot"])
        if "diff_identity_phc_band" in p2b:
            dd = p2b["diff_identity_phc_band"]
            n["DROP"] = f(dd["mean"], 3)
            n["DROPCD"] = f"{dd['lo']:.3f}--{dd['hi']:.3f}"

    if p3:
        alpha = "0.30"
        A = 0.30
        pp = p3["pooled_per_run"]

        def worst_of(proto, key):
            """FDR of the worst of the five wordings, and boxes kept over all five."""
            d = p3[proto][key]
            if proto == "per_wording":
                return (max(d[nm][alpha]["fdr"] for nm in VAR),
                        sum(d[nm][alpha]["kept"] for nm in VAR))
            rr = d[alpha]["per_run"] if proto == "worst" else pp[key][alpha]
            return (max(rr[nm]["fdr"] for nm in VAR),
                    sum(rr[nm]["kept"] for nm in VAR))

        ker = {}
        protos = [("POOL", "pooled"), ("PW", "per_wording")]
        if "worst" in p3:
            protos.insert(1, ("WC", "worst"))
        for tag, key in (("ID", "identity"), ("PHC", "phc"),
                         ("VEC", "vector"), ("TEMP", "temperature")):
            for ptag, proto in protos:
                fdr_, kept_ = worst_of(proto, key)
                n[f"FDR{ptag}{tag}"] = f"{fdr_:.2f}"
                n[f"KEPT{ptag}{tag}"] = f"{kept_:,}".replace(",", "{,}")
                ker[ptag + tag] = kept_

        n["ALPHAMAIN"] = f"{A:.2f}"
        n["ALPHAPCT"] = f"{100 * A:.0f}\\%"
        n["FDRWORST"] = f"{100 * worst_of('pooled', 'identity')[0]:.0f}\\%"
        n["FDROVERSHOOT"] = f"{worst_of('pooled', 'identity')[0] / A:.1f}"
        if "worst" in p3:
            n["WCOVERSHOOT"] = f"{worst_of('worst', 'identity')[0] / A:.2f}"
            for tag in ("ID", "PHC"):
                a_, b_ = ker["POOL" + tag], ker["WC" + tag]
                n[f"WCCOST{tag}"] = f"{100 * (a_ - b_) / max(a_, 1):.0f}\\%"
            n["WCGAIN"] = f"{100 * (ker['WCPHC'] - ker['WCID']) / max(ker['WCID'], 1):.0f}\\%"
            n["WCTOTALID"] = f"{ker['WCID']:,}".replace(",", "{,}")
            n["WCTOTALPHC"] = f"{ker['WCPHC']:,}".replace(",", "{,}")
            # the gain is larger when the target is strict, which is when it matters
            a5 = "0.05"
            k_id = sum(p3["worst"]["identity"][a5]["per_run"][nm]["kept"] for nm in VAR)
            k_ph = sum(p3["worst"]["phc"][a5]["per_run"][nm]["kept"] for nm in VAR)
            n["WCGAINFIVE"] = f"{100 * (k_ph - k_id) / max(k_id, 1):.0f}\\%"
            n["WCKEPTFIVEID"] = f"{k_id:,}".replace(",", "{,}")
            n["WCKEPTFIVEPHC"] = f"{k_ph:,}".replace(",", "{,}")

    if p4:
        d = p4.get("methods", p4)
        n["NLVISNDET"] = f"{p4['n']:,}".replace(",", "{,}")
        n["NLVISIMG"] = f"{p4['n_images']:,}".replace(",", "{,}")
        n["NLVISCAT"] = f"{p4['n_cats']:,}".replace(",", "{,}")
        n["TRBANDID"] = f(d["identity"]["band_ece"], 3)
        bp = "phc_band" if "phc_band" in d else "phc"
        n["TRBANDPHC"] = f(d[bp]["band_ece"], 3)
        n["TRBANDVEC"] = f(d["vector"]["band_ece"], 3)
        n["TRBANDTEMP"] = f(d["temperature"]["band_ece"], 3)
        n["TRBANDISO"] = f(d["isotonic"]["band_ece"], 3)
        n["TRAPISO"] = f(d["isotonic"]["ap"], 3)
        n["TRAPID"] = f(d["identity"]["ap"], 3)
        n["TRAPPHC"] = f(d[bp]["ap"], 3)
        n["VECSEEN"] = str(p4.get("vector_fitted", p4.get("vector_seen_cats", 0)))
        n["TRFACTOR"] = f"{d['identity']['band_ece'] / max(d[bp]['band_ece'], 1e-9):.1f}"
        # the wording ordering on the new vocabulary: if it repeats, the effect is a
        # property of the detector rather than of the COCO label style
        pw = d["identity"].get("per_wording", {})
        if pw:
            lo = min(pw, key=lambda k: pw[k]["ap"])
            hi = max(pw, key=lambda k: pw[k]["ap"])
            n["TRAPLO"] = f(pw[lo]["ap"], 3)
            n["TRAPLOID"] = lo
            n["TRAPHI"] = f(pw[hi]["ap"], 3)
            n["TRAPHIID"] = hi
            n["TRAPGAP"] = f(pw[hi]["ap"] - pw[lo]["ap"], 3)
            # the transfer is uneven across wordings, and the two that matter are the bare
            # name (where it is weakest) and the article (where it is strongest)
            bw = d[bp].get("per_wording", {})
            if "bare" in pw and "article" in pw and bw:
                n["TRBAREID"] = f(pw["bare"]["band_ece"], 3)
                n["TRBAREPHC"] = f(bw["bare"]["band_ece"], 3)
                n["TRARTID"] = f(pw["article"]["band_ece"], 3)
                n["TRARTPHC"] = f(bw["article"]["band_ece"], 3)

    # The transfer macros only exist once p4 has run; keep the manuscript compilable in
    # the meantime rather than failing on an undefined control sequence.
    for k in ("NLVISNDET", "NLVISIMG", "NLVISCAT", "TRBANDID", "TRBANDPHC",
              "TRBANDVEC", "TRBANDTEMP", "TRBANDISO", "TRAPID", "TRAPISO", "TRAPPHC",
              "VECSEEN", "TRFACTOR",
              "WCOVERSHOOT", "WCCOSTID", "WCCOSTPHC", "WCGAIN",
              "WCTOTALID", "WCTOTALPHC", "NPHRS", "FITROWS",
              "BANDIDCI", "BANDPHCCI", "BANDVECCI", "BOOTN", "DROP", "DROPCD",
              "TRAPLO", "TRAPLOID", "TRAPHI", "TRAPHIID", "TRAPGAP",
              "TRBAREID", "TRBAREPHC", "TRARTID", "TRARTPHC"):
        n.setdefault(k, "--")

    with open(os.path.join(PAPER, "nums.tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by `promptcal macros` -- do not edit\n")
        for k, v in n.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(n)} macros -> paper/nums.tex")

    if p1 and p2:
        tab_setup(p1, p2)
        tab_main(p2)
        tab_heldout(p2)
    if p3:
        tab_fdr(p3)
    if p4:
        tab_transfer(p4)
    print("tables written")


def _row(name, d, cols):
    return (f"{LAB[name]:<34} & " + " & ".join(cols) + r" \\")


def tab_setup(p1, p2):
    v = p1["variants"]
    lines = [r"\begin{table}[t]",
             r"\caption{The five wordings on the same images with the same weights. A row "
             r"is one phrasing; nothing but the text query changes between rows. ``Band'' "
             r"is $p\in[0.3,0.9)$ of the uncalibrated score.}",
             r"\label{tab:setup}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
             r"\toprule",
             r"Wording & detections & per image & positive rate & AP & share in band & "
             r"ECE \\",
             r"\midrule"]
    for w in VAR:
        d = v[w]
        lines.append(f"{NICE[w]:<18} & {d['n']:,} & {d['per_image']:.0f} & "
                     f"{100*d['pos_rate']:.1f}\\% & {d['ap']:.4f} & "
                     f"{100*d['frac_gt_0_3']:.1f}\\% & "
                     f"{d['ece']:.4f} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    _write("tables/tab_setup.tex", lines)


def tab_main(p2):
    m = p2["mixed"]
    lines = [r"\begin{table}[t]",
             r"\caption{Calibration on the COCO half the calibrator never saw. Metrics are "
             r"computed inside each wording run and averaged over the five runs. ``Band'' "
             r"restricts to $p\in[0.3,0.9)$ of the uncalibrated score of that run. Lower is "
             r"better except AP.}",
             r"\label{tab:main}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}",
             r"\toprule",
             r"Method & band ECE & band Brier & ECE & Brier & AP \\",
             r"\midrule"]
    best_be = min(m[k]["band_ece"] for k in METH if k in m)
    best_bb = min(m[k]["band_brier"] for k in METH if k in m)
    for k in METH:
        if k not in m:
            continue
        c = [f(m[k]["band_ece"], 4), f(m[k]["band_brier"], 4), f(m[k]["ece"], 4),
             f(m[k]["brier"], 4), f(m[k]["ap"], 4)]
        if abs(m[k]["band_ece"] - best_be) < 1e-9:
            c[0] = r"\textbf{" + c[0] + "}"
        if abs(m[k]["band_brier"] - best_bb) < 1e-9:
            c[1] = r"\textbf{" + c[1] + "}"
        lines.append(_row(k, m[k], c))
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    _write("tables/tab_main.tex", lines)


def tab_heldout(p2):
    h = p2.get("heldout")
    if not h:
        return
    lines = [r"\begin{table}[t]",
             r"\caption{Fitted on three wordings (bare, article, photo) and deployed on the "
             r"two it never saw. The uncalibrated row is the reference: a calibrator is only "
             r"useful here if it beats doing nothing on an unseen wording.}",
             r"\label{tab:heldout}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
             r"\toprule",
             r"Method & \multicolumn{2}{c}{``an image of a''} & \multicolumn{2}{c}{plural} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r" & band ECE & band Brier & band ECE & band Brier \\",
             r"\midrule"]
    for k in METH:
        if k not in h:
            continue
        c = [f(h[k]["image"]["band_ece"], 4), f(h[k]["image"]["band_brier"], 4),
             f(h[k]["plural"]["band_ece"], 4), f(h[k]["plural"]["band_brier"], 4)]
        lines.append(_row(k, h[k], c))
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    _write("tables/tab_heldout.tex", lines)


def _worst_over_runs(p3, method, proto, alpha):
    """The largest attained FDR over the five wordings, and the total kept.

    ``proto`` is one of the three keys of the result file: ``pooled`` reads the per-run
    columns of one pooled threshold, ``worst`` the same columns of the max-over-strata
    threshold, and ``per_wording`` five thresholds of their own.
    """
    d = p3[proto][method]
    if proto == "per_wording":
        return (max(d[nm][alpha]["fdr"] for nm in VAR),
                sum(d[nm][alpha]["kept"] for nm in VAR))
    rr = d[alpha]["per_run"] if proto == "worst" else p3["pooled_per_run"][method][alpha]
    return max(rr[nm]["fdr"] for nm in VAR), sum(rr[nm]["kept"] for nm in VAR)


def tab_fdr(p3):
    alpha = "0.30"
    lines = [r"\begin{table}[t]",
             r"\caption{Three ways to ship a threshold at target $\alpha=" + alpha + r"$, "
             r"fitted on the calibration half and measured on the other. ``Pooled'' is one "
             r"$\hat\lambda$ from the calibration rows of all five wordings, ``worst-case'' "
             r"is the most conservative of the five per-wording $\hat\lambda$, and "
             r"``per-wording'' selects inside each wording. A cell pair is the false "
             r"discovery rate of the \emph{worst} of the five wordings and the number of "
             r"boxes kept across all five; the worst wording is the one a guarantee has to "
             r"cover.}",
             r"\label{tab:fdr}"]
    pr = [p for p in ("pooled", "worst", "per_wording") if p in p3]
    head_name = {"pooled": r"pooled $\hat\lambda$",
                 "worst": r"worst-case $\hat\lambda$",
                 "per_wording": r"per-wording $\hat\lambda$"}
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "rr" * len(pr) + r"@{}}")
    lines.append(r"\toprule")
    hdr = " & " + " & ".join(r"\multicolumn{2}{c}{" + head_name[p] + "}" for p in pr) + r" \\"
    lines.append(hdr)
    cm = []
    for i in range(len(pr)):
        a, b = 2 + 2 * i, 3 + 2 * i
        cm.append(f"\\cmidrule(l{'r' if i < len(pr) - 1 else ''}){{{a}-{b}}}")
    lines.append("".join(cm))
    lines.append("Score & " + " & ".join(["max FDR & kept"] * len(pr)) + r" \\")
    lines.append(r"\midrule")
    rows = []
    pr = [p for p in ("pooled", "worst", "per_wording") if p in p3]
    for k in ("identity", "temperature", "vector", "phc"):
        if k not in p3["pooled"]:
            continue
        rows.append((k, [_worst_over_runs(p3, k, proto, alpha) for proto in pr]))
    best = {i: min(r[1][i][0] for r in rows) for i in range(len(pr))}
    for k, cells in rows:
        out = []
        for i, (mf, kp) in enumerate(cells):
            s = f"{mf:.3f}"
            if abs(mf - best[i]) < 5e-4:
                s = r"\textbf{" + s + "}"
            out.append(s + f" & {kp:,}".replace(",", "{,}"))
        lines.append(f"{LAB[k]} & " + " & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    _write("tables/tab_fdr.tex", lines)


def tab_transfer(p4):
    d = p4.get("methods", p4)
    lines = [r"\begin{table}[t]",
             r"\caption{Fitted on COCO and deployed on the LVIS minival split without "
             r"refitting anything, including the projection used by the phrase "
             r"descriptor. None of the 1{,}203 LVIS categories was seen during fitting, "
             r"so a calibrator with a table indexed by category has nothing to look up.}",
             r"\label{tab:transfer}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
             r"\toprule",
             r"Method & band ECE & band Brier & ECE & AP \\",
             r"\midrule"]
    best = min((d[k]["band_ece"] for k in METH if k in d), default=None)
    for k in METH:
        if k not in d:
            continue
        c = [f(d[k]["band_ece"], 4), f(d[k]["band_brier"], 4),
             f(d[k]["ece"], 4), f(d[k]["ap"], 4)]
        if best is not None and abs(d[k]["band_ece"] - best) < 1e-9:
            c[0] = r"\textbf{" + c[0] + "}"
        lines.append(_row(k, d[k], c))
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    _write("tables/tab_transfer.tex", lines)


if __name__ == "__main__":
    main()
