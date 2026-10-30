"""Figures for the 2026-10-30 lab talk (docs/talk_2026-10-30_outline.md) -> docs/figures/talk/*.png (+ .svg).

Slide-ready: sized for a 16:9 slide (13.3 x 7.5 in) at 1:1, Arial, large type, no figure titles (the slide
title carries the message), one colour per meaning: teal = PALM, red = standard tools, near-black = truth.
Uses saved benchmark outputs (causal_loop auto_m rows, dev_bench competitor arrays, practice_vs_palm rows);
the only computation is re-deriving the uncorrected effects and current-practice tests for split 0.
Usage: python talk_figures.py [SPLIT]
"""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import practice_vs_palm as pv

FIG = os.path.join(ROOT, "docs", "figures", "talk")
SPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
C_PRACT, C_PALM, C_TRUTH, C_GREY, C_LIGHT = "#d1495b", "#00798c", "#2b2d42", "#9aa0a6", "#eef1f3"
C_NEUT, C_MONO = "#e9a23b", "#6c5b9e"
plt.rcParams.update({"font.family": "Arial", "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
                     "axes.edgecolor": "#444444", "xtick.color": "#444444", "ytick.color": "#444444",
                     "axes.labelcolor": "#222222", "text.color": "#222222", "svg.fonttype": "none",
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.08})
LINE = {"LNCAPCLONEFGC": "LNCaP", "RCC10RGB": "RCC10RGB", "SNU1079": "SNU1079", "COLO680N": "COLO680N",
        "BICR31": "BICR31", "TEN": "TEN"}
P53_OK = {"LNCAPCLONEFGC", "RCC10RGB", "SNU1079"}


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, name + "." + ext))
    plt.close(fig)
    print("wrote", name)


def auto_m(dsn, split):
    a = json.load(open(os.path.join(cl.OUT, "%s__s%d__auto_m.json" % (dsn, split))))["arrays"]
    return {k: (np.asarray(v, float) if k not in ("cell_types", "programs") else v) for k, v in a.items()}


def practice(dsn, split, method, T, donors):
    z = np.load(os.path.join(pv.DEV, dsn, "s%d__%s.npz" % (split, method)))
    eff, lo, hi, p = pv.practice_test(z["model__activity"], T, donors)
    return eff, lo, hi, pv.bh(p) < 0.1


# ---- 2.1 concept: three reasons a bulk pathway signal goes up ----------------------------------------------
def fig_concept():
    """Same bulk rise, three causes (TB example: neutrophils and monocytes). Bars: bulk signal by source; dots:
    the 10 cells behind it (outlined = pathway switched on). Every scenario reaches the same bulk value."""
    per = dict(N=0.14, M=0.06)                                     # pathway signal per cell, at rest
    cases = [("Healthy", "", "", 5, 1.0, 1.0),
             ("1", "More neutrophils", "same activity per cell", 7, 1.0, 1.0),
             ("2", "Monocytes switched on", "spills into the\nneutrophil estimate", 5, 1.0, 46 / 30),
             ("3", "Neutrophils switched on", "what we want to know", 5, 86 / 70, 1.0)]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    w, gap = 0.6, 1.35
    top = 5 * per["N"] * 86 / 70 + 5 * per["M"]
    for i, (num, name, sub, nN, fN, fM) in enumerate(cases):
        x = i * gap
        nM = 10 - nN
        base_n, base_m = nN * per["N"], nM * per["M"]
        hN = base_n * fN
        if i == 3:
            ax.add_patch(FancyBboxPatch((x - 0.6, -0.98), 1.2, top + 1.12, boxstyle="round,pad=0,rounding_size=0.08",
                                        fc="#e3f1f3", ec=C_PALM, lw=1.8, zorder=0))
        ax.bar(x, base_n, w, color=C_NEUT, ec="white", lw=1.5, zorder=2)
        if fN > 1:
            ax.bar(x, hN - base_n, w, bottom=base_n, color=C_NEUT, ec="white", hatch="///", lw=1.5, zorder=2)
        ax.bar(x, base_m, w, bottom=hN, color=C_MONO, ec="white", lw=1.5, zorder=2)
        if fM > 1:
            ax.bar(x, base_m * (fM - 1), w, bottom=hN + base_m, color=C_MONO, ec="white", hatch="///", lw=1.5,
                   zorder=2)
        for k in range(10):                                        # the 10 cells behind the bulk sample
            is_n = k < nN
            on = (is_n and fN > 1) or (not is_n and fM > 1)
            ax.scatter(x - 0.24 + 0.12 * (k % 5), -0.16 - 0.14 * (k // 5), s=260,
                       color=C_NEUT if is_n else C_MONO, ec="#111111" if on else "white", lw=2.8 if on else 1.0,
                       zorder=3)
        if num == "Healthy":
            ax.text(x, -0.47, "Healthy", ha="center", va="top", fontsize=17, fontweight="bold")
        else:
            ax.text(x, -0.47, "%s.  %s" % (num, name.replace(" ", "\n", 1)), ha="center", va="top", fontsize=17,
                    fontweight="bold", color=C_PALM if i == 3 else "#222222", linespacing=1.1)
        ax.text(x, -0.80, sub, ha="center", va="top", fontsize=13.5, color=C_PALM if i == 3 else "#555555",
                fontweight="bold" if i == 3 else "normal", style="normal" if i == 3 else "italic")
    ax.plot([gap - 0.45, 3 * gap + 0.45], [top, top], ls=(0, (5, 3)), color="#333333", lw=1.5, zorder=1)
    ax.text(2 * gap - gap / 2, top + 0.04, "same bulk signal", ha="center", va="bottom", fontsize=14,
            color="#333333")
    ax.set_xlim(-0.75, 3 * gap + 0.7)
    ax.set_ylim(-1.12, 1.36)
    ax.axis("off")
    ax.text(-0.72, 0.55, "pathway signal\nin the bulk sample", rotation=90, ha="center", va="center",
            fontsize=15, color="#444444")
    handles = [Rectangle((0, 0), 1, 1, color=C_NEUT), Rectangle((0, 0), 1, 1, color=C_MONO),
               Rectangle((0, 0), 1, 1, fc="#bbbbbb", hatch="///", ec="white"),
               plt.Line2D([], [], marker="o", ls="", ms=13, mfc="#cccccc", mec="#111111", mew=2.6)]
    ax.legend(handles, ["signal from neutrophils", "signal from monocytes", "extra signal from activation",
                        "cell with the pathway switched on"], loc="upper center", bbox_to_anchor=(0.5, 0.0),
              frameon=False, ncol=4, fontsize=13, handlelength=1.4, columnspacing=1.8)
    save(fig, "2_1_three_causes")


# ---- backup: why correction matters, estimated vs true change ----------------------------------------------
def fig_attenuation(dsets=("mcf_ida", "mcf_tram")):
    from palm_bulk.core.realloc import reallocate
    xs, raw_e, palm_e, leak = [], [], [], []
    for dsn in dsets:
        pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, SPLIT)
        pi = np.median(src["fraction_draws"], 0)
        raw = pb.core.contrast(T)(reallocate(inp.Y, pi, inp.B, inp.Gmat))
        a = auto_m(dsn, SPLIT)
        xs.append(beta.ravel()); raw_e.append(raw.ravel()); palm_e.append(a["effects"].ravel())
        # non-responding lines' p53 under idasanutlin: where the leaked signal shows
        m = np.zeros(beta.shape, bool)
        if dsn == "mcf_ida":
            m[[c not in P53_OK for c in inp.cell_types], list(inp.programs).index("p53")] = True
        leak.append(m.ravel())
    x, r, p, lk = map(np.concatenate, (xs, raw_e, palm_e, leak))
    lim = [min(x.min(), r.min(), p.min()) - 0.06, max(x.max(), r.max(), p.max()) + 0.06]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), sharex=True, sharey=True)
    for ax, y, title, c in ((axes[0], r, "Uncorrected", C_PRACT), (axes[1], p, "PALM", C_PALM)):
        ax.plot(lim, lim, color=C_GREY, lw=1.2, ls=(0, (4, 3)), zorder=0)
        ax.scatter(x[~lk], y[~lk], s=38, color=c, alpha=0.7, ec="none")
        ax.scatter(x[lk], y[lk], s=70, color=c, ec="#111111", lw=1.6, zorder=3)
        slope = np.polyfit(x, y, 1)[0]
        ax.set_title(title, color=c, fontweight="bold", loc="left")
        ax.text(0.03, 0.95, "recovers %d%% of the true size" % round(100 * slope), transform=ax.transAxes,
                va="top", fontsize=14)
        ax.set_xlabel("true change in the cell line")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    axes[0].set_ylabel("estimated change")
    i = np.flatnonzero(lk)[np.argmax(r[lk])]
    axes[0].annotate("p53 in non-responding lines:\nlooks changed (leaked signal)", (x[i], r[i]),
                     xytext=(0.2, 0.24), textcoords="data", fontsize=12.5, va="center",
                     arrowprops=dict(arrowstyle="-", color="#333333", lw=1))
    axes[1].text(0.97, 0.04, "dashed line = perfect\neach dot: one cell line × pathway", transform=axes[1].transAxes,
                 ha="right", fontsize=11.5, color="#666666")
    save(fig, "2_3_estimated_vs_true")


# ---- 2.4 answer key and the two answers (idasanutlin) ------------------------------------------------------
def fig_heatmaps(dsn="mcf_ida", competitor="cibersort"):
    pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, SPLIT)
    ct, pg = list(inp.cell_types), list(inp.programs)
    v = float(np.ceil(np.abs(beta).max() * 10) / 10)
    if competitor == "lab":
        # the lab workflow: InstaPrism per-type expression -> GSVA -> Spearman with treatment (its own scale)
        from scipy import stats
        import lab_workflow as lw
        act = lw.lab_scores(dsn, SPLIT, inp, "instaprism")
        rho, p = np.zeros(beta.shape), np.ones(beta.shape)
        for c in range(len(ct)):
            for j in range(len(pg)):
                if act[:, c, j].std() > 0:
                    rho[c, j], p[c, j] = stats.spearmanr(act[:, c, j], T.astype(float))
        eff_p, calls_p, vp = rho, pv.bh(p) < 0.1, 1.0
        label, unit = "Standard workflow", "correlation with drug (Spearman ρ)"
        sub_p, name = "InstaPrism → GSVA → correlation", "2_5_heatmap_lab_workflow"
    else:
        eff_p, lo_p, hi_p, calls_p = practice(dsn, SPLIT, competitor, T, donors)
        vp, label, unit = v, "Current practice", None
        sub_p, name = "CIBERSORT → t-test", "2_5_heatmap_idasanutlin"
    a = auto_m(dsn, SPLIT)
    calls_palm = a["calls"].astype(bool)
    null = np.abs(beta) < 0.05
    order = [c for c in ct if c in P53_OK] + [c for c in ct if c not in P53_OK]
    ri = [ct.index(c) for c in order]
    panels = [("Answer key", "true change in each line", beta, None, v, C_TRUTH),
              (label, sub_p, eff_p, calls_p, vp, C_PRACT),
              ("PALM", "corrected change", a["effects"], calls_palm, v, C_PALM)]
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 5.9), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.80, bottom=0.30, wspace=0.08)
    ims = []
    for ax, (title, sub, e, calls, vm, col) in zip(axes, panels):
        ims.append(ax.imshow(e[ri], cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto"))
        if calls is not None:
            for i, r_ in enumerate(ri):
                for j in range(len(pg)):
                    if calls[r_, j]:
                        for lw_, ec_ in ((5.0, "white"), (2.4, "#111111")):     # white halo: visible on dark red
                            ax.add_patch(Rectangle((j - 0.40, i - 0.40), 0.80, 0.80, fill=False, lw=lw_, ec=ec_))
            n_false = int((calls & null).sum())
            ax.text(0.5, 1.035, "flags %d of %d unchanged" % (n_false, null.sum()), transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=14.5, fontweight="bold", color=col)
        ax.text(0.5, 1.16, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=18,
                fontweight="bold", color=col)
        ax.text(0.5, 1.10, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=13, color="#555555")
        ax.set_xticks(range(len(pg))); ax.set_xticklabels(pg, rotation=40, ha="right", rotation_mode="anchor")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(2.5, color="white", lw=5)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LINE[c] for c in order])
    for y0, y1, txt in ((-0.4, 2.2, "p53\nintact"), (2.8, 5.4, "other\nlines")):
        axes[0].annotate("", xy=(-0.50, 1 - (y0 + 0.5) / 6), xytext=(-0.50, 1 - (y1 + 0.5) / 6),
                         xycoords="axes fraction", arrowprops=dict(arrowstyle="-", lw=1.6, color="#555555"))
        axes[0].text(-0.56, 1 - ((y0 + y1) / 2 + 0.5) / 6, txt, transform=axes[0].transAxes, ha="right",
                     va="center", fontsize=13, color="#555555")
    # colour keys under the panels: truth and PALM share one; the workflow's correlation has its own
    for k, (lab_, im) in enumerate([("change in pathway activity\n(answer key and PALM, same scale)", ims[0]),
                                     (unit, ims[1])]):
        if lab_ is None:
            continue
        pos = axes[0 if k == 0 else 1].get_position()
        cax = fig.add_axes([pos.x0 + 0.15 * pos.width, 0.085, 0.7 * pos.width, 0.03])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(lab_, fontsize=12.5); cb.ax.tick_params(labelsize=12); cb.outline.set_visible(False)
    pos = axes[2].get_position()
    fig.legend([Rectangle((0, 0), 1, 1, fc="#f2c9b8", ec="#111111", lw=2.4)], ["flagged as changed"],
               loc="center", bbox_to_anchor=(pos.x0 + 0.5 * pos.width, 0.11), frameon=False, fontsize=14,
               handlelength=1.6, handleheight=1.2)
    save(fig, name)


# ---- 2.5 false flags across datasets and tools -------------------------------------------------------------
def fig_false_flags():
    rows = [r for f in glob.glob(os.path.join(pv.OUT, "*.json")) for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    df = df[df.split < 10]
    pract = df[df.kind.isin(["practice", "lab_gsva"]) & (df.rule == "bh10")].copy()
    pract.loc[pract.kind == "lab_gsva", "method"] = "lab_gsva"
    palm = pd.DataFrame(pv._palm_rows())
    palm = palm[palm.split.isin(pract.split.unique())]
    tools = [("lab_gsva", "Standard workflow\n(InstaPrism + GSVA)"), ("instaprism", "InstaPrism + t-test"),
             ("cibersort", "CIBERSORT + t-test"), ("reallocation", "NNLS + t-test"),
             ("naive", "Bulk score, no\ndeconvolution")]
    dsets = [("mcf_ida", "Idasanutlin", "p53 up in 3 of 6 lines"), ("mcf_tram", "Trametinib", "MAPK down"),
             ("kang", "Interferon-β", "all PBMC types respond")]
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 5.4), sharey=True)
    fig.subplots_adjust(wspace=0.12, bottom=0.2)
    y = np.arange(len(tools) + 1)
    for ax, (d, title, sub) in zip(axes, dsets):
        s = pract[pract.dataset == d].groupby("method").fpr_null.mean()
        vals = [s.get(m, np.nan) for m, _ in tools] + [palm[palm.dataset == d].fpr_null.mean()]
        cols = [C_PRACT] * len(tools) + [C_PALM]
        ax.barh(y, [100 * v for v in vals], color=cols, height=0.68)
        ax.plot([10, 10], [-0.6, len(tools) + 0.5], color="#555555", ls=(0, (3, 3)), lw=1.2, zorder=0)
        for i, v in enumerate(vals):
            inside = v > 0.8
            ax.text(100 * v - 2 if inside else max(100 * v + 2, 12), i, "%.0f%%" % (100 * v), va="center",
                    ha="right" if inside else "left", fontsize=14, fontweight="bold",
                    color="white" if inside else (C_PALM if i == len(tools) else "#222222"))
        ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xticklabels(["0", "50%", "100%"])
        ax.set_title(title, fontweight="bold", loc="left", pad=34)
        ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=13, color="#555555")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([t for _, t in tools] + ["PALM"])
    axes[0].invert_yaxis()
    for t in axes[0].get_yticklabels():
        if t.get_text() == "PALM":
            t.set_color(C_PALM); t.set_fontweight("bold")
        if t.get_text().startswith("Standard"):
            t.set_fontweight("bold")
    for ax in axes:
        ax.text(10, -0.62, "10%", ha="center", va="bottom", fontsize=12, color="#555555")
    fig.text(0.5, 0.06, "Unchanged cell type × pathway pairs flagged as changed", ha="center", fontsize=15)
    fig.text(0.5, 0.0, "When nothing responds strongly (chemotherapy, Zhang) no tool over-flags (0–1%). "
             "Development splits 0–2.", ha="center", fontsize=12, color="#666666")
    save(fig, "2_5_false_flags_all")


# ---- 2.5 positive and negative controls --------------------------------------------------------------------
def fig_controls():
    """Butterfly per dataset: left = negative control (unchanged pairs flagged, shorter is better), right =
    positive control (real changes in the drug's pathway found with the right sign, |true| >= 0.1; longer is
    better). Chemotherapy (Zhang) has no strong response: a negative-control dataset (left side only)."""
    rows = [r for f in glob.glob(os.path.join(pv.OUT, "*.json")) for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    df = df[df.split < 10]
    pract = df[df.kind.isin(["practice", "lab_gsva"]) & (df.rule == "bh10")].copy()
    pract.loc[pract.kind == "lab_gsva", "method"] = "lab_gsva"
    palm = pd.DataFrame(pv._palm_rows())
    palm = palm[palm.split.isin(pract.split.unique())]
    tools = [("lab_gsva", "Standard workflow\n(InstaPrism + GSVA)"), ("instaprism", "InstaPrism + t-test"),
             ("cibersort", "CIBERSORT + t-test"), ("reallocation", "NNLS + t-test"),
             ("naive", "Bulk score, no\ndeconvolution")]
    dsets = [("mcf_ida", "Idasanutlin", "p53 up in 3 of 6 lines"), ("mcf_tram", "Trametinib", "MAPK down"),
             ("kang", "Interferon-β", "all PBMC types respond"), ("zhang", "Chemotherapy", "no strong response")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.6), sharey=True, gridspec_kw=dict(width_ratios=[1, 1, 1, 0.62]))
    fig.subplots_adjust(wspace=0.14, bottom=0.2, top=0.78)
    y = np.arange(len(tools) + 1)
    for ax, (d, title, sub) in zip(axes, dsets):
        s = pract[pract.dataset == d].groupby("method")[["fpr_null", "tpr_t"]].mean()
        pm = palm[palm.dataset == d][["fpr_null", "tpr_t"]].mean()
        neg = [s.fpr_null.get(m, np.nan) for m, _ in tools] + [pm.fpr_null]
        pos = [s.tpr_t.get(m, np.nan) for m, _ in tools] + [pm.tpr_t]
        cols = [C_PRACT] * len(tools) + [C_PALM]
        only_neg = d == "zhang"
        ax.barh(y, [-100 * v for v in neg], color=cols, height=0.68, alpha=0.55)
        if not only_neg:
            ax.barh(y, [100 * v for v in pos], color=cols, height=0.68)
        ax.axvline(0, color="#333333", lw=1.2)
        for i, (vn, vp) in enumerate(zip(neg, pos)):
            bold = dict(fontsize=12.5, fontweight="bold", va="center")
            c_txt = C_PALM if i == len(tools) else "#222222"
            if vn > 0.25:                                           # large values inside their bars
                ax.text(-100 * vn + 3, i, "%.0f%%" % (100 * vn), ha="left", color="#222222", **bold)
            else:
                ax.text(-100 * vn - 3, i, "%.0f%%" % (100 * vn), ha="right", color=c_txt, **bold)
            if not only_neg:
                if vp > 0.25:
                    ax.text(100 * vp - 3, i, "%.0f%%" % (100 * vp), ha="right", color="white", **bold)
                else:
                    ax.text(100 * vp + 3, i, "%.0f%%" % (100 * vp), ha="left", color=c_txt, **bold)
        if only_neg:
            ax.set_xlim(-108, 40)
            ax.text(8, len(tools) / 2, "no real\nchanges\nto find", ha="left", va="center", fontsize=12.5,
                    color="#777777", style="italic")
            ax.set_xticks([-100, 0]); ax.set_xticklabels(["100%", "0"])
        else:
            ax.set_xlim(-108, 108)
            ax.set_xticks([-100, 0, 100]); ax.set_xticklabels(["100%", "0", "100%"])
        ax.set_title(title, fontweight="bold", loc="left", pad=46)
        ax.text(0, 1.075, sub, transform=ax.transAxes, fontsize=13, color="#555555")
        ax.text(0.47 if not only_neg else 0.76, 1.01, "← flags unchanged", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=11.5, color="#666666")
        if not only_neg:
            ax.text(0.53, 1.01, "finds real →", transform=ax.transAxes, ha="left", va="bottom", fontsize=11.5,
                    color="#666666")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([t for _, t in tools] + ["PALM"])
    axes[0].invert_yaxis()
    for t in axes[0].get_yticklabels():
        if t.get_text() == "PALM":
            t.set_color(C_PALM); t.set_fontweight("bold")
    fig.text(0.5, 0.075, "Left, negative control: unchanged cell type × pathway pairs flagged (shorter is better).   "
             "Right, positive control: real changes found (longer is better).", ha="center", fontsize=13.5)
    fig.text(0.5, 0.025, "Real change = drug's pathway, |true change| ≥ 0.1, found with the right sign. "
             "BH q < 0.1 for the tools; PALM 90% range. Development splits 0–2.", ha="center", fontsize=11.5,
             color="#666666")
    save(fig, "2_5_controls")


# ---- case studies: PALM vs InstaPrism + t-test, per cell type, for the drug's pathway ----------------------
def palm_arrays(dsn, split, name="auto_abs"):
    f = os.path.join(cl.OUT, "%s__s%d__%s.json" % (dsn, split, name))
    return auto_m(dsn, split) if not os.path.exists(f) else \
        {k: (np.asarray(v, float) if k not in ("cell_types", "programs") else v)
         for k, v in json.load(open(f))["arrays"].items()}


def fig_case_studies():
    """Forest plots: rows = cell lines / cell types, true change (diamond), InstaPrism + t-test and PALM with 90%
    ranges; filled marker = flagged as changed, open = not flagged."""
    cases = [("mcf_ida", "p53", "Idasanutlin (MDM2 inhibitor) → p53 pathway",
              lambda c: "p53 intact" if c in P53_OK else ""),
             ("mcf_tram", "MAPK", "Trametinib (MEK inhibitor) → MAPK pathway",
              lambda c: "MEK-independent" if c == "LNCAPCLONEFGC" else ""),
             ("kang", "JAK-STAT", "Interferon-β → JAK-STAT (all cells respond)", lambda c: ""),
             ("kang", "TGFb", "Interferon-β → TGF-β (monocytes only)", lambda c: "")]
    fig, axes = plt.subplots(2, 2, figsize=(13.3, 8.6))
    fig.subplots_adjust(hspace=0.42, wspace=0.42)
    for ax, (dsn, prog, title, tag) in zip(axes.ravel(), cases):
        pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, SPLIT)
        ct = list(inp.cell_types)
        j = list(inp.programs).index(prog)
        e_i, lo_i, hi_i, c_i = practice(dsn, SPLIT, "instaprism", T, donors)
        a = palm_arrays(dsn, SPLIT)
        order = np.argsort(-np.abs(beta[:, j]))
        y = np.arange(len(ct))
        ax.axvline(0, color="#888888", lw=1.1, zorder=0)
        for k, i in enumerate(order):
            for dy, e, lo, hi, call, col in ((0.2, e_i[i, j], lo_i[i, j], hi_i[i, j], c_i[i, j], C_PRACT),
                                             (-0.2, a["effects"][i, j], a["lo"][i, j], a["hi"][i, j],
                                              bool(a["calls"][i, j]), C_PALM)):
                ax.plot([lo, hi], [k - dy] * 2, color=col, lw=3, solid_capstyle="round", alpha=0.85)
                ax.scatter(e, k - dy, s=70, color=col if call else "white", ec=col, lw=2, zorder=4)
            ax.scatter(beta[i, j], k, marker="D", s=70, color=C_TRUTH, zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels([LINE.get(ct[i], ct[i]) + ("\n" + tag(ct[i]) if tag(ct[i]) else "") for i in order],
                           fontsize=12.5)
        for t, i in zip(ax.get_yticklabels(), order):
            if tag(ct[i]):
                t.set_fontweight("bold")
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=14.5, fontweight="bold")
        ax.set_xlabel("change in pathway activity", fontsize=13)
        lim = np.nanmax(np.abs(np.r_[lo_i[:, j], hi_i[:, j], a["lo"][:, j], a["hi"][:, j], beta[:, j]])) * 1.08
        ax.set_xlim(-lim, lim)
    handles = [plt.Line2D([], [], marker="D", ls="", ms=9, color=C_TRUTH),
               plt.Line2D([], [], marker="o", ms=9, color=C_PRACT, lw=3),
               plt.Line2D([], [], marker="o", ms=9, color=C_PALM, lw=3),
               plt.Line2D([], [], marker="o", ls="", ms=9, mfc="#777777", mec="#777777"),
               plt.Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec="#777777", mew=2)]
    fig.legend(handles, ["true change", "InstaPrism + t-test", "PALM", "flagged as changed", "not flagged"],
               loc="lower center", bbox_to_anchor=(0.5, 0.94), ncol=5, frameon=False, fontsize=13)
    fig.text(0.5, 0.0, "Lines: 90%% range. Development split %d; PALM with the per-type leakage allowance "
             "(sim_widen=\"abs\", identical to v1 for the drug datasets)." % SPLIT, ha="center", fontsize=11,
             color="#666666")
    save(fig, "case_studies_palm_vs_instaprism")


# ---- case studies scorecard: PALM vs InstaPrism, every aspect x dataset ------------------------------------
def scorecard_values(splits=(0, 1, 2)):
    """(aspect, dataset) -> (PALM, InstaPrism) on the same splits. Proportions: fraction RMSE in percentage
    points (dev_bench model rows); the rest: practice_vs_palm rows (InstaPrism + t-test, BH q < 0.1) and PALM
    (auto_m)."""
    rows = [r for f in glob.glob(os.path.join(pv.OUT, "*.json")) for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    ip = df[(df.kind == "practice") & (df.method == "instaprism") & (df.rule == "bh10") & df.split.isin(splits)]
    pm = pd.DataFrame(pv._palm_rows())
    pm = pm[pm.split.isin(splits)]
    fr = {}
    for ds in ("mcf_ida", "mcf_tram", "kang", "zhang"):
        for lab, f in (("PALM", "joint_nb"), ("InstaPrism", "instaprism")):
            v = []
            for s in splits:
                p = os.path.join(ROOT, "benchmarks", "dev", ds, "s%d__%s.json" % (s, f))
                if os.path.exists(p):
                    v += [r["frac_rmse"] for r in json.load(open(p)) if r.get("row") == "model"]
            fr[(ds, lab)] = 100 * np.mean(v) if v else np.nan
    out = {}
    for ds in ("mcf_ida", "mcf_tram", "kang", "zhang"):
        a, b = pm[pm.dataset == ds], ip[ip.dataset == ds]
        out[("frac", ds)] = (fr[(ds, "PALM")], fr[(ds, "InstaPrism")])
        out[("neg", ds)] = (100 * a.fpr_null.mean(), 100 * b.fpr_null.mean())
        out[("pos", ds)] = (100 * a.tpr_t.mean(), 100 * b.tpr_t.mean())
        out[("r", ds)] = (a.r_t.mean(), b.r_t.mean())
        out[("cov", ds)] = (100 * a["cov"].mean(), 100 * b["cov"].mean())
    return out


def fig_scorecard():
    """Rows = aspects (proportions, negative control, positive control, effect accuracy, honest ranges), columns =
    datasets; each cell a dumbbell PALM (teal) vs InstaPrism (red) on the aspect's scale, the better one bold."""
    v = scorecard_values()
    aspects = [("frac", "Proportion error\n(pp, lower better)", False, (0, 10)),
               ("neg", "Negative control:\nunchanged flagged (%)", False, (0, 100)),
               ("pos", "Positive control:\nreal changes found (%)", True, (0, 100)),
               ("r", "Effect accuracy\n(r with truth)", True, (0.5, 1)),
               ("cov", "Honest ranges:\ntruth inside 90% range (%)", True, (0, 100))]
    dsets = [("mcf_ida", "Idasanutlin\np53 in 3 of 6 lines"), ("mcf_tram", "Trametinib\nMAPK down in 5 of 6"),
             ("kang", "Interferon-β\nall PBMC types"), ("zhang", "Chemotherapy\nno strong response")]
    fig, axes = plt.subplots(len(aspects), len(dsets), figsize=(13.3, 8.2))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.86, bottom=0.08, hspace=0.75, wspace=0.18)
    for i, (key, lab, hb, lim) in enumerate(aspects):
        for j, (ds, dlab) in enumerate(dsets):
            ax = axes[i, j]
            p, q = v[(key, ds)]
            ax.set_xlim(*lim); ax.set_ylim(-1, 1)
            ax.set_yticks([])
            for s in ("left", "right", "top"):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis="x", labelsize=11)
            ax.set_xticks(np.linspace(lim[0], lim[1], 3))
            if i == 0:
                ax.set_title(dlab, fontsize=13.5, fontweight="bold", pad=14)
            if j == 0:
                ax.text(-0.12, 0, lab, transform=ax.transAxes, ha="right", va="center", fontsize=13)
            if not (np.isfinite(p) and np.isfinite(q)):
                ax.text(np.mean(lim), 0, "no real changes to find" if key in ("pos", "r") else "–",
                        ha="center", va="center", fontsize=11.5, color="#888888", style="italic")
                continue
            ax.plot([p, q], [0, 0], color="#cccccc", lw=4, zorder=1, solid_capstyle="round")
            better_p = (p >= q) if hb else (p <= q)
            tie = abs(p - q) < (0.02 if key == "r" else 1.0)
            fmt = "%.2f" if key == "r" else "%.0f" if key != "frac" else "%.1f"
            for val, col, win, dy in ((q, C_PRACT, not better_p and not tie, -0.62), (p, C_PALM, better_p and not tie, 0.62)):
                ax.scatter(np.clip(val, *lim), 0, s=150 if win else 90, color=col, zorder=3, ec="white", lw=1.2)
                ax.text(np.clip(val, *lim), dy, fmt % val, ha="center", va="center", fontsize=12,
                        fontweight="bold" if win else "normal", color=col)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=11, color=C_PALM),
               plt.Line2D([], [], marker="o", ls="", ms=11, color=C_PRACT)]
    fig.legend(handles, ["PALM", "InstaPrism (+ t-test for calls)"], loc="upper center", ncol=2, frameon=False,
               fontsize=14, bbox_to_anchor=(0.58, 0.995))
    fig.text(0.58, 0.01, "Larger dot and bold number = better of the two. Same development splits (0–2) for both "
             "methods. PALM v1 setting.", ha="center", fontsize=11, color="#666666")
    save(fig, "case_scorecard_palm_vs_instaprism")


# ---- 2.5 proportions accuracy ------------------------------------------------------------------------------
def fig_fractions():
    vals = {"PALM": 2.5, "InstaPrism": 6.0, "CIBERSORT": 6.5, "cellGeometry": 8.0, "NNLS": 10.1}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(vals))
    ax.barh(y, list(vals.values()), color=[C_PALM] + [C_GREY] * 4, height=0.66)
    for i, v in enumerate(vals.values()):
        ax.text(v + 0.15, i, "%.1f" % v, va="center", fontsize=14, fontweight="bold",
                color=C_PALM if i == 0 else "#333333")
    ax.set_yticks(y); ax.set_yticklabels(list(vals)); ax.invert_yaxis()
    ax.get_yticklabels()[0].set_color(C_PALM); ax.get_yticklabels()[0].set_fontweight("bold")
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 11.5)
    ax.set_xlabel("error in cell-type proportions\n(RMSE, percentage points; lower is better)")
    ax.set_title("Interferon-β PBMC", fontweight="bold", loc="left")
    ax.text(2.5 + 1.3, 0, "2.4× smaller than the next best", va="center", fontsize=13, color=C_PALM)
    save(fig, "2_6_fraction_error")


# ---- backup: interferon response per cell type -------------------------------------------------------------
def fig_kang_limit(competitor="cibersort"):
    pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup("kang", SPLIT)
    ct, pg = list(inp.cell_types), list(inp.programs)
    j = pg.index("JAK-STAT")
    eff_p, lo_p, hi_p, _ = practice("kang", SPLIT, competitor, T, donors)
    a = auto_m("kang", SPLIT)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(ct))[::-1]
    ax.axvline(0, color="#777777", lw=1.2, zorder=0)
    ax.errorbar(eff_p[:, j], y + 0.2, xerr=[eff_p[:, j] - lo_p[:, j], hi_p[:, j] - eff_p[:, j]], fmt="o", ms=8,
                color=C_PRACT, capsize=0, elinewidth=3, label="CIBERSORT + t-test")
    ax.errorbar(a["effects"][:, j], y - 0.2, xerr=[a["effects"][:, j] - a["lo"][:, j],
                                                   a["hi"][:, j] - a["effects"][:, j]], fmt="o", ms=8,
                color=C_PALM, capsize=0, elinewidth=3, label="PALM (90% range)")
    ax.scatter(beta[:, j], y, marker="D", s=80, color=C_TRUTH, zorder=4, label="true change")
    ax.set_yticks(y); ax.set_yticklabels(ct)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("change in interferon (JAK-STAT) pathway activity")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=13)
    ax.text(0.0, -0.2, "14 samples with similar proportions: the four cell types can't be told apart,\n"
            "so PALM's ranges include 0 (\"can't tell\") instead of guessing.", transform=ax.transAxes, ha="left",
            va="top", fontsize=13, color="#555555")
    save(fig, "2_6_interferon_ranges")


if __name__ == "__main__":
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    figs = dict(concept=fig_concept, fractions=fig_fractions, false_flags=fig_false_flags, controls=fig_controls,
                cases=fig_case_studies, scorecard=fig_scorecard,
                heat=fig_heatmaps,
                heat_lab=lambda: fig_heatmaps(competitor="lab"), attenuation=fig_attenuation, kang=fig_kang_limit)
    for k, f in figs.items():
        if only and k not in only:
            continue
        try:
            f()
        except Exception as e:
            print("FAILED", k, repr(e))
