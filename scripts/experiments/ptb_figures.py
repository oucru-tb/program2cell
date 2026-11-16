"""Figures for the PTB vs Healthy application (run_ptb.py outputs) -> results/ptb/figures/*.png (+ .svg).

Versions (freeze v2, Reactome immune6 programs, no run correction: PTB is all Run1, healthy all Run2):
  l2      results/ptb/l2_reactome_immune6/        17 celltype.l2 types (user's Seurat reference + neutrophils)
  combat  results/ptb/combat_reactome_immune6/    8 types (COMBAT healthy PBMC + GSE188288 neutrophils)
Per version: effects heatmap (one colour scale shared by both versions), effect intervals per program,
"cells changed" vs "more cells" per program and cell type, cell-type fractions PTB vs Healthy.
Comparison: each l2 type against its broad COMBAT type (L2_TO_BROAD; Plasmablast, MAIT, OtherT, pDC have none),
effects with intervals, call agreement map, numbers in compare.csv.
Figure sets (FIGSET): refs (default) -> results/ptb/figures/ (17-type vs 8-type, Reactome);
  response -> results/ptb/figures_response/: 8-type reference, Hallmark IFNb/TNF + Reactome TLR/NLR/BCR/TCR
  (immune_response6), study self-offsets vs offsets anchored on healthy CyTOF composition (ptb_anchor.py);
  coarse -> results/ptb/figures_coarse/: anchored offsets, 8 types vs the 5 CyTOF groups (each 8-type type against
  its group).
  innate7 (with FIGSET=response): anchored decomposition for TLR, NLR, TNF, type I IFN, IFN-gamma
  (combat_immune_response7_anchored), panel titles with gene counts and key genes; innate8: the same with the IFN
  sets split into type I only, IFN-gamma only and shared ISGs (combat_immune_response8_anchored).
Usage: [FIGSET=refs|response|coarse] python ptb_figures.py [--demo] [innate7|innate7r|innate8|innate8r|heat7r]   (--demo: synthetic tables, layout check only)
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "src"))
from program2cell import plots

DEMO = "--demo" in sys.argv
BASE = os.path.join(ROOT, "results", "ptb")
FIGSET = os.environ.get("FIGSET", "refs")
FIG = os.path.join(BASE, "figures" + ("" if FIGSET == "refs" else "_" + FIGSET) + ("_demo" if DEMO else ""))
VERSIONS = {"l2": ("l2_reactome_immune6", "17-type reference (celltype.l2)"),
            "combat": ("combat_reactome_immune6", "8-type reference (COMBAT + neutrophils)")}
if FIGSET == "response":
    VERSIONS = {"self": ("combat_immune_response6", "Study self-offsets (8-type reference)"),
                "anchored": ("combat_immune_response6_anchored", "Anchored offsets (8-type reference)")}
if FIGSET == "coarse":
    VERSIONS = {"anchored8": ("combat_immune_response6_anchored", "8 types, anchored offsets"),
                "anchored5": ("combat5_immune_response6_anchored", "5 types, anchored offsets")}
EIGHT_TO_FIVE = {"B": "B", "CD4 T": "T", "CD8 T": "T", "NK": "NK", "CD14 Mono": "Myeloid", "CD16 Mono": "Myeloid",
                 "DC": "Myeloid", "Neutro": "Neutro"}
COMPARE_TITLE = {"refs": "17-type vs 8-type reference", "response": "self vs anchored offsets",
                 "coarse": "8 vs 5 types"}[FIGSET]
A, B = list(VERSIONS)
L2_TO_BROAD = {"B naive": "B", "B memory": "B", "CD4 Naive": "CD4 T", "CD4 TM": "CD4 T", "Treg": "CD4 T",
               "CD4 CTL": "CD4 T", "CD8 Naive": "CD8 T", "CD8 TM": "CD8 T", "NK": "NK", "CD14 Mono": "CD14 Mono",
               "CD16 Mono": "CD16 Mono", "cDC": "DC", "Neutro": "Neutro"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MORE, CHANGED = "#2a78d6", "#eb6834"                  # dataviz slots 1-2 (as plots.py)
PTB_C, HV_C = "#e34948", "#2a78d6"
AGREE = {"both": "#1e2733", "%s only" % A: "#2a78d6", "%s only" % B: "#eb6834", "neither": "#e4e3df"}
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, name + "." + ext))
    plt.close(fig)
    print("wrote", os.path.join(FIG, name + ".png"))


def demo_tables():
    rng = np.random.default_rng(0)
    progs = ["TLR", "NLR", "IFNb", "TNF", "BCR", "TCR"]
    out = {}
    for v in VERSIONS:
        types = list(L2_TO_BROAD) + ["Plasmablast", "pDC"] if v == A else sorted(set(L2_TO_BROAD.values()))
        e = rng.normal(0, 0.08, (len(types), len(progs)))
        w = rng.uniform(0.05, 0.2, e.shape)
        rows = [dict(cell_type=c, program=p, effect=e[i, j], lo=e[i, j] - w[i, j], hi=e[i, j] + w[i, j],
                     called=abs(e[i, j]) > w[i, j], intrinsic=e[i, j] * 0.7, composition=e[i, j] * 0.3)
                for i, c in enumerate(types) for j, p in enumerate(progs)]
        fr = pd.DataFrame(rng.dirichlet(np.ones(len(types)) * 2, 326), columns=types)
        fr.insert(0, "group", ["PTB"] * 296 + ["Healthy"] * 30)
        out[v] = (pd.DataFrame(rows), {}, fr)
    return out, None


def load():
    if DEMO:
        return demo_tables()
    out, fr = {}, None
    for v, (d, _) in VERSIONS.items():
        p = os.path.join(BASE, d)
        if not os.path.exists(os.path.join(p, "effects.csv")):
            print("missing", p)
            continue
        info = json.load(open(os.path.join(p, "info.json")))
        out[v] = (pd.read_csv(os.path.join(p, "effects.csv")), info,
                  pd.read_csv(os.path.join(p, "fractions.csv"), index_col=0))
    return out, fr


# ---- per version ----------------------------------------------------------------------------------------------
def fig_version(v, tab, info, vmax):
    label = VERSIONS[v][1]
    ax = plots.plot_effect_heatmap(tab, vmax=vmax, title="%s\nPTB − Healthy (no run correction)\n" % label)
    ax.figure.set_size_inches(6.5, 7.5)
    save(ax.figure, "%s_effects_heatmap" % v)
    fig = plots.plot_effect_intervals(tab, ncols=4)
    fig.suptitle("%s: change per cell type with 90%% interval (filled = called)" % label, x=0.01, y=1.03, ha="left",
                 fontsize=10, color=INK)
    save(fig, "%s_effects_intervals" % v)
    if {"intrinsic", "composition"} <= set(tab.columns):
        fig_decomposition(v, tab)
        fig_decomposition(v, tab, exclude=("BCR", "TCR"), name="decomposition_innate", conditions=True, ncols=2,
                          order=("TLR", "NLR", "IFNb", "TNF"))


def fig_decomposition(v, tab, exclude=(), name="decomposition", conditions=False, ncols=4, order=None):
    """Per program: each cell type's share of the bulk change, split into cells changed (intrinsic) and more cells
    (composition). Same-sign parts stack away from 0. exclude: programs left out; conditions: label each panel's
    sides with the group that is higher (PTB right, Healthy left)."""
    from matplotlib.ticker import MaxNLocator
    progs = [p for p in dict.fromkeys(tab.program) if p not in set(exclude)]
    if order:                                                   # listed programs first, in that order
        progs = [p for p in order if p in progs] + [p for p in progs if p not in order]
    types = list(dict.fromkeys(tab.cell_type))
    ncols = min(ncols, len(progs))
    nrows = int(np.ceil(len(progs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.75 * ncols, (0.22 * len(types) + (0.55 if conditions else 0.3))
                                                    * nrows + 0.9), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    y = np.arange(len(types))
    for ax, p in zip(axes, progs):
        t = tab[tab.program == p].set_index("cell_type").reindex(types)
        I, K = t["intrinsic"].fillna(0).values, t["composition"].fillna(0).values
        same = np.sign(I) == np.sign(K)
        ax.barh(y, K, color=MORE, height=0.72, edgecolor="white", linewidth=0.6)
        ax.barh(y, I, left=np.where(same, K, 0), color=CHANGED, height=0.72, edgecolor="white", linewidth=0.6)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_title(p, fontsize=9, color=INK, loc="left")
        ax.tick_params(axis="x", labelsize=7, colors=INK2)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.xaxis.set_major_locator(MaxNLocator(4, symmetric=False))
        ax.grid(axis="x", color=GRID, lw=0.5)
        lo, hi = ax.get_xlim()
        m = max(abs(lo), abs(hi))
        ax.set_xlim(-m, m)                                      # 0 centred, so the condition sides are equal
        if conditions:
            for x, txt, col, ha in ((0.0, "← higher in Healthy", HV_C, "left"), (1.0, "higher in PTB →", PTB_C, "right")):
                ax.annotate(txt, (x, 0), xycoords="axes fraction", xytext=(0, -22), textcoords="offset points",
                            fontsize=7, color=col, ha=ha, va="top")   # fixed distance below the axis
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(types, fontsize=7.5)
    axes[0].set_ylim(len(types) - 0.5, -0.5)                    # first type on top (shared axis: set once)
    for ax in axes[len(progs):]:
        ax.set_visible(False)
    fig.legend([Patch(color=CHANGED), Patch(color=MORE)], ["cells changed (activity per cell)",
                                                           "more cells (composition)"],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.text(0.0, -0.01, "%s. Each bar: that cell type's part of the bulk program change, PTB − Healthy, split into "
             "a change of activity within the cells and a change in how many cells there are.%s" % (
                 VERSIONS[v][1], " %s not shown." % " and ".join(exclude) if exclude else ""),
             fontsize=7.5, color=INK2, va="top", wrap=True)
    fig.tight_layout()
    save(fig, "%s_%s" % (v, name))


def fig_fractions(fr, ver):
    types = [c for c in fr.columns if c != "group"]
    order = list(fr[types].mean().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(order))
    rng = np.random.default_rng(0)
    for k, (g, col, dx) in enumerate((("Healthy", HV_C, -0.18), ("PTB", PTB_C, 0.18))):
        sub = fr[fr.group == g]
        for i, c in enumerate(order):
            v = sub[c].values
            ax.scatter(x[i] + dx + rng.uniform(-0.07, 0.07, len(v)), v, s=4, color=col, alpha=0.35, lw=0)
            q = np.percentile(v, [25, 50, 75])
            ax.plot([x[i] + dx] * 2, [q[0], q[2]], color=col, lw=2.2, solid_capstyle="round")
            ax.plot(x[i] + dx, q[1], "o", ms=4.5, mfc="white", mec=col, mew=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=40, ha="right")
    ax.set_ylabel("read fraction")
    ax.grid(axis="y", color=GRID, lw=0.5)
    n = fr.group.value_counts()
    ax.legend([Line2D([], [], color=HV_C, marker="o", mfc="white", lw=2), Line2D([], [], color=PTB_C, marker="o",
                                                                                   mfc="white", lw=2)],
              ["Healthy (n=%d)" % n.get("Healthy", 0), "PTB (n=%d)" % n.get("PTB", 0)], frameon=False, loc="upper right")
    ax.set_title("%s: cell-type read fractions (PALM-lite; dot = median, bar = IQR)" % VERSIONS[ver][1], loc="left",
                 fontsize=10)
    save(fig, "%s_fractions" % ver)


# ---- comparison -----------------------------------------------------------------------------------------------
def compare(tabs):
    cmap = {"refs": L2_TO_BROAD, "coarse": EIGHT_TO_FIVE}.get(FIGSET, {t: t for t in tabs[A][0].cell_type})
    ta = tabs[A][0][tabs[A][0].cell_type.isin(cmap)].copy()
    ta["broad"] = ta.cell_type.map(cmap)
    tb = tabs[B][0].rename(columns={"cell_type": "broad"})
    cols = ["effect", "lo", "hi", "called"]
    m = ta.merge(tb[["broad", "program"] + cols], on=["broad", "program"], suffixes=("_v2", "_v4g"))
    m["cell_type"] = np.where(m.cell_type == m.broad, m.cell_type, m.cell_type + " / " + m.broad)
    m = m.drop(columns="broad").set_index(["cell_type", "program"])
    m["agreement"] = np.select([m.called_v2 & m.called_v4g, m.called_v2, m.called_v4g], ["both", "%s only" % A, "%s only" % B],
                               "neither")
    sign_flip = m.called_v2 & m.called_v4g & (np.sign(m.effect_v2) != np.sign(m.effect_v4g))
    m.loc[sign_flip, "agreement"] = "both"
    m.reset_index().to_csv(os.path.join(FIG, "compare.csv"), index=False)
    r = np.corrcoef(m.effect_v2, m.effect_v4g)[0, 1]
    counts = m.agreement.value_counts()
    print("r(%s, %s) %.3f; calls: %s; sign flips among both-called: %d" % (A, B, r, counts.to_dict(),
                                                                         int(sign_flip.sum())))

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lim = np.nanpercentile(np.abs(m[["lo_v2", "hi_v2", "lo_v4g", "hi_v4g"]].values), 90) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.axhline(0, color=GRID, lw=0.6); ax.axvline(0, color=GRID, lw=0.6)
    for k in ("neither", "%s only" % A, "%s only" % B, "both"):
        s = m[m.agreement == k]
        if s.empty:
            continue
        z = 3 if k == "both" else 2
        ax.errorbar(s.effect_v2, s.effect_v4g, xerr=[s.effect_v2 - s.lo_v2, s.hi_v2 - s.effect_v2],
                    yerr=[s.effect_v4g - s.lo_v4g, s.hi_v4g - s.effect_v4g], fmt="none", ecolor=AGREE[k],
                    alpha=0.35 if k == "neither" else 0.6, lw=0.6, zorder=z)
        ax.scatter(s.effect_v2, s.effect_v4g, s=16, color=AGREE[k], edgecolor="white", lw=0.4, zorder=z + 1,
                   label="%s (%d)" % (k, len(s)))
    big = m[(m.agreement != "neither")].copy()
    big["mag"] = big[["effect_v2", "effect_v4g"]].abs().max(axis=1)
    for k, ((c, p), row) in enumerate(big.sort_values("mag", ascending=False).head(4).iterrows()):
        dy = (12 if k % 2 == 0 else -14) * (1 if row.effect_v4g >= 0 else -1)
        ax.annotate("%s · %s" % (c, p), (row.effect_v2, row.effect_v4g), xytext=(-60 if row.effect_v2 > 0 else 8, dy),
                    textcoords="offset points", fontsize=6.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("%s: effect (PTB − Healthy)" % VERSIONS[A][1], fontsize=8)
    ax.set_ylabel("%s: effect" % VERSIONS[B][1], fontsize=8)
    ax.set_aspect("equal")
    ax.legend(title="called (90% range excludes 0)", frameon=False, fontsize=7.5, title_fontsize=7.5,
              loc="upper left")
    ax.set_title("%s, %d matched pairs (r = %.2f)" % (COMPARE_TITLE, len(m), r), loc="left", fontsize=9.5)
    save(fig, "compare_scatter")

    types = list(dict.fromkeys(m.index.get_level_values(0)))
    progs = list(dict.fromkeys(tabs[A][0].program))
    AG = m.reset_index().pivot(index="cell_type", columns="program", values="agreement").reindex(index=types,
                                                                                                columns=progs)
    fig, ax = plt.subplots(figsize=(4.8, 0.28 * len(types) + 1.2))
    for i, c in enumerate(types):
        for j, p in enumerate(progs):
            k = AG.loc[c, p]
            ax.add_patch(plt.Rectangle((j, i), 0.92, 0.92, color=AGREE.get(k, "#ffffff"), lw=0))
            if k != "neither":
                e2, e4 = m.loc[(c, p), "effect_v2"], m.loc[(c, p), "effect_v4g"]
                ax.text(j + 0.46, i + 0.46, "%+.2f\n%+.2f" % (e2, e4), ha="center", va="center", fontsize=5.5,
                        color="white" if k == "both" else INK)
    ax.set_xlim(0, len(progs)); ax.set_ylim(len(types), 0)
    ax.set_xticks(np.arange(len(progs)) + 0.46); ax.set_xticklabels(progs, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(types)) + 0.46); ax.set_yticklabels(types)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.legend([Patch(color=AGREE[k]) for k in AGREE], ["%s (%d)" % (k, counts.get(k, 0)) for k in AGREE],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5)
    fig.text(0.0, -0.02, "Called pairs show both effects (top %s, bottom %s)." % (A, B), fontsize=7, color=INK2)
    save(fig, "compare_calls")


def main():
    tabs, fr = load()
    if not tabs:
        return
    vmax = max(np.nanmax(np.abs(t[0].effect)) for t in tabs.values())
    for v, (t, info, f) in tabs.items():
        fig_version(v, t, info, vmax)
        fig_fractions(f, v)
    if len(tabs) == 2:
        compare(tabs)



# ---- innate programs with type I and type II IFN (user request 2026-11-12) -----------------------------------
INNATE7 = [("TLR", "TLR signaling", "Reactome"), ("NLR", "NLR signaling", "Reactome"),
           ("TNF", "TNF via NF-κB", "Hallmark"), ("IFNI", "Type I IFN (α/β)", "Hallmark"),
           ("IFNg", "IFN-γ", "Hallmark")]
INNATE8 = INNATE7[:3] + [("IFNI", "Type I IFN only", "Hallmark"), ("IFNg", "IFN-γ only", "Hallmark"),
                         ("ISG", "Shared ISGs (both IFNs)", "Hallmark")]
KEY_GENES = {"TLR": ["TLR2", "TLR4", "MYD88", "TLR8"], "NLR": ["NLRP3", "NOD2", "CASP1", "PYCARD"],
             "TNF": ["TNF", "NFKBIA", "IL1B", "CXCL8"],
             "IFNI": ["ISG15", "IFIT1", "MX1", "IFI44L", "OAS1", "IFITM1", "SAMD9"],
             "IFNg": ["IRF1", "CXCL10", "CXCL9", "CIITA", "IDO1", "STAT1"],
             "ISG": ["ISG15", "MX1", "IFI44L", "RSAD2"]}


KEY_GENES_R = dict(KEY_GENES, IFNI=["ISG15", "IFIT1", "MX1", "RSAD2", "IFI27"],
                   IFNg=["CIITA", "GBP1", "FCGR1A", "HLA-DRA"], ISG=["STAT1", "IRF1", "OAS1", "SOCS1"])
NOTE_HALLMARK = ("Programs: Reactome member sets (TLR, NLR), Hallmark response sets (TNF, IFN). The Hallmark IFN-α and "
                 "IFN-γ sets are split by membership into genes only in one and genes in both (shared ISGs). Gene count: "
                 "program genes in the model.")
NOTE_REACTOME = ("Programs: Reactome member sets (TLR, NLR, IFN), Hallmark response set (TNF). The Reactome "
                 "'Interferon alpha/beta signaling' and 'Interferon gamma signaling' sets are split by membership into "
                 "genes only in one and genes in both (shared). Gene count: program genes in the model.")


def fig_innate7(d="combat_immune_response7_anchored", name="anchored_decomposition_innate_ifn", progs=None,
                keys=None, note=None):
    """Slide figure: per innate program, each cell type's part of the PTB − Healthy bulk change, split into cells
    changed and more cells; panel title = program, its gene count in the model and 3 key member genes."""
    from matplotlib.ticker import MaxNLocator
    p_ = os.path.join(BASE, d)
    tab = pd.read_csv(os.path.join(p_, "effects.csv"))
    info = json.load(open(os.path.join(p_, "info.json")))
    ngen, glist = info.get("program_genes", {}), info.get("program_genes_list", {})
    types = list(dict.fromkeys(tab.cell_type))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), sharey=True)
    axes = axes.ravel()
    y = np.arange(len(types))
    progs = progs or INNATE7
    six = len(progs) == 6
    for ax, (p, label, src) in zip(axes, progs):
        t = tab[tab.program == p].set_index("cell_type").reindex(types)
        I, K = t["intrinsic"].fillna(0).values, t["composition"].fillna(0).values
        same = np.sign(I) == np.sign(K)
        ax.barh(y, K, color=MORE, height=0.72, edgecolor="white", linewidth=0.6)
        ax.barh(y, I, left=np.where(same, K, 0), color=CHANGED, height=0.72, edgecolor="white", linewidth=0.6)
        ax.axvline(0, color=MUTED, lw=0.6)
        kg = (keys or KEY_GENES)[p]
        kshow = [g for g in kg if g in set(glist.get(p, kg))][:3]
        ax.set_title(label, fontsize=11.5, color=INK, loc="left", fontweight="bold", pad=20)
        ax.text(0, 1.015, "%s genes · %s, …" % (ngen.get(p, "?"), ", ".join(kshow)), transform=ax.transAxes,
                fontsize=8.5, color=INK2, va="bottom", style="italic")
        ax.tick_params(axis="x", labelsize=7.5, colors=INK2)
        ax.xaxis.set_major_locator(MaxNLocator(4, symmetric=False))
        ax.grid(axis="x", color=GRID, lw=0.5)
        lo, hi = ax.get_xlim(); m = max(abs(lo), abs(hi)); ax.set_xlim(-m, m)
        for x, txt, col, ha in ((0.0, "← higher in Healthy", HV_C, "left"), (1.0, "higher in PTB →", PTB_C, "right")):
            ax.annotate(txt, (x, 0), xycoords="axes fraction", xytext=(0, -20), textcoords="offset points",
                        fontsize=7.5, color=col, ha=ha, va="top")
    axes[0].set_yticks(y); axes[0].set_yticklabels(types, fontsize=9)
    axes[0].set_ylim(len(types) - 0.5, -0.5)
    axes[3].tick_params(axis="y", labelleft=True)
    axes[3].set_yticks(y); axes[3].set_yticklabels(types, fontsize=9)
    if six:                                 # every slot used: legend on top, inputs in the footer
        fig.legend([Patch(color=CHANGED), Patch(color=MORE)], ["cells changed (activity per cell)",
                                                               "more cells (composition)"],
                   loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=10.5)
        fig.text(0.0, -0.045, note or NOTE_HALLMARK, fontsize=7.5, color=INK2, va="top", wrap=True)
        fig.text(0.0, -0.01, "PTB − Healthy, anchored offsets, 8-type reference. Each bar: that cell type's part of "
                 "the bulk program change, split into a change of activity within the cells and a change in how "
                 "many cells there are.", fontsize=7.5, color=INK2, va="top", wrap=True)
        fig.tight_layout(h_pad=2.6)
        save(fig, name)
        return
    # sixth slot: legend and the inputs
    lx = axes[5]; lx.axis("off")
    lx.legend([Patch(color=CHANGED), Patch(color=MORE)], ["cells changed\n(activity per cell)",
                                                           "more cells\n(composition)"],
              loc="upper left", frameon=False, fontsize=10, handlelength=1.4, handleheight=1.4, labelspacing=1.0)
    src = ("Reactome member sets (TLR, NLR, IFN)\nand a Hallmark response set (TNF)." if note else
           "Reactome member sets (TLR, NLR)\nand Hallmark response sets (TNF, IFN).")
    lx.text(0.0, 0.32, "Programs: %s\nGene count: program genes in the model.\nType I IFN and IFN-γ share %d genes\n"
            "in the model." % (src, len(set(glist.get("IFNI", [])) & set(glist.get("IFNg", [])))),
            transform=lx.transAxes, fontsize=8, color=INK2, va="top", linespacing=1.4)
    fig.text(0.0, -0.01, "PTB − Healthy, anchored offsets, 8-type reference. Each bar: that cell type's part of the "
             "bulk program change, split into a change of activity within the cells and a change in how many cells "
             "there are.", fontsize=7.5, color=INK2, va="top", wrap=True)
    fig.tight_layout(h_pad=2.6)
    save(fig, name)



def fig_heatmap_innate(d="combat_immune_reactome_ifn7_anchored", name="anchored_heatmap_innate_ifn_reactome",
                       progs=(("TLR", "TLR"), ("NLR", "NLR"), ("TNF", "TNF"), ("IFNI", "Type I IFN"), ("IFNg", "IFN-γ"))):
    """Slide heatmap: per-cell change in program activity (PTB − Healthy), cell types x innate programs, value in
    every cell; pairs whose 90% interval excludes 0 are outlined. Column labels carry the gene count in the model."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    p_ = os.path.join(BASE, d)
    tab = pd.read_csv(os.path.join(p_, "effects.csv"))
    info = json.load(open(os.path.join(p_, "info.json")))
    ngen = info.get("program_genes", {})
    types = list(dict.fromkeys(tab.cell_type))
    keys = [k for k, _ in progs]
    Ev = tab.pivot(index="cell_type", columns="program", values="effect").loc[types, keys].values.astype(float)
    Cv = tab.pivot(index="cell_type", columns="program", values="called").loc[types, keys].values.astype(bool)
    lim = float(np.ceil(np.nanmax(np.abs(Ev)) * 10) / 10)
    cmap = LinearSegmentedColormap.from_list("div", [HV_C, "#f4f3ef", PTB_C])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(Ev, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(types)):
        for j in range(len(keys)):
            dark = abs(Ev[i, j]) > 0.6 * lim
            ax.text(j, i, "%+.2f" % Ev[i, j], ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if dark else INK, fontweight="bold" if Cv[i, j] else "normal")
            if Cv[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=2.2))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["%s\n%s genes" % (lab, ngen.get(k, "?")) for k, lab in progs], fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(types))); ax.set_yticklabels(types, fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("change in activity per cell (PTB − Healthy)", fontsize=9, color=INK2)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, colors=INK2)
    n_called = int(Cv.sum())
    fig.text(0.0, 0.02, ("Outlined: 90%% interval excludes 0 (%d of %d pairs). " % (n_called, Cv.size) if n_called else
                         "No pair's 90%% interval excludes 0 (0 of %d). " % Cv.size) +
             "Anchored offsets, 8-type reference; programs: Reactome member sets (TLR, NLR, IFN), Hallmark (TNF); "
             "fitted jointly with BCR and TCR (not shown).", fontsize=7.5, color=INK2, va="top", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, name)


if __name__ == "__main__":
    if "innate7" in sys.argv:              # FIGSET=response python ptb_figures.py innate7
        fig_innate7()
    elif "heat7r" in sys.argv:             # heatmap for the whole-Reactome-IFN run
        fig_heatmap_innate()
    elif "innate7r" in sys.argv:           # whole Reactome IFN sets, not split (immune_reactome_ifn7)
        fig_innate7("combat_immune_reactome_ifn7_anchored", "anchored_decomposition_innate_ifn_reactome",
                    INNATE7[:3] + [("IFNI", "Type I IFN (α/β)", "Reactome"), ("IFNg", "IFN-γ", "Reactome")],
                    dict(KEY_GENES_R, IFNI=["ISG15", "IFIT1", "MX1", "STAT1"], IFNg=["CIITA", "GBP1", "IRF1", "STAT1"]),
                    "Programs: Reactome member sets (TLR, NLR, IFN α/β signaling, IFN-γ signaling), Hallmark response "
                    "set (TNF). The two IFN sets are used whole and share 28 genes. Gene count: program genes in the "
                    "model.")
    elif "innate8r" in sys.argv:           # IFN split from Reactome (immune_reactome_ifn8)
        fig_innate7("combat_immune_reactome_ifn8_anchored", "anchored_decomposition_innate_ifn_split_reactome",
                    INNATE8[:5] + [("ISG", "Shared (both IFNs)", "Reactome")], KEY_GENES_R, NOTE_REACTOME)
    elif "innate8" in sys.argv:            # IFN split by membership (immune_response8)
        fig_innate7("combat_immune_response8_anchored", "anchored_decomposition_innate_ifn_split", INNATE8)
    else:
        main()
