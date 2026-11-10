"""Benchmark figures for a split set -> docs/figures/heldout/<tag>/*.png (+ .svg).

Built on the development splits 20-29 (layout only, PALM_EST=auto_v2 there because v4g was not run on every
dataset), then rendered unchanged on the held-out splits 40-49 (PALM_EST=v4g) after heldout_v4_compare.py.
Reads saved rows only; no fitting, no rescoring rules of its own (scores come from causal_loop.score).

  calling     which cells changed, per dataset: (a) false calls on null pairs (|true| < 0.05, all programs),
              (b) real changes found (drug's program, |true| >= 0.1, right sign). PALM (90% range) vs current
              practice (deconvolution -> per-type score -> t-test, BH q < 0.1). Dot = mean over splits,
              line = 90% percentile bootstrap interval over splits.
  accuracy    PALM's estimators on the same splits: (a) r of estimated vs true change, (b) coverage of the 90%
              range (target 0.85). PALM_EST vs freeze v2 (auto_v2) vs no correction (diff).
  transport   the same two panels for the transport perturbations (lane e; PALM only).
  heatmap     benchmark_heatmaps.calling on these splits (funky heatmap).

Usage: python heldout_figures.py 20-29 [tag]      PALM_EST=v4g python heldout_figures.py 40-49 heldout_v4
"""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import practice_vs_palm as pv

SPLITS = cl.db._splits(sys.argv[1] if len(sys.argv) > 1 else "20-29")
TAG = sys.argv[2] if len(sys.argv) > 2 else "dev_%d-%d" % (min(SPLITS), max(SPLITS))
FIG = os.path.join(ROOT, "docs", "figures", "heldout", TAG)
PALM_EST = pv.PALM_EST

# datasets in reading order: cell lines (known biology), PBMC (paired donors), cytokines (Parse)
DS = {"mcf_ida": "McFarland idasanutlin", "mcf_tram": "McFarland trametinib",
      "tahoe_rmc": "Tahoe RMC-6236", "tahoe_enc": "Tahoe encorafenib", "tahoe_tram": "Tahoe trametinib",
      "tahoe_gem": "Tahoe gemcitabine", "kang": "Kang IFN-β", "zhang": "Zhang chemotherapy (null)",
      "parse_ifng": "Parse IFN-γ", "parse_ifnb": "Parse IFN-β", "parse_gmcsf": "Parse GM-CSF",
      "parse_il4": "Parse IL-4", "parse_tnf": "Parse TNF"}
TRANSPORT = {"parse_ifna": "Parse IFN-α", "parse_il10": "Parse IL-10 (one donor)", "parse_il1b": "Parse IL-1β",
             "parse_il6": "Parse IL-6", "tahoe_afa": "Tahoe afatinib", "tahoe_alp": "Tahoe alpelisib",
             "tahoe_bor": "Tahoe bortezomib", "tahoe_evr": "Tahoe everolimus", "tahoe_ptx": "Tahoe paclitaxel"}

# categorical slots in fixed order (dataviz reference palette, validated there); marker = second encoding
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARK = ["o", "s", "D", "^", "v", "P", "X"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
                     "svg.fonttype": "none", "savefig.dpi": 300, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.04})
MM = 1 / 25.4                     # Nature Methods: 89 mm single, 183 mm double column


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, name + "." + ext))
    plt.close(fig)
    print("wrote", os.path.join(FIG, name))


def boot_ci(v, level=0.9, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    if len(v) == 1:
        return v[0], np.nan, np.nan
    m = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    a = (1 - level) / 2
    return v.mean(), np.quantile(m, a), np.quantile(m, 1 - a)


# ---- data ---------------------------------------------------------------------------------------------------
def calling_rows():
    rows = [r for f in glob.glob(os.path.join(pv.OUT, "*.json")) for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    df = df[df.split.isin(SPLITS) & (df.kind == "practice") & (df.rule == "bh10")]
    lab = {"instaprism": "InstaPrism + t-test", "cibersort": "CIBERSORT + t-test", "reallocation": "NNLS + t-test",
           "toast": "TOAST + t-test", "tca": "TCA + t-test", "naive": "Bulk score + t-test"}
    df = df[df.method.isin(lab)].assign(label=lambda d: d.method.map(lab))
    palm = estimator_rows(DS, [PALM_EST]).assign(label="PALM")
    return pd.concat([palm, df], ignore_index=True), ["PALM"] + list(lab.values())


def _load(dsn, split, name):
    """Saved row (with arrays) for an estimator; Kang/Zhang use the paired-design file where it exists.
    v4g missing on a development split (< 40): rebuilt offline exactly as the frozen gate does
    (scripts/experiments/v4_gate.py): v4 stacked (v4k) where core.gls.gate_statistics passes, else auto_v2.
    The held-out splits always use the frozen run's own v4g file."""
    pair = dsn in ("kang", "zhang")
    for nm in (["%s_pair" % name, name] if pair else [name]):
        f = os.path.join(cl.OUT, "%s__s%d__%s.json" % (dsn, split, nm))
        if os.path.exists(f):
            return json.load(open(f))
    if name == "v4g" and split < 40:
        from program2cell.core.gls import gate_statistics, GATE_HETEROGENEITY, GATE_AGREEMENT
        fk = os.path.join(cl.OUT, "%s__s%d__v4k%s.json" % (dsn, split, "_pair" if pair else ""))
        f2 = os.path.join(cl.OUT, "%s__s%d__auto_v2.json" % (dsn, split))
        if os.path.exists(fk) and os.path.exists(f2):
            k, v2 = json.load(open(fk)), json.load(open(f2))
            a, b = k["arrays"], v2["arrays"]
            g = gate_statistics(*(np.asarray(a[x], float) for x in ("effects", "lo", "hi")),
                                np.asarray(b["effects"], float))
            use_v4 = g["heterogeneity"] >= GATE_HETEROGENEITY and g["agreement"] >= GATE_AGREEMENT
            return dict(v2 if not use_v4 else k, gate="offline:" + ("v4" if use_v4 else "v2"))
    return None


def estimator_rows(datasets, names):
    rows = []
    for dsn in datasets:
        for name in names:
            for split in SPLITS:
                s = _load(dsn, split, name)
                if s is None:
                    continue
                a = s.pop("arrays")
                s.update(cl.score(*(np.asarray(a[k], float) for k in ("effects", "lo", "hi")),
                                  np.asarray(a["calls"], bool), np.asarray(a["beta"], float),
                                  a["cell_types"], a["programs"], cl.KNOWN.get(dsn)))
                rows.append(dict(s, label=name, dataset=dsn, split=split))
    return pd.DataFrame(rows)


# ---- drawing ------------------------------------------------------------------------------------------------
def dot_matrix(ax, df, metric, datasets, methods, xlim, xlabel, ref=None, better=None):
    """Rows = datasets; per method a dot (mean over splits) and a 90% bootstrap line, offset within the row."""
    k = len(methods)
    off = (np.arange(k) - (k - 1) / 2) * min(0.8 / k, 0.13)
    names = list(datasets)
    for i in range(len(names)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f4f3f0", lw=0, zorder=0)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for j, m in enumerate(methods):
        for i, dsn in enumerate(names):
            sub = df[(df.label == m) & (df.dataset == dsn)]
            if metric not in sub or not len(sub):
                continue
            mu, lo, hi = boot_ci(sub[metric].values)
            if not np.isfinite(mu):
                continue
            y = i + off[j]
            if np.isfinite(lo):
                ax.plot([lo, hi], [y, y], color=PAL[j], lw=1.0, solid_capstyle="round", zorder=2)
            ax.plot(mu, y, MARK[j], ms=3.6, mfc=PAL[j], mec="white", mew=0.5, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([datasets[d] for d in names])
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel + ("  (%s is better)" % better if better else ""))
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)


def legend(fig, methods):
    top = 1 - 13 * MM / fig.get_figheight()          # 13 mm band for the legend, independent of figure height
    fig.subplots_adjust(top=top, wspace=0.08)
    y = top + 4 * MM / fig.get_figheight()
    h = [Line2D([], [], color=PAL[j], marker=MARK[j], ms=4, mec="white", mew=0.5, lw=1.0) for j in range(len(methods))]
    fig.legend(h, methods, loc="lower center", bbox_to_anchor=(0.5, y), ncol=min(len(methods), 4), frameon=False,
               handlelength=1.6, columnspacing=1.2)


def note(fig, text, y=-0.02):
    fig.text(0.0, y, text, ha="left", va="top", fontsize=6.5, color=INK2, wrap=True)


def panel(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


# ---- figures ------------------------------------------------------------------------------------------------
def fig_calling():
    df, methods = calling_rows()
    methods = [m for m in methods if (df.label == m).any()]
    datasets = {d: n for d, n in DS.items() if (df.dataset == d).any()}
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, (22 + 7 * len(datasets)) * MM), sharey=True)
    dot_matrix(axes[0], df, "fpr_null", datasets, methods, (-0.02, 1.0), "Null pairs called", ref=0.1,
               better="lower")
    dot_matrix(axes[1], df, "tpr_t", datasets, methods, (-0.02, 1.02), "Real changes found", better="higher")
    panel(axes[0], "a"); panel(axes[1], "b")
    legend(fig, methods)
    note(fig, "a, share of cell type × program pairs with no true change (|true| < 0.05) called changed; dashed line "
              "at 0.1. b, share of the drug's program changes (|true| ≥ 0.1) called with the right sign (blank: no "
              "drug program pre-specified: Zhang, Parse). PALM (%s): 90%% range excludes 0. Tools: per-type score, "
              "t-test (paired where donors have both arms), BH q < 0.1. Dot: mean over %d splits (%d–%d); line: 90%% "
              "bootstrap interval over splits." % (PALM_EST, len(SPLITS), min(SPLITS), max(SPLITS)))
    save(fig, "calling")


def _accuracy(datasets, name, title_note):
    # held-out v4 decision (2026-11-10): freeze v2 is the Paper 1 method, gated v4 secondary; both shown
    names = ["auto_v2", "v4g", "diff"]
    df = estimator_rows(datasets, names)
    if df.empty:
        print("no rows for", name)
        return
    lab = {"auto_v2": "PALM (freeze v2)", "v4g": "PALM gated v4 (secondary)", "diff": "No correction"}
    df["label"] = df.label.map(lab)
    methods = [lab[n] for n in names if (df.label == lab[n]).any()]
    datasets = {d: n for d, n in datasets.items() if (df.dataset == d).any()}
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, (22 + 6 * len(datasets)) * MM), sharey=True)
    dot_matrix(axes[0], df, "r_t", datasets, methods, (-0.2, 1.0), "r, estimated vs true change", better="higher")
    dot_matrix(axes[1], df, "cov", datasets, methods, (0.0, 1.02), "Coverage of the 90% range", ref=0.85)
    panel(axes[0], "a"); panel(axes[1], "b")
    legend(fig, methods)
    note(fig, "a, Pearson r over all cell type × program pairs (blank where the truth has no change ≥ 0.1). b, share "
              "of pairs whose 90%% range contains the true change; dashed line at the 0.85 target. %s Dot: mean over "
              "splits %d–%d; line: 90%% bootstrap interval over splits." % (title_note, min(SPLITS), max(SPLITS)))
    save(fig, name)


def fig_heatmap():
    import benchmark_heatmaps as bh
    bh.FIG = FIG
    bh.calling(splits=tuple(SPLITS), name="heatmap", split_note="Splits %d–%d." % (min(SPLITS), max(SPLITS)))


if __name__ == "__main__":
    print("splits", SPLITS, "PALM_EST", PALM_EST, "->", FIG)
    fig_calling()
    _accuracy(DS, "accuracy", "Kang and Zhang: paired design (donors).")
    _accuracy(TRANSPORT, "transport", "Perturbations never run in development (amendment 2026-11-09); PALM only.")
    fig_heatmap()
