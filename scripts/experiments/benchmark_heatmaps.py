"""Benchmark summary heatmaps (scIB / funky-heatmap style) -> docs/figures/talk/bench_*.png (+ .svg).

Rows = methods, columns = metrics per dataset in groups; each cell is a bar whose length and colour are the
min-max scaled score in that column (1 = best, direction-aware), top-3 ranks printed in the bar. Overall =
mean over groups of the group mean score; methods sorted by it. Missing = not run (grey dash).

  bench_calling   which cells changed: negative control (unchanged pairs flagged), positive control (real
                  changes found), effect accuracy (r with truth, responder AUROC). practice_vs_palm rows,
                  BH q < 0.1 for tools; PALM 90% range. Development splits 0-2.
  bench_deconv    proportions (fraction RMSE) and per-type activity (Spearman across samples; leakage) on six
                  datasets, every competitor with its own output. dev_bench rows, development splits 0-9.
Usage: python benchmark_heatmaps.py
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import colormaps, colors as mcolors
from scipy.stats import rankdata

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import practice_vs_palm as pv

FIG = os.path.join(ROOT, "docs", "figures", "talk")
C_PALM = "#00798c"
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "svg.fonttype": "none", "savefig.dpi": 300,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.1})
DS = {"mcf_ida": "Idasanutlin", "mcf_tram": "Trametinib", "kang": "Interferon-β", "zhang": "Chemotherapy",
      "pbmc": "PBMC", "o1k": "OneK1K"}
GROUP_CMAP = ["Blues", "Greens", "Purples", "Oranges", "RdPu"]


def scaled(col, higher_better, fixed=None):
    """Score in [0.08, 1], 1 = best: min-max within the column, or on a fixed range (lo, hi) for rates, so a
    1% vs 0% difference is not stretched to the full scale."""
    v = np.asarray(col, float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return v
    lo, hi = fixed if fixed is not None else (np.nanmin(v), np.nanmax(v))
    v = np.clip(v, lo, hi)
    s = np.ones_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)
    s = s if higher_better else 1 - s
    return np.where(ok, 0.08 + 0.92 * s, np.nan)


def ranks(col):
    """Ranks 1-3 (ties share the rank); a rank shared by more than 2 methods is not printed."""
    v = np.asarray(col, float)
    out = np.zeros(len(v), int)
    ok = np.isfinite(v)
    r = np.zeros(len(v))
    r[ok] = rankdata(-np.round(v[ok], 3), method="min")
    for k in (1, 2, 3):
        if 0 < (r == k).sum() <= 2:
            out[r == k] = k
    return out


def funky(table, groups, info, name, title=None, palm_rows=("PALM",), note=None, fixed=None):
    """table: DataFrame (methods x metric columns, raw values); groups: [(group label, [(col, short label,
    higher_better)])]; info: DataFrame (methods x info columns, strings: 'yes'/'no'/text)."""
    cols = [c for _, g in groups for c, _, _ in g]
    fx = fixed or {}
    S = pd.DataFrame({c: scaled(table[c], hb, fx.get(c)) for _, g in groups for c, _, hb in g}, index=table.index)
    gm = pd.DataFrame({gl: S[[c for c, _, _ in g]].mean(1, skipna=True) for gl, g in groups})
    overall = gm.mean(1)
    order = overall.sort_values(ascending=False).index
    S, table, info, overall = S.loc[order], table.loc[order], info.loc[order], overall.loc[order]
    n, ninfo = len(order), info.shape[1]
    # x layout: name | info columns | overall | gap | groups (gap between)
    x = 0.0
    xs_info = []
    for _ in range(ninfo):
        xs_info.append(x); x += 0.9
    x += 0.3
    x_over = x; x += 1.6
    xs, gx = [], []
    for gi, (gl, g) in enumerate(groups):
        x += 0.45
        start = x
        for _ in g:
            xs.append(x); x += 1.0
        gx.append((start, x - 1.0))
    width = x
    fig, ax = plt.subplots(figsize=(3.2 + 0.42 * width, 0.9 + 0.48 * n + 2.0))
    ax.set_xlim(-0.6, width + 0.2)
    ax.set_ylim(n + 0.2, -2.2)
    ax.axis("off")
    # zebra rows
    for i in range(n):
        if i % 2 == 0:
            ax.add_patch(Rectangle((-0.6, i - 0.5), width + 0.8, 1, color="#f3f4f6", zorder=0, lw=0))
        lab = order[i]
        is_palm = lab in palm_rows or lab.startswith("PALM")
        ax.text(-0.75, i, lab, ha="right", va="center", fontsize=12.5,
                fontweight="bold" if is_palm else "normal", color=C_PALM if is_palm else "#222222")
    # info columns
    for k, (xc, cname) in enumerate(zip(xs_info, info.columns)):
        ax.text(xc + 0.35, n - 0.1, cname, rotation=45, ha="right", va="top", fontsize=11.5)
        for i, v in enumerate(info[cname]):
            if v == "yes":
                ax.text(xc + 0.35, i, "✔", ha="center", va="center", fontsize=15, color="#2e9d4f",
                        fontfamily="DejaVu Sans")
            elif v == "no":
                ax.text(xc + 0.35, i, "✘", ha="center", va="center", fontsize=14, color="#d1495b",
                        fontfamily="DejaVu Sans")
            else:
                ax.add_patch(FancyBboxPatch((xc + 0.05, i - 0.25), 0.6, 0.5, boxstyle="round,pad=0.02",
                                            fc="#e7ebef", ec="none"))
                ax.text(xc + 0.35, i, v, ha="center", va="center", fontsize=10.5, color="#333333")
    # overall
    ov = overall.values
    so = (ov - ov.min()) / max(ov.max() - ov.min(), 1e-12)
    ro = ranks(ov)
    cmap_o = colormaps["Greys"]
    for i, s in enumerate(so):
        w = 0.15 + 1.25 * s
        ax.add_patch(Rectangle((x_over, i - 0.32), w, 0.64, color=cmap_o(0.35 + 0.6 * s), lw=0))
        if ro[i]:
            ax.text(x_over + w - 0.08, i, str(ro[i]), ha="right", va="center", color="white", fontsize=11,
                    fontweight="bold")
    ax.text(x_over + 0.2, n - 0.1, "Overall score", rotation=45, ha="right", va="top", fontsize=11.5,
            fontweight="bold")
    # metric columns
    j = 0
    for gi, (gl, g) in enumerate(groups):
        cmap = colormaps[GROUP_CMAP[gi % len(GROUP_CMAP)]]
        for c, short, hb in g:
            xc = xs[j]; j += 1
            r_ = ranks(S[c].values)
            for i, s in enumerate(S[c].values):
                if not np.isfinite(s):
                    ax.text(xc + 0.45, i, "–", ha="center", va="center", color="#aaaaaa", fontsize=12)
                    continue
                w = 0.9 * s
                ax.add_patch(Rectangle((xc, i - 0.32), w, 0.64, color=cmap(0.25 + 0.7 * s), lw=0))
                if r_[i]:
                    ax.text(xc + max(w, 0.32) - 0.06, i, str(r_[i]), ha="right", va="center",
                            color="white" if s > 0.45 else "#222222", fontsize=10.5, fontweight="bold")
            ax.text(xc + 0.45, n - 0.1, short, rotation=45, ha="right", va="top", fontsize=11.5)
        a, b = gx[gi]
        ax.add_patch(FancyBboxPatch((a - 0.05, -1.55), b + 0.95 - a, 0.75, boxstyle="round,pad=0.02,rounding_size=0.35",
                                    fc=mcolors.to_rgba(cmap(0.55), 0.25), ec="none"))
        ax.text((a + b + 0.9) / 2, -1.17, gl, ha="center", va="center", fontsize=12.5, fontweight="bold")
    ax.add_patch(FancyBboxPatch((x_over - 0.05, -1.55), 1.5, 0.75, boxstyle="round,pad=0.02,rounding_size=0.35",
                                fc="#dddddd", ec="none"))
    ax.text(x_over + 0.7, -1.17, "Overall", ha="center", va="center", fontsize=12.5, fontweight="bold")
    if title:
        ax.text(-0.75, -2.0, title, ha="left", va="center", fontsize=15, fontweight="bold",
                transform=ax.transData)
    if note:
        ax.text(-0.6, n + 3.3, note, fontsize=10.5, color="#555555", ha="left", va="top", clip_on=False)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, name + "." + ext))
    plt.close(fig)
    print("wrote", name)
    print(pd.DataFrame({"overall": overall.round(3)}).to_string())


# ---- 1. which cells changed --------------------------------------------------------------------------------
def calling(splits=(0, 1, 2)):
    rows = [r for f in glob.glob(os.path.join(pv.OUT, "*.json")) for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    df = df[df.split.isin(splits)]
    names = {("lab_gsva", "lab_gsva"): "Standard workflow (InstaPrism + GSVA)",
             ("practice", "instaprism"): "InstaPrism + t-test", ("practice", "cibersort"): "CIBERSORT + t-test",
             ("practice", "reallocation"): "NNLS + t-test", ("practice", "naive"): "Bulk score + t-test",
             ("palm_module", "instaprism"): "PALM with InstaPrism fractions",
             ("palm_module", "cibersort"): "PALM with CIBERSORT fractions",
             ("palm_module", "reallocation"): "PALM with NNLS fractions"}
    keep = []
    for (kind, m), lab in names.items():
        sub = df[(df.kind == kind) & ((df.method == m) | (kind == "lab_gsva"))]
        sub = sub[sub.rule == ("palm" if kind == "palm_module" else "bh10")]
        keep.append(sub.assign(label=lab))
    palm = pd.DataFrame(pv._palm_rows())
    palm = palm[palm.split.isin(splits)].assign(label="PALM")
    d = pd.concat(keep + [palm])
    g = d.groupby(["label", "dataset"])[["fpr_null", "tpr_t", "r_t", "auroc", "cov"]].mean()
    T = pd.DataFrame(index=sorted(d.label.unique()))
    groups = [("Negative control", []), ("Positive control", []), ("Effect accuracy", []),
              ("Honest ranges", [])]
    for ds in ("mcf_ida", "mcf_tram", "kang", "zhang"):
        T["neg_" + ds] = [g.fpr_null.get((m, ds), np.nan) for m in T.index]
        groups[0][1].append(("neg_" + ds, DS[ds], False))
    for ds in ("mcf_ida", "mcf_tram", "kang"):
        T["pos_" + ds] = [g.tpr_t.get((m, ds), np.nan) for m in T.index]
        groups[1][1].append(("pos_" + ds, DS[ds], True))
    for ds in ("mcf_ida", "mcf_tram", "kang"):
        T["r_" + ds] = [g.r_t.get((m, ds), np.nan) for m in T.index]
        groups[2][1].append(("r_" + ds, "r, " + DS[ds], True))
    T["auroc"] = [g.auroc.get((m, "mcf_ida"), np.nan) for m in T.index]
    groups[2][1].append(("auroc", "Responder AUROC", True))
    for ds in ("mcf_ida", "mcf_tram", "kang", "zhang"):                 # 90% range contains the true change
        T["cov_" + ds] = [g["cov"].get((m, ds), np.nan) for m in T.index]
        groups[3][1].append(("cov_" + ds, DS[ds], True))
    lang = {"PALM": "Py", "Standard workflow (InstaPrism + GSVA)": "R", "InstaPrism + t-test": "R",
            "CIBERSORT + t-test": "R", "NNLS + t-test": "Py", "Bulk score + t-test": "Py"}
    info = pd.DataFrame(index=T.index)
    info["Language"] = [lang.get(m, "Py") for m in T.index]
    info["Corrects leakage"] = ["yes" if m.startswith("PALM") else "no" for m in T.index]
    info["Calibrated ranges"] = ["yes" if m.startswith("PALM") else "no" for m in T.index]
    print(T.round(3).to_string())
    fixed = {c: ((0.5, 1) if c == "auroc" else (0, 1)) for c in T.columns}
    funky(T, groups, info, "bench_calling", fixed=fixed, title="Which cell types changed? Calling cell-type pathway changes",
          note="Negative control: unchanged cell type × pathway pairs flagged (fewer is better). Positive control: "
               "real changes in the drug's pathway (|true| ≥ 0.1) found with the right sign.\nEffect accuracy: "
               "correlation of the estimated with the true change; responder AUROC ranks the 3 p53-intact lines "
               "(idasanutlin).\nHonest ranges: share of 90% ranges containing the true change (tools: t-test "
               "interval; the standard workflow gives none). Tools: BH q < 0.1; PALM: 90% range.\nBars scaled per column (longest = best); "
               "1–3 = rank. Development splits 0–2. \"PALM with X fractions\": PALM's effect step on another "
               "tool's proportions.")


# ---- 2. proportions and per-type activity ------------------------------------------------------------------
def deconv():
    rows = []
    for ds in DS:
        for f in glob.glob(os.path.join(ROOT, "benchmarks", "dev", ds, "s*__*.json")):
            m = re.match(r"s(\d+)__(.+)\.json", os.path.basename(f))
            if not m or int(m.group(1)) >= 10:
                continue
            for r in json.load(open(f)):
                rows.append(dict(r, ds=ds, sp=int(m.group(1)), file=m.group(2)))
    df = pd.DataFrame(rows)
    pick = {"PALM": ("joint_nb", "posterior realloc"), "InstaPrism": ("instaprism", "model"),
            "CIBERSORT": ("cibersort", "model"), "cellGeometry": ("cellgeometry", "model"),
            "NNLS": ("reallocation", "model"), "TCA": ("tca", "model"), "TOAST": ("toast", "model")}
    T = pd.DataFrame(index=list(pick))
    groups = [("Cell-type proportions", []), ("Per-type activity", []), ("Low leakage", [])]
    for key, metric, gi, hb, short in (("p", "frac_rmse", 0, False, ""), ("a", "act_spearman_across_samples", 1, True, ""),
                                       ("l", "act_leakage", 2, False, "")):
        for ds in DS:
            col = "%s_%s" % (key, ds)
            vals = []
            for lab, (f, row) in pick.items():
                sub = df[(df.file == f) & (df.row == row) & (df.ds == ds)]
                if metric == "frac_rmse" and lab == "PALM":
                    sub = df[(df.file == f) & (df.row == "model") & (df.ds == ds)]
                vals.append(sub[metric].mean() if len(sub) and metric in sub else np.nan)
            T[col] = vals
            groups[gi][1].append((col, DS[ds], hb))
    T = T.loc[:, T.notna().any()]
    groups = [(gl, [c for c in g if c[0] in T.columns]) for gl, g in groups]
    lang = {"PALM": "Py", "InstaPrism": "R", "CIBERSORT": "R", "cellGeometry": "R", "NNLS": "Py", "TCA": "R",
            "TOAST": "R"}
    info = pd.DataFrame(index=T.index)
    info["Language"] = [lang[m] for m in T.index]
    info["Per-sample uncertainty"] = ["yes" if m == "PALM" else "no" for m in T.index]
    print(T.round(3).to_string())
    funky(T, groups, info, "bench_deconv", title="Deconvolution: proportions and per-cell-type activity",
          note="Proportions: RMSE of estimated cell-type fractions (lower is better). Per-type activity: Spearman "
               "correlation with the true per-sample pathway activity of each cell type.\nLeakage: share of one "
               "type's signal assigned to the others (lower is better). Each tool with its own output; PALM: "
               "posterior reallocation.\nBars scaled per column (longest = best); 1–3 = rank; – = not run. "
               "Development splits 0–9 (InstaPrism, TCA, TOAST: 0–2).")


if __name__ == "__main__":
    calling()
    deconv()
