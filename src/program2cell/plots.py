"""Minimal matplotlib helpers: PalmResult (first four) and program_deconvolution results (plot_effect_heatmap,
plot_effect_intervals, plot_contributions)."""
import numpy as np
import matplotlib.pyplot as plt


def plot_fractions(res, ax=None):
    ax = ax or plt.subplots(figsize=(max(5, 0.35 * len(res.inputs.samples) + 2), 3.5))[1]
    df = res.fractions()["median"]
    df.plot.bar(stacked=True, ax=ax, width=0.85, colormap="tab20")
    ax.set_ylabel("read fraction"); ax.set_xlabel(""); ax.legend(fontsize=7, bbox_to_anchor=(1, 1))
    return ax


def plot_attribution(res, program, ax=None, max_samples=30):
    """Stacked bars: per-sample share of `program` carried by each cell type."""
    ax = ax or plt.subplots(figsize=(max(5, 0.35 * min(len(res.inputs.samples), max_samples) + 2), 3.5))[1]
    a = res.attribution()
    d = a[a.program == program].pivot(index="sample", columns="cell_type", values="share")
    d = d.loc[res.inputs.samples].iloc[:max_samples].clip(lower=0)
    d.plot.bar(stacked=True, ax=ax, width=0.85, colormap="tab20")
    ax.set_ylabel("attribution share"); ax.set_title(program); ax.legend(fontsize=7, bbox_to_anchor=(1, 1))
    return ax


def plot_effects(res, axes=None):
    """Intrinsic vs composition effect per program (summed over cell types) with intervals."""
    t = res.effect_totals()
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(1.2 * len(t) + 3, 3.5))
    x = np.arange(len(t))
    for off, k, col in [(-0.2, "intrinsic", "tab:blue"), (0.2, "composition", "tab:orange")]:
        axes.errorbar(x + off, t[k], yerr=[t[k] - t[k + "_lo"], t[k + "_hi"] - t[k]], fmt="o", color=col, label=k)
    axes.axhline(0, color="k", lw=0.6); axes.set_xticks(x); axes.set_xticklabels(t.program, rotation=45, ha="right")
    axes.set_ylabel("effect on bulk program signal"); axes.legend()
    return axes


def plot_cell_type_activity(res, axes=None):
    """Heatmaps of program activity inside each cell type (posterior median of delta, averaged
    over samples), one panel per arm when a treatment is present."""
    act = res.cell_type_activity()
    groups = [("all samples", res.inputs.samples)]
    if res.has_treatment:
        T = np.asarray(res.inputs.T)
        groups = [(name, [s for s, t in zip(res.inputs.samples, T) if t == v]) for name, v in
                  [("control", 0), ("treated", 1)]]
    if axes is None:
        _, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups) + 1.5, 0.45 * len(res.activity_levels) + 1.6),
                               squeeze=False)
        axes = axes[0]
    mats = [act[act["sample"].isin(ss)].groupby(["cell_type", "program"])["activity"].mean().unstack()
            .loc[res.activity_levels, res.inputs.programs] for _, ss in groups]
    lim = max(np.abs(m.values).max() for m in mats) or 1.0
    for ax, (name, _), m in zip(axes, groups, mats):
        im = ax.imshow(m.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index, fontsize=8); ax.set_title(name, fontsize=10)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, "%.2f" % m.values[i, j], ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=list(axes), label="program shift (log scale)")
    return axes


# --- Effects and contributions from program_deconvolution (frozen method) -----------------------------------
# Colours: dataviz reference palette. Diverging blue <-> red with a grey midpoint for signed effects; the
# categorical pair blue / orange (slots 1-2) for "more cells" / "cells changed"; text in ink, never series colour.
_INK, _INK2, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
_MORE, _CHANGED = "#2a78d6", "#eb6834"
_DIVERGING = ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a8282a"]


def _table(x, kind):
    """DataFrame from a DeconvolutionResult (effects), a contributions_table, or a CSV path."""
    import pandas as pd
    if isinstance(x, str):
        return pd.read_csv(x)
    if hasattr(x, "table"):
        return x.table("effects")
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError("give a DeconvolutionResult, a %s DataFrame or a CSV path" % kind)


def _style(ax):
    ax.set_facecolor(_SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_MUTED)
    ax.tick_params(colors=_INK2, labelsize=9)


def _grid(n, ncols):
    ncols = max(1, min(ncols, n))
    return ncols, int(np.ceil(n / ncols))


def plot_effect_heatmap(effects, ax=None, vmax=None, title="Change per cell type and program", label="effect"):
    """Heatmap of the effects table (cell types x programs; diverging, centred on 0). Called cells (90%
    interval excludes 0) are outlined and labelled with their value; the others are coloured but unlabelled.
    effects: DeconvolutionResult, the res.table("effects") DataFrame, or results/effects.csv."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    t = _table(effects, "effects")
    cts, pgs = list(dict.fromkeys(t.cell_type)), list(dict.fromkeys(t.program))
    E = t.pivot(index="cell_type", columns="program", values="effect").loc[cts, pgs].values.astype(float)
    C = t.pivot(index="cell_type", columns="program", values="called").loc[cts, pgs].values.astype(bool)
    lim = vmax or (np.nanmax(np.abs(E)) or 1.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(0.9 * len(pgs) + 2.6, 0.55 * len(cts) + 1.8))
    cmap = LinearSegmentedColormap.from_list("p2c_div", _DIVERGING)
    im = ax.imshow(E, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i, j in zip(*np.nonzero(C)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=_INK, lw=2))
        dark = abs(E[i, j]) > 0.6 * lim
        ax.text(j, i, "%+.2f" % E[i, j], ha="center", va="center", fontsize=8, color="#ffffff" if dark else _INK)
    ax.set_xticks(range(len(pgs)))
    ax.set_xticklabels(pgs, rotation=45, ha="right")
    ax.set_yticks(range(len(cts)))
    ax.set_yticklabels(cts)
    ax.tick_params(colors=_INK2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(label, color=_INK2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=_INK2)
    ax.set_title(title, loc="left", fontsize=11, color=_INK)
    ax.text(1.0, 1.02, "outlined = 90% interval excludes 0", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=_MUTED)
    return ax


def plot_effect_intervals(effects, programs=None, ncols=4, sharex=True, label="effect"):
    """Small multiples, one panel per program: each cell type's effect with its 90% interval. Filled dot =
    called (interval excludes 0), open dot = not called. Same x scale in every panel (sharex)."""
    t = _table(effects, "effects")
    pgs = programs or list(dict.fromkeys(t.program))
    cts = list(dict.fromkeys(t.cell_type))
    ncols, nrows = _grid(len(pgs), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, (0.42 * len(cts) + 1.0) * nrows + 0.4),
                             sharex=sharex, sharey=True, squeeze=False)
    y = np.arange(len(cts))[::-1]
    for k, ax in enumerate(axes.ravel()):
        if k >= len(pgs):
            ax.set_visible(False)
            continue
        d = t[t.program == pgs[k]].set_index("cell_type").loc[cts]
        _style(ax)
        ax.axvline(0, color=_MUTED, lw=1)
        ax.hlines(y, d["lo"], d["hi"], color=_INK2, lw=2)
        called = d["called"].values.astype(bool)
        ax.scatter(d["effect"].values[called], y[called], s=64, color=_INK, zorder=3, edgecolor=_SURFACE,
                   linewidth=2)
        ax.scatter(d["effect"].values[~called], y[~called], s=64, facecolor=_SURFACE, edgecolor=_INK2,
                   linewidth=1.5, zorder=3)
        ax.set_title(pgs[k], loc="left", fontsize=10, color=_INK)
        ax.set_yticks(y)
        ax.set_yticklabels(cts)
        ax.grid(axis="x", color=_GRID, lw=0.8)
        ax.set_axisbelow(True)
        if k >= len(pgs) - ncols:
            ax.set_xlabel(label, color=_INK2, fontsize=9)
            ax.xaxis.set_tick_params(labelbottom=True)
    fig.text(0.01, 0.005, "line = 90% interval; filled = interval excludes 0, open = not called", fontsize=8,
             color=_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_contributions(contrib, programs=None, ncols=4, only_clear=False):
    """Where each program's bulk change comes from: per cell type, the contribution split into "more cells"
    (composition) and "cells changed" (intrinsic), from pb.contributions_table(...) or a CSV of it. The share
    of the total (with its 90% interval) is printed only for programs whose total change is clearly non-zero
    (total_clear); the other panels are faded and say so."""
    from matplotlib.patches import Patch
    t = _table(contrib, "contributions_table")
    need = {"program", "cell_type", "more_cells", "cells_changed", "share", "total", "total_clear"}
    if not need <= set(t.columns):
        raise ValueError("needs the long table of pb.contributions_table (columns %s)" % sorted(need))
    pgs = programs or list(dict.fromkeys(t.program))
    if only_clear:
        pgs = [p for p in pgs if bool(t[t.program == p]["total_clear"].iloc[0])]
    cts = list(dict.fromkeys(t.cell_type))
    ncols, nrows = _grid(len(pgs), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, (0.42 * len(cts) + 1.1) * nrows + 0.6),
                             sharey=True, squeeze=False)
    y = np.arange(len(cts))[::-1]
    has_iv = {"share_lo", "share_hi"} <= set(t.columns)
    for k, ax in enumerate(axes.ravel()):
        if k >= len(pgs):
            ax.set_visible(False)
            continue
        d = t[t.program == pgs[k]].set_index("cell_type").loc[cts]
        clear = bool(d["total_clear"].iloc[0])
        _style(ax)
        m, c = d["more_cells"].values.astype(float), d["cells_changed"].values.astype(float)
        pm, nm, pc, nc = np.clip(m, 0, None), np.clip(m, None, 0), np.clip(c, 0, None), np.clip(c, None, 0)
        kw = dict(height=0.6, edgecolor=_SURFACE, linewidth=2, alpha=1.0 if clear else 0.35)
        ax.barh(y, pm, color=_MORE, **kw)               # same-sign parts stack away from 0
        ax.barh(y, pc, left=pm, color=_CHANGED, **kw)
        ax.barh(y, nm, color=_MORE, **kw)
        ax.barh(y, nc, left=nm, color=_CHANGED, **kw)
        ax.axvline(0, color=_MUTED, lw=1)
        right, left = (pm + pc).max(), (nm + nc).min()
        span = max(right - left, 1e-9)
        if clear:
            for i, yi in enumerate(y):
                r = d.iloc[i]
                txt = "%.0f%%" % (round(100 * r["share"]) + 0.0)
                if has_iv:
                    txt += " (%.0f to %.0f)" % (round(100 * r["share_lo"]) + 0.0, round(100 * r["share_hi"]) + 0.0)
                ax.text(max(pm[i] + pc[i], 0) + 0.03 * span, yi, txt, va="center", fontsize=7.5, color=_INK2)
            ax.set_xlim(min(left * 1.1, -0.03 * span), right + 0.9 * span)
        tot = round(float(d["total"].iloc[0]), 2) + 0.0
        ax.set_title("%s   total %+.2f%s" % (pgs[k], tot, "" if clear else ", not clear: no shares"),
                     loc="left", fontsize=10, color=_INK if clear else _MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels(cts)
        ax.grid(axis="x", color=_GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.legend(handles=[Patch(color=_MORE, label="more cells (composition)"),
                        Patch(color=_CHANGED, label="cells changed (intrinsic)")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.text(0.01, 0.005, "bar = contribution to the bulk change; % = share of the total (90% interval)",
             fontsize=8, color=_MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig
