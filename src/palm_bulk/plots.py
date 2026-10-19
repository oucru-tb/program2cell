"""Minimal matplotlib helpers for PalmResult."""
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
