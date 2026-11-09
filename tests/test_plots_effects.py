import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from program2cell import plots

CTS, PGS = ["B", "T", "Mono"], ["JAK-STAT", "p53"]


def _effects():
    rng = np.random.default_rng(0)
    rows = []
    for c in CTS:
        for p in PGS:
            e = rng.normal(0, 0.1)
            rows.append(dict(cell_type=c, program=p, effect=e, lo=e - 0.1, hi=e + 0.1, called=abs(e) > 0.1))
    return pd.DataFrame(rows)


def _contrib():
    rows = []
    for p, clear in zip(PGS, (True, False)):
        m, ch = np.array([0.01, -0.02, 0.05]), np.array([0.02, 0.03, 0.3])
        con = m + ch
        for c, mi, ci, k in zip(CTS, m, ch, con):
            rows.append(dict(program=p, cell_type=c, share=k / con.sum(), share_lo=0.0, share_hi=1.0, more_cells=mi,
                             cells_changed=ci, contribution=k, total=con.sum(), total_clear=clear))
    return pd.DataFrame(rows)


def test_effect_plots_from_table(tmp_path):
    eff = _effects()
    ax = plots.plot_effect_heatmap(eff)
    assert len(ax.get_yticklabels()) == len(CTS)
    fig = plots.plot_effect_intervals(eff, ncols=2)
    assert sum(a.get_visible() for a in fig.axes) == len(PGS)
    f = tmp_path / "effects.csv"
    eff.to_csv(f, index=False)
    plots.plot_effect_heatmap(str(f))                      # from the saved CSV too


def test_contributions_plot():
    fig = plots.plot_contributions(_contrib())
    assert sum(a.get_visible() for a in fig.axes) == len(PGS)
    fig = plots.plot_contributions(_contrib(), only_clear=True)
    assert sum(a.get_visible() for a in fig.axes) == 1
