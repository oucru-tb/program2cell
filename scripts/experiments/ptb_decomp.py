"""Per-cell-type decomposition intervals for the PTB figures (reporting only; no method change).

program2cell.causal.decompose returns 90% bootstrap intervals (over samples, stratified by group) for the
intrinsic part I ("cells changed") and the composition part K ("more cells") of each cell type's share of a bulk
program change, but only the bulk total's interval, not each cell type's total I + K. This repeats the same
bootstrap (same seed, same order of random draws, so I_iv and K_iv are reproduced exactly; checked) and also
returns the per-type total interval.

These are sample-bootstrap intervals, NOT the calibrated (leakage-widened) effect intervals of PALM's effects.
"""
import numpy as np
from program2cell import causal


def decomposition_table(res, inp, covariates=None, groups=None, level=0.9, n_boot=500, seed=0, fraction_draws=None):
    """-> DataFrame (cell_type, program, total, total_lo, total_hi, composition(_lo/_hi), intrinsic(_lo/_hi))."""
    import pandas as pd
    estimator = "ols" if (covariates is not None or groups is not None) else "diff"     # as program_deconvolution
    T = np.asarray(inp.T)
    pi = np.asarray(res.fractions, float)
    s = causal.program_signal(inp.B, inp.Gmat, res.activity)
    w = causal.estimator_weights(T, covariates, groups, estimator)
    I, K = causal._split(w, pi, s)
    rng = np.random.default_rng(seed)
    bI, bK = [], []
    idx1, idx0 = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(idx1, len(idx1)), rng.choice(idx0, len(idx0))])
        pb = pi[b] if fraction_draws is None else np.asarray(fraction_draws)[rng.integers(len(fraction_draws))][b]
        try:
            wb = causal.estimator_weights(T[b], None if covariates is None else np.asarray(covariates).reshape(len(T), -1)[b],
                                          None if groups is None else np.asarray(groups)[b], estimator)
        except (np.linalg.LinAlgError, ValueError):
            continue
        i_, k_ = causal._split(wb, pb, s[b])
        bI.append(i_); bK.append(k_)
    bI, bK = np.array(bI), np.array(bK)
    a = (1 - level) / 2
    q = lambda x: (np.quantile(x, a, 0), np.quantile(x, 1 - a, 0))
    d = getattr(res, "decomposition", None)
    if d is not None:                                  # same bootstrap as the saved result
        assert np.allclose(d["I"], I) and np.allclose(d["K"], K), "point parts differ from the saved decomposition"
        assert np.allclose(d["I_iv"][0], q(bI)[0]) and np.allclose(d["K_iv"][1], q(bK)[1]), "bootstrap not reproduced"
    rows = []
    parts = dict(intrinsic=(I, q(bI)), composition=(K, q(bK)), total=(I + K, q(bI + bK)))
    for c, ct in enumerate(inp.cell_types):
        for p, pg in enumerate(inp.programs):
            r = dict(cell_type=ct, program=pg)
            for name, (v, (lo, hi)) in parts.items():
                r.update({name: v[c, p], name + "_lo": lo[c, p], name + "_hi": hi[c, p]})
            rows.append(r)
    return pd.DataFrame(rows)
