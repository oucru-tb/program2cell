"""Contributions: where a bulk program change comes from.

For a treatment (or disease) contrast, the change in a program's bulk signal splits exactly into per-cell-type
parts, each split again into "more cells" (composition) and "cells changed" (intrinsic); the Shapley form of
core.equations.effect_decomposition:

    s_cp(t)  = sum_g G_pg E_cg(t),   E_c(0) = B_c,   E_c(1) = cell_profiles(B_c, beta_c)
    I_cp = 0.5 (pi_c(0) + pi_c(1)) (s_cp(1) - s_cp(0))         cells changed
    K_cp = 0.5 (pi_c(1) - pi_c(0)) (s_cp(0) + s_cp(1))         more cells
    total_p = sum_c (I_cp + K_cp) = sum_c pi_c(1) s_cp(1) - pi_c(0) s_cp(0)   (exact)
    share_cp = (I_cp + K_cp) / total_p

pi(t): mean read fractions of each arm; beta: the per-cell-type effects (program_deconvolution). Intervals:
Monte Carlo over the fraction draws (when given) and the effects (normal with the effect interval's sd).
Shares are only meaningful where the total change is clearly non-zero (`total_clear`); they can fall outside
0-100% when cell types move in opposite directions.
"""
import numpy as np
from scipy import stats

from .core.equations import cell_profiles


def _decompose(B, G, pi0, pi1, beta):
    s0 = np.einsum("pg,cg->cp", G, B)                                  # (C, P), reference = control profile
    C, P = beta.shape
    s1 = np.empty((C, P))
    for p in range(P):                                                  # program p's own change only
        d = np.zeros((C, P))
        d[:, p] = beta[:, p]
        s1[:, p] = np.einsum("g,cg->c", G[p], cell_profiles(B, G, d))
    I = 0.5 * (pi0 + pi1)[:, None] * (s1 - s0)
    K = 0.5 * (pi1 - pi0)[:, None] * (s0 + s1)
    return I, K


def contributions(B, G, fractions, T, effects, lo=None, hi=None, fraction_draws=None, level=0.9, n_draws=400,
                  seed=0):
    """B (C, Genes) reference profiles, G (P, Genes) program weights, fractions (N, C), T (N,) 0/1, effects
    (C, P) with interval lo/hi. Returns dict of (C, P) arrays: intrinsic, composition, contribution, share
    (point values) and their *_lo / *_hi; total (P,), total_lo/hi, total_clear (P,) bool."""
    B = np.asarray(B, float)
    B = B / B.sum(1, keepdims=True)
    G = np.asarray(G, float)
    T = np.asarray(T).astype(bool)
    eff = np.asarray(effects, float)
    pi = np.asarray(fractions, float)
    I, K = _decompose(B, G, pi[~T].mean(0), pi[T].mean(0), eff)
    out = dict(intrinsic=I, composition=K, contribution=I + K, total=(I + K).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["share"] = (I + K) / out["total"][None]
    rng = np.random.default_rng(seed)
    q = stats.norm.ppf(0.5 + level / 2)
    sd = np.zeros_like(eff) if lo is None else (np.asarray(hi) - np.asarray(lo)) / (2 * q)
    draws = None if fraction_draws is None else np.asarray(fraction_draws, float)
    sims = {k: [] for k in ("intrinsic", "composition", "contribution", "total", "share")}
    for _ in range(n_draws):
        p_ = pi if draws is None else draws[rng.integers(len(draws))]
        b_ = eff + sd * rng.standard_normal(eff.shape)
        i_, k_ = _decompose(B, G, p_[~T].mean(0), p_[T].mean(0), b_)
        tot = (i_ + k_).sum(0)
        for k, v in (("intrinsic", i_), ("composition", k_), ("contribution", i_ + k_), ("total", tot)):
            sims[k].append(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            sims["share"].append((i_ + k_) / tot[None])
    a = (1 - level) / 2
    for k, v in sims.items():
        v = np.asarray(v)
        out[k + "_lo"], out[k + "_hi"] = np.nanquantile(v, a, 0), np.nanquantile(v, 1 - a, 0)
    out["total_clear"] = (out["total_lo"] > 0) | (out["total_hi"] < 0)
    return out


def contributions_table(res, cell_types, programs, program=None):
    """Long table (pandas) of the contributions: one row per (program, cell type)."""
    import pandas as pd
    rows = []
    for p, pg in enumerate(programs):
        if program is not None and pg != program:
            continue
        for c, ct in enumerate(cell_types):
            rows.append(dict(program=pg, cell_type=ct, share=res["share"][c, p], share_lo=res["share_lo"][c, p],
                             share_hi=res["share_hi"][c, p], more_cells=res["composition"][c, p],
                             cells_changed=res["intrinsic"][c, p], contribution=res["contribution"][c, p],
                             total=res["total"][p], total_clear=bool(res["total_clear"][p])))
    return pd.DataFrame(rows)
