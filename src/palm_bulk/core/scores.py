"""Program activity of an observed expression profile, on the model's own scale.

The model defines activity as the shift delta in E_cg = softmax_g(log B_cg + sum_p delta_cp G_pg).
`estimate_delta` inverts that definition for profiles that are observed directly: a per-cell-type
pseudo-bulk of real cells (ground truth for benchmarks and Paper 2's Teacher 1), or the per-sample
per-cell-type expression another method outputs (BayesPrism Z, CIBERSORTx high-resolution, BLADE).
All are then compared on one scale.
"""
import numpy as np
from scipy.special import softmax


def estimate_delta(X, B, G, ridge=1e-3, scale_to=None, min_total=0.0, max_iter=100, tol=1e-8):
    """Maximum-likelihood program shifts of observed profiles:
        X_..c ~ Multinomial(n_..c, softmax(log B_c + delta_..c G)),  small ridge on delta.

    X: (..., C, G) counts (or any non-negative expression; then set `scale_to`).
    B: baseline broadcastable to X, e.g. (C, G) reference profiles; rows need not sum to 1.
    G: (P, G) program weights, genes in the same order as X and B.
    scale_to: rescale every profile to this many pseudo-counts first (for normalised expression,
        e.g. 1e5); None uses X as given, which weights each profile by its real depth.
    min_total: profiles with total below this (e.g. a cell type absent from a sample) give nan.
    Returns delta (..., C, P). The log-likelihood is concave, so Newton steps converge; steps are
    capped at 1 per coordinate for safety far from the optimum."""
    X = np.asarray(X, float)
    G = np.asarray(G, float)
    logB = np.log(np.broadcast_to(np.asarray(B, float), X.shape))
    if scale_to is not None:
        tot = X.sum(-1, keepdims=True)
        X = np.where(tot > 0, X / np.where(tot > 0, tot, 1) * scale_to, 0.0)
    n = X.sum(-1)
    P = G.shape[0]
    d = np.zeros(X.shape[:-1] + (P,))
    eye = np.eye(P)
    GX = np.einsum("pg,...g->...p", G, X)
    for _ in range(max_iter):
        E = softmax(logB + np.einsum("...p,pg->...g", d, G), axis=-1)
        GE = np.einsum("pg,...g->...p", G, E)
        grad = GX - n[..., None] * GE - ridge * d
        H = n[..., None, None] * (np.einsum("pg,qg,...g->...pq", G, G, E, optimize=True)
                                  - GE[..., :, None] * GE[..., None, :]) + ridge * eye
        step = np.clip(np.linalg.solve(H, grad[..., None])[..., 0], -1.0, 1.0)
        d = d + step
        if np.abs(step).max(initial=0.0) < tol:
            break
    return np.where((n > min_total)[..., None] if min_total > 0 else (n > 0)[..., None], d, np.nan)


def bulk_delta(Y, pi, B, G, **kw):
    """One program shift per bulk sample, relative to the reference mixture pi @ B (the 'naive'
    bulk-level activity: it cannot say which cell types carry it). Y (N,G), pi (N,C). Returns (N,P)."""
    mix = np.asarray(pi, float) @ np.asarray(B, float)
    return estimate_delta(np.asarray(Y, float)[:, None, :], mix[:, None, :], G, **kw)[:, 0, :]
