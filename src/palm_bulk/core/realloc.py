"""Fractions and per-cell-type activity without a Bayesian fit.

reallocate:    split each bulk sample's reads between cell types in proportion to pi_nc B_cg (the expectation
               step of BayesPrism with a fixed reference) and estimate each type's program shift with
               estimate_delta. With good fractions this is the best activity estimator we have
               (docs/method_spec.md, section 3).
em_fractions:  "PALM-lite": multinomial maximum-likelihood read fractions, y_n ~ Mult(n_n, pi_n B), by EM
               with SQUAREM acceleration (Varadhan & Roland 2008).
"""
import numpy as np

from .scores import estimate_delta


def realloc_counts(Y, pi, B, floor=1e-4):
    """(N, C, G) reads of each sample split between cell types: Y_ng pi_nc B_cg / sum_c' pi_nc' B_c'g.
    Fractions are floored at `floor` (and renormalised) so that a type with a zero fraction (NNLS) still
    gets a profile instead of no reads."""
    Y, B = np.asarray(Y, float), np.asarray(B, float)
    pi = np.clip(np.asarray(pi, float), floor, None)
    pi = pi / pi.sum(-1, keepdims=True)
    w = pi[..., :, None] * B
    return Y[..., None, :] * (w / w.sum(-2, keepdims=True))


def reallocate(Y, pi, B, G, floor=1e-4, **kw):
    """Per-sample per-type program shifts (N, C, P) from bulk counts Y (N, G) and read fractions pi (N, C):
    estimate_delta of the reallocated counts against B. `pi` may carry leading draw axes, e.g. (S, N, C);
    the output then has them too. Extra keywords go to estimate_delta."""
    pi = np.asarray(pi, float)
    if pi.ndim > 2:
        return np.stack([reallocate(Y, p, B, G, floor, **kw) for p in pi.reshape(-1, *pi.shape[-2:])]
                        ).reshape(pi.shape + (np.asarray(G).shape[0],))
    return estimate_delta(realloc_counts(Y, pi, B, floor), B, G, **kw)


def _em_step(pi, Y, Bn, n):
    return pi * ((Y / (pi @ Bn)) @ Bn.T) / n


def em_fractions(Y, B, pi0=None, max_iter=2000, tol=1e-8, accelerate=True):
    """Multinomial ML read fractions (N, C) of bulk counts Y (N, G) on reference profiles B (C, G), by EM.
    Each EM step keeps pi on the simplex and does not decrease the likelihood. With accelerate=True
    SQUAREM extrapolates two EM steps (scheme S3) and falls back to the plain step when the extrapolation
    leaves the simplex; typically 5-20x fewer iterations. Stops when max |change| < tol."""
    Y = np.asarray(Y, float)
    Bn = np.asarray(B, float)
    Bn = Bn / Bn.sum(1, keepdims=True)
    n = Y.sum(1, keepdims=True)
    C = Bn.shape[0]
    pi = np.full((Y.shape[0], C), 1.0 / C) if pi0 is None else np.asarray(pi0, float).copy()
    for _ in range(max_iter):
        p1 = _em_step(pi, Y, Bn, n)
        if not accelerate:
            new = p1
        else:
            p2 = _em_step(p1, Y, Bn, n)
            r, v = p1 - pi, p2 - 2 * p1 + pi
            vn = np.sqrt((v ** 2).sum(1, keepdims=True))
            alpha = -np.sqrt((r ** 2).sum(1, keepdims=True)) / np.where(vn > 0, vn, 1.0)
            alpha = np.minimum(alpha, -1.0)
            ext = pi - 2 * alpha * r + alpha ** 2 * v
            ok = (ext > 0).all(1, keepdims=True) & (vn > 0)
            ext = np.where(ok, ext, p2)
            ext = ext / ext.sum(1, keepdims=True)
            new = _em_step(ext, Y, Bn, n)                           # stabilising EM step after extrapolation
        if np.abs(new - pi).max() < tol:
            return new
        pi = new
    return pi
