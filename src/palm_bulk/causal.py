"""Causal effects of a treatment or exposure on cell-type program activity, and their mediation.

Effects. Every estimator here is linear in the outcome: beta_cp = sum_n w_n activity_ncp, with weights w (N,)
that depend only on the design (treatment, covariates, pairing). So the calibration, conformal and leakage-
aware intervals of core.calibrate apply unchanged (Calibrator.effects(weights=w)). estimator_weights builds w:

  diff   difference of means                         causal under randomisation
  ols    regression on [1, T, covariates, groups]    causal under conditional ignorability (linear outcome);
                                                     groups = donor or pair ids -> within-donor (paired) effect
  ipw    Hajek inverse-probability weighting         causal under ignorability, propensity model correct
  aipw   augmented IPW (doubly robust)               causal if the outcome model OR the propensity model is
                                                     correct; outcome model linear in covariates per arm

Mediation. The exposure changes a bulk program score through composition (the mix pi) and through intrinsic
activity (each type's program state). decompose() splits the bulk effect per cell type into intrinsic I_cp
and composition K_cp, a Shapley (order-averaged) split of the natural indirect / direct effects
(equations.effect_decomposition, here on observed fractions and calibrated activity):
    I_cp = 0.5 (pi_c(0) + pi_c(1)) (s_cp(1) - s_cp(0)),   K_cp = 0.5 (pi_c(1) - pi_c(0)) (s_cp(0) + s_cp(1)),
    s_ncp = sum_g G_pg E_cg(delta_nc)  (program signal of type c),  I + K summed over c = the bulk effect.
Interpretation as natural effects needs the usual mediation assumptions (no unmeasured exposure-mediator,
mediator-outcome or exposure-outcome confounding given the covariates).

Sensitivity. robustness_value (Cinelli & Hazlett 2020): the share of residual variance of both the exposure
and the outcome that an unmeasured confounder would need to explain to move the effect to 0 (or to make it
non-significant). Small values = fragile.
"""
import numpy as np
from scipy.stats import norm, t as tdist

from .core.equations import cell_profiles

ESTIMATORS = ("diff", "ols", "ipw", "aipw")


# ---- estimator weights -------------------------------------------------------------------------------------
def _design(X, groups, n):
    cols = []
    if X is not None:
        X = np.asarray(X, float).reshape(n, -1)
        cols.append(X - X.mean(0))
    if groups is not None:
        g = np.asarray(groups)
        lev = np.unique(g)
        cols.append((g[:, None] == lev[None, 1:]).astype(float))         # fixed effects, first level dropped
    return np.column_stack(cols) if cols else np.zeros((n, 0))


def propensity(T, X, ridge=1e-3, clip=0.02, iters=50):
    """Logistic-regression propensity P(T = 1 | X), by Newton steps; clipped to [clip, 1 - clip]."""
    T = np.asarray(T, float)
    Z = np.column_stack([np.ones(len(T)), np.asarray(X, float).reshape(len(T), -1)])
    b = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ b))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + ridge * np.eye(len(b))
        step = np.linalg.solve(H, Z.T @ (T - p) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return np.clip(1 / (1 + np.exp(-Z @ b)), clip, 1 - clip)


def estimator_weights(T, X=None, groups=None, estimator="diff"):
    """(N,) weights w with beta = sum_n w_n y_n for the chosen estimator (see module doc)."""
    T = np.asarray(T, float)
    n = len(T)
    if estimator not in ESTIMATORS:
        raise ValueError("estimator must be one of %s" % (ESTIMATORS,))
    if estimator == "diff":
        if X is not None or groups is not None:
            raise ValueError("'diff' ignores covariates; use 'ols', 'ipw' or 'aipw'")
        return np.where(T == 1, 1 / (T == 1).sum(), -1 / (T == 0).sum())
    D = _design(X, groups, n)
    if estimator == "ols":
        M = np.column_stack([np.ones(n), T, D])
        return np.linalg.pinv(M)[1]
    if X is None and groups is None:
        raise ValueError("'%s' needs covariates" % estimator)
    e = propensity(T, D)
    w1, w0 = T / e, (1 - T) / (1 - e)
    if estimator == "ipw":
        return w1 / w1.sum() - w0 / w0.sum()
    # AIPW with per-arm linear outcome models: linear in y, so its weights are the estimator applied to I
    A = np.column_stack([np.ones(n), D])
    I = np.eye(n)
    m = {}
    for arm in (0, 1):
        idx = T == arm
        m[arm] = A @ np.linalg.pinv(A[idx]) @ I[idx]                      # (n, n): predictions from arm's y
    W = (m[1] - m[0]) + (T / e)[:, None] * (I - m[1]) - ((1 - T) / (1 - e))[:, None] * (I - m[0])
    return W.mean(0)


def degrees_of_freedom(T, X=None, groups=None):
    n = len(np.asarray(T))
    return max(n - 2 - _design(X, groups, n).shape[1], 1)


# ---- sensitivity -------------------------------------------------------------------------------------------
def robustness_value(eff, lo, hi, df, level=0.9, q=1.0, significance=True):
    """Cinelli & Hazlett (2020) robustness value per effect. eff, lo, hi: (C, P) effect and interval (the
    standard error is taken from the interval width); df: residual degrees of freedom.
    significance=False: RV_q (confounding that reduces the effect by 100q %); True: RV_{q,alpha} (that makes
    it non-significant at 1 - level)."""
    z = norm.ppf(0.5 + level / 2)
    se = (np.asarray(hi) - np.asarray(lo)) / (2 * z)
    tval = np.abs(eff) / np.where(se > 0, se, np.nan)
    f = q * tval / np.sqrt(df)
    if significance:
        f = f - tdist.ppf(0.5 + level / 2, max(df - 1, 1)) / np.sqrt(max(df - 1, 1))
    f = np.clip(f, 0, None)
    return 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)


# ---- mediation: intrinsic vs composition --------------------------------------------------------------------
def program_signal(B, G, activity):
    """(N, C, P) s_ncp = sum_g G_pg E_cg(delta_nc): each type's program signal at its activity."""
    E = np.asarray(cell_profiles(np.asarray(B, float), np.asarray(G, float), np.asarray(activity, float)))
    return np.einsum("pg,ncg->ncp", np.asarray(G, float), E)


def _split(w, pi, s):
    wp, wn = np.clip(w, 0, None), np.clip(-w, 0, None)
    pi1, pi0 = wp @ pi / wp.sum(), wn @ pi / wn.sum()                   # arm-level (adjusted) means
    s1, s0 = np.einsum("n,ncp->cp", wp, s) / wp.sum(), np.einsum("n,ncp->cp", wn, s) / wn.sum()
    dpi, ds = w @ pi, np.einsum("n,ncp->cp", w, s)                      # estimator contrasts
    I = 0.5 * (pi0 + pi1)[:, None] * ds
    K = 0.5 * dpi[:, None] * (s0 + s1)
    return I, K


def decompose(fractions, activity, B, G, T, X=None, groups=None, estimator="diff", n_boot=500, level=0.9,
              fraction_draws=None, seed=0):
    """Intrinsic (I) and composition (K) parts of the exposure's effect on each bulk program, per cell type,
    with bootstrap intervals over samples (stratified by T; a fraction draw per replicate if given).
    Returns dict(I, K, total (P,), share_intrinsic (P,), I_iv, K_iv, total_iv)."""
    pi = np.asarray(fractions, float)
    s = program_signal(B, G, activity)
    T = np.asarray(T)
    w = estimator_weights(T, X, groups, estimator)
    I, K = _split(w, pi, s)
    rng = np.random.default_rng(seed)
    bI, bK = [], []
    idx1, idx0 = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(idx1, len(idx1)), rng.choice(idx0, len(idx0))])
        pb = pi[b] if fraction_draws is None else np.asarray(fraction_draws)[rng.integers(len(fraction_draws))][b]
        try:
            wb = estimator_weights(T[b], None if X is None else np.asarray(X).reshape(len(T), -1)[b],
                                   None if groups is None else np.asarray(groups)[b], estimator)
        except (np.linalg.LinAlgError, ValueError):
            continue
        i_, k_ = _split(wb, pb, s[b])
        bI.append(i_); bK.append(k_)
    bI, bK = np.array(bI), np.array(bK)
    a = (1 - level) / 2
    q = lambda x: (np.quantile(x, a, 0), np.quantile(x, 1 - a, 0))
    tot = I.sum(0) + K.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = I.sum(0) / (np.abs(I.sum(0)) + np.abs(K.sum(0)))
    return dict(I=I, K=K, total=tot, share_intrinsic=share, I_iv=q(bI), K_iv=q(bK),
                total_iv=q(bI.sum(1) + bK.sum(1)))
