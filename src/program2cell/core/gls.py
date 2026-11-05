"""v4 effect estimator (development; docs/v4_plan.md section 4, 6, 7): generalised least squares unmixing with
every sample's own leakage operator, calibrated by simulation.

Model (method_spec 4.6): reallocated activity r_n = a + T_n A_n beta + e_n, rows weighted by the within-arm sd;
paired designs centre within group (donor). beta (C, P) is reported as a shared response mu_p (read-fraction
weighted mean over types) plus type deviations delta_cp = beta_cp - mu_p.

Calibration (gls_calibrated): cohorts shaped like the study (its fractions and design) are built from reference
specimens with planted effects (core.unmix.simulate_cohort) including an off-program response of the size the
study shows (offprog_sd, core.unmix.offprogram_tau) and per-sample fraction error of a stated level (frac_sd);
fractions are re-estimated with PALM-lite and GLS applied; per (type, program) a slope k_cp of planted on
estimated and the level-quantile of |planted - k_cp * estimate| give the study's effects and intervals.

Fraction error (docs/v4_plan.md section 7): frac_sd is the sd of per-sample log-fraction error,
softmax(log pi + N(0, s^2)). It cannot be read off a bulk analysed with its own best-fitting fractions, so:
  assembled / pseudobulk with PALM-lite fractions   s = 0
  real bulk with PALM-lite fractions                s = FRAC_SD_REAL_BULK (COMBAT whole blood vs single cell)
  fractions from elsewhere (cytometry, other tool)  estimate_frac_sd (misfit beyond the bulk's own fit)
"""
import numpy as np

# COMBAT sepsis whole-blood bulk, PALM-lite vs single-cell read fractions, splits 20-24 (combat_fracnoise.py)
FRAC_SD_REAL_BULK = dict(raw=1.0, platform_corrected=0.8)


def gls_unmix(raw, A, T, groups=None):
    """raw (N, C, P) reallocated activity, A (N, C, P, C, P) leakage operators, T (N,) 0/1, groups (N,) or None
    -> (beta (C, P), se (C, P), residual scale s2)."""
    N, C, P = raw.shape
    K = C * P
    y = np.asarray(raw, float).reshape(N, K)
    T = np.asarray(T, float)
    res = y.copy()
    for t in (0, 1):
        res[T == t] -= y[T == t].mean(0)
    sd = np.sqrt((res ** 2).sum(0) / max(N - 2, 1)) + 1e-12
    X = np.concatenate([np.broadcast_to(np.eye(K), (N, K, K)), T[:, None, None] * np.asarray(A).reshape(N, K, K)], 2)
    Y = y.copy()
    if groups is not None:                         # per-group intercepts: centre rows within group
        g = np.asarray(groups).astype(str)
        X = X[:, :, K:].copy()
        for u in np.unique(g):
            m = g == u
            X[m] -= X[m].mean(0)
            Y[m] -= Y[m].mean(0)
    X = (X / sd[None, :, None]).reshape(-1, X.shape[2])
    Y = (Y / sd[None, :]).reshape(-1)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    M = np.linalg.pinv(X.T @ X)
    resid = Y - X @ coef
    s2 = float(resid @ resid / max(len(Y) - X.shape[1], 1))
    b = coef[-K:].reshape(C, P)
    se = np.sqrt(np.clip(np.diag(M)[-K:] * max(s2, 1.0), 0, None)).reshape(C, P)
    return b, se, s2


def perturb_fractions(pi, s, rng):
    """Per-sample fraction error: softmax(log pi + N(0, s^2)) independently per sample and type."""
    if not s:
        return pi
    from scipy.special import softmax
    return softmax(np.log(np.clip(pi, 1e-4, None)) + rng.normal(0, s, np.shape(pi)), 1)


def frac_misfit(Y, pi, B):
    """Depth-corrected misfit of bulk counts Y (N, G) to the mixture pi B: per sample the mean over the more
    expressed half of the genes of ((y - mu)^2 - mu) / mu^2; median over samples."""
    Y = np.asarray(Y, float)
    Bn = np.asarray(B, float) / np.asarray(B, float).sum(1, keepdims=True)
    mu = Y.sum(1, keepdims=True) * (np.asarray(pi, float) @ Bn)
    keep = mu.mean(0) >= np.quantile(mu.mean(0), 0.5)
    y, m = Y[:, keep], np.maximum(mu[:, keep], 1e-9)
    return float(np.median((((y - m) ** 2 - m) / m ** 2).mean(1)))


def estimate_frac_sd(cal, pi, Y, T, B, G, b_study, offprog_sd=0.0, n_sim=8, seed=0,
                     grid=(0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)):
    """Per-sample fraction error of fractions pi that did not come from this bulk: the study's misfit beyond
    its own PALM-lite fit, matched to simulated cohorts shaped like the study whose PALM-lite fractions are
    perturbed by s over the grid (dev splits 20-22: mean estimate 0.43 / 0.94 / 0.63 at true 0.40 / 0.87 /
    0.90; 0.07-0.36 with no per-sample error). Zero for PALM-lite fractions by construction.
    Returns (s, info)."""
    from . import unmix as um
    from .realloc import em_fractions
    rng = np.random.default_rng(seed + 202)
    m_study = frac_misfit(Y, pi, B) - frac_misfit(Y, em_fractions(np.asarray(Y, float), np.asarray(B, float)), B)
    scale = np.clip(np.abs(b_study).max(0), 0.05, 1.0)
    M = np.zeros((n_sim, len(grid)))
    for i in range(n_sim):
        Ys, pt, Bk, beta = um.simulate_cohort(cal, pi, T, rng, scale, offprog_sd=offprog_sd)
        pf = em_fractions(np.asarray(Ys, float), np.asarray(Bk, float))
        base = frac_misfit(Ys, pf, Bk)
        for j, s in enumerate(grid):
            M[i, j] = frac_misfit(Ys, perturb_fractions(pf, s, rng), Bk) - base
    m = np.maximum.accumulate(M.mean(0))
    s = float(np.interp(m_study, m, grid)) if m_study > m[0] else 0.0
    return s, dict(m_study=m_study, m_grid=m.tolist())


def shrink_deviations(b, se, w):
    """Shared response plus shrunk type deviations: mu_p = sum_c w_c b_cp (w the mean read fractions),
    d = b - mu, between-type spread tau_p^2 = max(0, mean_c d^2 - mean_c se^2) (method of moments), each
    deviation pulled towards mu by tau^2 / (tau^2 + se^2). A response shared by all types (PBMC cytokines)
    keeps mu; types that differ beyond their noise keep their deviation."""
    w = np.asarray(w, float) / np.sum(w)
    mu = w @ b
    d = b - mu[None]
    tau2 = np.clip((d ** 2).mean(0) - (se ** 2).mean(0), 0, None)
    return mu[None] + tau2[None] / (tau2[None] + se ** 2 + 1e-12) * d


def gls_calibrated(cal, pi, T, B, G, b_study, n_sim=30, level=0.9, seed=0, offprog_sd=0.0, frac_sd=0.0,
                   shrink=False, se_study=None):
    """Simulation calibration of gls_unmix (see module doc). shrink: shrink_deviations applied in every simulated
    cohort as on the study (b_study must then be the shrunk study estimate). se_study (C, P): the study's GLS
    standard errors; where their median exceeds the simulated cohorts' the error quantile is scaled up by the
    ratio (the study is noisier than the reference specimens the simulation is built from).
    Returns (effects, lo, hi, k, info)."""
    from . import unmix as um
    from .realloc import reallocate, em_fractions
    rng = np.random.default_rng(seed + 101)
    scale = np.clip(np.abs(b_study).max(0), 0.05, 1.0)        # planted sizes U(-scale, scale) per program
    est, true, ses = [], [], []
    for _ in range(n_sim):
        Y, pt, Bk, beta = um.simulate_cohort(cal, pi, T, rng, scale, offprog_sd=offprog_sd)
        pf = perturb_fractions(em_fractions(np.asarray(Y, float), np.asarray(Bk, float)), frac_sd, rng)
        raw = reallocate(Y, pf, Bk, G)
        A = um.leakage_operator(pf, Bk, G, diagonal=False)
        b, se, _ = gls_unmix(np.asarray(raw), np.asarray(A), T, None)
        if shrink:
            b = shrink_deviations(b, se, pf.mean(0))
        est.append(b); true.append(beta); ses.append(se)
    est, true = np.array(est), np.array(true)
    k = np.clip((est * true).sum(0) / np.maximum((est ** 2).sum(0), 1e-12), 0.0, 1.5)
    q = np.quantile(np.abs(true - k[None] * est), level, axis=0)
    ratio = 1.0
    if se_study is not None:
        ratio = float(max(1.0, np.median(se_study) / np.median(ses)))
        q = q * ratio
    eff = k * b_study
    return eff, eff - q, eff + q, k, dict(noise_ratio=ratio)


def v4_effects(Y, fractions, B, G, T, calibration, groups=None, level=0.9, seed=0, n_sim=30, frac_sd=0.0,
               min_effect=0.0, shrink=False, noise_anchor=False):
    """The v4 estimator on one study. frac_sd: number, or "estimate" (estimate_frac_sd; for fractions that did
    not come from this bulk). min_effect: effect floor for calls (|effect| >= min_effect and the interval
    excludes 0). Returns dict(effects, lo, hi, calls, shared, deviation, k, tau, frac_sd, ...)."""
    from . import unmix as um
    from .realloc import reallocate
    Y, B, G, T = (np.asarray(x, float) for x in (Y, B, G, T))
    pi = np.asarray(fractions, float)
    raw = reallocate(Y, pi, B, G)
    A = um.leakage_operator(pi, B, G, diagonal=False)
    b, se, s2 = gls_unmix(np.asarray(raw), np.asarray(A), T, groups)
    tau = float(um.offprogram_tau(Y, pi, B, G, T, b, groups=groups)[0])
    info = {}
    if isinstance(frac_sd, str):
        if frac_sd != "estimate":
            raise ValueError("frac_sd: a number or 'estimate'")
        frac_sd, info = estimate_frac_sd(calibration, pi, Y, T, B, G, b, offprog_sd=tau, seed=seed)
    bs = shrink_deviations(b, se, pi.mean(0)) if shrink else b
    eff, lo, hi, k, ci = gls_calibrated(calibration, pi, T, B, G, bs, n_sim=n_sim, level=level, seed=seed,
                                        offprog_sd=tau, frac_sd=float(frac_sd), shrink=shrink,
                                        se_study=se if noise_anchor else None)
    info.update(ci)
    w = pi.mean(0) / pi.mean(0).sum()
    mu = w @ eff
    calls = ((lo > 0) | (hi < 0)) & (np.abs(eff) >= min_effect)
    return dict(effects=eff, lo=lo, hi=hi, calls=calls, shared=mu, deviation=eff - mu[None], k=k, gls=b,
                gls_se=se, resid_scale=s2, tau=tau, frac_sd=float(frac_sd), frac_sd_info=info)
