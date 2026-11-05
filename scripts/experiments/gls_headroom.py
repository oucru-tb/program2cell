"""Headroom check (v3 development, 2026-11-04): does a plain generalised-least-squares unmixing with every
sample's own leakage operator (the design behind identifiability.py's mde_full) get closer to the truth than
PALM v2? If yes, the information the design limit says is there is being lost in PALM's estimator.

Model (method_spec 4.6): reallocated activity r_n = a + T_n A_n beta + e_n; rows weighted by the within-arm sd;
paired designs (kang, zhang) centre within donor. No calibration, no shrinkage.
Usage: python gls_headroom.py DATASETS SPLITS   -> benchmarks/dev/gls_headroom/{ds}__s{split}.json
"""
import json, os, sys
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl

OUT = os.path.join(ROOT, "benchmarks", "dev", "gls_headroom")


def gls_unmix(raw, A, T, groups=None):
    """raw (N, C, P), A (N, C, P, C, P), T (N,) -> beta (C, P), se (C, P)."""
    N, C, P = raw.shape
    K = C * P
    y = raw.reshape(N, K)
    T = np.asarray(T, float)
    res = y.copy()
    for t in (0, 1):
        res[T == t] -= y[T == t].mean(0)
    sd = np.sqrt((res ** 2).sum(0) / max(N - 2, 1)) + 1e-12
    X = np.concatenate([np.broadcast_to(np.eye(K), (N, K, K)), T[:, None, None] * A.reshape(N, K, K)], 2)
    Y = y.copy()
    if groups is not None:                         # per-donor intercepts: centre rows within donor
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
    s2 = float(resid @ resid / max(len(Y) - X.shape[1], 1))   # residual scale beyond the within-arm sd
    b = coef[-K:].reshape(C, P)
    se = np.sqrt(np.clip(np.diag(M)[-K:] * max(s2, 1.0), 0, None)).reshape(C, P)
    return b, se, s2


def lowrank_deviation(b, se, w, k, n_iter=50):
    """Low-rank type x program deviations (v4 development, idea register 1): b (C, P) = mu_p + D_cp with
    mu_p = sum_c w_c b_cp (w the mean read fractions) and D replaced by its rank-k fit, weighted by 1/se^2
    (alternating weighted least squares from the SVD start). k=None returns b unchanged."""
    if k is None:
        return b
    w = np.asarray(w, float) / np.sum(w)
    mu = w @ b
    D = b - mu[None]
    W = 1.0 / np.maximum(se, 1e-6) ** 2
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    U, V = U[:, :k] * s[:k], Vt[:k].T                                  # (C, k), (P, k)
    for _ in range(n_iter):
        for c in range(D.shape[0]):
            U[c] = np.linalg.lstsq(V * np.sqrt(W[c])[:, None], D[c] * np.sqrt(W[c]), rcond=None)[0]
        for p in range(D.shape[1]):
            V[p] = np.linalg.lstsq(U * np.sqrt(W[:, p])[:, None], D[:, p] * np.sqrt(W[:, p]), rcond=None)[0]
    Dk = U @ V.T
    Dk = Dk - (w @ Dk)[None]                                           # keep sum_c w_c D_cp = 0
    return mu[None] + Dk


def perturb_fractions(pi, s, rng):
    """Per-sample fraction error: softmax(log pi + N(0, s^2)) independently per sample and type."""
    if not s:
        return pi
    from scipy.special import softmax
    return softmax(np.log(np.clip(pi, 1e-4, None)) + rng.normal(0, s, np.shape(pi)), 1)


def frac_misfit(Y, pi, B):
    """Depth-corrected misfit of bulk counts Y (N, G) to the mixture pi B: per sample the mean over the more
    expressed half of the genes of ((y - mu)^2 - mu) / mu^2 (squared relative error beyond Poisson); median
    over samples."""
    Y = np.asarray(Y, float)
    Bn = np.asarray(B, float) / np.asarray(B, float).sum(1, keepdims=True)
    mu = Y.sum(1, keepdims=True) * (np.asarray(pi, float) @ Bn)
    keep = mu.mean(0) >= np.quantile(mu.mean(0), 0.5)
    y, m = Y[:, keep], np.maximum(mu[:, keep], 1e-9)
    return float(np.median((((y - m) ** 2 - m) / m ** 2).mean(1)))


def estimate_frac_sd(cal, pi, Y, T, B, G, b_study, offprog_sd=0.0, n_sim=8, seed=0,
                     grid=(0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0), anchor="excess"):
    """Fraction-error level anchored on the data (v4): the study's fraction misfit (frac_misfit of Y on the
    fractions in use) against simulated cohorts shaped like the study, whose PALM-lite fractions are perturbed
    by perturb_fractions(., s) over a grid of s; returns the s whose mean simulated misfit matches the study's
    (linear interpolation; 0 when the study fits no worse than the unperturbed simulations).
    anchor="absolute": raw misfits (2026-11-05: reads baseline misfit - biology, platform - as fraction error;
    Kang true fractions -> 2.0, rand 0.45 -> 1.51). anchor="excess": misfit beyond the study's own best fit
    (PALM-lite on Y), in the study and the simulations alike; detects fractions that disagree with the bulk
    (external fractions), 0 for PALM-lite itself, blind to a bias shared by the bulk fit."""
    from program2cell.core import unmix as um
    from program2cell.core.realloc import em_fractions
    rng = np.random.default_rng(seed + 202)
    m_study = frac_misfit(Y, pi, B)
    if anchor == "excess":
        m_study -= frac_misfit(Y, em_fractions(np.asarray(Y, float), np.asarray(B, float)), B)
    scale = np.clip(np.abs(b_study).max(0), 0.05, 1.0)
    M = np.zeros((n_sim, len(grid)))
    for i in range(n_sim):
        Ys, pt, Bk, beta = um.simulate_cohort(cal, pi, T, rng, scale, offprog_sd=offprog_sd)
        pf = em_fractions(np.asarray(Ys, float), np.asarray(Bk, float))
        for j, s in enumerate(grid):
            M[i, j] = frac_misfit(Ys, perturb_fractions(pf, s, rng), Bk)
        if anchor == "excess":
            M[i] -= frac_misfit(Ys, pf, Bk)
    m = M.mean(0)
    m = np.maximum.accumulate(m)                      # misfit grows with s; guard the interpolation
    s = float(np.interp(m_study, m, grid)) if m_study > m[0] else 0.0
    return s, dict(m_study=m_study, m_grid=m.tolist())


def gls_calibrated(cal, pi, T, B, G, b_study, n_sim=30, level=0.9, seed=0, groups=None, offprog_sd=0.0,
                   plant_mult=1.0, post=None, frac_sd=0.0):
    """Scale calibration of gls_unmix by simulation (v3 candidate 10 with the GLS estimator): cohorts shaped like
    the study (its fractions and design) built from reference specimens with planted effects
    (unmix.simulate_cohort, sizes from the study's own GLS effects), fractions re-estimated (PALM-lite) as on the
    study, GLS applied (offprog_sd: off-program response as unmix.offprogram_tau measures it); per (type, program) a slope through the origin of planted on estimated, k_cp, and the
    level-quantile of |planted - k_cp * estimate| / (k_cp * se). Returns (effects, lo, hi) for the study.
    post(b, se, pi) -> b: a transform of the GLS estimate (e.g. lowrank_deviation), applied in every simulated
    cohort as on the study; b_study must then be the transformed study estimate.
    frac_sd: per-sample fraction error added to the simulated cohorts' re-estimated fractions
    (perturb_fractions; level from estimate_frac_sd), so the intervals carry the study's fraction error."""
    from program2cell.core import unmix as um
    from program2cell.core.realloc import reallocate, em_fractions
    rng = np.random.default_rng(seed + 101)
    C, P = b_study.shape
    scale = np.clip(plant_mult * np.abs(b_study).max(0), 0.05, 1.0)    # sizes U(-scale, scale) per program
    est, true, ses = [], [], []
    for _ in range(n_sim):
        Y, pt, Bk, beta = um.simulate_cohort(cal, pi, T, rng, scale, offprog_sd=offprog_sd)
        pf = perturb_fractions(em_fractions(np.asarray(Y, float), np.asarray(Bk, float)), frac_sd, rng)
        raw = reallocate(Y, pf, Bk, G)
        A = um.leakage_operator(pf, Bk, G, diagonal=False)
        b, se, _ = gls_unmix(np.asarray(raw), np.asarray(A), T, None)
        if post is not None:
            b = post(b, se, pf)
        est.append(b); true.append(beta); ses.append(se)
    est, true, ses = np.array(est), np.array(true), np.array(ses)
    k = (est * true).sum(0) / np.maximum((est ** 2).sum(0), 1e-12)          # (C, P)
    k = np.clip(k, 0.0, 1.5)
    err = np.abs(true - k[None] * est)
    q = np.quantile(err, level, axis=0)                                       # absolute error quantile (C, P)
    eff = k * b_study
    return eff, eff - q, eff + q, k


def run(datasets, splits):
    from program2cell.core.realloc import reallocate, em_fractions
    from program2cell.core import unmix as um
    os.makedirs(OUT, exist_ok=True)
    for dsn in datasets:
        for split in splits:
            f = os.path.join(OUT, "%s__s%d.json" % (dsn, split))
            if os.path.exists(f):
                continue
            pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
            pi = (np.median(src["fraction_draws"], 0) if "fraction_draws" in src
                  else em_fractions(np.asarray(inp.Y, float), np.asarray(inp.B, float)))
            raw = reallocate(inp.Y, pi, inp.B, inp.Gmat)
            A = um.leakage_operator(pi, inp.B, inp.Gmat, diagonal=False)
            paired = dsn in ("kang", "zhang")
            b, se, s2 = gls_unmix(np.asarray(raw), np.asarray(A), T, donors if paired else None)
            eff, lo, hi, k = gls_calibrated(cal, pi, T, inp.B, inp.Gmat, b, seed=split)
            out = dict(dataset=dsn, split=split, beta=np.asarray(beta).tolist(), gls=b.tolist(), se=se.tolist(),
                       s2=s2, cell_types=list(inp.cell_types), programs=list(inp.programs),
                       gls_cal=eff.tolist(), lo=lo.tolist(), hi=hi.tolist(), k=k.tolist())
            json.dump(out, open(f, "w"))
            e = b - beta
            print("%s s%d  GLS rmse %.3f r %.3f  median se %.3f  resid scale %.1f" % (
                dsn, split, np.sqrt((e ** 2).mean()), np.corrcoef(b.ravel(), np.asarray(beta).ravel())[0, 1],
                np.median(se), s2), flush=True)


if __name__ == "__main__":
    run(sys.argv[1].split(","), cl.db._splits(sys.argv[2]))
