"""Unmixing of reallocated activity: cell-type-specific treatment effects from how the bulk effect varies with
the fractions across samples (docs/method_spec.md, section 4b).

Why. Reallocation splits each gene's reads between types by pi_c B_cg, with the UNSHIFTED profiles B. When one
type's program shifts, every type's reallocated profile is multiplied by nearly the same mixture-averaged
factor, so the shift leaks to all types (on MIX-Seq, a type keeps ~30% of its own shift and the rest spreads
evenly). No per-sample map can undo this, since one sample's bulk does not say which type moved. Across
samples it is identified: the leakage depends on the fractions, which vary.

leakage_operator   A[n, c, q, k, p] = d dhat_ncq / d delta_nkp, the derivative of the reallocated activity of
                   type c (program q) with respect to a true shift of type k along program p, at the reference
                   profiles, in closed form (linearisation of estimate_delta around delta = 0):
                     A_nc.k. = pi_nk S_c^{-1} [G diag(v) G' - (G v) m_k' - m_c (G v)' + (1'v) m_c m_k'],
                     v = B_c * B_k / (pi_n B),  m_k = G B_k,  S_c = G diag(B_c) G' - m_c m_c'
                   (a lone type, pi_nc = 1, gives the identity).
unmix_effects      per program p, stacked over samples n and types c:
                     dhat_ncp = a_cp + sum_k A_nckp (gamma_kp + T_n beta_kp + covariates + donor effects) + e
                   with beta_k = m + u_k and gamma_k = g + v_k shrunk toward their shared values m, g by a ridge
                   on u, v whose size is chosen by REML (empirical Bayes: u_k ~ N(0, tau^2)). A shared response
                   (Kang) selects strong shrinkage, a type-specific one with varied fractions (MIX-Seq) weak
                   shrinkage, a null effect the shared value. Residual variance per type, two passes.
                   Intervals: cluster bootstrap (donors if groups, else samples within arms), refitted
                   with the shrinkage sizes of the full fit, a fraction draw per replicate if given;
                   effect +- t_{units-1} sd_boot.
"""
import itertools

import numpy as np
from scipy.stats import t as tdist

from .realloc import reallocate
from .scores import estimate_delta

LAMBDAS = np.logspace(-4, 3, 29)


# ---- leakage operator -------------------------------------------------------------------------------------
def leakage_operator(pi, B, G, floor=1e-4, diagonal=True):
    """Leakage of true program shifts into reallocated activity, at the reference profiles.
    pi (N, C) read fractions, B (C, G), G (P, G). Returns A (N, C, C, P) [n, c, k, p] = d dhat_ncp / d delta_nkp
    (diagonal=True: same program), or the full (N, C, P, C, P) [n, c, q, k, p]."""
    pi = np.clip(np.asarray(pi, float), floor, None)
    pi = pi / pi.sum(-1, keepdims=True)
    B, G = np.asarray(B, float), np.asarray(G, float)
    B = B / B.sum(1, keepdims=True)
    N, C = pi.shape
    P = G.shape[0]
    mu = pi @ B                                                     # (N, G)
    m = B @ G.T                                                     # (C, P)
    Sinv = []
    for c in range(C):
        S = (G * B[c]) @ G.T - np.outer(m[c], m[c])
        Sinv.append(np.linalg.pinv(S + 1e-10 * np.eye(P)))
    out = np.zeros((N, C, C, P) if diagonal else (N, C, P, C, P))
    for c in range(C):
        for k in range(C):
            v = B[c] * B[k] / mu                                    # (N, G)
            Gv = v @ G.T                                            # (N, P)
            M = np.einsum("qg,ng,pg->nqp", G, v, G, optimize=True)
            M = M - Gv[:, :, None] * m[k][None, None, :] - m[c][None, :, None] * Gv[:, None, :] \
                + v.sum(1)[:, None, None] * np.outer(m[c], m[k])[None]
            Ack = pi[:, k, None, None] * np.einsum("qr,nrp->nqp", Sinv[c], M)
            if diagonal:
                out[:, c, k] = np.einsum("npp->np", Ack)
            else:
                out[:, c, :, k, :] = Ack
    return out


# ---- GLS + REML fit -----------------------------------------------------------------------------------------
def _design(Ap, T, X, grp):
    """Rows (n, c) of one program. Ap (n, C, C) leakage [n, c, k]. Columns: a_c (C) | g, v_k (1 + C) |
    m, u_k (1 + C) | covariates (q) | donor effects (ng - 1); the last two enter as shifts shared by the
    sample's types (through s = A_n 1). Returns Z (n, C, ncol), penalised mask (u and v), effect columns."""
    n, C, _ = Ap.shape
    s = Ap.sum(-1)
    blocks = [np.broadcast_to(np.eye(C), (n, C, C)), s[..., None], Ap, (T[:, None] * s)[..., None],
              T[:, None, None] * Ap]
    if X is not None:
        blocks.append(X[:, None, :] * s[:, :, None])
    if grp is not None:
        lev = np.unique(grp)
        blocks.append((grp[:, None] == lev[None, 1:]).astype(float)[:, None, :] * s[:, :, None])
    Z = np.concatenate(blocks, -1)
    pen = np.zeros(Z.shape[-1], bool)
    pen[C + 1:2 * C + 1] = True
    pen[2 * C + 2:3 * C + 2] = True
    return Z, pen, (2 * C + 1, slice(2 * C + 2, 3 * C + 2))


RHOS = np.concatenate([[0.0], np.logspace(-2, 3, 11)])


def _gls_reml(Z, y, Ap, pen, lam=None, gls=True):
    """Mixed model per program: y_n = Z_n theta + A_n eta_n + e_n, eta_n ~ N(0, rho sigma^2 I) (each sample's
    own per-type deviation, leaked like a real shift), e_n ~ N(0, sigma^2 I); ridge lam * mean diag on the
    type deviations u, v (prior u_k ~ N(0, sigma^2 / (lam scale))). (rho, lam) by REML on a grid, or fixed.
    pen: bool (one lam for every penalised column) or int labels 0 (free), 1..K (K penalty groups, each its own
    lam: full grid for K <= 2, else coordinate ascent from the best common lam; returns (rho, lam_1..lam_K)).
    Z (n, C, k), y (n, C), Ap (n, C, C). Returns (theta, (rho, lam) or (rho, lam_1, ..))."""
    n, C, k = Z.shape
    lab = np.asarray(pen).astype(int)
    groups = [1] if np.asarray(pen).dtype == bool else [g for g in range(1, lab.max() + 1) if (lab == g).any()]
    k0 = int((lab == 0).sum())
    kg = [int((lab == g).sum()) for g in groups]
    AA = np.einsum("nck,ndk->ncd", Ap, Ap) if gls else None
    best = None
    for rho in ((RHOS if gls else [0.0]) if lam is None else [lam[0]]):
        if rho == 0:
            Zw, yw, ldV = Z.reshape(n * C, k), y.ravel(), 0.0
        else:
            L = np.linalg.cholesky(rho * AA + np.eye(C))
            Zw = np.linalg.solve(L, Z).reshape(n * C, k)
            yw = np.linalg.solve(L, y[..., None])[..., 0].ravel()
            ldV = 2 * np.log(np.einsum("ncc->nc", L)).sum()
        ZZ, Zy, yy = Zw.T @ Zw, Zw.T @ yw, yw @ yw
        scale = np.trace(ZZ) / k
        jitter = 1e-9 * scale * np.eye(k)

        def ev(ls):
            d = np.zeros(k)
            for g, l1 in zip(groups, ls):
                d[lab == g] = l1 * scale
            Mx = ZZ + np.diag(d) + jitter
            try:
                theta = np.linalg.solve(Mx, Zy)
            except np.linalg.LinAlgError:
                return -np.inf, None
            prss = max(yy - theta @ Zy, 1e-300)
            return -0.5 * ((n * C - k0) * np.log(prss) + np.linalg.slogdet(Mx)[1]
                           - sum(m * np.log(l1 * scale) for m, l1 in zip(kg, ls)) + ldV), theta

        if lam is not None:
            cand = [tuple(lam[1:1 + len(groups)])]
        elif len(groups) <= 2:
            cand = list(itertools.product(*[LAMBDAS] * len(groups)))
        else:
            cand = None
        if cand is not None:
            for ls in cand:
                ll, theta = ev(ls)
                if theta is not None and (best is None or ll > best[0]):
                    best = (ll, theta, (rho, *ls))
            continue
        # many groups: best common lam, then coordinate ascent over the groups
        cur = max(((ev((l1,) * len(groups))[0], l1) for l1 in LAMBDAS))[1]
        cur = [cur] * len(groups)
        for _ in range(3):
            moved = False
            for i in range(len(groups)):
                lls = [ev(cur[:i] + [l1] + cur[i + 1:])[0] for l1 in LAMBDAS]
                b = LAMBDAS[int(np.argmax(lls))]
                moved |= b != cur[i]
                cur[i] = b
            if not moved:
                break
        ll, theta = ev(cur)
        if best is None or ll > best[0]:
            best = (ll, theta, (rho, *cur))
    return best[1], best[2]


def _design_joint(Af, T, X, grp, separate=False):
    # separate: False (one lam), True (lam_u, lam_v), "program" (lam_u per program, then lam_v)
    """Rows (n, c, q) of all programs jointly. Af (n, C, P, C, P) full leakage [n, c, q, k, p]. Columns:
    a_cq (C P) | per program p: g_p, v_kp (P (1 + C)) | m_p, u_kp (P (1 + C)) | covariates and donor effects
    per program (shifts shared by the sample's types). Returns Z (n, C P, ncol), penalised mask, and the index
    arrays of m_p (P,) and u_kp (C, P)."""
    n, C, P = Af.shape[:3]
    R = C * P
    s = Af.sum(3)                                                   # (n, C, P[q], P[p]): shared shift of p
    Ak = Af.reshape(n, R, C, P)                                     # [n, (c q), k, p]
    sk = s.reshape(n, R, P)
    blocks = [np.broadcast_to(np.eye(R), (n, R, R))]
    for TT in (np.ones(n), np.asarray(T, float)):
        blocks.append(TT[:, None, None] * sk)                       # g_p  or m_p
        blocks.append(TT[:, None, None] * Ak.reshape(n, R, C * P))  # v_kp or u_kp, column (k, p)
    if X is not None:
        blocks.append((X[:, None, :, None] * sk[:, :, None, :]).reshape(n, R, -1))
    if grp is not None:
        lev = np.unique(grp)
        D = (grp[:, None] == lev[None, 1:]).astype(float)
        blocks.append((D[:, None, :, None] * sk[:, :, None, :]).reshape(n, R, -1))
    Z = np.concatenate(blocks, -1)
    pen = np.zeros(Z.shape[-1], int)
    o_v = R + P
    o_m = o_v + C * P
    o_u = o_m + P
    pen[o_v:o_v + C * P] = (P + 1 if separate == "program" else 2) if separate else 1   # baseline deviations v
    pen[o_u:o_u + C * P] = (1 + np.tile(np.arange(P), C)) if separate == "program" else 1  # effect deviations u_kp
    pen = pen if separate else pen.astype(bool)
    return Z, pen, np.arange(o_m, o_m + P), (o_u + np.arange(C * P)).reshape(C, P)


def _pev(Zf, y, pen, lam, scale_of=None):
    """Prediction-error variance of the penalised fit at fixed lam: sigma^2 (Z'Z + Lambda)^-1, sigma^2 from the
    residuals (n - free columns). It contains the shrinkage bias of the type deviations (BLUP), which a
    bootstrap at a fixed lam does not see."""
    k = Zf.shape[1]
    lab = np.asarray(pen).astype(int)
    groups = [1] if np.asarray(pen).dtype == bool else [g for g in range(1, lab.max() + 1) if (lab == g).any()]
    ZZ = Zf.T @ Zf
    scale = np.trace(ZZ) / k
    d = np.zeros(k)
    for g, l1 in zip(groups, lam):
        d[lab == g] = l1 * scale
    Mx = ZZ + np.diag(d) + 1e-9 * scale * np.eye(k)
    Mi = np.linalg.inv(Mx)
    theta = Mi @ (Zf.T @ y)
    r = y - Zf @ theta
    s2 = (r @ r) / max(len(y) - int((lab == 0).sum()), 1)
    return s2 * Mi


def _fit_joint(dhat, Af, T, X, grp, lam=None, pi=None, B=None, G=None, iters=6, tol=1e-3, separate=False,
               want_pev=False, shared_only=False):
    """All programs jointly with the full operator (cross-program leakage), one shrinkage size by REML
    (separate=True: one for the effect deviations u and its own for the baseline deviations v, so a large real
    baseline heterogeneity does not switch off the shrinkage of the effects; P x (0, lam_u, lam_v)).
    Nonlinear chord iterations as _fit. Returns beta (C, P), shared m (P,), (0, lam) per program (P, 2), and
    with want_pev the prediction-error standard deviation of beta (C, P) (_pev) as a fourth element.
    shared_only: no effect deviations u (every type gets the shared effect m_p): the shared-response model."""
    n, C, P = dhat.shape
    Z, pen, mi, ui = _design_joint(Af, T, X, grp, separate)
    if shared_only:
        keep = np.ones(Z.shape[-1], bool)
        keep[ui.ravel()] = False
        Z, pen, mi = Z[..., keep], pen[keep], np.searchsorted(np.flatnonzero(keep), mi)
    Zf = Z.reshape(n * C * P, -1)
    delta = np.zeros((n, C, P))
    f0 = forward(pi, delta, B, G) if pi is not None else None
    last, prev, l = np.inf, None, None if lam is None else tuple(lam[0, 1:])
    for it in range(iters if pi is not None else 1):
        work = dhat if pi is None else             dhat - (forward(pi, delta, B, G) - f0) + np.einsum("ncqkp,nkp->ncq", Af, delta)
        th, lp = _gls_reml(Zf[None], work.reshape(1, -1), np.zeros((1, n * C * P, 1)), pen,
                           None if l is None else (0.0, *l), gls=False)
        l = tuple(lp[1:])
        beta, shared = (np.repeat(th[mi][None], C, 0) if shared_only else th[mi][None] + th[ui]), th[mi]
        if pi is None:
            wk = work
            break
        new = np.asarray(T, float)[:, None, None] * beta[None]
        change = np.abs(new - delta).max()
        if it > 0 and (change > last or not np.isfinite(change)):
            beta, shared, wk = prev
            break
        prev, last, delta, wk = (beta.copy(), shared.copy(), work), change, new, work
        if change < tol:
            break
    lam_out = np.tile([0.0, *l], (P, 1))
    if not want_pev:
        return beta, shared, lam_out
    V = _pev(Zf, wk.ravel(), pen, l)
    sd = np.sqrt(np.maximum(V[mi, mi][None] + V[ui, ui] + 2 * V[ui, mi[None].repeat(C, 0)], 0))
    return beta, shared, lam_out, sd


def forward(pi, delta, B, G, floor=1e-4):
    """Exact expected reallocated activity (N, C, P) when the types' true shifts are delta (N, C, P): the bulk
    is pi_n mixed shifted profiles softmax(log B_k + delta_nk G), reallocated with the reference B."""
    from scipy.special import softmax
    pi = np.clip(np.asarray(pi, float), floor, None)
    pi = pi / pi.sum(-1, keepdims=True)
    Bn = np.asarray(B, float) / np.asarray(B, float).sum(1, keepdims=True)
    Bs = softmax(np.log(Bn)[None] + np.einsum("nkp,pg->nkg", delta, np.asarray(G, float)), -1)
    return reallocate(np.einsum("nk,nkg->ng", pi, Bs) * 1e7, pi, Bn, G)


def _fit(dhat, A, T, X, grp, lam=None, pi=None, B=None, G=None, iters=6, tol=1e-3, gls=True):
    """All programs. dhat (n, C, P), A (n, C, C, P). With pi, B, G: nonlinear fit by chord iterations (working
    response dhat - (forward(delta_t) - forward(0)) + A delta_t with delta_t = T beta_t and the same operator
    A), else linear. The chord iteration stops (keeping the last contracting step) if a step grows: with few
    samples the fit can be noisy enough for it to diverge (Kang).
    Returns beta (C, P), shared m (P,), (rho, lam) per program (P, 2)."""
    n, C, P = dhat.shape
    beta, shared, lams = np.zeros((C, P)), np.zeros(P), np.zeros((P, 2))
    designs = [_design(A[..., p], T, X, grp) for p in range(P)]
    delta = np.zeros((n, C, P))                                    # fitted true shifts per sample
    f0 = forward(pi, delta, B, G) if pi is not None else None
    last, prev = np.inf, None
    for it in range(iters if pi is not None else 1):
        work = dhat if pi is None else             dhat - (forward(pi, delta, B, G) - f0) + np.einsum("nckp,nkp->ncp", A, delta)
        new = np.zeros_like(delta)
        for p in range(P):
            Z, pen, (mi, ui) = designs[p]
            th, l = _gls_reml(Z, work[:, :, p], A[..., p], pen, None if lam is None else lam[p], gls)
            beta[:, p] = th[mi] + th[ui]
            shared[p], lams[p] = th[mi], l
            # the nonlinear correction uses the treatment part only: the baseline gamma (and donor and covariate
            # shifts) are nearly confounded with the type intercepts a_c, so their size is arbitrary
            new[:, :, p] = T[:, None] * beta[None, :, p]
        if pi is None:
            break
        change = np.abs(new - delta).max()
        if it > 0 and (change > last or not np.isfinite(change)):
            beta, shared, lams = prev                               # not contracting: keep the previous step
            break
        prev, last, delta = (beta.copy(), shared.copy(), lams.copy()), change, new
        if change < tol:
            break
    return beta, shared, lams


# ---- estimator with intervals ------------------------------------------------------------------------------
def unmix_effects(dhat, A, T, X=None, groups=None, level=0.9, n_boot=200, dhat_draws=None, A_draws=None,
                  seed=0, pi=None, B=None, G=None, pi_draws=None, gls=True, separate=False, pev=False,
                  misfit_tau=None, Y=None, shared_fit=False):
    """Cell-type-specific treatment effects (C, P) from reallocated activity dhat (N, C, P) and the leakage
    operator A from leakage_operator on the same fractions: (N, C, C, P) (diagonal: programs one at a time) or
    (N, C, P, C, P) (diagonal=False: all programs jointly, with cross-program leakage; recommended). T (N,) 0/1; X (N, q) covariates;
    groups (N,) donor / pair ids (a donor-level shift shared by its types). pi, B, G: fractions and reference
    for the nonlinear fit (recommended; linear without). Fraction uncertainty: matched dhat_draws (S, N, C, P),
    A_draws (S, N, C, C, P) and pi_draws (S, N, C), one draw per bootstrap replicate.
    separate: REML penalty groups (_fit_joint). pev=True (joint operator): se = max(bootstrap sd, prediction-error
    sd), so the interval also carries the shrinkage bias of the type deviations. misfit_tau (offprogram_tau):
    add the error sd of an off-program treatment response of that size per type and gene (misfit_sd) in
    quadrature; "auto" (needs the bulk Y): its size measured by offprogram_tau at the point fit.
    shared_fit (joint operator): also the shared-response model (_fit_joint shared_only), bootstrapped on the
    same resamples: 'common' (P,) its effect with 'common_se', and 'deviation' (C, P) = effects - common with
    'deviation_se' (bootstrap sd, at least the prediction-error sd with pev, plus the misfit sd).
    Returns dict(effects, lo, hi, se, se_boot, pev_sd, shared (P,), lam (P, 2), n_units, boot)."""
    dhat, A, T = np.asarray(dhat, float), np.asarray(A, float), np.asarray(T, float)
    N = len(T)
    X = None if X is None else np.asarray(X, float).reshape(N, -1)
    X = None if X is None else X - X.mean(0)
    grp = None if groups is None else np.asarray(groups).astype(str)
    nl = dict(pi=np.asarray(pi, float), B=B, G=G) if pi is not None else {}
    pev_sd = None
    if pev and A.ndim == 5:
        beta, shared, lam, pev_sd = _fit_joint(dhat, A, T, X, grp, separate=separate, want_pev=True, **nl)
    fit = (lambda *a, **k: _fit_joint(*a, separate=separate, **k)) if A.ndim == 5 else         (lambda *a, **k: _fit(*a, gls=gls, **k))
    if pev_sd is None:
        beta, shared, lam = fit(dhat, A, T, X, grp, **nl)
    shared_fit = shared_fit and A.ndim == 5
    if shared_fit:
        sfit = lambda *a, **k: _fit_joint(*a, separate=separate, shared_only=True, **k)
        common, lam_c = sfit(dhat, A, T, X, grp, **nl)[1:]
        cboot = []
    rng = np.random.default_rng(seed)
    if grp is not None:
        units = [np.flatnonzero(grp == g) for g in np.unique(grp)]
    else:
        units = [np.array([i]) for i in range(N)]
    boot = []
    for _ in range(n_boot):
        if grp is not None:
            pick = rng.integers(len(units), size=len(units))
            idx = np.concatenate([units[i] for i in pick])
            gb = np.concatenate([np.full(len(units[i]), j) for j, i in enumerate(pick)]).astype(str)
        else:
            i1, i0 = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
            idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
            gb = None
        if len(np.unique(T[idx])) < 2:
            continue
        if dhat_draws is not None:
            s = rng.integers(len(dhat_draws))
            yb, Ab = np.asarray(dhat_draws[s])[idx], np.asarray(A_draws[s])[idx]
            pb_ = None if pi is None else np.asarray(pi_draws[s] if pi_draws is not None else pi)[idx]
        else:
            yb, Ab = dhat[idx], A[idx]
            pb_ = None if pi is None else np.asarray(pi)[idx]
        nlb = dict(pi=pb_, B=B, G=G, iters=3) if pi is not None else {}
        boot.append(fit(yb, Ab, T[idx], None if X is None else X[idx], gb, lam=lam, **nlb)[0])
        if shared_fit:
            cboot.append(sfit(yb, Ab, T[idx], None if X is None else X[idx], gb, lam=lam_c, **nlb)[1])
    boot = np.array(boot)
    se = boot.std(0)
    se_boot = se
    if pev_sd is not None:                     # shrinkage bias (prediction-error variance) as well
        se = np.maximum(se, pev_sd)
    mis = tau = None
    if isinstance(misfit_tau, str):            # "auto": its size from the bulk (offprogram_tau at the fit)
        tau = offprogram_tau(Y, pi, B, G, T, beta, groups=grp)[0]
        misfit_tau = tau
    if misfit_tau is not None and A.ndim == 5 and pi is not None:   # off-program treatment response
        mis = misfit_sd(pi, B, G, T, beta, misfit_tau, lam, None if X is None else X, grp, separate,
                        rng=np.random.default_rng(seed + 7))
        se = np.sqrt(se ** 2 + mis ** 2)
    q = tdist.ppf(0.5 + level / 2, max(len(units) - 1, 1))
    res = dict(effects=beta, lo=beta - q * se, hi=beta + q * se, se=se, se_boot=se_boot, pev_sd=pev_sd,
               misfit_sd=mis, tau=misfit_tau, shared=shared, lam=lam, n_units=len(units), boot=boot)
    if shared_fit:
        cboot = np.array(cboot)
        dse = (boot - cboot[:, None]).std(0)
        dse = dse if pev_sd is None else np.maximum(dse, pev_sd)
        dse = dse if mis is None else np.sqrt(dse ** 2 + mis ** 2)
        res.update(common=common, common_se=cboot.std(0), deviation=beta - common[None], deviation_se=dse)
    return res


def unmix(Y, fractions, B, G, T, X=None, groups=None, fraction_draws=None, n_draws=10, **kw):
    """Convenience: reallocate, build the leakage operator and unmix. fraction_draws (S, N, C): up to n_draws
    of them carry fraction uncertainty into the bootstrap."""
    dh = reallocate(Y, fractions, B, G)
    A = leakage_operator(fractions, B, G)
    dd = Ad = None
    if fraction_draws is not None:
        z = np.asarray(fraction_draws, float)
        z = z[np.linspace(0, len(z) - 1, min(n_draws, len(z))).astype(int)]
        dd = reallocate(Y, z, B, G)
        Ad = np.stack([leakage_operator(p, B, G) for p in z])
    return unmix_effects(dh, A, T, X, groups, dhat_draws=dd, A_draws=Ad, pi=fractions, B=B, G=G,
                         pi_draws=None if fraction_draws is None else z, **kw)


# ---- design-aware choice and interval calibration -----------------------------------------------------------
def simulate_cohort(cal, pi, T, rng, scale=0.6, kinds=(0.2, 0.3, 0.5), max_log=6.0, offprog_sd=0.0,
                    beta_fn=None):
    """One synthetic cohort shaped like the study: N = len(pi) bulks with the study's fractions pi (N, C) and
    treatment T, built from real per-type counts of reference specimens (one calibration fold, whose profiles
    B_k exclude those donors), with planted treatment effects beta (C, P) per program drawn as null / shared by
    all types / localised to a random proper subset of types with probabilities `kinds`, sizes U(-scale, scale)
    (scale: scalar or per program (P,)).
    Treated specimens' type counts are redrawn as Poisson(x e^{beta_c G}) (rescaled to their total); each bulk
    thins every type to its target read fraction. Returns (Y (N, G), true read fractions (N, C), B, beta), beta
    the realised shift (estimate_delta of the treated type counts minus the untreated, averaged).
    offprog_sd: the treatment also shifts every type's genes by independent N(0, offprog_sd^2) logs (drawn once
    per cohort: a response outside the programs, as offprogram_tau measures on real data).
    beta_fn(rng) -> (C, P): planted effects drawn by the caller instead (design_check plant="observed")."""
    real = [np.asarray(c)[~np.asarray(p, bool)] for c, p in zip(cal.counts, cal.planted)]
    folds = [k for k, r in enumerate(real) if len(r)]
    k = folds[rng.integers(len(folds))]
    X, B, G = real[k], cal.B[k], cal.G
    N, C = np.asarray(pi).shape
    P = G.shape[0]
    beta = np.zeros((C, P))
    sc = np.broadcast_to(np.asarray(scale, float), (P,))
    for p in (range(P) if beta_fn is None else []):
        kind = rng.choice(3, p=kinds)
        if kind == 1:
            beta[:, p] = rng.uniform(-sc[p], sc[p])
        elif kind == 2:
            on = rng.permutation(C)[:rng.integers(1, C)]
            beta[on, p] = rng.uniform(-sc[p], sc[p], len(on))
    if beta_fn is not None:
        beta = np.asarray(beta_fn(rng), float)
    logf = beta @ G
    if offprog_sd:
        logf = logf + rng.normal(0, offprog_sd, logf.shape)
    logf = np.clip(logf, -max_log, max_log)                         # (C, G)
    Y, pt = np.zeros((N, G.shape[1])), np.zeros((N, C))
    real_shift = []
    for n in range(N):
        x = X[rng.integers(len(X))].copy()
        if T[n] == 1:
            mu = x * np.exp(logf)
            x0 = x
            x = rng.poisson(mu * x.sum(-1, keepdims=True) / np.maximum(mu.sum(-1, keepdims=True), 1e-12)).astype(float)
            real_shift.append(estimate_delta(x, B, G) - estimate_delta(x0, B, G))
        tot = x.sum(-1)
        target = np.clip(np.asarray(pi[n], float), 1e-3, None)
        D = (tot / target).min()
        xs = rng.binomial(np.rint(x).astype(np.int64), np.clip(target * D / np.maximum(tot, 1), 0, 1)[:, None])
        Y[n] = xs.sum(0)
        pt[n] = xs.sum(-1) / max(xs.sum(), 1)
    # the truth is the realised shift (the clip of beta G and Poisson noise make it differ from beta)
    return Y, pt, B, (np.mean(real_shift, 0) if real_shift else beta)


# ---- off-program treatment response ------------------------------------------------------------------------
def offprogram_tau(Y, pi, B, G, T, beta, groups=None, rng=None, n_boot=100):
    """Size of the treatment response outside the programs: per gene, the bulk log fold change (treated vs
    untreated arm means of log CPM, the more expressed half of the genes) minus the one the forward model
    predicts at the program effects beta (C, P); its variance beyond the bootstrap sampling noise is tau_bulk^2.
    Returned per type, tau = tau_bulk / sqrt(sum_k w_k^2) (w the mean read fractions): independent responses
    of the types average in the bulk (conservative if they are shared). Returns (tau, tau_bulk)."""
    from scipy.special import softmax
    rng = np.random.default_rng(0) if rng is None else rng
    Y, T = np.asarray(Y, float), np.asarray(T)
    pi = np.clip(np.asarray(pi, float), 1e-4, None)
    pi = pi / pi.sum(-1, keepdims=True)
    Bn = np.asarray(B, float) / np.asarray(B, float).sum(1, keepdims=True)
    cpm = Y / Y.sum(1, keepdims=True)
    keep = cpm.mean(0) >= np.quantile(cpm.mean(0), 0.5)
    lc = np.log(cpm[:, keep] + 1e-7)
    mean_c = lambda a, idx: a[idx[T[idx] == 1]].mean(0) - a[idx[T[idx] == 0]].mean(0)
    lfc = mean_c(lc, np.arange(len(T)))
    Bs = softmax(np.log(Bn)[None] + np.einsum("nkp,pg->nkg", T[:, None, None] * np.asarray(beta)[None],
                                              np.asarray(G, float)), -1)
    f = np.log(np.einsum("nk,nkg->ng", pi, Bs)[:, keep] + 1e-7)
    res = lfc - mean_c(f, np.arange(len(T)))
    if groups is not None:
        g = np.asarray(groups).astype(str)
        units = [np.flatnonzero(g == u) for u in np.unique(g)]
    else:
        units = [np.array([i]) for i in range(len(T))]
    bs = []
    for _ in range(n_boot):
        if groups is not None:
            idx = np.concatenate([units[i] for i in rng.integers(len(units), size=len(units))])
        else:
            i1, i0 = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
            idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        if T[idx].min() == T[idx].max():
            continue
        bs.append(mean_c(lc, idx))
    noise = np.var(bs, 0).mean()
    tau_bulk = np.sqrt(max(res.var() - noise, 0.0))
    w = pi.mean(0)
    return tau_bulk / np.sqrt((w ** 2).sum()), tau_bulk


def _forward_genes(pi, logshift, Bn, G):
    """Expected reallocated activity (N, C, P) when each type's profile is shifted by logshift (N, C, G)."""
    from scipy.special import softmax
    Bs = softmax(np.log(Bn)[None] + logshift, -1)
    return reallocate(np.einsum("nk,nkg->ng", pi, Bs) * 1e7, pi, Bn, G)


def misfit_sd(pi, B, G, T, beta, tau, lam, X=None, groups=None, separate=False, n=40, rng=None, floor=1e-4):
    """Error sd (C, P) of the unmixed effects caused by an off-program treatment response: per draw, every
    type's treated profile gets independent N(0, tau^2) log shifts on every gene (on top of the program effects
    beta); the change of the reallocated activity (exact forward model) is unmixed with the fitted operator at
    the fitted shrinkage lam (linear), and compared with the change of the realised truth (each type's own
    shift, estimate_delta). The response is the treatment's, the same in every treated sample, so it does not
    average out over samples. Returns sqrt(mean squared error) over the draws."""
    rng = np.random.default_rng(0) if rng is None else rng
    T = np.asarray(T)
    pi = np.clip(np.asarray(pi, float), floor, None)
    pi = pi / pi.sum(-1, keepdims=True)
    Bn = np.asarray(B, float) / np.asarray(B, float).sum(1, keepdims=True)
    G = np.asarray(G, float)
    N, C = pi.shape
    Af = leakage_operator(pi, Bn, G, diagonal=False)
    tr = np.flatnonzero(T == 1)
    base = np.asarray(beta, float) @ G                              # (C, Gn) program shift of treated types
    d0 = _forward_genes(pi[tr], np.broadcast_to(base, (len(tr),) + base.shape), Bn, G)
    t0 = estimate_delta(np.exp(np.log(Bn) + base) / np.exp(np.log(Bn) + base).sum(1, keepdims=True) * 1e7, Bn, G)
    grp = None if groups is None else np.asarray(groups).astype(str)
    err = []
    for _ in range(n):
        eps = rng.normal(0, tau, Bn.shape)
        sh = base + eps
        dd = np.zeros((N, C, G.shape[0]))
        dd[tr] = _forward_genes(pi[tr], np.broadcast_to(sh, (len(tr),) + sh.shape), Bn, G) - d0
        e = np.exp(np.log(Bn) + sh)
        dt = estimate_delta(e / e.sum(1, keepdims=True) * 1e7, Bn, G) - t0
        b = _fit_joint(dd, Af, T, X, grp, lam=lam, separate=separate)[0]
        err.append(b - dt)
    return np.sqrt((np.array(err) ** 2).mean(0))


def contrast_se(dhat, T, rng, n_boot=100, groups=None):
    """Bootstrap standard error (C, P) of the difference of arm means of dhat: samples within arms, or, with
    groups (e.g. donors of a paired design), whole groups, so the pairing's shared variation cancels."""
    T = np.asarray(T)
    if groups is not None:
        g = np.asarray(groups).astype(str)
        units = [np.flatnonzero(g == u) for u in np.unique(g)]
        out = []
        for _ in range(n_boot):
            idx = np.concatenate([units[i] for i in rng.integers(len(units), size=len(units))])
            t = T[idx]
            if t.min() == t.max():
                continue
            out.append(dhat[idx[t == 1]].mean(0) - dhat[idx[t == 0]].mean(0))
        return np.std(out, 0)
    i1, i0 = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
    out = [dhat[rng.choice(i1, len(i1))].mean(0) - dhat[rng.choice(i0, len(i0))].mean(0) for _ in range(n_boot)]
    return np.std(out, 0)


def design_check(cal, pi, T, X=None, groups=None, n_sim=10, scale=0.6, level=0.9, n_boot=20, nonlinear=True,
                 seed=0, estimator="diff", joint=True, effect_hint=None, offprog_sd=0.0, plant="prior",
                 kinds=(0.2, 0.3, 0.5)):
    """Simulated error of the unmixing estimator vs reallocation (the effect of reallocated activity, rescaled
    per program by a slope fitted on the other cohorts, as calibration does) under this study's design: per program RMSE over cohorts and planted
    effects (simulate_cohort), and the level-quantile of the unmixing errors studentised by their bootstrap
    standard error (to widen the bootstrap interval for bias). effect_hint (C, P), e.g. the calibrated effects:
    planted sizes per program = clip(2 max_c |hint|, 0.05, 1), so the choice is made for effects of the size the
    study shows (with large planted effects unmixing wins even where the real effects are tiny). Returns dict(rmse_unmix (P,), rmse_realloc (P,),
    z_q (scalar), use_unmix (P,) bool, n_sim, slope (P,) of the rescaled reallocation, rmse_unmix_cp,
    rmse_realloc_cp, use_unmix_cp (C, P) per type and program (type x program model), z_q_realloc (P,) the
    level-quantile of its errors studentised by slope * contrast_se). offprog_sd: simulated cohorts carry an
    off-program treatment response of this size (simulate_cohort).
    plant: "prior" (default) plants null / shared / localised effects of size clip(2 max_c |hint|) per program;
    "observed" (needs effect_hint) plants the study's own shared effect (the mean of the hint over types) plus
    type deviations drawn like the prior's (none / every type / a random subset, probabilities `kinds`) of size
    clip(2 max_c |hint_c - mean|, 0.05, 1): the leakage check then matches how type-specific this study's
    response looks, instead of assuming large type-specific responses wherever the effects are large."""
    from ..causal import estimator_weights
    rng = np.random.default_rng(seed + 11)
    beta_fn = None
    if plant == "observed":
        hint = np.asarray(effect_hint, float)
        shared = hint.mean(0)
        dev_scale = np.clip(2 * np.abs(hint - shared[None]).max(0), 0.05, 1.0)

        def beta_fn(r):
            C, P = hint.shape
            b = np.broadcast_to(shared, (C, P)).copy()
            for p in range(P):
                kind = r.choice(3, p=kinds)
                if kind == 1:
                    b[:, p] += r.uniform(-dev_scale[p], dev_scale[p], C)
                elif kind == 2:
                    on = r.permutation(C)[:r.integers(1, C)]
                    b[on, p] += r.uniform(-dev_scale[p], dev_scale[p], len(on))
            return b
    elif effect_hint is not None:               # plant effects of the size this study shows, per program
        scale = np.clip(2 * np.abs(np.asarray(effect_hint, float)).max(0), 0.05, 1.0)
    T = np.asarray(T)
    w = estimator_weights(T, X, groups, "ols" if (X is not None or groups is not None) else estimator)
    G = cal.G
    eu, rr, bt, z, rse = [], [], [], [], []
    for _ in range(n_sim):
        Y, pt, B, beta = simulate_cohort(cal, pi, T, rng, scale, offprog_sd=offprog_sd, beta_fn=beta_fn)
        dh = reallocate(Y, pt, B, G)
        rr.append(np.einsum("n,ncp->cp", w, dh))
        rse.append(contrast_se(dh, T, rng, groups=groups))
        bt.append(beta)
        A = leakage_operator(pt, B, G, diagonal=not joint)
        r = unmix_effects(dh, A, T, X, groups, level=level, n_boot=n_boot, seed=int(rng.integers(1 << 30)),
                          **(dict(pi=pt, B=B, G=G) if nonlinear else {}))
        eu.append(r["effects"] - beta)
        z.append(np.abs(r["effects"] - beta) / np.maximum(r["se"], 1e-9))
    eu, rr, bt, z = np.array(eu), np.array(rr), np.array(bt), np.array(z)
    # reallocation rescaled like calibration: per program one slope (pooled over types), leave one cohort out
    er = np.zeros_like(rr)
    for i in range(n_sim):
        o = np.arange(n_sim) != i
        b = (rr[o] * bt[o]).sum((0, 1)) / np.maximum((rr[o] ** 2).sum((0, 1)), 1e-12)
        er[i] = b * rr[i] - bt[i]
    rmse_u, rmse_r = np.sqrt((eu ** 2).mean((0, 1))), np.sqrt((er ** 2).mean((0, 1)))
    slope = (rr * bt).sum((0, 1)) / np.maximum((rr ** 2).sum((0, 1)), 1e-12)
    zr = np.abs(er) / np.maximum(np.abs(slope)[None, None] * np.array(rse), 1e-9)
    # per (type, program): a multiplicative type x program model of the mean squared error (10 cohorts are too
    # few for each cell on its own; the types differ mostly by how much of their own shift they keep)
    def cell(e):
        m = (e ** 2).mean(0)                                        # (C, P)
        return np.sqrt(m.mean(1, keepdims=True) * m.mean(0, keepdims=True) / max(m.mean(), 1e-24))
    rmse_u_cp, rmse_r_cp = cell(eu), cell(er)
    # absolute leakage error of rescaled reallocation: level-quantile per program, per type by the cell model
    q_abs = np.quantile(np.abs(er), level, axis=(0, 1))
    q_abs_cp = q_abs[None] * rmse_r_cp / np.maximum(rmse_r[None], 1e-12)
    return dict(rmse_unmix=rmse_u, rmse_realloc=rmse_r, z_q=float(np.quantile(z, level)), q_abs_realloc_cp=q_abs_cp,
                use_unmix=rmse_u < rmse_r, n_sim=n_sim, slope=slope, z_q_realloc=np.quantile(zr, level, axis=(0, 1)),
                rmse_unmix_cp=rmse_u_cp, rmse_realloc_cp=rmse_r_cp, use_unmix_cp=rmse_u_cp < rmse_r_cp)
