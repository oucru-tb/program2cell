"""Direct engine: cell-type-specific treatment effects by maximum likelihood on the bulk counts.

No reallocation: each bulk sample is a multinomial over genes whose probabilities mix the cell types' shifted
profiles at the sample's read fractions,
    Y_n ~ Multinomial(D_n, q_n),   q_n = sum_k pi_nk softmax(log B_k + delta_nk G),
    delta_nk = gamma_k + T_n beta_k  [+ h_g(n) (donor shift shared by the types) + x_n Gamma (covariates)].
The fractions pi come from any engine (PALM posterior median, PALM-lite, ...); the profiles B from the reference.
Effects are identified, as in unmixing (core.unmix), by how the bulk response varies with the fractions across
samples, but the likelihood uses every gene's counts and is exactly nonlinear (no linearisation, no
reallocation summary).
  - shrinkage (optional): ridge `lam` on the type deviations of beta from their mean (per program), per read
  - intervals: cluster bootstrap (donors if groups, else samples within arms), warm-started refits, a fraction
    draw per replicate if given; effect +- t_{units-1} sd_boot
The multinomial ignores overdispersion and per-sample biological deviations; the bootstrap carries them.
Quasi-likelihood (weights="nb"): the multinomial gives the deepest genes almost all the weight, so a few
high-count genes whose change the programs do not describe (or describe badly) steer every effect. Instead a
weighted Poisson likelihood with each sample's depth profiled out,
    l_n = sum_g w_g Y_ng log q_ng - (sum_g w_g Y_ng) log(sum_g w_g q_ng)
(an unbiased estimating equation for any fixed weights), with NB weights w_g = 1 / (1 + phi_g mu_g): phi_g the
moment estimate of gene g's overdispersion from the residuals of an unweighted fit (Var = mu + phi mu^2).
Gene offsets (gene_offset=True): the reference profiles come from other specimens (another platform, donors,
batch), and every off-program difference between them and the study's cells is otherwise forced into the
program shifts. A per-gene log offset b_g shared by all types and samples, B_kg -> B_kg e^{b_g} (as the
batch correction of BayesPrism / CIBERSORTx), absorbs the common part; it is constant across samples, so it
does not absorb treatment effects (small ridge `lam_b` for identifiability).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import t as tdist


def _unpack(theta, C, P, nx, ng):
    o = 0
    gam = theta[o:o + C * P].reshape(C, P); o += C * P
    beta = theta[o:o + C * P].reshape(C, P); o += C * P
    Gx = theta[o:o + nx * P].reshape(nx, P); o += nx * P
    H = theta[o:o + ng * P].reshape(ng, P)
    return gam, beta, Gx, H


def _delta(theta, T, X, gi, C, P):
    nx = 0 if X is None else X.shape[1]
    ng = 0 if gi is None else int(gi.max())
    gam, beta, Gx, H = _unpack(theta, C, P, nx, ng)
    d = gam[None] + T[:, None, None] * beta[None]
    shared = np.zeros((len(T), P))
    if nx:
        shared += X @ Gx
    if ng:
        Hf = np.vstack([np.zeros((1, P)), H])                       # first group is the baseline
        shared += Hf[gi]
    return d + shared[:, None, :], (gam, beta)


def _n_base(C, P, X, gi):
    return 2 * C * P + (0 if X is None else X.shape[1]) * P + (0 if gi is None else int(gi.max())) * P


def _objective(theta, Y, logpi, logB, G, T, X, gi, lam, wtot, w=None, lam_b=1e-4):
    N, C = logpi.shape
    P = G.shape[0]
    kb = _n_base(C, P, X, gi)
    off = len(theta) > kb                                           # gene offsets b (G,) at the end
    d, (gam, beta) = _delta(theta[:kb], T, X, gi, C, P)
    eta = (logB if not off else logB + theta[kb:][None])[None] + np.einsum("nkp,pg->nkg", d, G)   # (N, C, G)
    lse = logsumexp(eta, -1, keepdims=True)
    lp = eta - lse                                                  # log softmax per type
    lq = logsumexp(logpi[:, :, None] + lp, axis=1)                  # (N, G) log mixture probability
    P_ = np.exp(lp)
    if w is None:
        ll = (Y * lq).sum()
        # gradient wrt delta_nkp: sum_g Y_ng resp_nkg (G_pg - E_k[G_p])
        resp = np.exp(logpi[:, :, None] + lp - lq[:, None, :])      # (N, C, G) posterior of type k for a read
        a = Y[:, None, :] * resp
        gd = np.einsum("nkg,pg->nkp", a, G) - a.sum(-1)[..., None] * np.einsum("nkg,pg->nkp", P_, G)
    else:                                                           # weighted Poisson, depth profiled out
        q = np.exp(lq)
        WY, Wq = Y @ w, q @ w                                       # (N,)
        ll = (w * Y * lq).sum() - (WY * np.log(Wq)).sum()
        cg = w[None] * (Y / q - (WY / Wq)[:, None])                 # (N, G) dl/dq
        a = np.exp(logpi)[:, :, None] * P_ * cg[:, None, :]         # (N, C, G)
        gd = np.einsum("nkg,pg->nkp", a, G) - a.sum(-1)[..., None] * np.einsum("nkg,pg->nkp", P_, G)
    nx = 0 if X is None else X.shape[1]
    ng = 0 if gi is None else int(gi.max())
    grad = [gd.sum(0).ravel(), (T[:, None, None] * gd).sum(0).ravel()]
    if nx:
        grad.append((X.T @ gd.sum(1)).ravel())
    if ng:
        gh = np.zeros((ng + 1, P))
        np.add.at(gh, gi, gd.sum(1))
        grad.append(gh[1:].ravel())
    if off:
        grad.append((a - a.sum(-1, keepdims=True) * P_).sum((0, 1)) - 2 * lam_b * wtot * theta[kb:])
        ll = ll - lam_b * wtot * (theta[kb:] ** 2).sum()
    grad = np.concatenate(grad)
    f, g = -ll / wtot, -grad / wtot
    if lam > 0:                                                     # shrink beta toward its per-program mean
        dev = beta - beta.mean(0, keepdims=True)
        f += lam * (dev ** 2).sum()
        gb = 2 * lam * dev
        g[C * P:2 * C * P] += (gb - gb.mean(0, keepdims=True)).ravel()
    return f, g


def nb_weights(Y, pi, B, G, theta, T, X=None, gi=None):
    """Gene weights 1 / (1 + phi_g mu_g) (mean 1) from the residuals of a fit theta: phi_g the moment estimate
    of the overdispersion (>= 0), mu_g the gene's mean fitted count."""
    N, C = pi.shape
    kb = _n_base(C, G.shape[0], X, gi)
    d, _ = _delta(theta[:kb], T, X, gi, C, G.shape[0])
    lB = np.log(B) + (theta[kb:][None] if len(theta) > kb else 0)
    eta = lB[None] + np.einsum("nkp,pg->nkg", d, G)
    q = np.einsum("nk,nkg->ng", pi, np.exp(eta - logsumexp(eta, -1, keepdims=True)))
    mu = q * Y.sum(1, keepdims=True)
    phi = np.maximum((((Y - mu) ** 2 - mu) / np.maximum(mu, 1e-9) ** 2).mean(0), 0)
    w = 1 / (1 + phi * mu.mean(0))
    return w / w.mean()


def fit_direct(Y, pi, B, G, T, X=None, groups=None, lam=0.0, theta0=None, floor=1e-4, maxiter=500, w=None,
               gene_offset=False):
    """Maximum-likelihood (penalised) gamma, beta (C, P); w (G,) gene weights (quasi-likelihood) or None;
    gene_offset: also a log offset per gene shared by the types. Returns (beta, gamma, theta)."""
    Y = np.asarray(Y, float)
    pi = np.clip(np.asarray(pi, float), floor, None)
    pi = pi / pi.sum(-1, keepdims=True)
    B = np.asarray(B, float)
    B = B / B.sum(1, keepdims=True)
    G = np.asarray(G, float)
    T = np.asarray(T, float)
    N, C = pi.shape
    P = G.shape[0]
    X = None if X is None else np.asarray(X, float).reshape(N, -1)
    gi = None
    if groups is not None:
        gi = np.unique(np.asarray(groups).astype(str), return_inverse=True)[1]
        if gi.max() == 0:
            gi = None
    nx = 0 if X is None else X.shape[1]
    ng = 0 if gi is None else int(gi.max())
    k = 2 * C * P + nx * P + ng * P + (G.shape[1] if gene_offset else 0)
    th0 = np.zeros(k) if theta0 is None or len(theta0) != k else theta0
    wtot = Y.sum() if w is None else (Y @ np.asarray(w, float)).sum()
    res = minimize(_objective, th0, args=(Y, np.log(pi), np.log(B), G, T, X, gi, lam, wtot, w), jac=True,
                   method="L-BFGS-B", options=dict(maxiter=maxiter, gtol=1e-9, ftol=1e-13))
    gam, beta, _, _ = _unpack(res.x, C, P, nx, ng)
    return beta, gam, res.x


def direct_effects(Y, pi, B, G, T, X=None, groups=None, lam=0.0, level=0.9, n_boot=100, pi_draws=None, seed=0,
                   weights=None, gene_offset=False):
    """Effects (C, P) with cluster-bootstrap intervals. weights: None (multinomial), "nb" (quasi-likelihood
    with NB gene weights from a first unweighted fit) or an array (G,); gene_offset: per-gene log offsets
    shared by the types (reference-vs-study mismatch, refitted in every bootstrap replicate).
    Returns dict(effects, lo, hi, se, gamma, n_units, boot, w)."""
    Y, T = np.asarray(Y, float), np.asarray(T, float)
    N = len(T)
    X = None if X is None else np.asarray(X, float).reshape(N, -1)
    X = None if X is None else X - X.mean(0)
    w = None
    if isinstance(weights, str):
        if weights != "nb":
            raise ValueError("weights must be None, 'nb' or an array (G,)")
        _, _, th1 = fit_direct(Y, pi, B, G, T, X, groups, lam, gene_offset=gene_offset)
        pin = np.clip(np.asarray(pi, float), 1e-4, None)
        Bn = np.asarray(B, float)
        gi = None if groups is None else np.unique(np.asarray(groups).astype(str), return_inverse=True)[1]
        gi = None if gi is None or gi.max() == 0 else gi
        w = nb_weights(Y, pin / pin.sum(-1, keepdims=True), Bn / Bn.sum(1, keepdims=True), np.asarray(G, float),
                       th1, T, X, gi)
    elif weights is not None:
        w = np.asarray(weights, float)
    beta, gam, th = fit_direct(Y, pi, B, G, T, X, groups, lam, w=w, gene_offset=gene_offset)
    rng = np.random.default_rng(seed)
    grp = None if groups is None else np.asarray(groups).astype(str)
    units = [np.flatnonzero(grp == g) for g in np.unique(grp)] if grp is not None else [np.array([i]) for i in range(N)]
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
        pb_ = (np.asarray(pi_draws)[rng.integers(len(pi_draws))] if pi_draws is not None else np.asarray(pi))[idx]
        th0 = th if grp is None else None                          # donor effects change with relabelling
        th0 = th if gene_offset else th0                            # (a start only; keeps the gene offsets)
        boot.append(fit_direct(Y[idx], pb_, B, G, T[idx], None if X is None else X[idx], gb, lam, theta0=th0,
                               maxiter=200, w=w, gene_offset=gene_offset)[0])
    boot = np.array(boot)
    se = boot.std(0)
    q = tdist.ppf(0.5 + level / 2, max(len(units) - 1, 1))
    return dict(effects=beta, lo=beta - q * se, hi=beta + q * se, se=se, gamma=gam, n_units=len(units), boot=boot,
                w=w)
