"""Two-stage (modular) PALM: the fast route. Entry point: `run_palm_fast(inputs)`.

The full model (bayes.model) samples every sample's fractions and program shifts jointly with the
population parameters. Here the work is split:

Stage 0  Gene-level plug-ins: NB dispersion theta_g by moments (core.data.gene_dispersion) and the
         platform scale m_g from a short pooled VI fit of the full model.
Stage 1  Each sample on its own (in parallel). Unknowns x_i = (eta_i: C composition logits,
         delta_i: C*P program shifts per cell type). A weak Normal prior keeps the mode finite;
         Fisher scoring finds it, and the sample's log-likelihood is summarised by a quadratic in
         canonical form, l_i(x) ~ -1/2 x' Lam_i x + h_i' x, with Lam_i the expected Fisher
         information at the mode and h_i = grad_i + Lam_i x_hat_i. The weak prior is not part of
         the summary, so stage 2 does not count it twice and effects are not shrunk by it.
Stage 2  The population model of bayes.model (composition: a + gamma T + covariates + noise;
         activity: kappa + beta T + covariates + per-sample variation, 'independent', 'shared' or
         'shared_only') on the summaries. Given the population parameters every x_i is Gaussian, so it
         is integrated out exactly: NUTS samples only the population parameters (~100-200 numbers, no
         per-sample latents, no funnel), and x_i is then drawn from its exact Gaussian conditional.

Approximations relative to the full model: m_g and theta_g are plugged in, each sample's likelihood is
replaced by its Gaussian (Laplace) summary, and stage 1 does not use the treatment or the other samples
(a "cut": the population model cannot pull a sample's estimate away from what its own counts say,
except through the exact stage-2 combination of prior and summary).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import xlogy
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import ClippedAdam

from ..core.data import gene_dispersion
from ..core.equations import effect_decomposition
from ..compute import resolve_chain_method
from .diagnostics import convergence
from .fit import model_args, svi_warm_start, CONVERGENCE_SITES
from .model import make_model, _horseshoe, _iid_normal, ACTIVITY_MODELS


# ---- stage 0: gene-level plug-ins ------------------------------------------------------------------
def gene_scale(args, n_prog, theta, groups=None, steps=2000, sigma_m="auto", seed=0):
    """Platform scale m_g (model gene order) from a short pooled VI fit of the full model.
    sigma_m='auto' learns the scale's size (HalfNormal(0.3) prior), so m_g stays near 1 when bulk and
    reference come from the same platform (e.g. pseudo-bulk); a number fixes it. Returns (m, sigma_m)."""
    model = make_model(True, sigma_m, theta, False, "fixed", n_prog, groups)
    guide = AutoNormal(model, init_loc_fn=init_to_median)
    svi = SVI(model, guide, ClippedAdam(0.01), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(seed + 30_000), steps, *args, progress_bar=False, stable_update=True)
    med = guide.median(res.params)
    sm = med["sigma_m"] if sigma_m == "auto" else sigma_m
    return jnp.exp(sm * med["zm"]), float(sm)


# ---- stage 1: one Gaussian summary per sample ------------------------------------------------------
LIKELIHOODS = ("nb", "quasi")


def gene_blocks(B, G, n_blocks=100, seed=0):
    """Group genes whose deviations from the model are likely to move together: k-means on each gene's
    reference profile shape (log B centred over cell types) and its program weights. Used by the block
    sandwich in sample_summaries. Returns (G,) integer labels."""
    from scipy.cluster.vq import kmeans2
    lb = np.log(np.asarray(B, float) + 1e-12)
    f = np.concatenate([(lb - lb.mean(0)).T, np.asarray(G, float).T], 1)
    f = (f - f.mean(0)) / (f.std(0) + 1e-12)
    k = int(min(n_blocks, max(2, f.shape[0] // 5)))
    _, lab = kmeans2(f, k, seed=seed, minit="++")
    return lab


def sample_summaries(Y, L, B, G, theta, m=None, eta0=None, sd_eta=3.0, sd_delta=1.0, iters=80,
                     likelihood="nb", sandwich=False, blocks=None, fix_eta=False):
    """Gaussian summary of every sample's information about x = (eta (C), delta (C*P), cell type major).
    Y (N,G) counts, L (N,), B (C,G), G (P,G), theta (G,) or scalar, m (G,) or None, eta0 (N,C) start.

    Each sample's x solves estimating equations psi(x) = sum_g psi_g(x) = 0 (plus a weak prior):
      likelihood='nb'     the negative-binomial score (dispersion theta) for all of x;
      likelihood='quasi'  the Poisson score for all of x: genes weighted by their counts, as in the
                          multinomial estimators that define program activity (a loss chosen for the
                          estimand rather than a full likelihood, i.e. generalised Bayes; pair it with
                          sandwich=True for honest uncertainty).
    (A 'mixed' loss, NB for eta and Poisson for delta, was tried and dropped: its alternating solution did
    not converge reliably and did not keep the NB fractions, see benchmarks notes 2026-10-26.)
    The summary is the quadratic approximation of the GMM criterion 1/2 psi' V^-1 psi around the root,
    i.e. information Lam = A' V^-1 A and h = A' V^-1 (psi + A x), with A = -d psi/dx (expected):
      sandwich=False  V = A (a likelihood) or phi*A for 'quasi' (phi = Pearson X^2/(G - D));
      sandwich=True   V = sum over gene blocks of the block's summed score outer product (blocks: (G,)
                      labels, see gene_blocks): the robust (Godambe) information, which widens intervals in
                      the directions where genes disagree with the model beyond its noise model.
    fix_eta=True holds eta at eta0 (e.g. fractions from another fit) and solves for delta alone; Lam and h
    then carry information about delta only (zero eta block), conditional on those fractions.
    Returns dict(x_hat, Lam, h, grad_norm, phi, inflation): grad_norm is the largest remaining Newton
    step (near 0 at the root); inflation is the median ratio of robust to
    model-based variance of delta (1 = the model's noise is right; > 1 = the model is too confident)."""
    if likelihood not in LIKELIHOODS:
        raise ValueError("likelihood must be one of %s" % (LIKELIHOODS,))
    if sandwich and blocks is None:
        raise ValueError("sandwich=True needs gene blocks (see gene_blocks)")
    N, Gn = Y.shape
    C, P = B.shape[0], G.shape[0]
    D = C + C * P
    logB = jnp.log(B)
    prec = jnp.concatenate([jnp.full(C, sd_eta ** -2.0), jnp.full(C * P, sd_delta ** -2.0)])
    nb = likelihood == "nb"
    if fix_eta and eta0 is None:
        raise ValueError("fix_eta=True needs eta0")
    free = jnp.concatenate([jnp.full(C, 0.0 if fix_eta else 1.0), jnp.ones(C * P)])
    nblk = 1 if blocks is None else int(np.max(blocks)) + 1
    blocks = jnp.zeros(Gn, int) if blocks is None else jnp.asarray(blocks)

    def frac(x):
        pi = jax.nn.softmax(x[:C])
        E = jax.nn.softmax(logB + x[C:].reshape(C, P) @ G, axis=-1)
        raw = pi @ E
        if m is not None:
            raw = raw * m
            raw = raw / raw.sum()
        return raw

    def parts(x, y, l):
        """Per-gene score contributions S (G, D) and expected A = -d psi/dx (D, D)."""
        J = jax.jacfwd(frac)(x)                                        # (G, D)
        mu = l * frac(x)
        r_nb, r_p = y / mu - (y + theta) / (mu + theta), y / mu - 1.0
        w_nb, w_p = l ** 2 / (mu + mu ** 2 / theta), l ** 2 / mu
        r, w = (r_nb, w_nb) if nb else (r_p, w_p)
        return J * (l * r)[:, None], (J * w[:, None]).T @ J, mu

    def loss(x, y, l, nb):                  # log-likelihood used as the objective of a block
        mu = l * frac(x)
        return jnp.sum(xlogy(y, mu) - (y + theta) * jnp.log(mu + theta)) if nb else jnp.sum(xlogy(y, mu) - mu)

    ts = jnp.concatenate([2.0 ** -jnp.arange(13.0), jnp.zeros(1)])   # backtrack far: count weighting on
    # overdispersed data makes the expected-information curvature a poor guide, so good steps can be short

    def newton(x, y, l):
        """Fisher-scoring step with a backtracking line search on the penalised objective."""
        obj = lambda z: loss(z, y, l, nb) - 0.5 * jnp.sum(prec * z ** 2)
        _, A, _ = parts(x, y, l)
        H = A * free[:, None] * free[None, :] + jnp.diag(prec * free + (1.0 - free))
        dx = jnp.linalg.solve(H, jax.grad(obj)(x) * free) * free
        dx = dx * jnp.minimum(1.0, 2.0 / (jnp.max(jnp.abs(dx)) + 1e-12))           # at most 2 log units
        vals = jax.vmap(lambda t: obj(x + t * dx))(ts)
        return x + ts[jnp.argmax(vals)] * dx

    def one(y, l, x0):
        x, _ = jax.lax.scan(lambda x, _: (newton(x, y, l), None), x0, None, length=iters)
        S, A, mu = parts(x, y, l)
        if fix_eta:                 # information about delta only, given eta (zero eta rows/columns)
            S, A = S * free[None, :], A * free[:, None] * free[None, :]
        g = S.sum(0)
        phi = jnp.maximum(jnp.sum((y - mu) ** 2 / mu) / (Gn - D), 1.0)
        if sandwich:
            Sb = jax.ops.segment_sum(S, blocks, num_segments=nblk)        # (blocks, D)
            V = Sb.T @ Sb
            V = V + 1e-8 * jnp.trace(V) / D * jnp.eye(D)
        else:
            V = A * (phi if likelihood == "quasi" else 1.0)
        V = V + jnp.diag(1.0 - free)             # keep V invertible when the eta block is held fixed
        VA = jnp.linalg.solve(V, A)                                        # V^-1 A
        Lam = A.T @ VA
        h = VA.T @ (g + A @ x)
        Am = 0.5 * (A + A.T) / (phi if likelihood == "quasi" else 1.0)     # model-based information
        dd = jnp.arange(C, D)
        infl = jnp.median(jnp.diag(jnp.linalg.pinv(Lam))[dd] / jnp.diag(jnp.linalg.pinv(Am))[dd])
        # convergence: size of the Newton step that would still be taken (scale-free, unlike the raw score)
        nstep = jnp.max(jnp.abs(jnp.linalg.solve(A + jnp.diag(prec * free + (1.0 - free)),
                                                 (g - prec * x) * free)))
        return x, Lam, h, nstep, phi, infl

    x0 = jnp.zeros((N, D))
    if eta0 is not None:
        x0 = x0.at[:, :C].set(eta0)
    x, Lam, h, gn, phi, infl = jax.jit(jax.vmap(one))(Y, L, x0)
    return dict(x_hat=x, Lam=0.5 * (Lam + jnp.swapaxes(Lam, 1, 2)), h=h, grad_norm=gn,
                phi=phi if likelihood == "quasi" else None, inflation=infl)


def _two_step_summaries(Y, L, B, G, theta, m, eta0, sd_eta, sd_delta, iters, likelihood, sandwich, blocks):
    """Stage 1 in two steps (see fit_fast, delta_given_eta): NB fit for eta, marginal information for eta
    (Schur complement over delta), then delta given the NB eta. Returns the same dict as sample_summaries,
    with a block-diagonal Lam (eta and delta treated as independent pieces of information)."""
    C = B.shape[0]
    a = sample_summaries(Y, L, B, G, theta, m, eta0, sd_eta, sd_delta, iters, "nb")
    La, ha = a["Lam"], a["h"]
    Lee, Led, Ldd = La[:, :C, :C], La[:, :C, C:], La[:, C:, C:]
    K = jnp.swapaxes(jnp.linalg.solve(Ldd, jnp.swapaxes(Led, 1, 2)), 1, 2)          # Led Ldd^-1
    Lm = Lee - K @ jnp.swapaxes(Led, 1, 2)
    hm = ha[:, :C] - jnp.einsum("nij,nj->ni", K, ha[:, C:])
    eta_hat = a["x_hat"][:, :C]
    b = sample_summaries(Y, L, B, G, theta, m, eta_hat, sd_eta, sd_delta, iters, likelihood, sandwich, blocks,
                         fix_eta=True)
    w, V = jnp.linalg.eigh(0.5 * (Lm + jnp.swapaxes(Lm, 1, 2)))      # Schur complement: clip the
    Lm = (V * jnp.clip(w, 0)[:, None, :]) @ jnp.swapaxes(V, 1, 2)     # rounding negatives (float32)
    Lam = b["Lam"].at[:, :C, :C].set(Lm)
    h = b["h"].at[:, :C].set(hm)
    x = b["x_hat"].at[:, :C].set(eta_hat)
    return dict(x_hat=x, Lam=Lam, h=h, grad_norm=jnp.maximum(a["grad_norm"], b["grad_norm"]), phi=b["phi"],
                inflation=b["inflation"])


# ---- stage 2: population model on the summaries ----------------------------------------------------
def _prior_moments(p, T, X, C, P, groups, activity):
    """Mean (N, D) and covariance (D, D) of x_i given the population parameters `p` (a dict)."""
    N = T.shape[0]
    mu_eta = p["a"] + p["gamma"] * T[:, None]
    if X is not None:
        mu_eta = mu_eta + X @ p["A_cov"]
    md = p["kappa"] + p["beta"] * T[:, None, None]
    if X is not None:
        md = md + jnp.einsum("ik,klp->ilp", X, p["D_cov"])
    gi = jnp.arange(C) if groups is None else jnp.asarray(groups)
    md = md[:, gi, :]                                                           # (N, C, P)
    same = (gi[:, None] == gi[None, :]).astype(md.dtype)                        # (C, C)
    if activity == "independent":
        sc, lam = jnp.full(C, p["sigma_d"]), jnp.zeros((C, P), md.dtype)
    else:
        lam = jnp.broadcast_to(p["lam"], (C, P)) if activity == "common" else p["lam"][gi]      # (C, P)
        sc = p["sigma_d"][gi] if activity in ("shared", "common") else jnp.zeros(C, md.dtype)
    # Cov[(c,p),(c',p')] = 1[p=p'] (lam_cp lam_c'p + sc_c sc_c' 1[same lineage])
    blk = lam.T[:, :, None] * lam.T[:, None, :] + (sc[:, None] * sc[None, :] * same)[None]   # (P, C, C)
    covd = jnp.einsum("pab,pq->apbq", blk, jnp.eye(P)).reshape(C * P, C * P)
    S = jax.scipy.linalg.block_diag(p["s_eta"] ** 2 * jnp.eye(C), covd)
    return jnp.concatenate([mu_eta, md.reshape(N, C * P)], 1), S


def _gauss_marginal(mu, S, Lam, h):
    """log of the integral of exp(-1/2 x'Lam x + h'x) N(x; mu, S) dx, per sample (S may be singular),
    and the conditional mean and covariance of x."""
    D = S.shape[0]
    M = jnp.eye(D) + S @ Lam                                                     # (N, D, D)
    r = h - jnp.einsum("nij,nj->ni", Lam, mu)
    Sr = jnp.einsum("ij,nj->ni", S, r)
    sol = jnp.linalg.solve(M, Sr[..., None])[..., 0]                             # (S^-1 + Lam)^-1 r
    ll = (-0.5 * jnp.linalg.slogdet(M)[1] + jnp.sum(h * mu, -1)
          - 0.5 * jnp.einsum("ni,nij,nj->n", mu, Lam, mu) + 0.5 * jnp.sum(r * sol, -1))
    return ll, mu + sol, M


def make_stage2_model(C, P, groups=None, activity="independent"):
    groups = None if groups is None else np.asarray(groups, int)
    Lg = C if groups is None else int(groups.max()) + 1

    def model(Lam, h, T, X=None):
        # a is zero-sum: softmax ignores a common shift, which would otherwise be a ridge for NUTS
        p = dict(a=_iid_normal("a", (C,), 1.5, True),
                 gamma=numpyro.sample("gamma", dist.Normal(0, 1.0).expand([C]).to_event(1)),
                 s_eta=numpyro.sample("s_eta", dist.HalfNormal(0.5)))
        if X is not None:
            K = X.shape[1]
            p["A_cov"] = numpyro.sample("A_cov", dist.Normal(0, 1.0).expand([K, C]).to_event(2))
            p["D_cov"] = numpyro.sample("D_cov", dist.Normal(0, 0.3).expand([K, Lg, P]).to_event(3))
        p["kappa"] = numpyro.deterministic("kappa", _horseshoe("kappa", (Lg, P), 0.3))
        p["beta"] = numpyro.deterministic("beta", _horseshoe("beta", (Lg, P), 0.3))
        if activity == "independent":
            p["sigma_d"] = numpyro.sample("sigma_d", dist.HalfNormal(0.3))
        else:
            lshape = [P] if activity == "common" else [Lg, P]
            p["lam"] = numpyro.sample("lam", dist.HalfNormal(0.3).expand(lshape).to_event(len(lshape)))
            if activity in ("shared", "common"):
                p["sigma_d"] = numpyro.sample("sigma_d", dist.HalfNormal(0.3).expand([Lg]).to_event(1))
        mu, S = _prior_moments(p, T, X, C, P, groups, activity)
        ll, _, _ = _gauss_marginal(mu, S, Lam, h)
        numpyro.factor("summaries", ll.sum())

    return model


def _sqrt_psd(A):
    """Symmetric square root factor F with F F' = A for PSD A (batched); tolerates singular A."""
    w, V = jnp.linalg.eigh(0.5 * (A + jnp.swapaxes(A, -1, -2)))
    return V * jnp.sqrt(jnp.clip(w, 0))[..., None, :]


def _draw_latents(post, Lam, h, T, X, C, P, groups, activity, seed, batch=None):
    """Exact draws of every x_i from its Gaussian conditional, one per posterior draw, by perturbation
    ("perturb and solve"): with x0 ~ N(mu, S) and e_i ~ N(0, Lam_i),
        x_i = x0 + (I + S Lam_i)^-1 S (h_i + e_i - Lam_i x0)
    has exactly the conditional's mean (S^-1 + Lam_i)^-1 (S^-1 mu + h_i) and covariance (S^-1 + Lam_i)^-1.
    It needs one factorisation of S per draw (shared by all samples) and of each Lam_i once, instead of
    one eigendecomposition per sample and draw (which made this step most of the run time)."""
    # The prior moments come from JAX (no linear algebra); the many small factorisations and solves run
    # in NumPy, whose stacked LAPACK calls cost microseconds per 32x32 matrix (JAX on CPU pays a large
    # fixed cost per small matrix, which made this step take minutes to hours).
    # Draws from parallel chains are sharded over the CPU devices; slicing them batch by batch forces
    # slow cross-device gathers (minutes), so take them to host memory once.
    post = {k: np.asarray(v) for k, v in post.items()}
    keys = list(post)
    n = post[keys[0]].shape[0]
    Lam, h = np.asarray(Lam, float), np.asarray(h, float)
    N, D = Lam.shape[:2]
    moments = jax.jit(jax.vmap(lambda pd_: _prior_moments(pd_, T, X, C, P, groups, activity)))
    rng = np.random.default_rng(seed + 40_000)
    sqL = np.asarray(_sqrt_psd(jnp.asarray(Lam)))                                # (N, D, D), once
    out = np.empty((n, N, D))
    batch = batch or max(1, int(2e7 // (N * D * D)))            # ~160 MB of (draws, N, D, D) at a time
    for s in range(0, n, batch):
        mu, S = (np.asarray(v, float) for v in moments({k: v[s:s + batch] for k, v in post.items()}))
        b = len(S)
        w, V = np.linalg.eigh(0.5 * (S + np.swapaxes(S, 1, 2)))
        sqS = V * np.sqrt(np.clip(w, 0, None))[:, None, :]                       # (b, D, D)
        x0 = mu + np.einsum("bij,bnj->bni", sqS, rng.standard_normal((b, N, D)))
        e = np.einsum("nij,bnj->bni", sqL, rng.standard_normal((b, N, D)))
        r = h[None] + e - np.einsum("nij,bnj->bni", Lam, x0)
        M = np.eye(D) + np.einsum("bij,njk->bnik", S, Lam)                        # (b, N, D, D)
        out[s:s + b] = x0 + np.linalg.solve(M, np.einsum("bij,bnj->bni", S, r)[..., None])[..., 0]
    return out


STAGE2 = ("nuts", "vi", "laplace")


def _vi_fit(model, args, steps, particles, seed):
    from numpyro.infer.autoguide import AutoMultivariateNormal
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_median, init_scale=0.05)
    svi = SVI(model, guide, ClippedAdam(0.01), Trace_ELBO(num_particles=particles))
    res = svi.run(jax.random.PRNGKey(seed + 50_000), steps, *args, progress_bar=False, stable_update=True)
    loss = np.asarray(res.losses)
    tail = max(10, steps // 10)
    a, b = loss[-2 * tail:-tail].mean(), loss[-tail:].mean()
    return guide, res.params, dict(steps=steps, final_loss=float(b), rel_change=float(abs(a - b) / (abs(b) + 1e-12)))


def _stage2_gauss(model, args, kind, n_draws, steps, particles, seed):
    """Gaussian approximation of the stage-2 posterior (unconstrained space). Returns (draws of every
    sample and deterministic site, dict with the final loss and its relative change over the last 10%
    of steps, a crude convergence check)."""
    from numpyro.infer import Predictive
    from numpyro.infer.autoguide import AutoLaplaceApproximation
    if kind == "vi":
        guide, params, info = _vi_fit(model, args, steps, particles, seed)
    else:
        guide = AutoLaplaceApproximation(model, init_loc_fn=init_to_median)
        svi = SVI(model, guide, ClippedAdam(0.01), Trace_ELBO())
        res = svi.run(jax.random.PRNGKey(seed + 50_000), steps, *args, progress_bar=False, stable_update=True)
        loss, params = np.asarray(res.losses), res.params
        tail = max(10, steps // 10)
        a, b = loss[-2 * tail:-tail].mean(), loss[-tail:].mean()
        info = dict(steps=steps, final_loss=float(b), rel_change=float(abs(a - b) / (abs(b) + 1e-12)))
    info["kind"] = kind
    lat = guide.sample_posterior(jax.random.PRNGKey(seed + 60_000), params, sample_shape=(n_draws,))
    det = Predictive(model, posterior_samples=lat, return_sites=["kappa", "beta"])(
        jax.random.PRNGKey(seed + 70_000), *args)
    s = jax.block_until_ready({**lat, **det})
    return s, info


# ---- driver -----------------------------------------------------------------------------------------
def fit_fast(Y, L, T, B, G, X=None, groups=None, activity="independent", use_m=True, num_warmup=500,
             num_samples=500, chains=2, seed=0, target_accept=0.9, chain_method="auto", svi_steps=2000,
             sd_eta=3.0, sd_delta=1.0, stage1_iters=80, progress_bar=False, likelihood="nb", sigma_m="auto",
             sandwich=False, n_blocks=100, stage2="nuts", vi_steps=5000, vi_particles=4,
             delta_given_eta=False):
    """Two-stage fit (see module docstring). Same inputs as bayes.fit.fit; returns a draws dict with
    the same keys (pi, delta, kappa, beta, a, gamma, dec, convergence, ...) plus `stage1`
    (summary quality) and `timing` (seconds per stage).

    delta_given_eta=True splits stage 1 in two: fractions (eta) from the NB likelihood, summarised by their
    marginal information (delta integrated out), then delta with eta held at the NB estimate, using
    `likelihood`/`sandwich`. Estimating eta and delta jointly lets program shifts trade off against
    composition (program genes are often cell-type markers), which made per-sample delta 2-4x too
    dispersed on PBMC/OneK1K; holding the fractions fixed is what the naive two-step estimator does.

    stage2 picks how the population parameters (~100-200 numbers) are inferred given the summaries:
      'nuts'     NUTS (reference; reports R-hat/ESS);
      'vi'       full-rank Gaussian VI in the unconstrained space (AutoMultivariateNormal, vi_steps Adam
                 steps, vi_particles draws per gradient): keeps the correlations between parameters;
      'laplace'  Gaussian at the MAP with the Hessian as precision (AutoLaplaceApproximation).
    'vi' and 'laplace' draw num_samples*chains independent draws; R-hat/ESS do not apply to them.
    They match NUTS for pi and delta (driven by each sample's summary) but not for the effects: the
    horseshoe on beta is far from Gaussian, so VI's effect intervals are too narrow and Laplace's too
    wide (PBMC/OneK1K split 0, 2026-10-26). Use them for fractions/activity only. (Starting NUTS from the
    full-rank VI fit, with its covariance as mass matrix, was tried and dropped: VI draws start chains in
    the horseshoe's narrow regions and NUTS mixed worse and ran slower than from its usual warm start.)"""
    if stage2 not in STAGE2:
        raise ValueError("stage2 must be one of %s" % (STAGE2,))
    if activity not in ACTIVITY_MODELS:
        raise ValueError("activity must be one of %s" % (ACTIVITY_MODELS,))
    timing, t = {}, time.time()
    chain_method = resolve_chain_method(chains, chain_method)
    args, n_prog, fdt = model_args(Y, L, T, B, G, X)
    Yj, Lj, Tj, Bj, Gj, Xj = args
    C, P = Bj.shape[0], Gj.shape[0]
    if P == 0:
        raise ValueError("run_palm_fast needs programs; use run_palm for composition-only inputs")
    theta = jnp.asarray(gene_dispersion(np.asarray(Yj)), fdt)
    m, sm = gene_scale(args, n_prog, theta, groups, svi_steps, sigma_m, seed=seed) if use_m else (None, 0.0)
    timing["stage0"] = time.time() - t

    t = time.time()
    from ..compare.methods import nnls_fractions
    pi0 = np.clip(nnls_fractions(np.asarray(Yj), np.asarray(Bj)), 1e-3, None)
    eta0 = jnp.asarray(np.log(pi0 / pi0.sum(1, keepdims=True)), fdt)
    blocks = gene_blocks(np.asarray(Bj), np.asarray(Gj), n_blocks, seed) if sandwich else None
    if not delta_given_eta:
        st1 = sample_summaries(Yj, Lj, Bj, Gj, theta, m, eta0, sd_eta, sd_delta, stage1_iters, likelihood,
                               sandwich, blocks)
    else:
        st1 = _two_step_summaries(Yj, Lj, Bj, Gj, theta, m, eta0, sd_eta, sd_delta, stage1_iters, likelihood,
                                  sandwich, blocks)
    jax.block_until_ready(st1["Lam"])
    timing["stage1"] = time.time() - t

    t = time.time()
    model = make_stage2_model(C, P, groups, activity)
    s2args = (st1["Lam"], st1["h"], Tj, Xj)
    vi_info = None
    if stage2 == "nuts":
        init, inv_mass, _ = svi_warm_start(model, s2args, chains, steps=1000, seed=seed)
        kernel = NUTS(model, init_strategy=init_to_median, target_accept_prob=target_accept,
                      inverse_mass_matrix=inv_mass)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=chains,
                    chain_method=chain_method, progress_bar=progress_bar)
        mcmc.run(jax.random.PRNGKey(seed), *s2args, init_params=init, extra_fields=("diverging", "num_steps"))
        s = jax.block_until_ready(mcmc.get_samples())      # JAX is asynchronous: time the sampler itself
    else:
        s, vi_info = _stage2_gauss(model, s2args, stage2, num_samples * chains, vi_steps, vi_particles, seed)
        mcmc = None
    timing["stage2_" + stage2] = time.time() - t

    t = time.time()
    names = ["a", "gamma", "s_eta", "kappa", "beta", "sigma_d", "lam", "A_cov", "D_cov"]
    post = {k: s[k] for k in names if k in s}
    x = _draw_latents(post, st1["Lam"], st1["h"], Tj, Xj, C, P, groups, activity, seed)
    timing["latent_draws"] = time.time() - t

    pi = np.asarray(jax.nn.softmax(jnp.asarray(x[..., :C]), -1))
    delta_ct = x[..., C:].reshape(x.shape[:2] + (C, P))
    if groups is None:
        delta = delta_ct
    else:                         # one value per lineage (cell types of a lineage share it exactly)
        first = [int(np.flatnonzero(np.asarray(groups) == l)[0]) for l in range(int(np.max(groups)) + 1)]
        delta = delta_ct[:, :, first, :]
    out = dict(pi=pi, delta=delta, a=np.asarray(s["a"]), gamma=np.asarray(s["gamma"]),
               kappa=np.asarray(s["kappa"]), beta=np.asarray(s["beta"]))
    ix = slice(None) if groups is None else np.asarray(groups, int)
    dec = effect_decomposition(Bj, Gj, s["a"], s["gamma"], s["kappa"][:, ix], s["beta"][:, ix])
    out["dec"] = {k: np.array(v) for k, v in dec.items()}
    out["groups"] = None if groups is None else np.asarray(groups, int)
    out.update(platform=jax.default_backend(), precision="float64" if fdt == jnp.float64 else "float32",
               stage2=stage2)
    if mcmc is not None:
        ef = mcmc.get_extra_fields()
        out["divergences"] = int(np.asarray(ef["diverging"]).sum())
        out["mean_steps"] = float(np.asarray(ef["num_steps"]).mean())
        out.update(chain_method=chain_method, chains=chains)
        by_chain = {k: out[k].reshape((chains, -1) + out[k].shape[1:]) for k in CONVERGENCE_SITES if k in out}
        by_chain.update({k: out["dec"][k].reshape((chains, -1) + out["dec"][k].shape[1:]) for k in ("I", "K")})
        conv = convergence(by_chain)
        out["convergence"] = conv
        out["rhat_max"] = max((v["rhat_max"] for v in conv.values()), default=float("nan"))
        finite = [v["ess_min"] for v in conv.values() if np.isfinite(v["ess_min"])]
        out["ess_min"] = float(min(finite)) if finite else float("nan")
    else:                          # independent draws: no chains to compare
        out.update(divergences=0, chains=1, convergence={}, rhat_max=float("nan"), ess_min=float("nan"),
                   stage2_fit=vi_info)
    gn = np.asarray(st1["grad_norm"])
    out["stage1"] = dict(grad_norm_max=float(gn.max()), n_not_converged=int((gn > 1e-3).sum()),
                         sd_eta=sd_eta, sd_delta=sd_delta, iters=stage1_iters, likelihood=likelihood,
                         sigma_m=round(sm, 4), sandwich=sandwich, delta_given_eta=delta_given_eta,
                         inflation_median=float(np.median(np.asarray(st1["inflation"]))))
    if st1["phi"] is not None:     # per-sample fit check: Pearson overdispersion far above the cohort's
        phi = np.asarray(st1["phi"])
        out["stage1"].update(phi_median=float(np.median(phi)), phi_max=float(phi.max()),
                             phi_outliers=np.flatnonzero(phi > 3 * np.median(phi)).tolist())
    out["timing"] = {k: round(v, 2) for k, v in timing.items()}
    xh = np.asarray(st1["x_hat"])          # each sample's own mode (weak prior, no pooling): diagnostics
    out["stage1_mode"] = dict(pi=np.asarray(jax.nn.softmax(jnp.asarray(xh[:, :C]), -1)),
                              delta=xh[:, C:].reshape(-1, C, P))
    return out


def run_palm_fast(inputs, activity="independent", use_m=True, num_warmup=500, num_samples=500, chains=2,
                  seed=0, level=0.9, check=True, **kw):
    """Fast two-stage PALM on prepared inputs; returns a PalmResult like run_palm (same tables).
    See the module docstring for what is approximated. Extra keywords go to fit_fast."""
    import warnings
    from .pipeline import PalmResult
    draws = fit_fast(inputs.Y, inputs.L, inputs.T, inputs.B, inputs.Gmat, X=inputs.X,
                     groups=getattr(inputs, "groups", None), activity=activity, use_m=use_m,
                     num_warmup=num_warmup, num_samples=num_samples, chains=chains, seed=seed, **kw)
    res = PalmResult(inputs, draws, level=level)
    if check:
        problems = res.diagnostics()["problems"]
        if draws["stage1"]["n_not_converged"]:
            problems.append("stage 1: %d samples did not reach their mode (raise stage1_iters)"
                            % draws["stage1"]["n_not_converged"])
        if problems:
            warnings.warn("two-stage fit checks failed:\n  " + "\n  ".join(problems))
    return res
