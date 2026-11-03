"""Inference for the Bayesian model: VI warm start and NUTS. Main entry point: `fit`."""
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import ClippedAdam

from ..core.equations import effect_decomposition
from ..core.data import gene_dispersion
from ..compute import resolve_chain_method
from .model import make_model
from .diagnostics import convergence


def svi_warm_start(model, args, chains=1, steps=3000, lr=0.01, seed=0):
    """Short mean-field VI fit used only to start NUTS: returns per-chain initial values
    (unconstrained, drawn from the VI posterior so chains start apart) and a diagonal initial
    inverse mass matrix from the VI scales. NUTS still produces every reported draw."""
    guide = AutoNormal(model, init_loc_fn=init_to_median)
    svi = SVI(model, guide, ClippedAdam(lr), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(seed + 10_000), steps, *args, progress_bar=False, stable_update=True)
    p = res.params
    sites = [k[:-len("_auto_loc")] for k in p if k.endswith("_auto_loc")]
    keys = jax.random.split(jax.random.PRNGKey(seed + 20_000), len(sites))
    init, inv_mass = {}, {}
    for k, name in zip(keys, sites):
        loc, scale = p[name + "_auto_loc"], p[name + "_auto_scale"]
        shape = (chains,) + loc.shape if chains > 1 else loc.shape
        init[name] = loc + scale * jax.random.normal(k, shape)
        inv_mass[(name,)] = jnp.ravel(scale ** 2)
    return init, inv_mass, np.asarray(res.losses)


def gene_order(G):
    """Order in which the model sees the genes: those with program weight first (stable)."""
    G = np.asarray(G)
    return np.argsort(~(G != 0).any(0), kind="stable")


# Low-dimensional parameters shared across samples or genes. Their posteriors are correlated (the
# horseshoe's global and local scales, baseline vs treatment logits), which a diagonal mass matrix
# cannot follow; a dense block for them costs little (about 100-200 dimensions).
GLOBAL_SITES = ("a", "gamma", "s_eta", "A_cov", "kappa_tau", "kappa_lam", "kappa_z", "beta_tau", "beta_lam",
                "beta_z", "sigma_d", "lam", "theta", "s_theta")


def _mass_blocks(inv_mass):
    """Turn the per-site diagonal VI mass into numpyro's structured form with one dense block of
    GLOBAL_SITES, started at the VI variances. Returns (inverse mass, dense_mass argument)."""
    block = tuple(k for k in GLOBAL_SITES if (k,) in inv_mass)
    if not block:
        return inv_mass, False
    inv_mass = dict(inv_mass)
    inv_mass[block] = jnp.diag(jnp.concatenate([inv_mass.pop((k,)) for k in block]))
    return inv_mass, [block]


def model_args(Y, L, T, B, G, X=None):
    """Model inputs as JAX arrays, genes reordered so those with program weight come first
    (the model then shifts only those, see core.equations.mix_profiles). Float precision follows
    configure_compute(precision=). Returns (args, n_prog, dtype)."""
    fdt = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    B, G, Y = np.asarray(B), np.asarray(G), np.asarray(Y)
    T = np.zeros(len(L)) if T is None else np.asarray(T, float)
    order = gene_order(G)
    n_prog = int((G != 0).any(0).sum())
    Xj = None
    if X is not None:
        X = np.asarray(X, float).reshape(len(L), -1)
        Xj = jnp.array(X - X.mean(0), fdt)
    args = (jnp.array(Y[:, order], fdt), jnp.array(L, fdt), jnp.array(T, fdt),
            jnp.array(B[:, order], fdt), jnp.array(G[:, order], fdt), Xj)
    return args, n_prog, fdt


CONVERGENCE_SITES = ["a", "gamma", "pi", "kappa", "beta", "delta", "theta"]


def fit(Y, L, T, B, G, theta=None, use_m=False, sigma_m=0.3, num_warmup=200, num_samples=300,
        chains=1, seed=0, target_accept=0.9, progress_bar=False, chain_method="auto", reparam=False,
        init="svi", svi_steps=3000, dispersion="fixed", X=None, groups=None, dense_mass=False,
        max_tree_depth=10, activity="independent"):
    """Run NUTS. Y (N,G) counts, L (N,) library sizes (sum of Y over the modelled genes),
    T (N,) 0/1 treatment or zeros, B (C,G), G (P,G). Returns a dict of posterior draws,
    the treatment-effect decomposition, and sampler diagnostics.
    chain_method: 'auto' (see compute.resolve_chain_method), 'sequential', 'parallel' or
    'vectorized'. It changes how chains are scheduled, not the model.
    reparam: see make_model.
    init: 'svi' (default; start from a `svi_steps` VI fit, see svi_warm_start, which allows a
    shorter num_warmup) or 'median' (prior medians, the original start).
    dispersion: 'fixed' (default; one per gene, estimated once from Y by moments, see
    core.data.gene_dispersion, and not sampled), 'gene' (one per gene, sampled and partially pooled)
    or 'shared' (one sampled dispersion). 'fixed' removes the funnel of the gene-level dispersion
    hierarchy and most of the likelihood cost: on the PBMC benchmark ~15-25x more effective draws per
    minute than 'gene' (benchmarks/sampler_speed_pbmc.json). Dispersion uncertainty is then ignored,
    which narrowed treatment-effect intervals there (90% coverage 0.89 -> 0.82); use 'gene' for a
    final analysis where that matters.
    theta: a fixed scalar, or per-gene array in the input gene order; overrides `dispersion`.
    dense_mass: False (diagonal mass matrix) or 'globals' (one dense block for GLOBAL_SITES, the
    parameters shared across samples and genes).
    max_tree_depth: NUTS takes at most 2^depth - 1 gradient steps per iteration (10 -> 1023).
    activity: 'independent', 'shared' or 'shared_only'; how per-sample program shifts vary, see make_model.
    X: optional (N,K) covariates (numeric; one-hot code categories), centred here.
    groups: optional (C,) group index per cell type; program shifts are shared within groups
    (see make_model). Draws of kappa, beta, delta are then per group; `dec` is per cell type.
    Diagnostics: `convergence` holds split R-hat and effective sample size for a, gamma, pi,
    kappa, beta, delta, theta and the effect decomposition (I, K); `rhat_max` and `ess_min` are the
    worst values over all of them. Defaults follow benchmarks/calibration (init='svi') and
    benchmarks/sampler_speed_pbmc.json (dispersion='fixed', reparam=False, diagonal mass, depth 10)."""
    chain_method = resolve_chain_method(chains, chain_method)
    args, n_prog, fdt = model_args(Y, L, T, B, G, X)
    if theta is None and dispersion == "fixed":
        theta = gene_dispersion(np.asarray(Y, float))
    if theta is not None and np.ndim(theta) == 1:
        theta = np.asarray(theta, float)[gene_order(G)]
    if dense_mass not in (False, "globals"):
        raise ValueError("dense_mass must be False or 'globals'")
    model = make_model(use_m, sigma_m, theta, reparam, dispersion, n_prog, groups, activity)
    Bj, Gj = jnp.array(B, fdt), jnp.array(G, fdt)
    init_params, inv_mass = None, None
    if init == "svi":
        init_params, inv_mass, _ = svi_warm_start(model, args, chains, svi_steps, seed=seed)
    elif init != "median":
        raise ValueError("init must be 'median' or 'svi'")
    dense = False
    if dense_mass == "globals":
        if inv_mass is None:
            from numpyro.infer.util import initialize_model
            sites = initialize_model(jax.random.PRNGKey(seed), model, model_args=args).param_info.z
            dense = [tuple(k for k in GLOBAL_SITES if k in sites)]
        else:
            inv_mass, dense = _mass_blocks(inv_mass)
    kernel = NUTS(model, init_strategy=init_to_median, target_accept_prob=target_accept,
                  inverse_mass_matrix=inv_mass, dense_mass=dense, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=chains,
                chain_method=chain_method, progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed), *args, init_params=init_params,
             extra_fields=("diverging", "num_steps"))
    s = mcmc.get_samples()
    keys = ["a", "gamma", "pi"] + (["kappa", "beta", "delta"] if "kappa" in s else [])
    out = {k: np.array(s[k]) for k in keys}
    if "kappa" in s:
        ix = slice(None) if groups is None else np.asarray(groups, int)
        dec = effect_decomposition(Bj, Gj, s["a"], s["gamma"], s["kappa"][:, ix], s["beta"][:, ix])
        out["dec"] = {k: np.array(v) for k, v in dec.items()}
    if "theta" in s:
        out["theta"] = np.array(s["theta"])
    out["groups"] = None if groups is None else np.asarray(groups, int)
    ef = mcmc.get_extra_fields()
    out["divergences"] = int(np.asarray(ef["diverging"]).sum())
    out["mean_steps"] = float(np.asarray(ef["num_steps"]).mean())
    out["chain_method"], out["platform"] = chain_method, jax.default_backend()
    out["precision"] = "float64" if fdt == jnp.float64 else "float32"
    out["chains"] = chains
    # draws are stored chain after chain, so (chains * draws, ...) reshapes to (chains, draws, ...)
    by_chain = {k: out[k].reshape((chains, -1) + out[k].shape[1:]) for k in CONVERGENCE_SITES if k in out}
    if "dec" in out:
        by_chain.update({k: out["dec"][k].reshape((chains, -1) + out["dec"][k].shape[1:]) for k in ("I", "K")})
    conv = convergence(by_chain)
    out["convergence"] = conv
    out["rhat_max"] = max((v["rhat_max"] for v in conv.values()), default=float("nan"))
    out["ess_min"] = float(np.nanmin([v["ess_min"] for v in conv.values()])) \
        if any(np.isfinite(v["ess_min"]) for v in conv.values()) else float("nan")
    return out
