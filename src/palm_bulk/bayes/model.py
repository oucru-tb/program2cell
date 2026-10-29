"""NumPyro model: priors and likelihood around the core forward model (palm_bulk.core.equations).

Unified model (T may be all zeros for a purely descriptive analysis):
    eta_i     = a + gamma * T_i + s_eta * eps_i,   pi_i = softmax(eta_i)       composition
    delta_icp = kappa_cp + beta_cp * T_i + sigma_d * z_icp                       program shift
    mu_ig     = expected_counts(L, pi, B, G, delta, m)                           core forward model
    Y_ig     ~ NegBin(mu_ig, theta)
kappa, beta: regularised horseshoe (sparse). m_g = exp(sigma_m * z_g) absorbs platform bias.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import gammaln, xlogy

from ..core.equations import expected_fractions


class _NegBin2(dist.NegativeBinomial2):
    """NegativeBinomial2 (mean mu, concentration theta) with the log density written out directly.
    Same value and gradient as numpyro's, which goes through GammaPoisson and costs ~8x more per
    gradient on CPU; the likelihood is most of the gradient cost, so this speeds up NUTS ~2-3x."""

    def log_prob(self, value):
        mu, th = self.mean, self.concentration
        return (gammaln(value + th) - gammaln(th) - gammaln(value + 1) + th * jnp.log(th)
                + xlogy(value, mu) - (value + th) * jnp.log(mu + th))


class _NegBin2Kernel(_NegBin2):
    """As _NegBin2 but only the terms that depend on mu, for a fixed (data, not sampled) theta.
    The log density is off by a constant, so posteriors and gradients are unchanged, but the
    log-gamma terms, most of the remaining likelihood cost, are skipped. Do not use its values for
    model comparison."""

    def log_prob(self, value):
        mu, th = self.mean, self.concentration
        return xlogy(value, mu) - (value + th) * jnp.log(mu + th)


def _horseshoe(name, shape, tau_scale, slab2=1.0):
    nd = len(shape)
    tau0 = numpyro.sample(name + "_tau", dist.HalfCauchy(tau_scale))
    lam = numpyro.sample(name + "_lam", dist.HalfCauchy(1.0).expand(list(shape)).to_event(nd))
    z = numpyro.sample(name + "_z", dist.Normal(0, 1).expand(list(shape)).to_event(nd))
    lam_t = jnp.sqrt(slab2 * lam ** 2 / (slab2 + tau0 ** 2 * lam ** 2))
    return tau0 * lam_t * z


def _iid_normal(name, shape, scale, zero_sum):
    """iid Normal(0, scale) over `shape`, or only its zero-sum part along the last axis.
    Use zero_sum when the likelihood ignores a constant shift along that axis (softmax inputs,
    renormalised gene scales): the mean component of an iid Normal is independent of the rest
    and carries no data, so dropping it leaves the posterior of everything else unchanged and
    removes a direction NUTS otherwise explores very slowly."""
    if zero_sum:
        d = dist.ZeroSumNormal(scale, event_shape=(shape[-1],))
        return numpyro.sample(name, d.expand(list(shape[:-1])).to_event(len(shape) - 1))
    return numpyro.sample(name, dist.Normal(0, scale).expand(list(shape)).to_event(len(shape)))


ACTIVITY_MODELS = ("independent", "common", "shared", "shared_only")


def make_model(use_m=False, sigma_m=0.3, theta=None, reparam=False, dispersion="shared", n_prog=None,
               groups=None, activity="independent"):
    """Build the model function. `theta=None` -> dispersion is sampled (LogNormal(log 30, 1));
    otherwise a fixed scalar or (G,) per-gene dispersion, in the model's gene order.
    reparam=True samples a, gamma, eps (softmax inputs) and zm (renormalised) as zero-sum
    Normals: same posterior for pi, delta, effects and fits; the reported a and gamma are then
    centred over cell types. reparam=False is the original sampler.
    dispersion='gene': one NB dispersion per gene, log theta_g ~ Normal(log theta0, s_theta)
    (ignored when `theta` is fixed). n_prog: see core.equations.mix_profiles (genes must be ordered).
    The model takes optional covariates X (N,K), centred; they shift composition (A, K x C) and
    program activity (D, K x C x P), so treatment effects are for a sample at the covariate mean.
    groups: optional (C,) integer array mapping each cell type to a group (e.g. lineage). Program
    shifts (kappa, beta, delta, D) are then estimated per group and shared by its cell types, while
    composition stays at cell-type level. With no programs (G has 0 rows) the model is
    composition-only: profiles are the reference baselines.
    activity: how a sample's program shift varies around kappa + beta*T across samples.
      'independent': sigma_d * z_icp, independent for every sample, cell type and program.
      'shared': lam_cp * u_ip + sigma_d,c * z_icp. u_ip ~ N(0,1) is sample i's level of program p,
          shared by all cell types; lam_cp >= 0 is how strongly type c follows it. One bulk sample
          cannot separate C independent shifts per program, but it can see the shared one in every
          cell type's genes; on PBMC one such factor per program explains 60-92% of the true
          across-sample variation, with loadings of one sign. The residual scale is per cell type.
      'shared_only': lam_cp * u_ip with no residual (fewest parameters, no residual funnel).
      'common': lam_p * u_ip + sigma_d,c * z_icp: one sample-level shift per program that every cell type
          shares equally (the structure of a bulk-level score), plus per-type deviations."""
    groups = None if groups is None else np.asarray(groups, int)
    if activity not in ACTIVITY_MODELS:
        raise ValueError("activity must be one of %s" % (ACTIVITY_MODELS,))

    if theta is not None:
        theta = jnp.asarray(theta)

    def model(Y, L, T, B, G, X=None):
        N, Gn = Y.shape
        C, P = B.shape[0], G.shape[0]
        a = _iid_normal("a", (C,), 1.5, reparam)
        gamma = _iid_normal("gamma", (C,), 1.0, reparam)
        s_eta = numpyro.sample("s_eta", dist.HalfNormal(0.5))
        eps = _iid_normal("eps", (N, C), 1.0, reparam)
        eta = a + gamma * T[:, None] + s_eta * eps
        if X is not None:
            K = X.shape[1]
            eta = eta + X @ _iid_normal("A_cov", (K, C), 1.0, reparam)
        pi = numpyro.deterministic("pi", jax.nn.softmax(eta, axis=-1))
        delta_ct = None
        if P > 0:
            Lg = C if groups is None else int(groups.max()) + 1
            kappa = numpyro.deterministic("kappa", _horseshoe("kappa", (Lg, P), 0.3))
            beta = numpyro.deterministic("beta", _horseshoe("beta", (Lg, P), 0.3))
            shift = kappa + beta * T[:, None, None]
            if activity == "independent":
                sigma_d = numpyro.sample("sigma_d", dist.HalfNormal(0.3))
                zr = numpyro.sample("zres", dist.Normal(0, 1).expand([N, Lg, P]).to_event(3))
                shift = shift + sigma_d * zr
            else:
                lshape = [P] if activity == "common" else [Lg, P]
                lam = numpyro.sample("lam", dist.HalfNormal(0.3).expand(lshape).to_event(len(lshape)))
                u = numpyro.sample("u", dist.Normal(0, 1).expand([N, P]).to_event(2))
                shift = shift + lam * u[:, None, :]
                if activity in ("shared", "common"):
                    sigma_d = numpyro.sample("sigma_d", dist.HalfNormal(0.3).expand([Lg]).to_event(1))
                    zr = numpyro.sample("zres", dist.Normal(0, 1).expand([N, Lg, P]).to_event(3))
                    shift = shift + sigma_d[:, None] * zr
            if X is not None:
                D = numpyro.sample("D_cov", dist.Normal(0, 0.3).expand([K, Lg, P]).to_event(3))
                shift = shift + jnp.einsum("ik,kcp->icp", X, D)
            delta = numpyro.deterministic("delta", shift)
            delta_ct = delta if groups is None else delta[:, groups, :]
        m = None
        if use_m:                          # sigma_m='auto': the scale's size is learned (HalfNormal(0.3))
            sm = numpyro.sample("sigma_m", dist.HalfNormal(0.3)) if sigma_m == "auto" else sigma_m
            m = jnp.exp(sm * _iid_normal("zm", (Gn,), 1.0, reparam))
        frac = expected_fractions(pi, B, G, delta_ct, m, n_prog)
        if theta is not None:            # fixed dispersion (scalar or per gene): mu-dependent terms only
            numpyro.sample("Y", _NegBin2Kernel(L[:, None] * frac, theta).to_event(2), obs=Y)
            return
        th = numpyro.sample("theta", dist.LogNormal(np.log(30.0), 1.0))
        if dispersion == "gene":
            s_th = numpyro.sample("s_theta", dist.HalfNormal(0.5))
            zt = numpyro.sample("ztheta", dist.Normal(0, 1).expand([Gn]).to_event(1))
            th = th * jnp.exp(s_th * zt)
        numpyro.sample("Y", _NegBin2(L[:, None] * frac, th).to_event(2), obs=Y)

    return model
