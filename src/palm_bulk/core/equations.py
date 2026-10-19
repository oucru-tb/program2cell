"""Model equations shared by the simulator, the Bayesian model, the truth computations and (Paper 2)
the fast network. No inference code here.

Notation: N samples, C cell types, G genes, P programs.
  B (C,G)   reference baseline: share of reads per gene in each cell type (rows sum to 1)
  G (P,G)   program dictionary (signed log-fold-change style weights)
  delta     (...,C,P) program shift of a cell type relative to its reference state
  pi        (...,C) read fractions

Every function accepts numpy, JAX or torch arrays and returns the same kind: torch if any input is
a torch tensor, else JAX if any input is a JAX array (including traced values inside jit/grad),
else numpy. Torch keeps autograd, so the forward model can sit inside a PyTorch training loop.
"""
import numpy as np
from scipy import special as _sp


class _Ops:
    """The few array operations the equations need, for one backend."""

    def __init__(self, name):
        self.name = name
        if name == "torch":
            import torch
            self.m = torch
        elif name == "jax":
            import jax
            import jax.numpy as jnp
            self.m, self.nn = jnp, jax.nn
        else:
            self.m = np

    def asarray(self, x, like=None):
        if self.name == "torch":
            t = self.m
            if isinstance(x, t.Tensor):
                return x if like is None else x.to(like.dtype)
            dt = like.dtype if like is not None else t.get_default_dtype()
            dev = like.device if like is not None else None
            return t.as_tensor(np.asarray(x), dtype=dt, device=dev)
        return self.m.asarray(x)

    def einsum(self, spec, *xs):
        return self.m.einsum(spec, *xs)

    def softmax(self, x, axis=-1):
        if self.name == "torch":
            return self.m.softmax(x, dim=axis)
        if self.name == "jax":
            return self.nn.softmax(x, axis=axis)
        return _sp.softmax(x, axis=axis)

    def logsumexp(self, x, axis=-1):
        if self.name == "torch":
            return self.m.logsumexp(x, dim=axis)
        if self.name == "jax":
            return self.nn.logsumexp(x, axis=axis)
        return _sp.logsumexp(x, axis=axis)

    def stack(self, xs, axis=0):
        return self.m.stack(xs, dim=axis) if self.name == "torch" else self.m.stack(xs, axis=axis)

    def concat(self, xs, axis=0):
        return self.m.cat(xs, dim=axis) if self.name == "torch" else self.m.concatenate(xs, axis=axis)

    def sum(self, x, axis, keepdims=False):
        return x.sum(dim=axis, keepdim=keepdims) if self.name == "torch" else x.sum(axis=axis, keepdims=keepdims)

    def nan_where_not(self, cond, x):
        return self.m.where(cond, x, self.m.full_like(x, float("nan")))


def _backend(*xs):
    name = "numpy"
    for x in xs:
        mod = type(x).__module__
        if mod.startswith("torch"):
            return "torch"
        if mod.startswith("jax") or mod.startswith("jaxlib"):
            name = "jax"
    return name


def _ops(*xs):
    """Backend for these inputs, and the inputs converted to it (dtype/device of the first
    array-like of that backend)."""
    ops = _Ops(_backend(*xs))
    like = next((x for x in xs if _backend(x) == ops.name and ops.name != "numpy"), None)
    return ops, [None if x is None else ops.asarray(x, like) for x in xs]


# ----- profiles and the mixture (forward model) ------------------------------
def cell_profiles(B, G, delta):
    """E_cg = B_cg exp(sum_p delta_cp G_pg), renormalised over genes. Returns (...,C,G)."""
    ops, (B, G, delta) = _ops(B, G, delta)
    logu = ops.m.log(B) + ops.einsum("...cp,pg->...cg", delta, G)
    return ops.softmax(logu, axis=-1)


def mix_profiles(pi, B, G, delta, n_prog=None):
    """sum_c pi_ic E_icg with E = cell_profiles(B, G, delta); pi (N,C), delta (N,C,P).

    If the first `n_prog` genes are the only ones with program weight, the remaining genes are not
    shifted by delta: they share one normaliser per (sample, cell type) and are computed as a
    (N,C)x(C,R) product. Exact; the expensive (N,C,G) work shrinks to (N,C,n_prog)."""
    ops, (pi, B, G, delta) = _ops(pi, B, G, delta)
    Gn = B.shape[1]
    if G.shape[0] == 0:
        return pi @ B
    if n_prog is None or n_prog >= Gn:
        return ops.einsum("ic,icg->ig", pi, cell_profiles(B, G, delta))
    BS, BR = B[:, :n_prog], B[:, n_prog:]
    logu = ops.m.log(BS) + ops.einsum("icp,pg->icg", delta, G[:, :n_prog])                  # (N,C,S)
    logZ = ops.m.logaddexp(ops.logsumexp(logu, axis=-1), ops.m.log(ops.sum(BR, -1)))         # (N,C)
    raw_s = ops.einsum("ic,icg->ig", pi, ops.m.exp(logu - logZ[..., None]))
    raw_r = (pi * ops.m.exp(-logZ)) @ BR
    return ops.concat([raw_s, raw_r], axis=-1)


def expected_fractions(pi, B, G, delta, m=None, n_prog=None):
    """Expected share of each gene in each bulk sample: normalise_g(m_g * sum_c pi_ic E_icg).
    m: optional (G,) positive gene-wise platform scale (scRNA reference vs bulk). Returns (N,G)."""
    raw = mix_profiles(pi, B, G, delta, n_prog)
    if m is None:
        return raw
    ops, (raw, m) = _ops(raw, m)
    raw = raw * m
    return raw / ops.sum(raw, -1, keepdims=True)


def expected_counts(L, pi, B, G, delta, m=None, n_prog=None):
    """Mean of the bulk counts, mu_ig = L_i * expected_fractions. L: (N,) library sizes over the
    modelled genes. The likelihood (negative binomial around mu) is left to the inference layer."""
    f = expected_fractions(pi, B, G, delta, m, n_prog)
    ops, (f, L) = _ops(f, L)
    return L[:, None] * f


# ----- derived quantities ------------------------------------------------------
def attribution(B, G, pi, delta):
    """Counterfactual attribution of each program's induced signal to cell types.
    C_icp = pi_ic * sum_g G_pg (E_icg - E^(-p)_icg), E^(-p) = profile with delta_.p = 0.
    pi: (...,C), delta: (...,C,P). Returns (C_icp (...,C,P), S_ip (...,P))."""
    ops, (B, G, pi, delta) = _ops(B, G, pi, delta)
    E = cell_profiles(B, G, delta)
    P = G.shape[0]
    contrib = []
    for p in range(P):
        keep = ops.asarray(1.0 - np.eye(P)[p], delta)
        E0 = cell_profiles(B, G, delta * keep)
        contrib.append(pi * ops.einsum("g,...cg->...c", G[p], E - E0))
    Cicp = ops.stack(contrib, axis=-1)
    return Cicp, ops.sum(Cicp, -2)


def shares(Cicp, thresh=1e-6):
    """A_icp = C_icp / S_ip; nan where |S_ip| < thresh."""
    ops, (Cicp,) = _ops(Cicp)
    S = ops.sum(Cicp, -2, keepdims=True)
    return ops.nan_where_not(ops.m.abs(S) > thresh, Cicp / S)


def effect_decomposition(B, G, a, gamma, kappa, beta):
    """Shapley split of a treatment's bulk program effect (typical sample, noise-free).
    a,gamma: (...,C); kappa,beta: (...,C,P).
      I_cp (intrinsic)   = 0.5 (pi_c(0)+pi_c(1)) (s_cp(1)-s_cp(0))
      K_cp (composition) = 0.5 (pi_c(1)-pi_c(0)) (s_cp(0)+s_cp(1))
      I_cp + K_cp = pi_c(1) s_cp(1) - pi_c(0) s_cp(0)   (exact)
    with s_cp(t) = sum_g G_pg E_cg(kappa + beta t)."""
    ops, (B, G, a, gamma, kappa, beta) = _ops(B, G, a, gamma, kappa, beta)
    pi0 = ops.softmax(a, axis=-1)
    pi1 = ops.softmax(a + gamma, axis=-1)
    s0 = ops.einsum("pg,...cg->...cp", G, cell_profiles(B, G, kappa))
    s1 = ops.einsum("pg,...cg->...cp", G, cell_profiles(B, G, kappa + beta))
    I = 0.5 * (pi0 + pi1)[..., None] * (s1 - s0)
    K = 0.5 * (pi1 - pi0)[..., None] * (s0 + s1)
    return dict(I=I, K=K, total=ops.sum(I + K, -2), pi0=pi0, pi1=pi1, s0=s0)
