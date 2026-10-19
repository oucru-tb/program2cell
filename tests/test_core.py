import pytest
import numpy as np
import jax.numpy as jnp
import palm_bulk as pb
from palm_bulk import sim


def _ref(seed=0, C=3, P=2, G=120):
    rng = np.random.default_rng(seed)
    B, Gm = sim.make_reference(rng, C=C, P=P, Gn=G, genes_per_prog=15)
    return rng, B, Gm


def test_profiles_normalised():
    _, B, G = _ref()
    E = pb.cell_profiles(jnp.array(B), jnp.array(G), jnp.ones((3, 2)) * 0.5)
    assert np.allclose(np.array(E).sum(-1), 1, atol=1e-5)


def test_zero_delta_recovers_reference():
    _, B, G = _ref()
    E = pb.cell_profiles(jnp.array(B), jnp.array(G), jnp.zeros((3, 2)))
    assert np.allclose(np.array(E), B, atol=1e-6)


def test_decomposition_is_exact():
    rng, B, G = _ref()
    a, gam = rng.normal(size=3), rng.normal(size=3)
    kap, bet = rng.normal(size=(3, 2)) * 0.2, rng.normal(size=(3, 2))
    d = pb.effect_decomposition(*map(jnp.array, (B, G, a, gam, kap, bet)))
    pi0 = np.exp(a) / np.exp(a).sum(); pi1 = np.exp(a + gam) / np.exp(a + gam).sum()
    s0 = np.array(d["s0"])
    E1 = np.array(pb.cell_profiles(jnp.array(B), jnp.array(G), jnp.array(kap + bet)))
    s1 = np.einsum("pg,cg->cp", G, E1)
    direct = (pi1[:, None] * s1 - pi0[:, None] * s0).sum(0)
    assert np.allclose(np.array(d["total"]), direct, atol=1e-5)
    assert np.allclose(np.array(d["I"] + d["K"]).sum(0), direct, atol=1e-5)


def test_attribution_shares_sum_to_one():
    rng, B, G = _ref()
    pi = rng.dirichlet(np.ones(3), size=4)
    delta = np.abs(rng.normal(size=(4, 3, 2)))
    C, S = pb.attribution(jnp.array(B), jnp.array(G), jnp.array(pi), jnp.array(delta))
    A = np.array(pb.shares(C, 1e-9))
    assert np.allclose(np.nansum(A, axis=1), 1, atol=1e-4)


def test_mix_profiles_sparse_matches_dense():
    import jax.numpy as jnp
    from palm_bulk.core import mix_profiles
    rng = np.random.default_rng(0)
    N, C, P, Gn, S = 5, 3, 2, 40, 12
    B = rng.dirichlet(np.ones(Gn), C)
    G = np.zeros((P, Gn)); G[:, :S] = rng.normal(0, 1, (P, S))         # program genes first
    pi = rng.dirichlet(np.ones(C), N); delta = rng.normal(0, 1, (N, C, P))
    args = [jnp.array(x, jnp.float32) for x in (pi, B, G, delta)]
    dense = mix_profiles(*args)
    sparse = mix_profiles(*args, n_prog=S)
    np.testing.assert_allclose(np.asarray(sparse), np.asarray(dense), rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(sparse).sum(1), 1, rtol=1e-5)


def _random_problem(seed=0, N=4, C=3, P=2, Gn=40, S=12):
    rng = np.random.default_rng(seed)
    B = rng.dirichlet(np.ones(Gn), C)
    G = np.zeros((P, Gn)); G[:, :S] = rng.normal(0, 1, (P, S))
    pi = rng.dirichlet(np.ones(C), N); delta = rng.normal(0, 0.7, (N, C, P))
    m = np.exp(rng.normal(0, 0.3, Gn)); L = rng.uniform(1e5, 2e5, N)
    return dict(B=B, G=G, pi=pi, delta=delta, m=m, L=L, S=S)


def test_equations_same_on_numpy_jax_torch():
    # One forward model for the sampler (JAX), truth tables (numpy) and Paper 2 training (torch)
    torch = pytest.importorskip("torch")
    from palm_bulk.core import expected_counts
    p = _random_problem()
    calls = {
        "profiles": lambda B, G, pi, d, m, L: pb.cell_profiles(B, G, d),
        "counts": lambda B, G, pi, d, m, L: expected_counts(L, pi, B, G, d, m, n_prog=p["S"]),
        "attribution": lambda B, G, pi, d, m, L: pb.attribution(B, G, pi, d)[0],
        "decomposition": lambda B, G, pi, d, m, L: pb.effect_decomposition(B, G, d[0, :, 0], d[1, :, 0],
                                                                           d[2], d[3])["I"],
    }
    args = [p[k] for k in ("B", "G", "pi", "delta", "m", "L")]
    for name, f in calls.items():
        ref = f(*args)
        assert isinstance(ref, np.ndarray), name
        out_t = f(*[torch.as_tensor(a, dtype=torch.float64) for a in args])
        assert isinstance(out_t, torch.Tensor), name
        np.testing.assert_allclose(out_t.numpy(), ref, rtol=1e-9, atol=1e-12, err_msg=name)
        out_j = f(*[jnp.asarray(a) for a in args])                        # float32 unless x64 is on
        assert type(out_j).__module__.startswith("jax"), name
        scale = np.abs(ref).max()
        np.testing.assert_allclose(np.asarray(out_j), ref, rtol=2e-4, atol=2e-5 * scale, err_msg=name)


def test_forward_model_differentiable_in_torch():
    torch = pytest.importorskip("torch")
    from palm_bulk.core import expected_counts
    p = _random_problem()
    d = torch.tensor(p["delta"], requires_grad=True)
    pi = torch.tensor(p["pi"], requires_grad=True)
    mu = expected_counts(p["L"], pi, p["B"], p["G"], d, p["m"], n_prog=p["S"])
    mu.log().sum().backward()
    assert d.grad is not None and torch.isfinite(d.grad).all() and d.grad.abs().sum() > 0
    assert pi.grad is not None and torch.isfinite(pi.grad).all()


def test_mixed_inputs_follow_jax():
    p = _random_problem()
    out = pb.cell_profiles(p["B"], p["G"], jnp.asarray(p["delta"]))       # numpy B, G with JAX delta
    assert type(out).__module__.startswith("jax")


def test_estimate_delta_recovers_shift():
    # Per-cell-type pseudo-bulk drawn from the model's profiles -> estimate_delta returns delta
    rng = np.random.default_rng(3)
    C, P, Gn, N = 4, 3, 400, 6
    B, G = sim.make_reference(rng, C=C, P=P, Gn=Gn, genes_per_prog=40)
    delta = rng.normal(0, 0.8, (N, C, P))
    E = np.asarray(pb.cell_profiles(B, G, delta))
    X = np.stack([[rng.multinomial(2_000_000, e / e.sum()) for e in Ei] for Ei in E])
    est = pb.estimate_delta(X, B, G)
    assert np.abs(est - delta).max() < 0.05
    X[0, 1] = 0                                                           # cell type absent -> nan
    assert np.isnan(pb.estimate_delta(X, B, G)[0, 1]).all()
    # normalised expression gives the same answer once rescaled to pseudo-counts
    est_n = pb.estimate_delta(X[1:] / X[1:].sum(-1, keepdims=True), B, G, scale_to=2e6)
    np.testing.assert_allclose(est_n, est[1:], atol=1e-6)


def test_bulk_delta_shared_shift():
    rng = np.random.default_rng(4)
    B, G = sim.make_reference(rng, C=3, P=2, Gn=300, genes_per_prog=30)
    pi = rng.dirichlet(np.ones(3), 5)
    d = np.repeat(rng.normal(0, 0.8, (5, 1, 2)), 3, axis=1)              # same shift in every cell type
    f = np.einsum("ic,icg->ig", pi, np.asarray(pb.cell_profiles(B, G, d)))
    est = pb.bulk_delta(f * 1e7, pi, B, G)
    np.testing.assert_allclose(est, d[:, 0], atol=0.05)


def test_convergence_flags_disagreeing_chains():
    from palm_bulk.bayes import convergence, convergence_problems
    rng = np.random.default_rng(0)
    good = rng.normal(size=(2, 500, 3))
    stuck = good.copy(); stuck[1] += 3.0                                  # chain 2 in another mode
    conv = convergence(dict(good=good, stuck=stuck, const=np.ones((2, 500, 2))))
    assert conv["good"]["rhat_max"] < 1.01 and conv["good"]["ess_min"] > 400
    assert conv["stuck"]["rhat_max"] > 1.5 and "const" not in conv
    msgs = convergence_problems(conv, chains=2)
    assert len(msgs) == 2 and all("stuck" in m and "good" not in m for m in msgs)   # R-hat and ESS
    assert any("one chain" in m for m in convergence_problems(conv, chains=1))


def test_convergence_ignores_invalid_ess(monkeypatch):
    import palm_bulk.bayes.diagnostics as dg
    monkeypatch.setattr(dg, "effective_sample_size", lambda x: np.array([-9952.0, 300.0, 500.0]))
    conv = dg.convergence(dict(x=np.random.default_rng(0).normal(size=(2, 100, 3))))
    assert conv["x"]["ess_min"] == 300 and conv["x"]["n_ess_invalid"] == 1
    assert any("not estimable" in m for m in dg.convergence_problems(conv, chains=1))
