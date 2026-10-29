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


def test_estimate_delta_large_weights_converge():
    # Raw PROGENy-like weights (up to ~17) and an interferon-sized shift: a capped Newton step
    # cycled between 0 and 1 here and returned 0; the line search must reach the true shift
    rng = np.random.default_rng(5)
    Gn = 300
    B = rng.dirichlet(np.ones(Gn))[None]
    G = np.zeros((1, Gn))
    G[0, :30] = rng.uniform(1, 17, 30)
    E = np.asarray(pb.cell_profiles(B, G, np.array([[[0.3]]])))[0, 0]
    X = rng.multinomial(5_000_000, E / E.sum())[None, None].astype(float)
    assert abs(pb.estimate_delta(X, B, G)[0, 0, 0] - 0.3) < 0.01


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


def test_negbin_log_prob_matches_numpyro():
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from palm_bulk.bayes.model import _NegBin2
    rng = np.random.default_rng(3)
    mu = jnp.array(rng.gamma(2.0, 20.0, (6, 50)))
    th = jnp.array(rng.gamma(2.0, 10.0, 50))
    y = jnp.array(rng.poisson(np.asarray(mu)).astype(float))
    ref = lambda m, t: dist.NegativeBinomial2(m, t).log_prob(y).sum()
    new = lambda m, t: _NegBin2(m, t).log_prob(y).sum()
    np.testing.assert_allclose(new(mu, th), ref(mu, th), rtol=1e-5)
    for a, b in zip(jax.grad(new, (0, 1))(mu, th), jax.grad(ref, (0, 1))(mu, th)):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)     # float32: theta-gradient terms cancel


def test_mass_blocks_dense_globals():
    import jax.numpy as jnp
    from palm_bulk.bayes.fit import _mass_blocks
    inv = {("a",): jnp.array([1.0, 2.0]), ("sigma_d",): jnp.array([3.0]), ("zres",): jnp.ones(5)}
    out, dense = _mass_blocks(inv)
    assert dense == [("a", "sigma_d")]
    np.testing.assert_allclose(out[("a", "sigma_d")], np.diag([1.0, 2.0, 3.0]))
    assert ("zres",) in out and ("a",) not in out


def test_activity_models_trace():
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    from palm_bulk.bayes.model import make_model
    rng = np.random.default_rng(5)
    N, C, P, Gn = 5, 3, 2, 30
    B = jnp.array(rng.dirichlet(np.ones(Gn), C))
    G = jnp.array(rng.normal(0, 1, (P, Gn)))
    Y = jnp.array(rng.poisson(20, (N, Gn)).astype(float))
    args = (Y, Y.sum(1), jnp.array([0.0, 1, 0, 1, 1]), B, G)
    want = {"independent": {"zres"}, "common": {"lam", "u", "zres"}, "shared": {"lam", "u", "zres"},
            "shared_only": {"lam", "u"}}
    for act, sites in want.items():
        tr = handlers.trace(handlers.seed(make_model(theta=30.0, activity=act), 0)).get_trace(*args)
        assert sites <= set(tr) and not ({"lam", "u", "zres"} - sites) & set(tr)
        assert tr["delta"]["value"].shape == (N, C, P)
    with pytest.raises(ValueError):
        make_model(activity="nope")


def test_gauss_marginal_matches_closed_form():
    # log integral of exp(-x'Lam x/2 + h'x) N(x; mu, S): compare with the textbook form (S invertible)
    import jax.numpy as jnp
    from palm_bulk.bayes.twostage import _gauss_marginal
    rng = np.random.default_rng(7)
    D = 4
    A = rng.normal(size=(D, D)); S = A @ A.T + 0.5 * np.eye(D)
    Bm = rng.normal(size=(2, D, D)); Lam = Bm @ np.swapaxes(Bm, 1, 2)
    h, mu = rng.normal(size=(2, D)), rng.normal(size=(2, D))
    ll, mean, _ = _gauss_marginal(jnp.array(mu), jnp.array(S), jnp.array(Lam), jnp.array(h))
    Si = np.linalg.inv(S)
    for i in range(2):
        P = Si + Lam[i]; b = Si @ mu[i] + h[i]
        ref = (0.5 * b @ np.linalg.solve(P, b) - 0.5 * mu[i] @ Si @ mu[i]
               - 0.5 * np.linalg.slogdet(S)[1] - 0.5 * np.linalg.slogdet(P)[1])
        np.testing.assert_allclose(ll[i], ref, rtol=1e-4)
        np.testing.assert_allclose(mean[i], np.linalg.solve(P, b), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("activity", ["independent", "shared_only"])
def test_latent_draws_match_gaussian_conditional(activity):
    # perturb-and-solve draws have the exact conditional mean and covariance (S may be singular)
    import jax.numpy as jnp
    from palm_bulk.bayes.twostage import _draw_latents, _prior_moments
    rng = np.random.default_rng(1)
    C, P, N, n = 2, 2, 2, 40000
    D = C + C * P
    A = rng.normal(size=(N, D, D)); Lam = A @ np.swapaxes(A, 1, 2) / D
    h, T = rng.normal(size=(N, D)), np.array([0.0, 1.0])
    p = dict(a=np.array([0.3, -0.3]), gamma=np.array([0.5, -0.5]), s_eta=0.7,
             kappa=np.array([[0.1, -0.2], [0.0, 0.3]]), beta=np.array([[0.2, 0.0], [-0.1, 0.1]]))
    p.update(sigma_d=0.4) if activity == "independent" else p.update(lam=np.array([[0.5, 0.2], [0.3, 0.6]]))
    post = {k: jnp.repeat(jnp.asarray(v, float)[None], n, 0) for k, v in p.items()}
    x = _draw_latents(post, jnp.asarray(Lam), jnp.asarray(h), jnp.asarray(T), None, C, P, None, activity, 0)
    mu, S = (np.asarray(v) for v in _prior_moments({k: jnp.asarray(v, float) for k, v in p.items()},
                                                     jnp.asarray(T), None, C, P, None, activity))
    for i in range(N):
        M = np.eye(D) + S @ Lam[i]
        cov = np.linalg.solve(M, S)
        mean = mu[i] + cov @ (h[i] - Lam[i] @ mu[i])
        np.testing.assert_allclose(x[:, i].mean(0), mean, atol=0.02)
        np.testing.assert_allclose(np.cov(x[:, i].T), 0.5 * (cov + cov.T), atol=0.02)


def test_sandwich_matches_model_when_model_is_right():
    # Poisson data from the model itself: robust and model-based variances agree (inflation ~ 1)
    import jax
    import jax.numpy as jnp
    from palm_bulk.bayes.twostage import sample_summaries, gene_blocks
    rng = np.random.default_rng(4)
    C, P, Gn, N = 3, 2, 400, 6
    B = rng.dirichlet(np.full(Gn, 0.5), C)
    G = np.where(rng.random((P, Gn)) < 0.15, rng.normal(0, 1, (P, Gn)), 0.0)
    pi = rng.dirichlet(np.ones(C), N)
    d = rng.normal(0, 0.3, (N, C, P))
    E = np.exp(np.log(B)[None] + np.einsum("ncp,pg->ncg", d, G)); E /= E.sum(-1, keepdims=True)
    L = np.full(N, 2e6)
    Y = rng.poisson(L[:, None] * np.einsum("nc,ncg->ng", pi, E)).astype(float)
    args = [jnp.asarray(v) for v in (Y, L, B, G)]
    blocks = gene_blocks(B, G, 80)
    q = sample_summaries(*args, jnp.full(Gn, 1e8), likelihood="quasi", sandwich=True, blocks=blocks)
    assert 0.6 < float(np.median(q["inflation"])) < 1.6


def test_fixed_eta_summaries():
    """fix_eta holds the fractions at eta0, gives no information about eta and still finds delta."""
    import jax.numpy as jnp
    from palm_bulk.bayes.twostage import sample_summaries, _two_step_summaries
    rng = np.random.default_rng(3)
    B, G = sim.make_reference(rng, C=3, P=2, Gn=150, genes_per_prog=20)
    s = sim.simulate_causal(rng, B, G, N=6)
    eta = jnp.log(jnp.asarray(s["pi"]))
    Y, L, B_, G_ = (jnp.asarray(v, float) for v in (s["Y"], s["L"], B, G))
    out = sample_summaries(Y, L, B_, G_, 50.0, eta0=eta, likelihood="quasi", fix_eta=True)
    np.testing.assert_allclose(np.asarray(out["x_hat"][:, :3]), np.asarray(eta))
    assert np.abs(np.asarray(out["Lam"][:, :3, :])).max() == 0
    assert float(out["grad_norm"].max()) < 0.05            # float32 floor (exactly 0 in float64)
    two = _two_step_summaries(Y, L, B_, G_, 50.0, None, eta, 3.0, 1.0, 80, "quasi", False, None)
    w = np.linalg.eigvalsh(np.asarray(two["Lam"]))
    assert (w.min(1) > -1e-6 * w.max(1)).all()             # PSD (zero only along the softmax shift)
