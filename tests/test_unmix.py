import numpy as np
from scipy.special import softmax

from palm_bulk.core.realloc import reallocate
from palm_bulk.core.unmix import leakage_operator, forward, unmix, unmix_effects


def _world(seed=0, C=3, G=400, P=2, N=60):
    rng = np.random.default_rng(seed)
    B = rng.dirichlet(np.full(G, 0.5), size=C)
    W = np.zeros((P, G))
    for p in range(P):
        idx = rng.choice(G, 40, replace=False)
        W[p, idx] = rng.normal(0, 1, 40)
    pi = rng.dirichlet(np.full(C, 2.0), size=N)
    T = np.arange(N) % 2
    return rng, B, W, pi, T


def _bulk(rng, B, W, pi, delta, depth=2e6):
    Bs = softmax(np.log(B)[None] + np.einsum("nkp,pg->nkg", delta, W), -1)
    mu = np.einsum("nk,nkg->ng", pi, Bs)
    return np.stack([rng.multinomial(int(depth), m / m.sum()) for m in mu]).astype(float)


def test_leakage_operator_identity_for_a_lone_type():
    _, B, W, _, _ = _world()
    pi = np.array([[1.0, 0.0, 0.0]])
    A = leakage_operator(pi, B, W, floor=1e-12)
    assert np.allclose(A[0, 0, 0], 1, atol=1e-3)


def test_leakage_operator_matches_finite_difference():
    _, B, W, pi, _ = _world()
    pi = pi[:4]
    A = leakage_operator(pi, B, W)
    eps = 1e-4
    d0 = reallocate(pi @ B * 1e8, pi, B, W)
    for k in range(B.shape[0]):
        for p in range(W.shape[0]):
            Bk = B.copy()
            Bk[k] = softmax(np.log(B[k]) + eps * W[p])
            fd = (reallocate(pi @ Bk * 1e8, pi, B, W)[:, :, p] - d0[:, :, p]) / eps
            assert np.abs(fd - A[:, :, k, p]).max() < 0.01


def test_forward_reproduces_reallocated_activity():
    rng, B, W, pi, T = _world()
    delta = rng.normal(0, 0.3, (len(pi), B.shape[0], W.shape[0]))
    Y = _bulk(rng, B, W, pi, delta, depth=1e8)
    assert np.abs(forward(pi, delta, B, W) - reallocate(Y, pi, B, W)).max() < 0.02


def test_unmix_localises_a_type_specific_effect():
    rng, B, W, pi, T = _world(seed=1)
    N, C, P = len(pi), B.shape[0], W.shape[0]
    beta = np.zeros((C, P))
    beta[0, 0] = 0.6                                             # only type 0, program 0 responds
    delta = rng.normal(0, 0.05, (N, C, P)) + T[:, None, None] * beta[None]
    Y = _bulk(rng, B, W, pi, delta)
    raw = reallocate(Y, pi, B, W)
    raw_eff = raw[T == 1].mean(0) - raw[T == 0].mean(0)
    r = unmix(Y, pi, B, W, T, n_boot=30)
    e = r["effects"]
    # reallocation alone leaks the effect into the other types; unmixing puts it back
    assert raw_eff[1:, 0].max() > 0.1
    assert abs(e[0, 0] - 0.6) < 0.15
    assert np.abs(e[1:, 0]).max() < 0.1
    assert np.all((r["lo"] <= r["effects"]) & (r["effects"] <= r["hi"]))


def test_unmix_shared_effect_is_shrunk_to_shared():
    rng, B, W, pi, T = _world(seed=2)
    N, C, P = len(pi), B.shape[0], W.shape[0]
    beta = np.zeros((C, P))
    beta[:, 1] = 0.4                                             # every type, program 1
    delta = rng.normal(0, 0.05, (N, C, P)) + T[:, None, None] * beta[None]
    Y = _bulk(rng, B, W, pi, delta)
    r = unmix(Y, pi, B, W, T, n_boot=10)
    assert np.abs(r["effects"][:, 1] - 0.4).max() < 0.12
    assert np.abs(r["effects"][:, 0]).max() < 0.1


def test_design_check_and_simulated_cohort():
    from palm_bulk.core.calibrate import build_calibration_set
    from palm_bulk.core.unmix import design_check, simulate_cohort
    rng, B, W, pi, T = _world(seed=4, N=30)
    C, G = B.shape
    counts = np.stack([np.stack([rng.multinomial(20000, b) for b in B]) for _ in range(12)]).astype(float)
    cal = build_calibration_set(counts, np.arange(12).astype(str), W, n_folds=3)
    Y, pt, Bk, beta = simulate_cohort(cal, pi, T, rng)
    assert Y.shape == (30, G) and np.allclose(pt.sum(1), 1) and beta.shape == (C, W.shape[0])
    assert np.abs(pt - pi).max() < 0.05                          # the cohort has the study's fractions
    r = design_check(cal, pi, T, n_sim=3, n_boot=5, nonlinear=False)
    assert r["rmse_unmix"].shape == (W.shape[0],) and np.isfinite(r["z_q"])


def test_unmix_effects_with_groups_and_covariates_runs():
    rng, B, W, pi, T = _world(seed=3)
    N = len(pi)
    delta = rng.normal(0, 0.05, (N, B.shape[0], W.shape[0]))
    Y = _bulk(rng, B, W, pi, delta)
    dh = reallocate(Y, pi, B, W)
    A = leakage_operator(pi, B, W)
    r = unmix_effects(dh, A, T, X=rng.normal(size=(N, 1)), groups=np.arange(N) // 2, n_boot=10)
    assert r["effects"].shape == (B.shape[0], W.shape[0]) and np.isfinite(r["se"]).all()
    assert r["n_units"] == N // 2


def test_direct_engine_gradient_and_localisation():
    from palm_bulk.core import direct as dr
    rng, B, W, pi, T = _world(seed=1)
    N, C, P = len(pi), B.shape[0], W.shape[0]
    beta = np.zeros((C, P))
    beta[0, 0] = 0.6
    delta = rng.normal(0, 0.05, (N, C, P)) + T[:, None, None] * beta[None]
    Y = _bulk(rng, B, W, pi, delta)
    X = rng.normal(size=(N, 1))
    gi = np.unique((np.arange(N) // 2).astype(str), return_inverse=True)[1]
    k = 2 * C * P + P + gi.max() * P
    th = rng.normal(0, 0.1, k)
    args = (Y, np.log(pi), np.log(B / B.sum(1, keepdims=True)), W, T.astype(float), X, gi, 0.3, Y.sum())
    f, g = dr._objective(th, *args)
    num = np.array([(dr._objective(th + 1e-6 * np.eye(k)[i], *args)[0] - f) / 1e-6 for i in range(k)])
    assert np.abs(num - g).max() < 1e-5
    r = dr.direct_effects(Y, pi, B, W, T, n_boot=10)
    assert abs(r["effects"][0, 0] - 0.6) < 0.1 and np.abs(r["effects"][1:, 0]).max() < 0.08
    r2 = dr.direct_effects(Y, pi, B, W, T, X=X, groups=np.arange(N) // 2, n_boot=5)
    assert r2["n_units"] == N // 2 and abs(r2["effects"][0, 0] - 0.6) < 0.15


def test_unmix_penalty_groups_pev_and_misfit():
    from palm_bulk.core.unmix import offprogram_tau, misfit_sd
    rng, B, W, pi, T = _world(seed=1)
    N, C, P = len(pi), B.shape[0], W.shape[0]
    beta = np.zeros((C, P))
    beta[0, 0] = 0.6
    delta = rng.normal(0, 0.05, (N, C, P)) + T[:, None, None] * beta[None]
    Y = _bulk(rng, B, W, pi, delta)
    dh = reallocate(Y, pi, B, W)
    A = leakage_operator(pi, B, W, diagonal=False)
    base = unmix_effects(dh, A, T, n_boot=5, pi=pi, B=B, G=W)
    for sep in (True, "program"):
        r = unmix_effects(dh, A, T, n_boot=5, pi=pi, B=B, G=W, separate=sep)
        assert abs(r["effects"][0, 0] - 0.6) < 0.15 and np.abs(r["effects"][1:, 0]).max() < 0.1
        assert r["lam"].shape[1] == (3 if sep is True else 2 + P)
    r = unmix_effects(dh, A, T, n_boot=5, pi=pi, B=B, G=W, pev=True)
    assert np.allclose(r["effects"], base["effects"]) and np.all(r["se"] >= r["se_boot"])
    # no off-program response in this world: tau small; a planted one is measured
    tau, tb = offprogram_tau(Y, pi, B, W, T, r["effects"])
    off = rng.normal(0, 0.3, B.shape[1])
    Bt = B * np.exp(off)
    Yt = np.stack([_bulk(rng, Bt if t else B, W, pi[[i]], delta[[i]])[0] for i, t in enumerate(T)])
    tau2, tb2 = offprogram_tau(Yt, pi, B, W, T, r["effects"])
    assert tb < 0.1 and 0.15 < tb2 < 0.45
    m = misfit_sd(pi, B, W, T, r["effects"], 0.3, r["lam"], n=5)
    assert m.shape == (C, P) and np.all(m > 0) and np.all(np.isfinite(m))
    r2 = unmix_effects(dh, A, T, n_boot=5, pi=pi, B=B, G=W, misfit_tau="auto", Y=Y)
    assert r2["tau"] is not None and np.all(r2["se"] >= r2["se_boot"])


def test_direct_quasi_likelihood_and_gene_offsets():
    from palm_bulk.core import direct as dr
    rng, B, W, pi, T = _world(seed=1, G=60)
    N, C, P = len(pi), B.shape[0], W.shape[0]
    delta = rng.normal(0, 0.05, (N, C, P))
    delta[:, 0, 0] += 0.6 * T
    Y = _bulk(rng, B, W, pi, delta)
    k = 2 * C * P + B.shape[1]
    th = rng.normal(0, 0.1, k)
    for w in (None, rng.uniform(0.2, 2, B.shape[1])):
        wt = Y.sum() if w is None else (Y @ w).sum()
        args = (Y, np.log(pi), np.log(B / B.sum(1, keepdims=True)), W, T.astype(float), None, None, 0.0, wt, w)
        f, g = dr._objective(th, *args)
        num = np.array([(dr._objective(th + 1e-6 * np.eye(k)[i], *args)[0] - f) / 1e-6 for i in range(k)])
        assert np.abs(num - g).max() < 1e-5
    # a gene-wise batch difference between reference and study: the offsets absorb it
    rng, B, W, pi, T = _world(seed=1)
    Bs = B * np.exp(rng.normal(0, 0.5, B.shape[1]))
    Bs /= Bs.sum(1, keepdims=True)
    delta = rng.normal(0, 0.05, (len(pi), C, P))
    delta[:, 0, 0] += 0.6 * T
    Y = _bulk(rng, Bs, W, pi, delta)
    r = dr.direct_effects(Y, pi, B, W, T, n_boot=3, gene_offset=True)
    assert abs(r["effects"][0, 0] - 0.6) < 0.08
    r = dr.direct_effects(Y, pi, B, W, T, n_boot=3, weights="nb")
    assert r["w"].shape == (B.shape[1],) and np.isclose(r["w"].mean(), 1)
