"""Causal effect estimators, mediation decomposition and sensitivity (palm_bulk.causal)."""
import numpy as np
import pytest
import palm_bulk as pb
from palm_bulk import causal, sim


def _confounded(n=3000, seed=0, effect=1.0, nonlinear=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n)
    T = (rng.random(n) < 1 / (1 + np.exp(-1.5 * X))).astype(float)     # confounded assignment
    m = 2 * X + (1.5 * X ** 2 if nonlinear else 0)
    y = effect * T + m + rng.normal(0, 0.5, n)
    return y, T, X


def test_diff_weights_match_contrast_and_ols_without_covariates():
    T = np.array([0, 0, 1, 1, 1, 0, 1])
    y = np.arange(7.0)
    w = causal.estimator_weights(T)
    assert np.isclose(w @ y, y[T == 1].mean() - y[T == 0].mean())
    np.testing.assert_allclose(causal.estimator_weights(T, estimator="ols"), w, atol=1e-10)
    np.testing.assert_allclose(w, pb.core.calibrate.contrast_weights(T))


def test_adjusted_estimators_remove_confounding():
    y, T, X = _confounded()
    naive = causal.estimator_weights(T) @ y
    assert naive > 1.8                                                  # strongly biased upwards
    for est in ("ols", "ipw", "aipw"):
        assert abs(causal.estimator_weights(T, X, estimator=est) @ y - 1.0) < 0.15, est


def test_aipw_doubly_robust_when_outcome_model_wrong():
    y, T, X = _confounded(nonlinear=True, seed=1)
    aipw = causal.estimator_weights(T, X, estimator="aipw") @ y         # linear outcome model is wrong here
    assert abs(aipw - 1.0) < 0.2


def test_paired_design_removes_donor_effects():
    # unbalanced within donors (2 + 1 samples per donor, arms alternating): the difference of means then
    # carries donor effects, the within-donor (fixed-effect) estimate does not
    rng = np.random.default_rng(2)
    donors = np.repeat(np.arange(30), 3)
    T = np.concatenate([[0, 0, 1] if i % 2 else [0, 1, 1] for i in range(30)])
    est_d, est_p = [], []
    for _ in range(200):
        y = rng.normal(0, 3, 30)[donors] + 0.5 * T + rng.normal(0, 0.3, 90)
        est_d.append(causal.estimator_weights(T) @ y)
        est_p.append(causal.estimator_weights(T, groups=donors, estimator="ols") @ y)
    assert abs(np.mean(est_p) - 0.5) < 0.03
    assert np.std(est_p) < 0.3 * np.std(est_d)


def test_decomposition_exact_and_attributes_correctly():
    rng = np.random.default_rng(3)
    B, G = sim.make_reference(rng, C=3, P=2, Gn=150, genes_per_prog=30)
    N = 40
    T = np.repeat([0, 1], N // 2)
    pi0 = rng.dirichlet(np.full(3, 20.0), N)
    # (a) composition only: type 0 expands in treated, activity unchanged
    pi = pi0.copy(); pi[T == 1, 0] += 0.2; pi /= pi.sum(1, keepdims=True)
    delta = np.tile(rng.normal(0, 0.3, (1, 3, 2)), (N, 1, 1))
    d = causal.decompose(pi, delta, B, G, T, n_boot=50)
    assert np.abs(d["I"]).max() < 1e-10 and np.abs(d["K"]).max() > 1e-4
    # identity: I + K = pi1 s1 - pi0 s0 with arm means
    s = causal.program_signal(B, G, delta)
    pi1, pi0_ = pi[T == 1].mean(0), pi[T == 0].mean(0)
    np.testing.assert_allclose((d["I"] + d["K"]).sum(0),
                               (pi1[:, None] * s[T == 1].mean(0) - pi0_[:, None] * s[T == 0].mean(0)).sum(0),
                               atol=1e-12)
    # (b) intrinsic only: same mix, type 1 program 0 switches on in treated
    delta2 = delta.copy(); delta2[T == 1, 1, 0] += 1.0
    d2 = causal.decompose(pi0[:1].repeat(N, 0), delta2, B, G, T, n_boot=50)
    assert np.abs(d2["K"]).max() < 1e-10
    assert np.argmax(np.abs(d2["I"][:, 0])) == 1 and d2["share_intrinsic"][0] > 0.99


def test_robustness_value_behaviour():
    eff = np.array([[0.0, 0.5, 2.0]])
    lo, hi = eff - 0.3, eff + 0.3
    rv = causal.robustness_value(eff, lo, hi, df=50)
    assert rv[0, 0] == 0 and rv[0, 1] < rv[0, 2] < 1


def test_program_deconvolution_with_covariates_and_decomposition():
    from test_deconvolution import _study
    s = _study(4)
    inp = s["inputs"]
    cal = pb.build_calibration_set(s["ref_counts"], ["d%d" % i for i in range(40)], s["G"])
    X = np.random.default_rng(0).normal(size=len(s["T"]))
    res = pb.program_deconvolution(inp, cal, fractions=s["pi"], covariates=X, decompose=True, estimator="diff")
    assert res.info["estimator"] == "ols"
    assert res.calls[0, 0] or res.effects[0, 0] > 0.5
    tab = res.table()
    assert {"robustness", "intrinsic", "composition"} <= set(tab.columns)
    assert np.isfinite(res.decomposition["total"]).all()


def test_known_answer_metrics_and_comparison_stats():
    from palm_bulk.compare import known_answer as ka
    resp = np.zeros((4, 3), bool); resp[[0, 1], 2] = True             # program 2 responds in types 0 and 1
    eff = np.zeros((4, 3)); eff[[0, 1], 2] = [1.0, 0.8]; eff[2, 2] = 0.3
    calls = np.abs(eff) > 0.2
    assert ka.responder_auroc(eff, resp) == 1.0
    assert np.isclose(ka.leakage_fpr(calls, resp), 1 / 10)
    assert ka.responder_tpr(eff, calls, resp) == 1.0
    spec = ka.program_specificity(eff, np.array([False, False, True]), types=[0, 1])
    assert spec["top1"] == 1.0 and spec["auroc"] == 1.0
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.02, 10); b = a - 0.03 + rng.normal(0, 0.005, 10)
    t = ka.paired_test(a, b)
    assert t["lo"] > 0 and t["p_wilcoxon"] < 0.01 and t["win_rate"] == 1.0
    np.testing.assert_allclose(ka.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    m = ka.random_effects([0.03, 0.02, 0.025], [0.005, 0.006, 0.004])
    assert m["lo"] > 0 and m["p"] < 1e-3
