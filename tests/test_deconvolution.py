"""Reallocation, PALM-lite fractions, calibration and program_deconvolution on simulated studies."""
import types
import numpy as np
import pytest
import palm_bulk as pb
from palm_bulk import sim


def _study(seed=0, C=3, P=2, Gn=240, n_ref_donors=40, n_bulk=40, effect=(0, 0, 1.0), sd=0.4, depth=4e5):
    """Reference specimens (one per donor) and bulk samples; every specimen has its own program shifts
    delta ~ N(0, sd); treated bulk samples get `effect` = (type, program, size) on top."""
    rng = np.random.default_rng(seed)
    B, G = sim.make_reference(rng, C=C, P=P, Gn=Gn, genes_per_prog=40)

    def specimens(n, T=None):
        delta = rng.normal(0, sd, (n, C, P))
        if T is not None:
            delta[T == 1, effect[0], effect[1]] += effect[2]
        frac = rng.dirichlet(np.full(C, 4.0), n)
        E = np.asarray(pb.cell_profiles(B, G, delta))                        # (n, C, Gn)
        counts = np.stack([[rng.multinomial(int(depth * f), e / e.sum()) for f, e in zip(fr, Ei)]
                           for fr, Ei in zip(frac, E)]).astype(float)
        return counts, delta

    ref_counts, _ = specimens(n_ref_donors)
    T = np.repeat([0, 1], n_bulk // 2)
    bulk_counts, bulk_delta = specimens(n_bulk, T)
    tot = bulk_counts.sum(-1)
    inputs = types.SimpleNamespace(Y=bulk_counts.sum(1), B=B, Gmat=G, T=T, cell_types=["t%d" % c for c in range(C)],
                                   programs=["p%d" % p for p in range(P)])
    return dict(B=B, G=G, ref_counts=ref_counts, inputs=inputs, pi=tot / tot.sum(1, keepdims=True),
                delta=bulk_delta, T=T)


def test_em_fractions_recovers_and_squarem_matches_plain():
    s = _study(1)
    Y, B = s["inputs"].Y, s["B"]
    fast = pb.em_fractions(Y, B, tol=1e-10)
    slow = pb.em_fractions(Y, B, tol=1e-10, accelerate=False, max_iter=20000)
    assert np.abs(fast - slow).max() < 1e-4
    np.testing.assert_allclose(fast.sum(1), 1)
    assert np.sqrt(((fast - s["pi"]) ** 2).mean()) < 0.03


def test_reallocate_shapes_and_signal():
    s = _study(2)
    inp = s["inputs"]
    a = pb.reallocate(inp.Y, s["pi"], inp.B, inp.Gmat)
    assert a.shape == s["delta"].shape
    # own-type signal survives reallocation (attenuated, positive correlation)
    r = [np.corrcoef(a[:, c, p], s["delta"][:, c, p])[0, 1] for c in range(3) for p in range(2)]
    assert np.mean(r) > 0.5
    draws = np.stack([s["pi"]] * 3)
    np.testing.assert_allclose(pb.reallocate(inp.Y, draws, inp.B, inp.Gmat)[1], a)
    # a zero fraction (NNLS) still gives finite activity thanks to the floor
    pi0 = s["pi"].copy(); pi0[0, 0] = 0
    assert np.isfinite(pb.reallocate(inp.Y, pi0, inp.B, inp.Gmat)).all()


def test_calibration_set_folds_by_donor():
    s = _study(3, n_ref_donors=12)
    donors = np.repeat(["d%d" % i for i in range(6)], 2)                  # two specimens per donor
    cal = pb.build_calibration_set(s["ref_counts"], donors, s["G"], n_folds=3)
    assert len(cal) == 12 and cal.info["n_folds"] == 3
    for k in range(3):                                                   # a donor's specimens stay in one fold
        assert len(set(cal.donors[k]) & set(np.concatenate([cal.donors[j] for j in range(3) if j != k]))) == 0
    dh, d, dn, fk = cal.dhat()
    assert dh.shape == d.shape == (12, 3, 2)


@pytest.mark.parametrize("source", ["true", "lite", "external"])
def test_program_deconvolution_calibrated_and_finds_effect(source):
    s = _study(4)
    inp = s["inputs"]
    cal = pb.build_calibration_set(s["ref_counts"], ["d%d" % i for i in range(40)], s["G"])
    kw = dict(true=dict(fractions=s["pi"]), lite=dict(engine="lite"),
              external=dict(fractions=np.clip(s["pi"] + np.random.default_rng(0).normal(0, 0.02, s["pi"].shape),
                                              0, None)))[source]
    res = pb.program_deconvolution(inp, cal, effect_interval="lka", estimator="diff", **kw)
    lo, hi = res.activity_interval
    cover = ((s["delta"] >= lo) & (s["delta"] <= hi)).mean()
    assert cover > 0.75                                                   # nominal 0.9, small calibration set
    beta = pb.core.contrast(s["T"])(s["delta"])
    assert res.calls[0, 0] and res.effects[0, 0] > 0                       # the planted effect is found
    elo, ehi = res.effects_interval
    assert ((beta >= elo) & (beta <= ehi)).mean() >= 0.8
    assert res.table().shape[0] == 3 * 2
    assert res.info["calibration"] == {"true": "PALM-lite proxy for fraction error",
                                       "lite": "engine's own fractions",
                                       "external": "PALM-lite proxy for fraction error"}[source]


@pytest.mark.parametrize("estimator", ["unmix", "auto", None])
def test_program_deconvolution_unmix_localises(estimator):
    s = _study(6, n_bulk=60, sd=0.1)
    inp = s["inputs"]
    cal = pb.build_calibration_set(s["ref_counts"], ["d%d" % i for i in range(40)], s["G"])
    kw = {} if estimator is None else dict(estimator=estimator, unmix_kw=dict(pev=False, misfit=False))
    res = pb.program_deconvolution(inp, cal, fractions=s["pi"], n_sim=3, **kw)
    beta = pb.core.contrast(s["T"])(s["delta"])
    if estimator is None:                                                # default: the causal setting
        assert res.info["unmix"]["tau"] is not None and res.info["unmix"]["misfit_sd"] is not None
        estimator = "auto"
    assert res.info["estimator"] == estimator and "unmix" in res.info
    assert res.calls[0, 0] and abs(res.effects[0, 0] - beta[0, 0]) < 0.3
    if estimator == "unmix":                                             # the leak into types 1, 2 is removed
        raw = pb.core.contrast(s["T"])(res.raw_activity)
        assert np.abs(res.effects[1:, 0]).max() < np.abs(raw[1:, 0]).max()


def test_calibrator_rejects_unknown_interval():
    s = _study(5, n_ref_donors=10)
    cal = pb.build_calibration_set(s["ref_counts"], ["d%d" % i for i in range(10)], s["G"])
    c = pb.Calibrator.from_set(cal, n_boot=5)
    with pytest.raises(ValueError):
        c.effects(np.zeros((4, 3, 2)), np.array([0, 0, 1, 1]), interval="wide")
