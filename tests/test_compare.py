import sys
import textwrap
import numpy as np
import pandas as pd
import pytest
import palm_bulk as pb
from palm_bulk import compare, sim


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    rng = np.random.default_rng(11)
    st = sim.simulate_study(rng, C=4, P=3, Gn=600, genes_per_prog=40, N=12, cells_per_type=120)
    ref = pb.build_reference(st["sc_counts"], st["genes"], st["sc_cell_types"], st["sc_donors"])
    inp = pb.prepare_inputs(st["bulk"], ref, st["program_table"], treatment=st["treatment"], max_genes=400)
    data = compare.CompareData(bulk=st["bulk"], sc_counts=st["sc_counts"], sc_genes=st["genes"],
                               sc_cell_types=st["sc_cell_types"], inputs=inp, programs=st["program_table"],
                               treatment=st["treatment"], workdir=str(tmp_path_factory.mktemp("work")))
    truth = sim.truth_on_genes(st, inp.genes, inp.cell_types, inp.programs)
    return st, data, truth


def test_truth_aligned(study):
    st, data, truth = study
    N, C, P = len(data.inputs.samples), len(data.inputs.cell_types), len(data.inputs.programs)
    assert truth["pi"].shape == (N, C) and np.allclose(truth["pi"].sum(1), 1)
    assert truth["delta"].shape == (N, C, P) and truth["beta"].shape == (C, P)


def test_python_baselines_and_scoring(study):
    st, data, truth = study
    naive = compare.naive_two_step(data)
    realloc = compare.reallocation(data)
    for r in (naive, realloc):
        assert r.fractions.shape == truth["pi"].shape and np.allclose(r.fractions.sum(1), 1)
        assert r.activity.shape == truth["delta"].shape and r.effects.shape == truth["beta"].shape
        s = compare.score(r, truth)
        assert s["frac_rmse"] < 0.08 and s["frac_pearson"] > 0.9
        assert {"act_spearman_across_samples", "act_mae", "eff_pearson", "eff_sign_agree"} <= set(s)
    # naive: the same activity for every cell type, so it cannot rank cell types
    assert np.allclose(naive.activity[:, 0], naive.activity[:, 1])
    assert compare.score(naive, truth)["eff_auroc"] == 0.5
    sn = compare.score(naive, truth)
    assert np.isfinite(sn["act_spearman_shared"]) and np.isnan(sn["act_spearman_specific"])   # shared only
    assert 0 < sn["act_specific_share"] < 1


def test_ulm_activity(study):
    pytest.importorskip("decoupler")                      # optional extra: palm-bulk[progeny]
    _, data, truth = study
    ulm = compare.reallocation(data, activity="ulm")
    assert ulm.activity.shape == truth["delta"].shape and np.isfinite(ulm.activity).any()
    assert ulm.activity_scale == "score" and "act_mae" not in compare.score(ulm, truth)


def test_score_of_truth_is_perfect(study):
    _, data, truth = study
    perfect = compare.MethodResult("oracle", fractions=truth["pi"], activity=truth["delta"], effects=truth["beta"],
                                   intervals=dict(effects=(truth["beta"] - 0.1, truth["beta"] + 0.1)))
    s = compare.score(perfect, truth)
    assert s["frac_rmse"] == 0 and s["act_mae"] == 0 and s["eff_mae"] == 0
    assert s["eff_pearson"] > 0.999 and s["eff_sign_agree"] == 1 and s["eff_cov90"] == 1 and s["eff_auroc"] == 1
    assert s["act_spearman_shared"] > 0.999 and s["act_spearman_specific"] > 0.999
    assert abs(s["act_gain"] - 1) < 1e-6 and s["act_leakage"] < 1e-6 and s["act_coupling"] < 1e-9
    assert s["eff_localise"] == 1 and np.isclose(s["eff_iscore"], 0.2)


def test_leakage_and_localisation_expose_copied_activity(study):
    """A bulk shift copied to every cell type mixes all types' signal into each and cannot say which type
    responds; one type's truth leaked into another's estimate shows up as leakage, not as gain."""
    _, data, truth = study
    t, b = truth["delta"], truth["beta"]
    C = t.shape[1]
    copied = np.repeat(t.mean(1, keepdims=True), C, axis=1)
    s = compare.score(compare.MethodResult("copied", activity=copied, effects=copied[:, 0].mean(0)[None].repeat(C, 0)),
                      truth)
    assert s["act_leakage"] > 0.05 and s["act_gain"] < 0.9
    assert np.isclose(s["eff_localise"], 1 / C)                  # all types tie: 1/C credit each
    leak = t + 0.8 * np.roll(t, 1, axis=1)                       # each type gets its neighbour's activity
    sl = compare.score(compare.MethodResult("leaky", activity=leak), truth)
    assert sl["act_leakage"] > 0.05 and abs(sl["act_gain"] - 1) < 1e-6


def test_interval_score_and_false_sign_rate():
    t = dict(pi=np.full((4, 2), 0.5), beta=np.array([[0.5, 0.0], [-0.5, 0.05]]))
    e = np.array([[0.4, 0.2], [0.3, 0.05]])
    iv = (e - 0.1, e + 0.1)                                      # calls: (0,0) right, (0,1) null, (1,0) wrong sign
    s = compare.score(compare.MethodResult("m", effects=e, intervals=dict(effects=iv)), t)
    assert s["eff_n_called"] == 3 and np.isclose(s["eff_fsr"], 2 / 3) and np.isclose(s["eff_fdr"], 2 / 3)
    s2 = compare.score(compare.MethodResult("m", effects=e, intervals=dict(effects=(e - 0.01, e + 0.01))),
                       dict(t, beta=np.array([[0.5, 0.05], [0.5, 0.05]])))   # right signs; two effects too small
    assert s2["eff_fsr"] == 0 and np.isclose(s2["eff_fdr"], 0.5)
    # interval score: width 0.2 everywhere, plus 20 x miss distance where the truth is outside
    miss = np.array([[0.0, 0.1], [0.7, 0.0]])
    assert np.isclose(s["eff_iscore"], np.mean(0.2 + 20 * miss))


def test_effects_from_logfc_recovers_program_shift(study):
    _, data, _ = study
    inp = data.inputs
    b = np.random.default_rng(0).normal(0, 1, (len(inp.cell_types), len(inp.programs)))
    lfc = b @ inp.Gmat + np.arange(len(inp.cell_types))[:, None]                  # plus a per-type constant
    np.testing.assert_allclose(compare.effects_from_logfc(lfc, inp), b, atol=1e-8)


def test_unavailable_methods_reported(study, monkeypatch):
    _, data, _ = study
    for var in ("PALM_RSCRIPT", "PALM_BLADE_PYTHON", "CIBERSORTX_USER", "CIBERSORTX_TOKEN"):
        monkeypatch.delenv(var, raising=False)                  # never launch the real tools here
    monkeypatch.setattr("palm_bulk.compare.methods.find_rscript", lambda: None)
    for name in ["instaprism", "toast", "cibersortx", "blade"]:
        res, err = compare.run_method(name, data)
        assert res is None and err.startswith("unavailable"), name


FAKE_R = textwrap.dedent('''
    # Stands in for Rscript: writes the files the real R scripts would write.
    import os, sys
    import numpy as np, pandas as pd
    script, wd = os.path.basename(sys.argv[1]), sys.argv[2]
    if script == "run_instaprism.R":
        labels = sorted(set(open(os.path.join(wd, "sc_labels.txt")).read().split()))
        bulk = pd.read_csv(os.path.join(wd, "bulk_counts.csv"), index_col=0)
        genes = list(bulk.index[:-5])                                    # pretend 5 genes were dropped
        theta = pd.DataFrame(1.0 / len(labels), index=bulk.columns, columns=labels)
        theta.to_csv(os.path.join(wd, "theta.csv"))
        for ct in labels:
            (bulk.loc[genes].T / len(labels)).to_csv(os.path.join(wd, "Z_%s.csv" % ct))
        open(os.path.join(wd, "genes_used.txt"), "w").write("\\n".join(genes))
    else:
        b = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "b.npy"))   # (C, G)
        cpm = pd.read_csv(os.path.join(wd, "bulk_cpm.csv"), index_col=0)
        types = pd.read_csv(os.path.join(wd, "fractions.csv"), index_col=0).columns
        pd.DataFrame(b.T, index=cpm.index, columns=types).to_csv(os.path.join(wd, "logfc.csv"))
''')


def test_r_wrappers_file_handoff(study, tmp_path):
    _, data, truth = study
    inp = data.inputs
    fake = tmp_path / "fake_r.py"
    fake.write_text(FAKE_R)
    beta = truth["beta"]
    np.save(tmp_path / "b.npy", beta @ inp.Gmat + 0.3)                       # lfc consistent with true beta
    rs = [sys.executable, str(fake)]
    bp = compare.instaprism(data, rscript=rs)
    assert bp.fractions.shape == truth["pi"].shape and np.allclose(bp.fractions.sum(1), 1)
    assert bp.activity.shape == truth["delta"].shape and np.isfinite(bp.activity).all()
    for method in (compare.toast, compare.tca):
        r = method(data, rscript=rs)
        np.testing.assert_allclose(r.effects, beta, atol=1e-6)
        assert compare.score(r, truth)["eff_pearson"] > 0.999


def test_blade_prior_uses_donor_pseudobulk(study):
    from palm_bulk.compare.methods import _blade_prior
    st, data, _ = study
    data2 = compare.CompareData(**{**data.__dict__, "sc_donors": st["sc_donors"]})
    mean, sd = _blade_prior(data2)
    c = data.inputs.cell_types[0]
    gi = [st["genes"].index(g) for g in data.inputs.genes]
    prof = []
    for dn in np.unique(st["sc_donors"]):
        s = st["sc_counts"][(st["sc_cell_types"] == c) & (st["sc_donors"] == dn)].sum(0)
        prof.append(np.log1p(s[gi] / s.sum() * 1e6))
    np.testing.assert_allclose(mean[0], np.mean(prof, 0))
    assert (np.asarray(sd) >= 0.1).all()


@pytest.mark.slow
def test_cellgeometry(study):
    """Runs only where R with cellGeometry is set up (PALM_RSCRIPT); skipped otherwise."""
    import subprocess
    from palm_bulk.compare.base import find_rscript, _env_for
    rs = find_rscript()
    if not rs or subprocess.run([rs, "-e", "library(cellGeometry)"], env=_env_for(rs), capture_output=True,
                                timeout=120).returncode != 0:
        pytest.skip("R with cellGeometry not available")
    _, data, truth = study
    r = compare.cellgeometry(data)
    assert r.fractions.shape == truth["pi"].shape and np.allclose(r.fractions.sum(1), 1)
    s = compare.score(r, truth)
    assert s["frac_rmse"] < 0.1 and r.activity.shape == truth["delta"].shape
