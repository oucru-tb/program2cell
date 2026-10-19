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
    ulm = compare.reallocation(data, activity="ulm")
    assert ulm.activity_scale == "score" and "act_mae" not in compare.score(ulm, truth)


def test_score_of_truth_is_perfect(study):
    _, data, truth = study
    perfect = compare.MethodResult("oracle", fractions=truth["pi"], activity=truth["delta"], effects=truth["beta"],
                                   intervals=dict(effects=(truth["beta"] - 0.1, truth["beta"] + 0.1)))
    s = compare.score(perfect, truth)
    assert s["frac_rmse"] == 0 and s["act_mae"] == 0 and s["eff_mae"] == 0
    assert s["eff_pearson"] > 0.999 and s["eff_sign_agree"] == 1 and s["eff_cov90"] == 1 and s["eff_auroc"] == 1


def test_effects_from_logfc_recovers_program_shift(study):
    _, data, _ = study
    inp = data.inputs
    b = np.random.default_rng(0).normal(0, 1, (len(inp.cell_types), len(inp.programs)))
    lfc = b @ inp.Gmat + np.arange(len(inp.cell_types))[:, None]                  # plus a per-type constant
    np.testing.assert_allclose(compare.effects_from_logfc(lfc, inp), b, atol=1e-8)


def test_unavailable_methods_reported(study, monkeypatch):
    _, data, _ = study
    monkeypatch.delenv("PALM_RSCRIPT", raising=False)
    monkeypatch.setattr("palm_bulk.compare.methods.find_rscript", lambda: None)
    for name in ["bayesprism", "toast", "cibersortx", "blade"]:
        res, err = compare.run_method(name, data)
        assert res is None and err.startswith("unavailable"), name


FAKE_R = textwrap.dedent('''
    # Stands in for Rscript: writes the files the real R scripts would write.
    import os, sys
    import numpy as np, pandas as pd
    script, wd = os.path.basename(sys.argv[1]), sys.argv[2]
    if script == "run_bayesprism.R":
        labels = sorted(set(open(os.path.join(wd, "sc_labels.txt")).read().split()))
        bulk = pd.read_csv(os.path.join(wd, "bulk_counts.csv"), index_col=0)
        genes = list(bulk.index[:-5])                                    # pretend cleanup dropped 5 genes
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
    bp = compare.bayesprism(data, rscript=rs)
    assert bp.fractions.shape == truth["pi"].shape and np.allclose(bp.fractions.sum(1), 1)
    assert bp.activity.shape == truth["delta"].shape and np.isfinite(bp.activity).all()
    for method in (compare.toast, compare.tca):
        r = method(data, rscript=rs)
        np.testing.assert_allclose(r.effects, beta, atol=1e-6)
        assert compare.score(r, truth)["eff_pearson"] > 0.999
