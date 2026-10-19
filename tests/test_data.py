import numpy as np
import pandas as pd
import pytest
import palm_bulk as pb
from palm_bulk import sim


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(1)
    B, G = sim.make_reference(rng, C=3, P=2, Gn=150, genes_per_prog=15)
    counts, types, donors = sim.simulate_reference_cells(rng, B, n_per_type=160, n_donors=4, depth=4000)
    genes = ["g%d" % i for i in range(150)]
    ref = pb.build_reference(counts, genes, ["ct%d" % t for t in types], donors)
    prog = pd.DataFrame(G, index=["P1", "P2"], columns=genes)
    return rng, B, G, ref, prog, genes


def test_build_reference_close_to_truth(world):
    _, B, _, ref, _, _ = world
    assert ref.B.shape == B.shape and np.allclose(ref.B.sum(1), 1)
    r = np.corrcoef(np.log(ref.B.ravel()), np.log(B.ravel()))[0, 1]
    assert r > 0.95
    assert (ref.n_donors == 4).all()


def test_load_programs_long_and_scale():
    long = pd.DataFrame(dict(program=["a"] * 6, gene=list("abcdef"), weight=[2, 4, 6, -2, -4, 8.0]))
    P = pb.load_programs(long)
    assert P.shape == (1, 6)
    nz = np.abs(P.values[P.values != 0])
    assert np.isclose(np.median(nz), 1.0)


def test_prepare_inputs_and_diagnostics(world):
    rng, B, G, ref, prog, genes = world
    s = sim.simulate_causal(rng, B, G, N=6)
    bulk = pd.DataFrame(s["Y"].T, index=genes, columns=["s%d" % i for i in range(6)])
    inp = pb.prepare_inputs(bulk, ref, prog, treatment=s["T"], max_genes=100, min_program_genes=5)
    assert inp.Y.shape[0] == 6 and inp.B.shape[1] == inp.Gmat.shape[1] == len(inp.genes)
    assert np.allclose(inp.B.sum(1), 1)
    assert set(inp.genes) >= set(prog.columns[(prog.abs() > 0).any()])      # program genes kept
    d = pb.diagnose_inputs(inp)
    # baseline-only NNLS fit degrades for strongly induced samples, but stays well correlated
    assert d.log_corr.min() > 0.9 and d.r2.median() > 0.8


def test_bad_treatment_coding(world):
    rng, B, G, ref, prog, genes = world
    s = sim.simulate_causal(rng, B, G, N=4)
    bulk = pd.DataFrame(s["Y"].T, index=genes, columns=list("wxyz"))
    with pytest.raises(ValueError):
        pb.prepare_inputs(bulk, ref, prog, treatment=[0, 1, 2, 1])


def test_select_genes_union_when_programs_exceed_budget():
    # PROGENy-like: program genes alone exceed max_genes; markers, hvg and anchors must survive
    rng = np.random.default_rng(9)
    C, Gn, P = 5, 3000, 4
    B = rng.gamma(0.6, 1.0, (C, Gn)) * np.exp(rng.normal(0, 1.5, (C, Gn)))
    B /= B.sum(1, keepdims=True)
    genes = ["g%d" % i for i in range(Gn)]
    ref = pb.Reference(B=B, genes=genes, cell_types=["t%d" % c for c in range(C)], rna_content=np.ones(C),
                       n_cells=np.full(C, 100), hvg_score=rng.normal(size=Gn))
    W = np.zeros((P, Gn))
    for p in range(P):
        W[p, rng.choice(Gn, 600, replace=False)] = rng.normal(0, 1, 600)
    prog = pd.DataFrame(W, index=["P%d" % p for p in range(P)], columns=genes)
    bulk = pd.DataFrame(rng.poisson(50, (Gn, 6)), index=genes)
    sel, info = pb.select_genes(ref, bulk, prog, max_genes=400, return_info=True)
    assert len(sel) == len(set(sel)) == 400
    assert info["sizes"] == {"programs": 180, "markers": 120, "hvg": 40, "anchors": 60}
    per_prog = (prog[sel].abs() > 0).sum(axis=1)
    assert per_prog.min() >= 40                                   # every program represented
    assert min(info["markers_per_type"].values()) >= 15           # every cell type has markers
    for p in prog.index:                                          # strongest weights first
        top = prog.loc[p].abs().sort_values(ascending=False).index[:20]
        assert set(top) <= set(sel)
    assert not (prog[sel[-60:]].abs() > 0).any().any()           # anchors carry no program weight


def test_select_genes_filters_unexpressed_and_undetected(world):
    _, B, G, ref, prog, genes = world
    bulk = pd.DataFrame(np.ones((len(genes), 5)), index=genes)
    bulk.iloc[0] = 0                                              # never detected in bulk
    sel = pb.select_genes(ref, bulk, prog, max_genes=150)
    assert genes[0] not in sel
    assert all(ref.B[:, genes.index(g)].max() * 1e6 >= 2 for g in sel)


def test_hvg_scores_rank_variable_gene_first():
    rng = np.random.default_rng(0)
    X = rng.poisson(5, (500, 50)).astype(float)
    X[:250, 7] = rng.poisson(40, 250)                            # bimodal across cells
    assert np.argmax(pb.data.hvg_scores(X)) == 7


def test_prepare_inputs_covariates():
    rng = np.random.default_rng(3)
    B, G = sim.make_reference(rng, C=3, P=2, Gn=100, genes_per_prog=12)
    s = sim.simulate_causal(rng, B, G, N=8)
    genes = ["g%d" % i for i in range(100)]
    ref = pb.Reference(B=B, genes=genes, cell_types=["a", "b", "c"], rna_content=np.ones(3),
                       n_cells=np.array([100, 100, 100]))
    prog = pd.DataFrame(G, index=["P1", "P2"], columns=genes)
    samples = ["s%d" % i for i in range(8)]
    bulk = pd.DataFrame(s["Y"].T, index=genes, columns=samples)
    cov = pd.DataFrame({"batch": list("xxyyzzxy"), "age": np.arange(8.0)}, index=samples[::-1])
    inp = pb.prepare_inputs(bulk, ref, prog, max_genes=100, covariates=cov)
    assert inp.X.shape == (8, 3) and set(inp.covariates) == {"age", "batch_y", "batch_z"}
    np.testing.assert_allclose(inp.X.mean(0), 0, atol=1e-12)
    assert inp.X[0, inp.covariates.index("age")] > 0          # s0 is the oldest: aligned by name, not position


def test_many_cell_types_warns():
    import warnings
    rng = np.random.default_rng(0)
    C, Gn = 21, 200
    B = rng.dirichlet(np.ones(Gn), C)
    genes = ["g%d" % i for i in range(Gn)]
    ref = pb.Reference(B=B, genes=genes, cell_types=["t%d" % c for c in range(C)], rna_content=np.ones(C),
                       n_cells=np.full(C, 50))
    prog = pd.DataFrame(rng.normal(size=(1, Gn)), index=["P"], columns=genes)
    bulk = pd.DataFrame(rng.poisson(20, (Gn, 4)), index=genes)
    with pytest.warns(UserWarning, match="two stages"):
        pb.prepare_inputs(bulk, ref, prog, max_genes=200)


def test_covariates_with_missing_values_rejected():
    cov = pd.DataFrame({"age": [30.0, np.nan, 50.0]}, index=list("abc"))
    with pytest.raises(ValueError, match="missing"):
        pb.data.encode_covariates(cov, list("abc"))
