import numpy as np
import pandas as pd
import pytest
import palm_bulk as pb
from palm_bulk import sim
from palm_bulk.core import PseudobulkTable, make_benchmark


@pytest.fixture(scope="module")
def world():
    """Specimen x cell-type rows drawn from known profiles: reference donors at delta = 0, the others
    with a known per-specimen delta and a COVID-like group shift in one cell type."""
    rng = np.random.default_rng(5)
    C, P, Gn, n_donors = 4, 3, 500, 40
    B, G = sim.make_reference(rng, C=C, P=P, Gn=Gn, genes_per_prog=40)
    types = ["B cell", "Monocyte", "NK cell", "T cell"]
    genes = ["g%03d" % g for g in range(Gn)]
    rows, X, true = [], [], {}
    for d in range(n_donors):
        disease = "COVID-19" if d % 2 else "HC"
        delta = rng.normal(0, 0.3, (C, P))
        delta[1, 0] += 1.0 if disease == "COVID-19" else 0.0         # monocytes respond to "disease"
        true["S%02d" % d] = delta
        E = np.asarray(pb.cell_profiles(B, G, delta))
        for c, t in enumerate(types):
            n_cells = int(rng.integers(5, 200))
            X.append(rng.multinomial(n_cells * 20000, E[c] / E[c].sum()))
            rows.append(dict(specimen="S%02d" % d, donor_id="D%02d" % d, cell_type=t, n_cells=n_cells,
                             disease=disease, project="P1"))
            if t == "Monocyte":                                       # an extra row to merge as "DC"
                X.append(rng.multinomial(5 * 20000, E[c] / E[c].sum()))
                rows.append(dict(rows[-1], cell_type="DC", n_cells=5))
    obs = pd.DataFrame(rows)
    obs.index = obs["specimen"] + "::" + obs["cell_type"]
    table = PseudobulkTable(np.array(X, float), genes, obs)
    return table, B, G, types, genes, true


def test_relabel_merges_rows(world):
    table = world[0]
    t2 = table.relabel({"DC": "Monocyte"})
    assert "DC" not in set(t2.obs.cell_type)
    one = t2.obs.loc["S00::Monocyte"]
    src = table.obs[(table.obs.specimen == "S00") & table.obs.cell_type.isin(["Monocyte", "DC"])]
    assert one.n_cells == src.n_cells.sum()
    rows = [table.obs.index.get_loc(i) for i in src.index]
    np.testing.assert_allclose(t2.counts[t2.obs.index.get_loc("S00::Monocyte")].toarray().ravel(),
                               np.asarray(table.counts)[rows].sum(0))


def test_benchmark_split_and_truth(world):
    table, B, G, types, genes, true = world
    table = table.relabel(keep_types=types)
    bench = make_benchmark(table, min_cells=30, test_frac=0.5, contrast=("disease", "COVID-19", "HC"), seed=1)
    test_donors = set(bench.meta.donor_id)
    assert test_donors.isdisjoint(set(bench.reference_rows["donors"]))            # cross-fitting
    assert bench.bulk.shape[0] == len(genes) and set(bench.treatment) == {0.0, 1.0}
    np.testing.assert_allclose(bench.bulk.values.T, bench.type_counts.sum(1))       # bulk = sum of its rows
    assert (bench.cell_counts >= 30).all()
    prog = pd.DataFrame(G, index=["p0", "p1", "p2"], columns=genes)
    inp = pb.prepare_inputs(bench.bulk, bench.reference, prog, treatment=bench.treatment, max_genes=400)
    tr = bench.truth(inp)
    assert np.allclose(tr["pi"].sum(1), 1) and np.allclose(tr["cell_fractions"].sum(1), 1)
    # delta truth recovers the generating shifts relative to the reference (which averages donors at
    # their own deltas, so compare centred values)
    ci = [types.index(c) for c in inp.cell_types]
    pj = [int(p[1:]) for p in inp.programs]
    gen = np.stack([true[s.split("|")[0]] for s in bench.meta.specimen])[:, ci][:, :, pj]
    est = tr["delta"]
    r = np.corrcoef((est - est.mean(0)).ravel(), (gen - gen.mean(0)).ravel())[0, 1]
    assert r > 0.95
    mono = inp.cell_types.index("Monocyte")
    assert tr["beta"][mono, inp.programs.index("p0")] > 0.7                        # the disease response
    others = np.delete(tr["beta"], mono, axis=0)
    assert np.abs(others).max() < 0.4


def test_benchmark_truth_split(world):
    table, B, G, types, genes, true = world
    table = table.relabel(keep_types=types)
    kw = dict(min_cells=30, test_frac=0.5, contrast=("disease", "COVID-19", "HC"), seed=1)
    full = make_benchmark(table, **kw)
    split = make_benchmark(table, truth_split=True, **kw)
    assert list(split.meta.index) == list(full.meta.index)                          # same donor split
    np.testing.assert_allclose(split.type_counts + split.truth_counts, full.type_counts)   # halves add up
    np.testing.assert_allclose(split.bulk.values.T, split.type_counts.sum(1))       # bulk = its own half
    assert abs(split.bulk.values.sum() / full.bulk.values.sum() - 0.5) < 0.01
    prog = pd.DataFrame(G, index=["p0", "p1", "p2"], columns=genes)
    inp = pb.prepare_inputs(split.bulk, split.reference, prog, treatment=split.treatment, max_genes=400)
    tr, tf = split.truth(inp), full.truth(inp)
    assert np.corrcoef(tr["delta"].ravel(), tf["delta"].ravel())[0, 1] > 0.95      # same target, other half
    assert not np.allclose(tr["delta"], tf["delta"])
