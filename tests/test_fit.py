import pytest
import numpy as np
import pandas as pd
import palm_bulk as pb
from palm_bulk import sim

pytestmark = pytest.mark.slow       # every test here runs NUTS


def test_end_to_end_small():
    rng = np.random.default_rng(2)
    B, G = sim.make_reference(rng, C=3, P=2, Gn=120, genes_per_prog=15)
    s = sim.simulate_causal(rng, B, G, N=12)
    res = pb.fit(s["Y"], s["L"], s["T"], B, G, theta=s["theta"], num_warmup=150, num_samples=150, seed=0)
    pi_hat = res["pi"].mean(0)
    assert np.sqrt(((pi_hat - s["pi"]) ** 2).mean()) < 0.06
    I_hat = np.median(res["dec"]["I"].sum(1), 0)
    assert np.corrcoef(I_hat, s["truth"]["I"].sum(0))[0, 1] > 0.5
    assert res["divergences"] <= 5
    assert {"pi", "kappa", "beta", "delta", "I", "K"} <= set(res["convergence"])   # diagnostics for one chain too
    assert np.isfinite(res["rhat_max"]) and res["ess_min"] > 0


def test_run_palm_tables():
    rng = np.random.default_rng(3)
    B, G = sim.make_reference(rng, C=3, P=2, Gn=100, genes_per_prog=12)
    s = sim.simulate_causal(rng, B, G, N=8)
    genes = ["g%d" % i for i in range(100)]
    ref = pb.Reference(B=B, genes=genes, cell_types=["a", "b", "c"], rna_content=np.array([1.0, 2.0, 3.0]),
                       n_cells=np.array([100, 100, 100]))
    prog = pd.DataFrame(G, index=["P1", "P2"], columns=genes)
    bulk = pd.DataFrame(s["Y"].T, index=genes, columns=["s%d" % i for i in range(8)])
    inp = pb.prepare_inputs(bulk, ref, prog, treatment=s["T"], max_genes=100, min_program_genes=5)
    res = pb.run_palm(inp, use_m=False, theta=40.0, num_warmup=100, num_samples=100)
    assert res.fractions()["median"].shape == (8, 3)
    assert abs(res.cell_fractions()["median"].sum(axis=1).mean() - 1) < 1e-3
    assert {"sample", "program", "cell_type", "share"} <= set(res.attribution().columns)
    assert len(res.effects()) == 2 * 3 * 2
    assert set(res.effect_totals().columns) >= {"total", "intrinsic", "composition"}
    d = res.diagnostics()
    assert d["chains"] == 2 and res.draws["pi"].shape[0] == 200          # run_palm default: 2 chains
    assert isinstance(d["problems"], list) and "rhat_max" in res.convergence().columns


def test_attribution_batching_matches_single_batch():
    rng = np.random.default_rng(4)
    B, G = sim.make_reference(rng, C=3, P=2, Gn=80, genes_per_prog=10)
    s = sim.simulate_causal(rng, B, G, N=6)
    genes = ["g%d" % i for i in range(80)]
    ref = pb.Reference(B=B, genes=genes, cell_types=["a", "b", "c"], rna_content=np.ones(3),
                       n_cells=np.array([100, 100, 100]))
    prog = pd.DataFrame(G, index=["P1", "P2"], columns=genes)
    bulk = pd.DataFrame(s["Y"].T, index=genes, columns=["s%d" % i for i in range(6)])
    inp = pb.prepare_inputs(bulk, ref, prog, max_genes=80, min_program_genes=5)
    res = pb.run_palm(inp, use_m=False, theta=40.0, num_warmup=50, num_samples=50)
    one = res._attribution_draws()
    res._attr = None
    many = res._attribution_draws(max_elements=6 * 3 * 80 * 7)   # batches of 7 draws
    for a, b in zip(one, many):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)


PARALLEL_SCRIPT = """
import numpy as np, palm_bulk as pb
from palm_bulk import sim
info = pb.configure_compute("cpu", parallel_chains=2)
assert info["n_devices"] == 2, info
rng = np.random.default_rng(5)
B, G = sim.make_reference(rng, C=3, P=2, Gn=80, genes_per_prog=10)
s = sim.simulate_causal(rng, B, G, N=6)
kw = dict(theta=40.0, num_warmup=60, num_samples=60, chains=2, seed=1)
par = pb.fit(s["Y"], s["L"], s["T"], B, G, **kw)
seq = pb.fit(s["Y"], s["L"], s["T"], B, G, chain_method="sequential", **kw)
assert par["chain_method"] == "parallel" and seq["chain_method"] == "sequential"
# Same posterior; individual draws differ because float rounding differs between the two
# compiled programs and MCMC trajectories amplify it.
np.testing.assert_allclose(par["pi"].mean(0), seq["pi"].mean(0), atol=0.03)
print("ok")
"""


def test_parallel_chains_match_sequential():
    # Device count must be set before JAX starts, so this runs in a fresh interpreter.
    import subprocess, sys
    out = subprocess.run([sys.executable, "-c", PARALLEL_SCRIPT], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]


def test_gene_dispersion_and_covariates_fit():
    rng = np.random.default_rng(6)
    B, G = sim.make_reference(rng, C=3, P=2, Gn=120, genes_per_prog=15)
    s = sim.simulate_causal(rng, B, G, N=16, theta_sd=0.5, confounding=0.8)
    res = pb.fit(s["Y"], s["L"], s["T"], B, G, num_warmup=100, num_samples=100,
                 dispersion="gene", X=s["X"])
    assert res["pi"].shape == (100, 16, 3)
    assert np.sqrt(((res["pi"].mean(0) - s["pi"]) ** 2).mean()) < 0.08


def _lineage_world(seed=8, C=6, N=10):
    rng = np.random.default_rng(seed)
    B, G = sim.make_reference(rng, C=C, P=2, Gn=150, genes_per_prog=15)
    counts, ct, dn = sim.simulate_reference_cells(rng, B, n_per_type=120, n_donors=3, depth=3000)
    genes = ["g%d" % i for i in range(150)]
    fine = ["sub%d" % c for c in range(C)]
    lin_of = {"sub0": "T", "sub1": "T", "sub2": "T", "sub3": "Myeloid", "sub4": "Myeloid", "sub5": "B"}
    ref = pb.build_reference(counts, genes, [fine[c] for c in ct], dn, lineages=[lin_of[fine[c]] for c in ct])
    s = sim.simulate_causal(rng, B, G, N=N, balanced=True)
    bulk = pd.DataFrame(s["Y"].T, index=genes, columns=["s%d" % i for i in range(N)])
    prog = pd.DataFrame(G, index=["P1", "P2"], columns=genes)
    return ref, bulk, prog, s, fine


def test_composition_only_mode():
    ref, bulk, prog, s, fine = _lineage_world()
    assert ref.lineage == ["T", "T", "T", "Myeloid", "Myeloid", "B"]
    inp = pb.prepare_inputs(bulk, ref, None, max_genes=150)
    assert inp.Gmat.shape[0] == 0
    res = pb.run_palm(inp, num_warmup=100, num_samples=100, svi_steps=500)
    assert not res.has_programs and "delta" not in res.draws
    pi_true = s["pi"][:, [fine.index(c) for c in inp.cell_types]]
    assert np.sqrt(((res.fractions()["median"].values - pi_true) ** 2).mean()) < 0.1
    lf = res.lineage_fractions()["median"]
    assert list(lf.columns) == ["B", "Myeloid", "T"] and np.allclose(lf.sum(axis=1), 1, atol=0.03)   # medians: ~1
    with pytest.raises(ValueError):
        res.attribution()


def test_lineage_shared_programs():
    ref, bulk, prog, s, fine = _lineage_world()
    inp = pb.prepare_inputs(bulk, ref, prog, treatment=s["T"], max_genes=150, program_groups="lineage")
    assert inp.group_names == ["B", "Myeloid", "T"] and len(inp.groups) == 6
    res = pb.run_palm(inp, num_warmup=100, num_samples=100, svi_steps=500)
    assert res.draws["delta"].shape[2] == 3 and res.draws["dec"]["I"].shape[1:] == (6, 2)
    assert set(res.cell_type_activity().cell_type) == {"B", "Myeloid", "T"}
    assert res.attribution().cell_type.nunique() == 6          # attribution stays per fine cell type
    assert len(res.effects()) == 2 * 6 * 2

