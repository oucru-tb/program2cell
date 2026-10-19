"""How many modelled genes are enough? Accuracy, calibration and cost versus MAX_GENES.

Simulates a 3000-gene transcriptome with PROGENy-sized programs, a single-cell reference
(built through the real data path, including HVG scores) and bulk samples, then runs the real
gene selection (prepare_inputs) and fit at several max_genes. Scored on quantities that keep
their meaning for any gene subset:
  * beta, kappa: per-cell-type program activity (treatment shift, baseline shift);
  * read fractions recomputed on the selected genes.
Usage: python run_gene_budget.py [reps] [out.json] [max_genes ...]
"""
import sys, time, json
import numpy as np
import pandas as pd
import palm_bulk as pb
from palm_bulk import sim
from palm_bulk.core import cell_profiles

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = sys.argv[2] if len(sys.argv) > 2 else "gene_budget.json"
SIZES = [int(x) for x in sys.argv[3:]] or [300, 600, 1000, 1500]
N, C, P, GN, PER_PROG = 40, 5, 4, 3000, 150


def cover(draws, truth, level=0.9):
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return float(((truth >= lo) & (truth <= hi)).mean())


PROV = pb.provenance(reps=REPS, sizes=SIZES, N=N, C=C, P=P, genes=GN, genes_per_program=PER_PROG)
rows = []
for r in range(REPS):
    rng = np.random.default_rng(2000 + r)
    B, G = sim.make_reference(rng, C=C, P=P, Gn=GN, genes_per_prog=PER_PROG)
    counts, ct, dn = sim.simulate_reference_cells(rng, B, n_per_type=200, n_donors=4, depth=4000)
    s = sim.simulate_causal(rng, B, G, N=N, depth=2e6, platform_sd=0.2)
    genes = ["g%d" % i for i in range(GN)]
    types = ["t%d" % c for c in range(C)]
    ref = pb.build_reference(counts, genes, [types[c] for c in ct], dn)
    progs = pd.DataFrame(G, index=["p%d" % p for p in range(P)], columns=genes)
    samples = ["s%d" % i for i in range(N)]
    bulk = pd.DataFrame(s["Y"].T, index=genes, columns=samples)
    E_true = np.asarray(cell_profiles(B, G, s["delta"]))                           # (N,C,G)
    for m in SIZES:
        inp = pb.prepare_inputs(bulk, ref, progs, treatment=s["T"], max_genes=m)
        gi = [genes.index(g) for g in inp.genes]
        w = s["pi"][:, :, None] * E_true[:, :, gi]
        pi_sub = w.sum(-1) / w.sum((-1, -2))[:, None]                               # true fractions on subset
        ci = [types.index(c) for c in inp.cell_types]
        pj = [int(p[1:]) for p in inp.programs]
        t = time.time()
        res = pb.run_palm(inp, use_m=True, num_warmup=300, num_samples=300, chains=1, check=False)   # as in the runs so far
        el = time.time() - t
        d = res.draws
        row = dict(rep=r, max_genes=m, n_genes=len(inp.genes), time=el, mean_steps=d["mean_steps"],
                   divergences=d["divergences"], gene_sets=inp.info["gene_sets"]["sizes"],
                   pi_rmse=float(np.sqrt(((d["pi"].mean(0) - pi_sub[:, ci]) ** 2).mean())),
                   beta_mae=float(np.abs(np.median(d["beta"], 0) - s["beta"][np.ix_(ci, pj)]).mean()),
                   kappa_mae=float(np.abs(np.median(d["kappa"], 0) - s["kappa"][np.ix_(ci, pj)]).mean()),
                   beta_cov90=cover(d["beta"], s["beta"][np.ix_(ci, pj)]),
                   kappa_cov90=cover(d["kappa"], s["kappa"][np.ix_(ci, pj)]))
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        json.dump(dict(rows=rows, provenance=PROV), open(OUT, "w"), indent=1)

df = pd.DataFrame(rows).drop(columns=["gene_sets"]).groupby("max_genes").mean(numeric_only=True)
print(df.drop(columns=["rep"]).round(4).to_string())
