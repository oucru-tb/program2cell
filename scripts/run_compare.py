"""Head-to-head comparison of PALM with competing methods on a simulated study with exact truth.

Runs every requested method through palm_bulk.compare, scores fractions, per-cell-type program
activity and treatment effects, and saves the scores with provenance. Methods whose software is
missing (R packages, Docker, BLADE environment) are skipped and listed. The same pipeline will
run on real PBMC pseudo-bulk once its truth tables exist.

Usage: python scripts/run_compare.py [--reps 1] [--N 24] [--cell-types 5] [--programs 4] [--genes 1500]
                                     [--max-genes 600] [--methods palm,naive_two_step,...]
                                     [--warmup 200] [--samples 200] [--out benchmarks/compare/compare_sim.json]
                                     [--method-kw JSON]
"""
import argparse
import dataclasses
import importlib.util
import json
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--reps", type=int, default=1)
p.add_argument("--N", type=int, default=24)
p.add_argument("--cell-types", type=int, default=5)
p.add_argument("--programs", type=int, default=4)
p.add_argument("--genes", type=int, default=1500)
p.add_argument("--max-genes", type=int, default=600)
p.add_argument("--methods", default="palm,naive_two_step,reallocation,instaprism,cibersortx,blade,toast,tca")
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--samples", type=int, default=200)
p.add_argument("--chains", type=int, default=2)
p.add_argument("--precision", default="float32")
p.add_argument("--device", default="cpu", choices=["cpu", "gpu", "auto"], help="JAX device for PALM")
p.add_argument("--out", default="benchmarks/compare/compare_sim.json")
p.add_argument("--workdir", default="benchmarks/compare/work")
p.add_argument("--method-kw", default="{}",
               help='JSON of extra arguments per method, e.g. \'{"blade": {"grid": "small", "n_rep": 1}}\'')
g = p.add_argument_group("real pseudo-bulk instead of simulation (each rep = a different donor split)")
g.add_argument("--pseudobulk", help="specimen x cell-type h5ad (scripts/build_pbmc_pseudobulk.py output)")
g.add_argument("--keep-types", default="B cell,Monocyte,NK cell,T cell")
g.add_argument("--label-map", default="{}", help='JSON, e.g. \'{"DC": "Monocyte"}\'')
g.add_argument("--min-cells", type=int, default=20)
g.add_argument("--truth-split", action="store_true",
               help="bulk and truth from independent halves of the counts (binomial thinning)")
g.add_argument("--n-test", type=int, default=None)
g.add_argument("--contrast", default="disease,COVID-19,HC", help="column,treated,control or 'none'")
g.add_argument("--programs-csv", default="data/progeny_human_top500.csv")
g.add_argument("--program-list", default=None, help="comma-separated subset of programs")
g.add_argument("--cells", default=None,
               help="cell-level h5ad (build_pseudobulk_from_cells.py *_cells.h5ad): reference cells for the "
                    "competitors and HVG scores for PALM's gene selection; test donors' cells are never used")
args = p.parse_args()
method_kw = json.loads(args.method_kw)

import numpy as np
import pandas as pd
import palm_bulk as pb
print("compute:", pb.configure_compute(args.device, parallel_chains=args.chains if args.device == "cpu" else 1,
                                         precision=args.precision), flush=True)
from palm_bulk import compare, sim

methods = args.methods.split(",")
HAS_DECOUPLER = importlib.util.find_spec("decoupler") is not None
if not HAS_DECOUPLER:
    print("decoupler not installed: skipping the '+ulm' variants (pip install 'palm-bulk[progeny]')")


def simulated(r):
    rng = np.random.default_rng(3000 + r)
    st = sim.simulate_study(rng, C=args.cell_types, P=args.programs, Gn=args.genes, N=args.N)
    ref = pb.build_reference(st["sc_counts"], st["genes"], st["sc_cell_types"], st["sc_donors"])
    inp = pb.prepare_inputs(st["bulk"], ref, st["program_table"], treatment=st["treatment"],
                            max_genes=args.max_genes)
    data = compare.CompareData(bulk=st["bulk"], sc_counts=st["sc_counts"], sc_genes=st["genes"],
                               sc_cell_types=st["sc_cell_types"], sc_donors=st["sc_donors"], inputs=inp,
                               programs=st["program_table"], treatment=st["treatment"],
                               workdir=os.path.join(args.workdir, "rep%d" % r))
    return data, sim.truth_on_genes(st, inp.genes, inp.cell_types, inp.programs)


TABLE, CELLS = None, None


def reference_cells(path, test_donors, cell_types, genes):
    """Cells of reference donors only (test donors excluded: no leakage), kept cell types, genes aligned."""
    import anndata as ad
    global CELLS
    if CELLS is None:
        CELLS = ad.read_h5ad(path)
        CELLS.obs["cell_type"] = CELLS.obs["cell_type"].astype(str).replace(json.loads(args.label_map))
    c = CELLS[(~CELLS.obs["donor_id"].astype(str).isin(test_donors)).values
              & CELLS.obs["cell_type"].isin(cell_types).values]
    X = c.X if list(c.var_names) == list(genes) else c[:, c.var_names.isin(genes)].to_df().reindex(
        columns=genes, fill_value=0).values
    import scipy.sparse as sps
    return dict(counts=sps.csr_matrix(X), cell_types=c.obs["cell_type"].astype(str).values,
                donors=c.obs["donor_id"].astype(str).values, kind="reference cells")


def pseudobulk(r):
    """Real data: reference rows of the reference donors play the role of single cells for the
    competitors (one profile per donor and cell type)."""
    from palm_bulk.core import PseudobulkTable, make_benchmark
    global TABLE
    if TABLE is None:
        TABLE = PseudobulkTable.from_h5ad(args.pseudobulk, json.loads(args.label_map), args.keep_types.split(","))
    contrast = None if args.contrast == "none" else tuple(args.contrast.split(","))
    b = make_benchmark(TABLE, min_cells=args.min_cells, n_test=args.n_test, contrast=contrast, seed=r,
                       truth_split=args.truth_split)
    prog = pb.load_programs(args.programs_csv)
    if args.program_list:
        prog = prog.loc[args.program_list.split(",")]
    rr = b.reference_rows
    sc = dict(counts=rr["counts"], cell_types=rr["cell_types"], donors=rr["donors"], kind="donor pseudo-bulk rows")
    if args.cells:
        sc = reference_cells(args.cells, set(b.meta["donor_id"].astype(str)), b.cell_types, b.reference.genes)
        b.reference.hvg_score = pb.data.hvg_scores(sc["counts"])
    inp = pb.prepare_inputs(b.bulk, b.reference, prog, treatment=b.treatment, max_genes=args.max_genes)
    data = compare.CompareData(bulk=b.bulk, sc_counts=sc["counts"], sc_genes=b.reference.genes,
                               sc_cell_types=sc["cell_types"], sc_donors=sc["donors"], inputs=inp, programs=prog,
                               treatment=b.treatment, workdir=os.path.join(args.workdir, "rep%d" % r))
    print("rep %d: %d test specimens (%s treated), %d genes, %d programs; competitor reference: %d %s"
          % (r, len(b.meta), "none" if b.treatment is None else int(b.treatment.sum()), len(inp.genes),
             len(inp.programs), sc["counts"].shape[0], sc["kind"]), flush=True)
    return data, b.truth(inp)


rows, skipped = [], {}
t_all = time.time()
for r in range(args.reps):
    data, truth = pseudobulk(r) if args.pseudobulk else simulated(r)
    inp = data.inputs
    for m in methods:
        kw = dict(num_warmup=args.warmup, num_samples=args.samples, chains=args.chains) if m in ("palm", "palm_fast") else {}
        kw.update(method_kw.get(m, {}))
        print("running %s %s" % (m, kw or ""), flush=True)
        res, err = compare.run_method(m, data, **kw)
        if res is None:
            skipped[m] = err
            print("  skipped:", err, flush=True)
            continue
        results = [(m, res)]
        if res.type_expression is not None and HAS_DECOUPLER:   # same deconvolution, ULM instead of delta
            act, scale = compare.activity_from_expression(res.type_expression, inp, "ulm", data.programs)
            results.append((m + "+ulm", dataclasses.replace(
                res, activity=act, activity_scale=scale, effects_scale=scale,
                effects=None if data.treatment is None else compare.effects_from_activity(act, data.treatment))))
        for name, rr in results:
            row = dict(rep=r, **compare.score(rr, truth))
            row["method"] = name
            row["notes"] = rr.notes
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "notes"},
                  flush=True)
    # progress so far, so a long run that dies later keeps its finished reps
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(provenance=pb.provenance(**vars(args)), skipped=skipped, rows=rows, finished_reps=r + 1),
              open(os.path.splitext(args.out)[0] + ".partial.json", "w"), indent=1, default=float)

df = pd.DataFrame(rows)
cols = [c for c in ["frac_rmse", "frac_pearson", "frac_cov90", "act_spearman_across_samples",
                    "act_spearman_across_types", "act_mae", "act_cov90", "eff_pearson", "eff_auroc", "eff_mae",
                    "eff_cov90", "runtime"] if c in df]
summary = df.groupby("method", sort=False)[cols].mean()
print("\n" + summary.round(3).to_string())
if skipped:
    print("\nskipped:", json.dumps(skipped, indent=1))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(dict(provenance=pb.provenance(**vars(args)), skipped=skipped,
               summary=summary.reset_index().to_dict("records"), rows=rows),
          open(args.out, "w"), indent=1, default=float)
print("wrote %s (%.0fs)" % (args.out, time.time() - t_all))
