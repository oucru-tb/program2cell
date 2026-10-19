"""Head-to-head comparison of PALM with competing methods on a simulated study with exact truth.

Runs every requested method through palm_bulk.compare, scores fractions, per-cell-type program
activity and treatment effects, and saves the scores with provenance. Methods whose software is
missing (R packages, Docker, BLADE environment) are skipped and listed. The same pipeline will
run on real PBMC pseudo-bulk once its truth tables exist.

Usage: python scripts/run_compare.py [--reps 1] [--N 24] [--cell-types 5] [--programs 4] [--genes 1500]
                                     [--max-genes 600] [--methods palm,naive_two_step,...]
                                     [--warmup 200] [--samples 200] [--out benchmarks/compare/compare_sim.json]
"""
import argparse
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
p.add_argument("--methods", default="palm,naive_two_step,reallocation,bayesprism,cibersortx,blade,toast,tca")
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--samples", type=int, default=200)
p.add_argument("--chains", type=int, default=2)
p.add_argument("--precision", default="float32")
p.add_argument("--out", default="benchmarks/compare/compare_sim.json")
p.add_argument("--workdir", default="benchmarks/compare/work")
args = p.parse_args()

import numpy as np
import pandas as pd
import palm_bulk as pb
pb.configure_compute("cpu", parallel_chains=args.chains, precision=args.precision)
from palm_bulk import compare, sim

methods = args.methods.split(",")
rows, skipped = [], {}
t_all = time.time()
for r in range(args.reps):
    rng = np.random.default_rng(3000 + r)
    st = sim.simulate_study(rng, C=args.cell_types, P=args.programs, Gn=args.genes, N=args.N)
    ref = pb.build_reference(st["sc_counts"], st["genes"], st["sc_cell_types"], st["sc_donors"])
    inp = pb.prepare_inputs(st["bulk"], ref, st["program_table"], treatment=st["treatment"],
                            max_genes=args.max_genes)
    data = compare.CompareData(bulk=st["bulk"], sc_counts=st["sc_counts"], sc_genes=st["genes"],
                               sc_cell_types=st["sc_cell_types"], sc_donors=st["sc_donors"], inputs=inp,
                               programs=st["program_table"], treatment=st["treatment"],
                               workdir=os.path.join(args.workdir, "rep%d" % r))
    truth = sim.truth_on_genes(st, inp.genes, inp.cell_types, inp.programs)
    for m in methods:
        kw = dict(num_warmup=args.warmup, num_samples=args.samples, chains=args.chains) if m == "palm" else {}
        variants = [("", kw)] + ([("+ulm", dict(activity="ulm"))] if m in ("reallocation", "bayesprism",
                                                                         "cibersortx", "blade") else [])
        for suffix, vkw in variants:
            res, err = compare.run_method(m, data, **vkw)
            if res is None:
                skipped[m] = err
                break
            row = dict(rep=r, **compare.score(res, truth))
            row["method"] = m + suffix
            if m == "palm":
                row["notes"] = res.notes
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "notes"},
                  flush=True)

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
