"""Kang et al. 2018 (IFN-beta stimulated vs control PBMC, 8 lupus donors; via pertpy) -> pseudo-bulk table
in the format of PseudobulkTable.from_h5ad (one row per specimen x cell type).

Cell types are mapped to the broad set used by dev_bench (B cell, Monocyte, NK cell, T cell); dendritic
cells and megakaryocytes are dropped. Each donor x condition is split at random into
max(1, n_cells // CELLS_PER_REP) pseudo-replicate specimens, so that a split with a few test donors still
has enough bulk samples. Pseudo-replicates of one donor are not independent: benchmarks keep a donor's
specimens on one side of the split (make_benchmark splits by donor).
Usage: python build_kang.py [CELLS_PER_REP]"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
CELLS_PER_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 600
MAP = {"CD4 T cells": "T cell", "CD8 T cells": "T cell", "CD14+ Monocytes": "Monocyte",
       "FCGR3A+ Monocytes": "Monocyte", "B cells": "B cell", "NK cells": "NK cell"}

src = os.path.join(ROOT, "data", "kang", "kang_2018_cells.h5ad")
if not os.path.exists(src):
    import pertpy as pt
    pt.dt.kang_2018().write_h5ad(src)
a = ad.read_h5ad(src)
obs = a.obs
X = sp.csr_matrix(a.X)
genes = list(a.var["name"]) if "name" in a.var else list(a.var_names)
keep = obs["cell_type"].astype(str).isin(MAP).values
obs, X = obs[keep].copy(), X[np.flatnonzero(keep)]
obs["ct"] = obs["cell_type"].astype(str).map(MAP)
rng = np.random.default_rng(0)
rep = np.zeros(len(obs), int)
for _, idx in obs.groupby(["replicate", "label"], observed=True).indices.items():
    r = max(1, len(idx) // CELLS_PER_REP)
    rep[idx] = rng.permutation(np.arange(len(idx)) % r)
obs["specimen"] = obs["replicate"].astype(str) + "|" + obs["label"].astype(str) + "|r" + rep.astype(str)
key = obs["specimen"] + "::" + obs["ct"]
codes, uniq = pd.factorize(key)
M = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
P = (M @ X).tocsr()
g = obs.groupby(codes, sort=True)
out = pd.DataFrame({"donor_id": g["replicate"].first().astype(str).values,
                    "label": g["label"].first().astype(str).values,
                    "specimen": g["specimen"].first().values,
                    "cell_type": g["ct"].first().values,
                    "n_cells": g.size().values}, index=uniq)
out["project"] = "kang2018"
out["n_counts"] = np.asarray(P.sum(1)).ravel()
res = ad.AnnData(P.astype(np.float32), obs=out, var=pd.DataFrame(index=pd.Index(genes)))
res.var_names_make_unique()
dst = os.path.join(ROOT, "data", "kang", "kang_pseudobulk.h5ad")
res.write_h5ad(dst)
wide = out.pivot_table(index="specimen", columns="cell_type", values="n_cells", aggfunc="sum", fill_value=0)
print(wide.to_string())
print("%d specimens, %d rows -> %s" % (len(wide), len(out), dst))
