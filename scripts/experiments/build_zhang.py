"""Zhang et al. 2021 (TNBC, paclitaxel +/- atezolizumab; via pertpy) -> pseudo-bulk table of BLOOD samples, in
the format of PseudobulkTable.from_h5ad (one row per specimen x cell type).

Design for the benchmark (dev_bench DATASETS["zhang"]):
  - specimens: patient x time point (Pre-treatment, Post-treatment; Progression dropped) x pseudo-replicate;
    each patient x time point is split at random into REPS replicates so that calibration has more specimens
  - cell types mapped to the dev_bench set: T cell, B cell, Myeloid cell -> Monocyte, ILC cell -> NK cell
    (blood myeloid cells are mostly monocytes, blood ILCs mostly NK cells)
  - obs: donor_id = patient, label = pre / post, arm = Chemo / Anti-PD-L1+Chemo
The contrast post vs pre is paired within patients (estimator 'ols' with groups = patient); the arm contrast
(difference in differences) is the randomised effect of adding anti-PD-L1.
Usage: python build_zhang.py [REPS]"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MAP = {"T cell": "T cell", "B cell": "B cell", "Myeloid cell": "Monocyte", "ILC cell": "NK cell"}
LABEL = {"Pre-treatment": "pre", "Post-treatment": "post"}

a = ad.read_h5ad(os.path.join(ROOT, "data", "pert", "zhang_2021.h5ad"), backed="r")
o = a.obs
keep = ((o["Tissue"].astype(str) == "blood") & o["Group"].astype(str).isin(LABEL)
        & o["Major celltype"].astype(str).isin(MAP)).values
idx = np.flatnonzero(keep)
print("blood pre/post cells:", len(idx), flush=True)
X = sp.csr_matrix(a.X[idx]) if not sp.issparse(a.X) else a[idx].to_memory().X.tocsr()
obs = o.iloc[idx].copy()
genes = list(a.var_names)
obs["ct"] = obs["Major celltype"].astype(str).map(MAP)
obs["label"] = obs["Group"].astype(str).map(LABEL)
rng = np.random.default_rng(0)
rep = np.zeros(len(obs), int)
for _, ix in obs.groupby(["Patient", "label"], observed=True).indices.items():
    rep[ix] = rng.permutation(np.arange(len(ix)) % REPS)
obs["specimen"] = obs["Patient"].astype(str) + "|" + obs["label"] + "|r" + rep.astype(str)
key = obs["specimen"] + "::" + obs["ct"]
codes, uniq = pd.factorize(key)
M = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
P = (M @ X).tocsr()
g = obs.groupby(codes, sort=True)
out = pd.DataFrame({"donor_id": g["Patient"].first().astype(str).values,
                    "label": g["label"].first().values,
                    "arm": g["Treatment"].first().astype(str).values,
                    "efficacy": g["Efficacy"].first().astype(str).values,
                    "specimen": g["specimen"].first().values,
                    "cell_type": g["ct"].first().values,
                    "n_cells": g.size().values}, index=uniq)
out["project"] = "zhang2021"
out["n_counts"] = np.asarray(P.sum(1)).ravel()
res = ad.AnnData(P.astype(np.float32), obs=out, var=pd.DataFrame(index=pd.Index(genes)))
res.var_names_make_unique()
dst = os.path.join(ROOT, "data", "pert", "zhang_blood_pseudobulk.h5ad")
res.write_h5ad(dst)
wide = out.pivot_table(index="specimen", columns="cell_type", values="n_cells", aggfunc="sum", fill_value=0)
print(wide.describe().round(0).to_string())
print("%d specimens (%d with >= 20 cells of every type), %d patients -> %s" % (
    len(wide), int((wide >= 20).all(1).sum()), out["donor_id"].nunique(), dst))
