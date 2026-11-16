"""COMBAT healthy-volunteer PBMC + GSE188288 neutrophils -> data/pbmc_ref/combat_hv_neutro_pseudobulk.h5ad
(PseudobulkTable: rows = specimen x cell type). For the PTB application (user request 2026-11-10).

PBMC: COMBAT single-cell pseudobulk (data/combat/combat_sc_pseudobulk.h5ad), the 10 healthy volunteers (label HV;
one specimen each), 7 broad types (B, CD4 T, CD8 T, CD14 Mono, CD16 Mono, DC, NK), 63-5196 cells per type.
Neutrophils: the 2,950 neutrophils (one donor, GSE188288) in the user's Seurat object
(data/pbmc_ref/#4-Azimuth_Neutro(SubsetA).RDS), split at random (seed 0) into 10 groups, one per COMBAT specimen,
so every specimen has all 8 types (the calibration needs that). Neutrophil donor-to-donor variation is that of one
donor (understated), and the neutrophils come from another study and platform than the PBMC.
Genes: those in both sources (the Seurat object has 9,991 genes).
Coarse (--coarse; user choice 2026-11-11, PTB power): the 8 types merged into the 5 CyTOF groups B, T (CD4 + CD8 T),
NK, Myeloid (CD14 + CD16 Mono + DC), Neutro, rows summed per specimen -> combat_hv_neutro_coarse_pseudobulk.h5ad.
Usage: python build_combat_neutro_ref.py [--coarse]
"""
import os, subprocess, tempfile
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
RDS = os.path.join(ROOT, "data", "pbmc_ref", "#4-Azimuth_Neutro(SubsetA).RDS")
COMBAT = os.path.join(ROOT, "data", "combat", "combat_sc_pseudobulk.h5ad")
OUT = os.path.join(ROOT, "data", "pbmc_ref", "combat_hv_neutro_pseudobulk.h5ad")
OUT_COARSE = os.path.join(ROOT, "data", "pbmc_ref", "combat_hv_neutro_coarse_pseudobulk.h5ad")
COARSE = {"B": "B", "CD4 T": "T", "CD8 T": "T", "NK": "NK", "CD14 Mono": "Myeloid", "CD16 Mono": "Myeloid",
          "DC": "Myeloid", "Neutro": "Neutro"}
RENV = os.environ.get("PALM_R_ENV", "C:/Users/duy.pham/AppData/Local/miniforge3/envs/palm-r")

R = r"""
suppressMessages({library(SeuratObject); library(Matrix)})
x <- readRDS("%(rds)s")
md <- x@meta.data
cnt <- LayerData(x, "counts")[, md$celltype.l2 == "Neutro"]
set.seed(0)
grp <- factor(sample(rep(seq_len(%(k)d), length.out = ncol(cnt))))
M <- sparseMatrix(i = seq_along(grp), j = as.integer(grp), x = 1, dims = c(length(grp), nlevels(grp)))
agg <- as.matrix(cnt %%*%% M)
colnames(agg) <- paste0("g", levels(grp))
write.csv(agg, gzfile("%(tmp)s/neutro.csv.gz"))
write.csv(data.frame(group = paste0("g", levels(grp)), n_cells = as.vector(table(grp))), "%(tmp)s/neutro_n.csv",
          row.names = FALSE)
"""


def main():
    c = ad.read_h5ad(COMBAT)
    c = c[c.obs["label"] == "HV"].copy()
    specs = sorted(c.obs["specimen"].unique())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = tmp.replace("\\", "/")
        script = os.path.join(tmp, "neutro.R")
        open(script, "w").write(R % dict(rds=RDS.replace("\\", "/"), tmp=tmp, k=len(specs)))
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([os.path.join(RENV, "Lib", "R", "bin", "x64"), RENV,
                                       os.path.join(RENV, "Library", "mingw-w64", "bin"),
                                       os.path.join(RENV, "Library", "bin"), env["PATH"]])
        subprocess.run([os.path.join(RENV, "Lib", "R", "bin", "x64", "Rscript.exe"), script], check=True, env=env)
        neu = pd.read_csv(os.path.join(tmp, "neutro.csv.gz"), index_col=0)
        nn = pd.read_csv(os.path.join(tmp, "neutro_n.csv")).set_index("group")["n_cells"]
    genes = [g for g in c.var_names if g in set(neu.index)]
    X = sp.csr_matrix(c[:, genes].X)
    obs = c.obs[["specimen", "donor_id", "cell_type", "n_cells", "label"]].copy()
    spec_donor = obs.drop_duplicates("specimen").set_index("specimen")["donor_id"]
    nrows = pd.DataFrame({"specimen": specs, "donor_id": [spec_donor[s] for s in specs], "cell_type": "Neutro",
                          "n_cells": nn.loc[neu.columns].values, "label": "HV (neutrophils: GSE188288)"},
                         index=[s + "::Neutro" for s in specs])
    NX = sp.csr_matrix(neu.loc[genes].T.values.astype(np.float32))
    out = ad.AnnData(sp.vstack([X, NX]).tocsr(), obs=pd.concat([obs, nrows]))
    out.var_names = genes
    out.obs["n_counts"] = np.asarray(out.X.sum(1)).ravel()
    out.write_h5ad(OUT, compression="gzip")
    print("wrote", OUT, out.shape, "genes shared:", len(genes))
    print(out.obs.pivot_table(index="specimen", columns="cell_type", values="n_cells", aggfunc="sum").to_string())


def coarse():
    a = ad.read_h5ad(OUT)
    key = a.obs["specimen"].astype(str) + "::" + a.obs["cell_type"].map(COARSE)
    keys = list(dict.fromkeys(key))
    M = sp.csr_matrix((np.ones(len(key)), ([keys.index(k) for k in key], np.arange(len(key)))),
                      shape=(len(keys), len(key)))
    obs = a.obs.assign(key=key.values, cell_type=a.obs["cell_type"].map(COARSE)).groupby("key", sort=False).agg(
        specimen=("specimen", "first"), donor_id=("donor_id", "first"), cell_type=("cell_type", "first"),
        n_cells=("n_cells", "sum"), label=("label", "first")).loc[keys]
    out = ad.AnnData(sp.csr_matrix(M @ a.X), obs=obs, var=a.var.copy())
    out.obs["n_counts"] = np.asarray(out.X.sum(1)).ravel()
    out.write_h5ad(OUT_COARSE, compression="gzip")
    print("wrote", OUT_COARSE, out.shape)
    print(out.obs.pivot_table(index="specimen", columns="cell_type", values="n_cells", aggfunc="sum").to_string())


if __name__ == "__main__":
    import sys
    coarse() if "--coarse" in sys.argv else main()
