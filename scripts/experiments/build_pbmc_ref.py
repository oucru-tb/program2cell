"""PBMC + neutrophil single-cell reference (data/pbmc_ref/#4-Azimuth_Neutro(SubsetA).RDS, Seurat, user-supplied
2026-11-10) -> data/pbmc_ref/pbmc_neutro_pseudobulk.h5ad, a PseudobulkTable (rows = specimen x cell type).

Source: 43,380 cells, 9,991 genes, raw counts; celltype.l2 (17 labels); PBMC donors P1-P3, each in 5 libraries
(orig.ident L1-L5); neutrophils (2,950 cells) from one other donor/study (HealthyN, GSE188288) only.

Broad types (the 7 COMBAT types + neutrophils):
  B = B naive, B memory, Plasmablast;  CD4 T = CD4 Naive, CD4 TM, Treg, CD4 CTL;
  CD8 T = CD8 Naive, CD8 TM, MAIT, OtherT;  NK;  CD14 Mono;  CD16 Mono;  DC = cDC, pDC;  Neutro.
Specimen = donor::library (15 specimens, 3 donors). The calibration needs every type in each specimen, but no
PBMC specimen has neutrophils: the neutrophil cells are split at random (seed 0) into the 15 specimens. Their
donor-to-donor variation is therefore that of one donor (understated); stated as a limitation of the PTB run.
LABELS=l2 (default since 2026-11-10, user decision): celltype.l2 labels as they are (17 types); several are rare
(CD4 CTL, OtherT, Plasmablast, pDC, MAIT: 1-20 cells per library), so specimen = donor (libraries pooled: 3
specimens) and the run uses min_cells=1 (user choice; rare types' profiles rest on few cells). Output
data/pbmc_ref/pbmc_neutro_l2_pseudobulk.h5ad; LABELS=broad: the 8 broad types above, pbmc_neutro_pseudobulk.h5ad.
Usage: [LABELS=l2|broad] python build_pbmc_ref.py
"""
import os, subprocess, tempfile
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
SRC = os.path.join(ROOT, "data", "pbmc_ref", "#4-Azimuth_Neutro(SubsetA).RDS")
LABELS = os.environ.get("LABELS", "l2")
OUT = os.path.join(ROOT, "data", "pbmc_ref", "pbmc_neutro_%spseudobulk.h5ad" % ("l2_" if LABELS == "l2" else ""))
RENV = os.environ.get("PALM_R_ENV", "C:/Users/duy.pham/AppData/Local/miniforge3/envs/palm-r")
BROAD = {"B naive": "B", "B memory": "B", "Plasmablast": "B", "CD4 Naive": "CD4 T", "CD4 TM": "CD4 T",
         "Treg": "CD4 T", "CD4 CTL": "CD4 T", "CD8 Naive": "CD8 T", "CD8 TM": "CD8 T", "MAIT": "CD8 T",
         "OtherT": "CD8 T", "NK": "NK", "CD14 Mono": "CD14 Mono", "CD16 Mono": "CD16 Mono", "cDC": "DC",
         "pDC": "DC", "Neutro": "Neutro"}
TYPES = ["B", "CD4 T", "CD8 T", "NK", "CD14 Mono", "CD16 Mono", "DC", "Neutro"]
if LABELS == "l2":
    BROAD = {k: k for k in BROAD}
    TYPES = list(BROAD)

R = r"""
suppressMessages({library(SeuratObject); library(Matrix)})
x <- readRDS("%(src)s")
md <- x@meta.data
cnt <- LayerData(x, "counts")
map <- c(%(map)s)
ct <- unname(map[as.character(md$celltype.l2)])
stopifnot(!any(is.na(ct)))
spec <- %(spec)s
pb <- spec[md$donor != "HealthyN"]
specs <- sort(unique(pb))
set.seed(0)
neu <- which(md$donor == "HealthyN")
spec[neu] <- sample(rep(specs, length.out = length(neu)))          # neutrophils split over the PBMC specimens
key <- paste(spec, ct, sep = "||")
f <- factor(key)
M <- sparseMatrix(i = seq_along(f), j = as.integer(f), x = 1, dims = c(length(f), nlevels(f)))
agg <- as.matrix(cnt %%*%% M)                                       # genes x (specimen || type)
colnames(agg) <- levels(f)
write.csv(agg, gzfile("%(tmp)s/agg.csv.gz"))
write.csv(data.frame(key = levels(f), n_cells = as.vector(table(f))), "%(tmp)s/rows.csv", row.names = FALSE)
"""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = tmp.replace("\\", "/")
        script = os.path.join(tmp, "ref.R")
        mp = ",".join('"%s"="%s"' % kv for kv in BROAD.items())
        spec = 'as.character(md$donor)' if LABELS == "l2" else 'paste(md$donor, md$orig.ident, sep = "::")'
        open(script, "w").write(R % dict(src=SRC.replace("\\", "/"), tmp=tmp, map=mp, spec=spec))
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([os.path.join(RENV, "Lib", "R", "bin", "x64"), RENV,
                                       os.path.join(RENV, "Library", "mingw-w64", "bin"),
                                       os.path.join(RENV, "Library", "bin"), env["PATH"]])
        subprocess.run([os.path.join(RENV, "Lib", "R", "bin", "x64", "Rscript.exe"), script], check=True, env=env)
        agg = pd.read_csv(os.path.join(tmp, "agg.csv.gz"), index_col=0)
        rows = pd.read_csv(os.path.join(tmp, "rows.csv")).set_index("key").loc[agg.columns]
    parts = rows.index.to_series().str.split(r"\|\|", expand=True)
    obs = pd.DataFrame({"specimen": parts[0].values, "cell_type": parts[1].values,
                        "donor_id": parts[0].str.split("::").str[0].values,
                        "n_cells": rows["n_cells"].values}, index=rows.index)
    a = ad.AnnData(sp.csr_matrix(agg.T.values.astype(np.float32)), obs=obs)
    a.obs_names, a.var_names = list(map(str, obs.index)), list(map(str, agg.index))
    a.write_h5ad(OUT, compression="gzip")
    print("wrote", OUT, a.shape)
    print(obs.pivot_table(index="specimen", columns="cell_type", values="n_cells", aggfunc="sum").reindex(
        columns=TYPES).to_string())


if __name__ == "__main__":
    main()
