"""Parse Biosciences 10M PBMC cytokines (12 donors; fetch_parse.py) -> pseudo-bulk table in the format of
PseudobulkTable.from_h5ad (one row per specimen x cell type), like build_kang.py.

Input: Parse's published pseudobulk files, one per cell type x condition (rows = donors, summed raw counts,
psbulk_n_cells). Fine types are merged into a broad PBMC set (MAP; the minor types NKT, HSPC, ILC, granulocyte and
platelet, ~4% of cells, are dropped). There are no cells, so each donor x condition is partitioned at random
(multinomial, per gene and cell type) into N_REP disjoint pseudo-replicate specimens of about CELLS_PER_REP cells'
worth of counts each (the rest unused), enough for >= 20 dendritic cells per specimen (calibration min_cells). Partitioning the summed counts is the
same in distribution as partitioning the cells' counts, but cell-to-cell heterogeneity between replicates is lost,
so replicates of one donor are close: benchmarks keep a donor's specimens on one side of the split.
Labels: "PBS" and short cytokine names (LABELS). Output: data/parse/parse_pseudobulk.h5ad
Usage: python build_parse.py
"""
import os, glob
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
D = os.path.join(ROOT, "data", "parse")
MAP = {"CD14_Mono": "CD14 Mono", "CD16_Mono": "CD16 Mono",
       "CD4_Naive_T_cell_": "CD4 T", "CD4_Memory_T_cell_": "CD4 T", "Treg": "CD4 T",
       "CD8_Naive_T_cell": "CD8 T", "CD8_Memory_T_cell": "CD8 T", "MAIT": "CD8 T",
       "NK_CD56low": "NK", "NK_CD56hi": "NK",
       "Naive_B_cell": "B", "Intermediate_B_cell": "B", "Plasmablast": "B",
       "cDC": "DC", "pDC": "DC"}
LABELS = {"PBS": "PBS", "IFN-gamma": "ifng", "IFN-beta": "ifnb", "IFN-alpha1": "ifna", "TNF-alpha": "tnf",
          "IL-1-beta": "il1b", "IL-6": "il6", "IL-10": "il10", "GM-CSF": "gmcsf", "IL-4": "il4"}
N_REP, CELLS_PER_REP = 3, 5000                 # >= 20 DCs per specimen (DC ~0.5-1.5% of PBMC), for calibration


def main():
    rng = np.random.default_rng(0)
    acc, genes = {}, None                                # (donor, label, broad type) -> [counts, n_cells]
    for f in sorted(glob.glob(os.path.join(D, "pseudobulk", "pbulk_*.h5ad"))):
        stem = os.path.basename(f)[6:-5]
        ct, cyt = stem.rsplit("_", 1)
        if ct not in MAP or cyt not in LABELS:
            continue
        a = ad.read_h5ad(f)
        genes = list(a.var_names) if genes is None else genes
        assert list(a.var_names) == genes, f
        X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        for i, dn in enumerate(a.obs["donor"].astype(str)):
            k = (dn, LABELS[cyt], MAP[ct])
            c, n = acc.get(k, (0, 0.0))
            acc[k] = (c + X[i].astype(np.int64), n + float(a.obs["psbulk_n_cells"].iloc[i]))
    rows, mats = [], []
    for dn, lab in sorted({(d, l) for d, l, _ in acc}):
        keys = [k for k in acc if k[0] == dn and k[1] == lab]
        total = sum(acc[k][1] for k in keys)
        q = min(1.0 / N_REP, CELLS_PER_REP / max(total, 1.0))
        for k in sorted(keys):
            c, n = acc[k]
            parts = rng.multinomial(c, [q] * N_REP + [1 - q * N_REP])             # disjoint parts, per gene
            for r in range(N_REP):
                spec = "%s|%s|r%d" % (dn, lab, r)
                mats.append(sp.csr_matrix(parts[:, r].astype(np.float32)[None]))
                rows.append(dict(donor_id=dn, label=lab, specimen=spec, cell_type=k[2],
                                 n_cells=int(round(n * q)), project="parse10m"))
    obs = pd.DataFrame(rows)
    obs.index = obs["specimen"] + "::" + obs["cell_type"]
    P = sp.vstack(mats).tocsr()
    obs["n_counts"] = np.asarray(P.sum(1)).ravel()
    res = ad.AnnData(P, obs=obs, var=pd.DataFrame(index=pd.Index(genes)))
    res.var_names_make_unique()
    dst = os.path.join(D, "parse_pseudobulk.h5ad")
    res.write_h5ad(dst, compression="gzip")
    print(res)
    print(obs.groupby("cell_type").n_cells.sum().to_string())
    print(obs.groupby("label").specimen.nunique().to_string())


if __name__ == "__main__":
    main()
