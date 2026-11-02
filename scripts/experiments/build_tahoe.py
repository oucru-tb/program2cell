"""Tahoe-100M plate 3 (5 uM; Arc Virtual Cell Atlas h5ad) -> semi-synthetic pooled-cell-line mixtures with REAL drug
responses, in the format of PseudobulkTable.from_h5ad (as build_mcfarland.py).

In Tahoe each well holds one drug at one dose with ~50 cancer lines pooled; plate 3 has 93 drugs at 5 uM and two DMSO
wells (plate-matched vehicle). Known answers, from the lines' genetics (cell_line_metadata) and checked against
the measured target-gene response:
  tram    trametinib (MEK)          MAPK down in every line                         shared positive
  dab     dabrafenib (BRAF V600)    MAPK down in BRAF-V600 lines only               type-specific positive
  adag    adagrasib (KRAS G12C)     MAPK down in KRAS-G12C lines only               type-specific positive
  oxa     oxaliplatin (DNA damage)  p53 up in TP53-wild-type lines only             type-specific positive
  gem     gemcitabine (DNA damage)  p53 up in TP53-wild-type lines only             type-specific positive
  dmso2   second DMSO well          nothing                                         negative control
(plus paclitaxel, alpelisib, everolimus, afatinib, bortezomib: exploratory). Control arm: the first DMSO well.

Stages
  scan   obs only: cells per (line, well) on plate 3 -> data/tahoe/plate3_cells.csv
  cells  one streaming pass over X: the chosen lines x wells -> data/tahoe/plate3_subset.h5ad (cells)
  mix    mixtures (McFarland design): each line's cells per arm split 50/50 into reference and test pools; test
         mixtures N_TEST with Dirichlet line proportions, the same proportions for every arm (paired by mixture);
         reference mixtures from DMSO-1 reference cells only -> data/tahoe/tahoe_mix.h5ad
Usage: python build_tahoe.py scan|cells|mix
"""
import os, sys, json
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
D = os.path.join(ROOT, "data", "tahoe")
RAW = os.path.join(D, "raw", "plate3.h5ad")
DRUGS = {"DMSO_TF": "ctrl", "Trametinib": "tram", "Dabrafenib": "dab", "Adagrasib": "adag", "Oxaliplatin": "oxa",
         "Gemcitabine": "gem", "Paclitaxel": "ptx", "Alpelisib": "alp", "Everolimus": "evr", "Afatinib": "afa",
         "Bortezomib": "bor"}
LINES_FILE = os.path.join(D, "plate3_lines.json")       # the chosen panel (written after the scan)
N_TEST, N_REF, CELLS, MIN_PER_LINE = 40, 30, 400, 20


def _obs():
    import h5py
    f = h5py.File(RAW, "r")
    g = f["obs"]

    def col(k):
        x = g[k]
        if isinstance(x, h5py.Group):                                   # categorical
            cats = x["categories"][:].astype(str)
            return cats[x["codes"][:]]
        v = x[:]
        return v.astype(str) if v.dtype.kind in "SO" else v
    obs = pd.DataFrame({k: col(k) for k in ("sample", "drug", "cell_name", "pass_filter", "tscp_count")})
    f.close()
    return obs


def scan():
    obs = _obs()
    obs = obs[obs.pass_filter == "full"]
    obs["drug"] = obs["drug"].str.strip()
    t = obs[obs.drug.isin(DRUGS)].groupby(["cell_name", "sample", "drug"]).size().rename("n").reset_index()
    t.to_csv(os.path.join(D, "plate3_cells.csv"), index=False)
    print(len(obs), "cells pass filter;", obs.cell_name.nunique(), "lines;", obs["sample"].nunique(), "wells")
    w = t.pivot_table(index="cell_name", columns="drug", values="n", aggfunc="sum", fill_value=0)
    print(w.to_string())
    print("median UMI per cell", float(np.median(obs.tscp_count)))


def cells(chunk=60_000):
    import h5py
    import anndata as ad
    lines = json.load(open(LINES_FILE))["lines"]
    obs = _obs()
    obs["drug"] = obs["drug"].str.strip()
    keep = (obs.pass_filter == "full") & obs.drug.isin(DRUGS) & obs.cell_name.isin(lines)
    rows = np.flatnonzero(keep.values)
    print(len(rows), "cells to extract", flush=True)
    f = h5py.File(RAW, "r")
    X = f["X"]
    indptr = X["indptr"][:]
    n_genes = f["var"][f["var"].attrs["_index"]].shape[0]
    genes = f["var"][f["var"].attrs["_index"]][:].astype(str)
    parts = []
    for a in range(0, len(indptr) - 1, chunk):
        b = min(a + chunk, len(indptr) - 1)
        sel = rows[(rows >= a) & (rows < b)]
        if not len(sel):
            continue
        lo, hi = indptr[a], indptr[b]
        dat, ind = X["data"][lo:hi], X["indices"][lo:hi]
        M = sp.csr_matrix((dat, ind, indptr[a:b + 1] - lo), shape=(b - a, n_genes))
        parts.append(M[sel - a].astype(np.float32))
        print("rows %d-%d: %d kept" % (a, b, len(sel)), flush=True)
    f.close()
    Xs = sp.vstack(parts).tocsr()
    o = obs.iloc[rows].reset_index(drop=True)
    o.index = o.index.astype(str)
    res = ad.AnnData(Xs, obs=o, var=pd.DataFrame(index=pd.Index(genes)))
    res.var_names_make_unique()
    res.write_h5ad(os.path.join(D, "plate3_subset.h5ad"), compression="gzip")
    print("wrote", res)


def mix():
    import anndata as ad
    spec = json.load(open(LINES_FILE))
    lines = spec["lines"]
    a = ad.read_h5ad(os.path.join(D, "plate3_subset.h5ad"))
    X = sp.csr_matrix(a.X)
    o = a.obs
    dmso = sorted(o.loc[o.drug == "DMSO_TF", "sample"].unique())
    arm = np.array(o["drug"].astype(str).map(DRUGS), dtype=object)
    arm[(o["sample"] == dmso[1]).values] = "dmso2"
    line = o["cell_name"].astype(str).values
    arms = ["ctrl"] + [v for v in dict.fromkeys(DRUGS.values()) if v != "ctrl"] + ["dmso2"]
    rng = np.random.default_rng(0)
    pool = {}
    for ln in lines:
        for ar in arms:
            idx = rng.permutation(np.flatnonzero((line == ln) & (arm == ar)))
            h = len(idx) // 2
            pool[ln, ar, "ref"], pool[ln, ar, "test"] = idx[:h], idx[h:]
    rows, mats = [], []

    def mixture(donor, label, arm_key, which, props):
        counts = np.maximum(np.rint(props * CELLS).astype(int), MIN_PER_LINE)
        for ln, n in zip(lines, counts):
            cand = pool[ln, arm_key, which]
            pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
            mats.append(sp.csr_matrix(X[pick].sum(0)))
            rows.append(dict(donor_id=donor, label=label, specimen="%s|%s" % (donor, label), cell_type=ln,
                             n_cells=len(pick), **{k: ln in v for k, v in spec["groups"].items()}))

    for k in range(N_TEST):
        props = rng.dirichlet(np.full(len(lines), 2.0))
        for ar in arms:
            mixture("mix%02d" % k, ar, ar, "test", props)
    for k in range(N_REF):
        mixture("ref%02d" % k, "refctrl", "ctrl", "ref", rng.dirichlet(np.full(len(lines), 2.0)))
    obs = pd.DataFrame(rows)
    obs.index = obs["specimen"] + "::" + obs["cell_type"]
    obs["project"] = "tahoe100m_plate3"
    P = sp.vstack(mats).tocsr()
    obs["n_counts"] = np.asarray(P.sum(1)).ravel()
    res = ad.AnnData(P.astype(np.float32), obs=obs, var=pd.DataFrame(index=a.var_names))
    dst = os.path.join(D, "tahoe_mix.h5ad")
    res.write_h5ad(dst)
    print("%d specimens (%d mixtures x %d arms, %d reference) -> %s" % (obs.specimen.nunique(), N_TEST, len(arms),
                                                                     N_REF, dst))


if __name__ == "__main__":
    dict(scan=scan, cells=cells, mix=mix)[sys.argv[1]]()
