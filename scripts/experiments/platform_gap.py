"""Platform gap between real bulk and single-cell pseudobulk on paired samples (COMBAT; build_combat.py).

Per paired sample: real whole-blood bulk vs the sum of its PBMC single-cell pseudobulk over cell types, both as
log2 CPM over the shared expressed genes. Reports
  - per-gene offset g = median over samples of the log2 ratio (bulk / pseudobulk): the systematic platform part;
  - its stability: correlation of offsets estimated on two random halves of the donors;
  - how much of the sample x gene log-ratio variance a per-gene offset explains;
  - per-sample correlation of bulk and pseudobulk, before and after removing the per-gene offset;
  - the genes with the largest offsets (expected: neutrophil and red-cell genes, absent from PBMC).
Writes benchmarks/dev/platform_gap/combat_gene_offsets.csv and summary.json.
Usage: python platform_gap.py
"""
import os, json
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
D = os.path.join(ROOT, "data", "combat")
OUT = os.path.join(ROOT, "benchmarks", "dev", "platform_gap")


def _cpm(M):
    return np.log2(M / M.sum(1, keepdims=True) * 1e6 + 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = ad.read_h5ad(os.path.join(D, "combat_sc_pseudobulk.h5ad"))
    bk = ad.read_h5ad(os.path.join(D, "combat_bulk.h5ad"))
    bk = bk[bk.obs["paired"].values]
    bk = bk[~bk.obs_names.duplicated()]
    genes = sorted(set(sc.var_names) & set(bk.var_names))
    si, bi = {g: i for i, g in enumerate(sc.var_names)}, {g: i for i, g in enumerate(bk.var_names)}
    gs, gb = [si[g] for g in genes], [bi[g] for g in genes]
    S = sp.csr_matrix(sc.X)
    specs = [s for s in bk.obs_names if (sc.obs["specimen"] == s).any()]
    ps = np.vstack([np.asarray(S[np.flatnonzero((sc.obs["specimen"] == s).values)].sum(0)).ravel()[gs] for s in specs])
    B = sp.csr_matrix(bk[specs].X).toarray()[:, gb]
    lb, ls = _cpm(B), _cpm(ps)
    expr = (np.exp2(lb).mean(0) > 10) & (np.exp2(ls).mean(0) > 10)
    lb, ls, g_e = lb[:, expr], ls[:, expr], np.array(genes)[expr]
    R = lb - ls                                                      # samples x genes, log2 ratio
    off = np.median(R, 0)
    donors = bk[specs].obs["donor_id"].astype(str).values
    rng = np.random.default_rng(0)
    ud = rng.permutation(np.unique(donors))
    h1 = np.isin(donors, ud[: len(ud) // 2])
    stab = float(np.corrcoef(np.median(R[h1], 0), np.median(R[~h1], 0))[0, 1])
    resid = R - off[None]
    var_expl = float(1 - resid.var() / R.var())
    r_sample = [float(np.corrcoef(lb[i], ls[i])[0, 1]) for i in range(len(specs))]
    r_sample_corr = [float(np.corrcoef(lb[i] - off, ls[i])[0, 1]) for i in range(len(specs))]
    out = dict(n_samples=len(specs), n_genes=int(expr.sum()), offset_sd=float(off.std()),
               offset_split_half_r=stab, var_explained_by_gene_offset=var_expl,
               sample_r_median=float(np.median(r_sample)), sample_r_after_offset_median=float(np.median(r_sample_corr)),
               resid_sd=float(resid.std()))
    pd.DataFrame(dict(gene=g_e, offset=off, resid_sd=resid.std(0), mean_bulk=lb.mean(0), mean_sc=ls.mean(0))
                 ).sort_values("offset").to_csv(os.path.join(OUT, "combat_gene_offsets.csv"), index=False)
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for k, v in out.items():
        print("%-32s %s" % (k, v))


if __name__ == "__main__":
    main()
