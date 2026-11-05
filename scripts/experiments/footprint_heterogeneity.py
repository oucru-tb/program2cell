"""Idea register 10 gate (review R17; 2026-11-05): are program footprints shared across cell types?

From the single-cell pseudobulk table (truth side, no deconvolution): per cell type, per gene, the treatment log
fold change (mean log CPM treated - control specimens, genes with mean CPM >= 10 in that type). Per program,
over its genes:
  between-type agreement  mean pairwise Pearson r of the types' gene log fold changes (1 = one footprint)
  weight agreement        mean over types of r(gene log fold change, program weight)
  effect                  per type, the program-weighted mean log fold change (size of the response)
Programs reported where the largest |effect| >= 0.1.
Usage: python footprint_heterogeneity.py DATASETS -> benchmarks/dev/footprint/{ds}.json
"""
import json, os, sys
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import program2cell as pb

OUT = os.path.join(ROOT, "benchmarks", "dev", "footprint")


def run(datasets):
    os.makedirs(OUT, exist_ok=True)
    prog = cl.db._programs(pb, "progeny7")
    for dsn in datasets:
        ds = cl.db.DATASETS[dsn]
        t = cl.db._table(ds)
        col, tr, ct = ds["contrast"]
        o = t.obs
        X = t.counts.tocsr() if hasattr(t.counts, "tocsr") else t.counts
        genes = np.asarray(t.genes)
        types = ds.get("types") or sorted(o["cell_type"].unique())
        L, E = {}, {}
        for c in types:
            lc = {}
            for arm in (tr, ct):
                rows = np.flatnonzero(((o["cell_type"] == c) & (o[col] == arm)).values)
                M = np.asarray(X[rows].todense(), float)
                M = M / np.maximum(M.sum(1, keepdims=True), 1) * 1e6
                lc[arm] = (np.log2(M + 1).mean(0), M.mean(0))
            ok = (lc[tr][1] >= 10) & (lc[ct][1] >= 10)
            L[c] = np.where(ok, lc[tr][0] - lc[ct][0], np.nan)
        out = dict(dataset=dsn, types=types, programs={})
        gi = {g: i for i, g in enumerate(genes)}
        for p, w in prog.iterrows():
            w = w[w != 0]
            idx = np.array([gi[g] for g in w.index if g in gi])
            wv = np.array([w[g] for g in w.index if g in gi])
            F = np.array([L[c][idx] for c in types])                     # types x genes
            good = ~np.isnan(F).any(0)
            F, wv = F[:, good], wv[good]
            if F.shape[1] < 20:
                continue
            eff = (F * wv).sum(1) / np.abs(wv).sum()
            R = np.corrcoef(F)
            pair = R[np.triu_indices(len(types), 1)]
            wr = [np.corrcoef(F[k], wv)[0, 1] for k in range(len(types))]
            out["programs"][p] = dict(n_genes=int(F.shape[1]), between_r=float(np.nanmean(pair)),
                                      weight_r=float(np.nanmean(wr)), effects=eff.tolist(),
                                      max_abs_effect=float(np.abs(eff).max()))
            if np.abs(eff).max() >= 0.1:
                print("%-11s %-8s genes %3d  between-type r %.2f  weight r %.2f  effects %s" % (
                    dsn, p, F.shape[1], np.nanmean(pair), np.nanmean(wr), np.round(eff, 2)), flush=True)
        json.dump(out, open(os.path.join(OUT, "%s.json" % dsn), "w"))


if __name__ == "__main__":
    run(sys.argv[1].split(","))
