"""Idea register F6 (CETYGO-style per-sample fit score as a missing-cell-type detector; 2026-11-05).

Score per sample: RMSE over the modelled genes (more expressed half) of log(observed CPM) - log(expected CPM
from the PALM-lite fractions and the reference). Also the depth-corrected relative misfit
(core.gls.frac_misfit per sample).
Tests:
  1. Drop-one-type, single-cell pseudobulk (truth known): remove type k from the reference, refit, score; Spearman
     of the score increase with the true read fraction of k across samples.
  2. COMBAT real whole-blood bulk: the reference (PBMC) lacks granulocytes; Spearman of the score with the
     CyTOF granulocyte (neutrophil + eosinophil + basophil) frequency of the same specimens; control: the same
     specimens' single-cell pseudobulk (no granulocytes).
Usage: python missing_type_score.py [SPLIT=20] -> benchmarks/dev/missing_type/combat__s{split}.json
"""
import json, os, re, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import combat_real as cr
import program2cell as pb
from program2cell.core.realloc import em_fractions

OUT = os.path.join(ROOT, "benchmarks", "dev", "missing_type")
CYTOF = os.path.join(ROOT, "data", "combat", "cytof_wb", "CBD-KEY-CYTOF-WB", "WB_Cell_Counts_Freq.xlsx")


def fit_score(Y, B):
    Y, B = np.asarray(Y, float), np.asarray(B, float)
    Bn = B / B.sum(1, keepdims=True)
    pi = em_fractions(Y, B)
    mu = pi @ Bn
    p = Y / Y.sum(1, keepdims=True)
    keep = mu.mean(0) >= np.quantile(mu.mean(0), 0.5)
    lr = np.log(p[:, keep] + 1e-7) - np.log(mu[:, keep] + 1e-7)
    rmse = np.sqrt(((lr - lr.mean(1, keepdims=True)) ** 2).mean(1))
    m = Y.sum(1, keepdims=True) * mu
    rel = ((((Y - m) ** 2 - m) / np.maximum(m, 1e-9) ** 2)[:, keep]).mean(1)
    return rmse, rel, pi


def cytof_freqs():
    x = pd.ExcelFile(CYTOF)
    d = pd.concat([x.parse(s) for s in x.sheet_names if s.endswith("CellFreqs")])
    key = d["CyTOF_sample_ID"].str.extract(r"^([A-Z]\d+)-([A-Za-z]+\d+)")
    d.index = key[0] + "|" + key[1]
    g = d[["Neutrophil", "Eosinophil", "Basophil"]].sum(1)
    return g[~g.index.duplicated()]


def run(split=20):
    import anndata as ad
    from program2cell.core import PseudobulkTable, make_benchmark
    os.makedirs(OUT, exist_ok=True)
    table = PseudobulkTable.from_h5ad(os.path.join(cr.D, "combat_sc_pseudobulk_sepsis.h5ad"), {}, cr.TYPES)
    b = make_benchmark(table, cell_types=cr.TYPES, min_cells=20, seed=split, truth_split=True,
                       contrast=("arm", "T", "C"), n_test=20)
    prog = cl.db._programs(pb, "progeny7")
    out = dict(split=split)
    # 1. drop-one-type on the single-cell pseudobulk
    inp = pb.prepare_inputs(b.bulk, b.reference, prog, treatment=b.treatment, max_genes=1000)
    Y, B = np.asarray(inp.Y, float), np.asarray(inp.B, float)
    pt = np.asarray(b.truth(inp)["pi"], float)
    base, _, _ = fit_score(Y, B)
    out["drop_one"] = {}
    for k, c in enumerate(inp.cell_types):
        s, _, _ = fit_score(Y, np.delete(B, k, 0))
        rho = spearmanr(s - base, pt[:, k])[0]
        out["drop_one"][c] = dict(rho=float(rho), mean_increase=float((s - base).mean()), mean_frac=float(pt[:, k].mean()))
        print("drop %-10s mean read fraction %.3f  score increase %.3f  Spearman(increase, fraction) %.2f" % (
            c, pt[:, k].mean(), (s - base).mean(), rho), flush=True)
    # 2. COMBAT real bulk vs CyTOF granulocytes (all bulk specimens with CyTOF; reference from split's donors)
    bk = ad.read_h5ad(os.path.join(cr.D, "combat_bulk.h5ad"))
    bk = bk[~bk.obs_names.duplicated()]
    gran = cytof_freqs()
    specs = [s for s in bk.obs_names if s in gran.index]
    bulk = pd.DataFrame(sp.csr_matrix(bk[specs].X).toarray().T, index=bk.var_names, columns=specs)
    genes = [g for g in inp.genes if g in bulk.index]
    gi = [list(inp.genes).index(g) for g in genes]
    Yr = bulk.loc[genes].values.T.astype(float)
    rmse, rel, pi = fit_score(Yr, B[:, gi])
    gv = gran.loc[specs].values
    out["real"] = dict(n=len(specs), rho_rmse=float(spearmanr(rmse, gv)[0]), rho_rel=float(spearmanr(rel, gv)[0]),
                       gran_mean=float(gv.mean()), gran_range=[float(gv.min()), float(gv.max())])
    print("COMBAT real bulk, %d specimens with CyTOF (granulocytes %.0f%% [%.0f-%.0f]): Spearman(score, granulocyte %%)"
          " rmse %.2f  relative %.2f" % (len(specs), gv.mean(), gv.min(), gv.max(), out["real"]["rho_rmse"],
                                         out["real"]["rho_rel"]), flush=True)
    # control: the same specimens' single-cell pseudobulk where present
    o = table.obs
    X = sp.csr_matrix(table.counts)
    gidx = {g: i for i, g in enumerate(table.genes)}
    sc_specs = [s for s in specs if (o["specimen"] == s).any()]
    Ysc = np.vstack([np.asarray(X[np.flatnonzero((o["specimen"] == s).values)].sum(0)).ravel()[[gidx[g] for g in genes]]
                     for s in sc_specs]).astype(float)
    r2, rel2, _ = fit_score(Ysc, B[:, gi])
    gs = gran.loc[sc_specs].values
    out["sc_control"] = dict(n=len(sc_specs), rho_rmse=float(spearmanr(r2, gs)[0]), rho_rel=float(spearmanr(rel2, gs)[0]))
    print("control, single-cell pseudobulk of %d of those specimens: Spearman rmse %.2f relative %.2f" % (
        len(sc_specs), out["sc_control"]["rho_rmse"], out["sc_control"]["rho_rel"]), flush=True)
    json.dump(out, open(os.path.join(OUT, "combat__s%d.json" % split), "w"))


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
