"""PALM-lite test: closed-form fractions (multinomial EM) + reallocation, vs NNLS and PALM fractions.
Usage: python lite.py DATASET N_SPLITS"""
import os, sys, time
import numpy as np, pandas as pd
from scipy.special import softmax

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dev_bench as db
import palm_bulk as pb
from palm_bulk.core import PseudobulkTable, make_benchmark
from palm_bulk.core.scores import estimate_delta
from palm_bulk.compare.base import MethodResult
from palm_bulk.compare.methods import nnls_fractions

ds_name, n_splits = sys.argv[1], int(sys.argv[2])
ds = db.DATASETS[ds_name]
table = PseudobulkTable.from_h5ad(ds["path"], {}, db.TYPES)
prog = pb.load_programs(os.path.join(ROOT, "data", "progeny_human_top500.csv")).loc[db.PROGS]


def em_fractions(Y, B, iters=500, tol=1e-10):
    """Multinomial MLE of read fractions: y_i ~ Mult(n_i, pi_i B) with B rows summing to 1 (EM)."""
    B = B / B.sum(1, keepdims=True)
    n = Y.sum(1, keepdims=True)
    pi = np.full((Y.shape[0], B.shape[0]), 1 / B.shape[0])
    for _ in range(iters):
        mix = pi @ B                                                     # (N, G)
        new = pi * ((Y / mix) @ B.T) / n
        if np.abs(new - pi).max() < tol:
            return new
        pi = new
    return pi


def em_program(Y, B, G, rounds=3):
    """Alternate: EM fractions with each sample's type profiles shifted by the reallocated delta."""
    pi = em_fractions(Y, B)
    for _ in range(rounds):
        w = pi[:, :, None] * B[None]
        d = estimate_delta(Y[:, None, :] * w / w.sum(1, keepdims=True), B, G)
        prof = softmax(np.log(B)[None] + np.einsum("ncp,pg->ncg", np.nan_to_num(d), G), -1)   # (N, C, G)
        n = Y.sum(1, keepdims=True)
        for _ in range(300):
            mix = np.einsum("nc,ncg->ng", pi, prof)
            pi = pi * np.einsum("ng,ncg->nc", Y / mix, prof) / n
    return pi


rows = []
for split in range(n_splits):
    b = make_benchmark(table, min_cells=20, seed=split, truth_split=True, contrast=ds["contrast"], n_test=ds["n_test"])
    inp = pb.prepare_inputs(b.bulk, b.reference, prog, treatment=b.treatment, max_genes=1000)
    tz = np.load(os.path.join(ROOT, "benchmarks", "dev", ds_name, "s%d__truth.npz" % split))
    design = db._design(tz["T"])
    truth = dict(pi=tz["pi"], delta=tz["delta"], beta=design(tz["delta"]))
    truth_ulm = dict(pi=tz["pi"], delta=tz["ulm"])
    Y, B, G = inp.Y, inp.B, inp.Gmat
    fr = {}
    t0 = time.time(); fr["NNLS"] = nnls_fractions(Y, B); tn = time.time() - t0
    t0 = time.time(); fr["EM (lite)"] = em_fractions(Y, B); te = time.time() - t0
    t0 = time.time(); fr["EM + programs"] = em_program(Y, B, G); tp = time.time() - t0
    z = np.load(os.path.join(ROOT, "benchmarks", "dev", ds_name, "s%d__joint_nb.npz" % split))
    fr["PALM (joint nb)"] = np.median(z["pi"], 0)
    fr["truth"] = tz["pi"]
    times = {"NNLS": tn, "EM (lite)": te, "EM + programs": tp}
    for name, pi in fr.items():
        t0 = time.time()
        w = pi[:, :, None] * B[None]
        a = estimate_delta(Y[:, None, :] * w / w.sum(1, keepdims=True), B, G)
        tr = time.time() - t0
        s = db._score(MethodResult(name, fractions=pi, activity=a, effects=design(a)), truth, truth_ulm)
        s.update(split=split, method=name, sec=times.get(name, np.nan) + tr)
        rows.append(s)
    print("split %d done (EM %.2fs, EM+programs %.2fs, realloc %.2fs)" % (split, te, tp, tr), flush=True)

df = pd.DataFrame(rows)
cols = dict(frac_rmse="frac", frac_r_per_type="frac_r", act_spearman_across_samples="rho", act_spearman_specific="spec",
            act_spearman_across_types="types", ulm_act_spearman_across_samples="ulm_rho", eff_pearson="eff_r",
            eff_localise="local", sec="sec")
t = df.groupby("method", sort=False)[[c for c in cols if c in df]].mean().rename(columns=cols)
pd.set_option("display.width", 200)
print("== %s, %d splits" % (ds_name, df.split.nunique()))
print(t.round(3).to_string())
