"""Freeze v2 and freeze v5 on COMBAT real bulk (method_spec "Method freeze v5", criterion S3; written with the freeze,
before any split 30-39 result). Same benchmark, bulks, truth and real-bulk scores as combat_v4.py (sepsis vs HV;
bulks sc, real, real_offset, real_selfoffset; within-program r, null and wrong-sign calls, coverage).
Estimators: v2 (estimator="auto") and v5 (estimator="v5": the same estimates, size-matched calibrated interval).
Usage: python combat_v5.py SPLITS -> benchmarks/dev/combat_v5/sepsis__s{split}.json
       python combat_v5.py summary [SPLITS]
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import combat_real as cr
import program2cell as pb
from program2cell.core.gls import FRAC_SD_REAL_BULK

OUT = os.path.join(ROOT, "benchmarks", "dev", "combat_v5")
FRAC_SD = dict(sc=0.0, real=FRAC_SD_REAL_BULK["raw"], real_offset=FRAC_SD_REAL_BULK["platform_corrected"],
               real_selfoffset=FRAC_SD_REAL_BULK["platform_corrected"])


def rscore(e, lo, hi, b):
    c = (lo > 0) | (hi < 0)
    ws = [np.corrcoef(e[:, p], b[:, p])[0, 1] for p in range(b.shape[1]) if b[:, p].std() >= 0.05]
    return dict(within_r=float(np.mean(ws)) if ws else None, null_calls=int((c & (np.abs(b) < 0.05)).sum()),
                wrong_sign=int((c & (np.sign(e) != np.sign(b))).sum()), calls=int(c.sum()),
                right_big=int((c & (np.abs(b) >= 0.1) & (np.sign(e) == np.sign(b))).sum()),
                r=float(np.corrcoef(e.ravel(), b.ravel())[0, 1]), slope=float(np.polyfit(b.ravel(), e.ravel(), 1)[0]),
                cov=float(((b >= lo) & (b <= hi)).mean()), width=float((hi - lo).mean()))


def run(splits, contrast="sepsis"):
    import anndata as ad
    from program2cell.core import PseudobulkTable, make_benchmark
    os.makedirs(OUT, exist_ok=True)
    table = PseudobulkTable.from_h5ad(os.path.join(cr.D, "combat_sc_pseudobulk_%s.h5ad" % contrast), {}, cr.TYPES)
    bk = ad.read_h5ad(os.path.join(cr.D, "combat_bulk.h5ad"))
    bk = bk[bk.obs["paired"].values & ~bk.obs_names.duplicated()]
    bulk_all = pd.DataFrame(sp.csr_matrix(bk.X).toarray().T, index=bk.var_names, columns=bk.obs_names)
    prog = cl.db._programs(pb, "progeny7")
    for split in splits:
        f = os.path.join(OUT, "%s__s%d.json" % (contrast, split))
        if os.path.exists(f):
            continue
        b = make_benchmark(table, cell_types=cr.TYPES, min_cells=20, seed=split, truth_split=True,
                           contrast=("arm", "T", "C"), n_test=cr.CONTRASTS[contrast][2])
        specs = list(b.bulk.columns)
        genes = [g for g in b.bulk.index if g in bulk_all.index]
        test_donors = set(b.meta["donor_id"].astype(str))
        other = [s for s in bulk_all.columns if s.split("|")[0] not in test_donors]
        off = cr._offsets(bulk_all, table, other, genes)
        real = bulk_all.loc[genes, specs]
        corr = (real.T / np.exp2(off.clip(-8, 8)).values).T
        corr = corr / corr.sum(0) * real.sum(0)
        bulks = {"sc": b.bulk.loc[genes], "real": real, "real_offset": corr.round(),
                 "real_selfoffset": pb.correct_platform(real, b.reference)}
        out = dict(contrast=contrast, split=split)
        for name, Y in bulks.items():
            inp = pb.prepare_inputs(Y, b.reference, prog, treatment=b.treatment, max_genes=1000)
            T = np.asarray(inp.T)
            beta = np.asarray(pb.core.contrast(T)(b.truth(inp)["delta"]))
            cal = pb.calibration_from_table(table, test_donors, inp.cell_types, inp.genes, inp.Gmat, seed=split)
            res = {}
            for est, kw in (("v2", dict(estimator="auto")), ("v5", dict(estimator="v5"))):
                t0 = time.time()
                r = pb.program_deconvolution(inp, cal, engine="lite", treatment=T, seed=split, **kw)
                lo, hi = (np.asarray(x) for x in r.effects_interval)
                res[est] = dict(rscore(np.asarray(r.effects), lo, hi, beta), seconds=time.time() - t0,
                                effects=np.asarray(r.effects).tolist(), lo=lo.tolist(), hi=hi.tolist())
            res["beta"] = beta.tolist()
            out[name] = res
            for est in ("v2", "v5"):
                s = res[est]
                print("[%s] s%d %-15s %-7s within %5.2f r %5.2f slope %5.2f calls %d null %d wrong %d cov %.2f "
                      "width %.2f" % (time.strftime("%H:%M"), split, name, est, s["within_r"] or np.nan, s["r"],
                                      s["slope"], s["calls"], s["null_calls"], s["wrong_sign"], s["cov"], s["width"]),
                      flush=True)
        json.dump(out, open(f, "w"))


def summary(contrast="sepsis", splits=None):
    fs = sorted(glob.glob(os.path.join(OUT, "%s__s*.json" % contrast)))
    if splits is not None:                                      # e.g. 40-49 (held-out v5: 30-39)
        fs = [f for f in fs if int(os.path.basename(f).split("__s")[1].split(".")[0]) in splits]
    R = [json.load(open(f)) for f in fs]
    print("COMBAT %s, %d splits; per-split means (calls are counts of 49)" % (contrast, len(R)))
    print("%-16s %-8s %7s %6s %6s %6s %5s %5s %6s %5s %6s %10s" % (
        "bulk", "est", "within", "r", "slope", "calls", "null", "wrong", "rightbig", "cov", "width", "retention"))
    keys = ["within_r", "r", "slope", "calls", "null_calls", "wrong_sign", "right_big", "cov", "width"]
    sc = {est: np.nanmean([o["sc"][est]["within_r"] or np.nan for o in R]) for est in ("v2", "v5")}
    for name in ("sc", "real", "real_offset", "real_selfoffset"):
        for est in ("v2", "v5"):
            m = [np.nanmean([np.nan if o[name][est][k] is None else o[name][est][k] for o in R]) for k in keys]
            print("%-16s %-8s %7.2f %6.2f %6.2f %6.1f %5.1f %5.1f %6.1f %5.2f %6.2f %10.2f" % (
                (name, est) + tuple(m) + (m[0] / sc[est],)))

if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(splits=cl.db._splits(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        run(cl.db._splits(sys.argv[1]))
