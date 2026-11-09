"""v2, v4 stacked and gated v4 on COMBAT real bulk, scored by the real-bulk rules (2026-11-06; docs/v4_plan.md
section 12): within-program r, null and wrong-sign calls (primary); overall r with slope; retention of
within-program r vs the pseudobulk of the same specimens; coverage marked scale-dependent.

Per split (sepsis vs HV, make_benchmark as combat_real.py), bulks for the same specimens and truth:
  sc, real (raw), real_offset (non-test-donor platform offsets), real_selfoffset (correct_platform).
Estimators: v2 (auto, freeze v2 defaults), v4k (v4 stacked, noise anchor; frac_sd 0 for sc,
FRAC_SD_REAL_BULK raw / platform_corrected for real bulk), v4gated = v4k or v2 by core.gls.gate_statistics
(computed from the two runs; the same choice as estimator="v4gated", section 12).
Usage: python combat_v4.py SPLITS -> benchmarks/dev/combat_v4/sepsis__s{split}.json
       python combat_v4.py summary
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import combat_real as cr
import program2cell as pb
from program2cell.core.gls import FRAC_SD_REAL_BULK, gate_statistics

OUT = os.path.join(ROOT, "benchmarks", "dev", "combat_v4")
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
            for est, kw in (("v2", dict(estimator="auto")),
                            ("v4k", dict(estimator="v4", v4_kw=dict(stack=True, noise_anchor=True,
                                                                    frac_sd=FRAC_SD[name])))):
                t0 = time.time()
                r = pb.program_deconvolution(inp, cal, engine="lite", treatment=T, seed=split, **kw)
                lo, hi = (np.asarray(x) for x in r.effects_interval)
                res[est] = dict(rscore(np.asarray(r.effects), lo, hi, beta), seconds=time.time() - t0,
                                effects=np.asarray(r.effects).tolist(), lo=lo.tolist(), hi=hi.tolist())
            g = gate_statistics(res["v4k"]["effects"], res["v4k"]["lo"], res["v4k"]["hi"], res["v2"]["effects"])
            res["v4gated"] = dict(res["v4k" if g["use_v4"] else "v2"], gate=g)
            res["beta"] = beta.tolist()
            out[name] = res
            for est in ("v2", "v4k", "v4gated"):
                s = res[est]
                print("[%s] s%d %-15s %-7s within %5.2f r %5.2f slope %5.2f calls %d null %d wrong %d cov %.2f "
                      "width %.2f%s" % (time.strftime("%H:%M"), split, name, est, s["within_r"] or np.nan, s["r"],
                                        s["slope"], s["calls"], s["null_calls"], s["wrong_sign"], s["cov"], s["width"],
                                        "  gate het %.2f agree %.2f -> %s" % (g["heterogeneity"], g["agreement"],
                                                                             "v4" if g["use_v4"] else "v2")
                                        if est == "v4gated" else ""), flush=True)
        json.dump(out, open(f, "w"))


def summary(contrast="sepsis"):
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "%s__s*.json" % contrast)))]
    print("COMBAT %s, %d splits; per-split means (calls are counts of 49)" % (contrast, len(R)))
    print("%-16s %-8s %7s %6s %6s %6s %5s %5s %6s %5s %6s %10s" % (
        "bulk", "est", "within", "r", "slope", "calls", "null", "wrong", "rightbig", "cov", "width", "retention"))
    keys = ["within_r", "r", "slope", "calls", "null_calls", "wrong_sign", "right_big", "cov", "width"]
    sc = {est: np.nanmean([o["sc"][est]["within_r"] or np.nan for o in R]) for est in ("v2", "v4k", "v4gated")}
    for name in ("sc", "real", "real_offset", "real_selfoffset"):
        for est in ("v2", "v4k", "v4gated"):
            m = [np.nanmean([np.nan if o[name][est][k] is None else o[name][est][k] for o in R]) for k in keys]
            n4 = sum(o[name]["v4gated"]["gate"]["use_v4"] for o in R) if est == "v4gated" else None
            print("%-16s %-8s %7.2f %6.2f %6.2f %6.1f %5.1f %5.1f %6.1f %5.2f %6.2f %10.2f%s" % (
                (name, est) + tuple(m) + (m[0] / sc[est], "  (v4 on %d/%d)" % (n4, len(R)) if n4 is not None else "")))


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(cl.db._splits(sys.argv[1]))
