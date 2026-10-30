"""Real-truth validation on GSE60424 (Linsley et al. 2014): whole blood + sorted cells from the same donors.

Experimental design
- Data: 20 donors, RNA-seq of whole blood (Tempus) and of sorted neutrophils, monocytes, B, CD4 and CD8 T
  cells from the same blood draw (NK cells in 14 donors only: not modelled; they remain in whole blood as an
  unmodelled type, like eosinophils and platelets). NCBI-generated raw counts (GRCh38), globin genes removed.
- Contrasts: sepsis vs healthy (3 vs 4); MS after vs before interferon-beta (3 vs 3; treated as unpaired,
  the GEO donor ids differ); negative-control contrasts T1D vs healthy (4 vs 4) and ALS vs healthy (3 vs 4),
  where little is expected in blood.
- Bulk = whole blood of the contrast's donors. Truth = the same donors' sorted cells: per donor and type,
  program activity (estimate_delta against the reference profiles), per type the difference of arm means;
  its uncertainty by bootstrapping donors within arms (90% interval). A truth call = interval excluding 0.
- Reference profiles and PALM's calibration: sorted cells of the OTHER donors only (no test donor's cells).
- Programs: PROGENy, 7 pathways (as the benchmarks).
- Methods: PALM (full fractions, default effects 'auto_m'; and 'auto_abs'), InstaPrism (sorted reference
  samples as reference "cells"; per-type expression -> activity -> Welch t-test, BH q < 0.1), NNLS
  reallocation + t-test, bulk score without deconvolution + t-test.
- Metrics per method over type x program pairs: r with the true change; negative control = pairs whose
  truth interval includes 0 and |true| < 0.05, flagged; positive control = pairs with a truth call, flagged
  with the right sign; coverage of the method's 90% range.

Usage: python real_truth_gse60424.py run [CONTRASTS]   |   python real_truth_gse60424.py summary
Writes benchmarks/real/gse60424/{contrast}.json
"""
import gzip, json, os, sys, time
import numpy as np, pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
DATA = os.path.join(ROOT, "data", "real", "gse60424")
OUT = os.path.join(ROOT, "benchmarks", "real", "gse60424")
TYPES = ["Neutrophils", "Monocytes", "B-cells", "CD4", "CD8"]
CONTRASTS = {"sepsis": ("Sepsis", "Healthy Control"), "ms_ifnb": ("MS posttreatment", "MS pretreatment"),
             "t1d": ("Type 1 Diabetes", "Healthy Control"), "als": ("ALS", "Healthy Control")}


def load():
    lines = [l.rstrip("\n").split("\t") for l in gzip.open(os.path.join(DATA, "GSE60424_series_matrix.txt.gz"), "rt")
             if l.startswith("!Sample_")]
    acc = [x.strip('"') for x in next(l for l in lines if l[0] == "!Sample_geo_accession")[1:]]
    meta = pd.DataFrame(index=acc)
    for l in lines:
        if l[0] == "!Sample_characteristics_ch1":
            v = [x.strip('"') for x in l[1:]]
            meta[v[0].split(":")[0]] = [x.split(": ", 1)[1] if ": " in x else x for x in v]
    counts = pd.read_csv(os.path.join(DATA, "GSE60424_raw_counts_GRCh38.p13_NCBI.tsv.gz"), sep="\t", index_col=0)
    info = pd.read_csv(os.path.join(DATA, "Homo_sapiens.gene_info.gz"), sep="\t", usecols=["GeneID", "Symbol"])
    sym = info.set_index("GeneID").Symbol.reindex(counts.index)
    counts = counts[sym.notna().values].groupby(sym.dropna().values).sum()
    globin = [g for g in counts.index if g.startswith(("HBA", "HBB", "HBD", "HBG", "HBM", "HBQ", "HBZ", "HBE"))]
    counts = counts.drop(index=globin)
    return counts, meta


def build(counts, meta, contrast):
    import palm_bulk as pb
    from palm_bulk.core.data import Reference
    from palm_bulk.core.calibrate import build_calibration_set
    import dev_bench as db
    a, b = CONTRASTS[contrast]
    test = meta[meta.diseasestatus.isin([a, b])].donorid.unique()
    ref_donors = [d for d in meta.donorid.unique() if d not in set(test)]
    wb = meta[(meta.celltype == "Whole Blood") & meta.donorid.isin(test)]
    bulk = counts[wb.index]
    T = (wb.diseasestatus == a).astype(int).values

    def sorted_counts(donors):
        out = np.zeros((len(donors), len(TYPES), len(counts)))
        for i, d in enumerate(donors):
            for j, t in enumerate(TYPES):
                s = meta[(meta.donorid == d) & (meta.celltype == t)].index
                out[i, j] = counts[list(s)].sum(1).values
        return out

    R = sorted_counts(ref_donors)                                 # (D, C, G) reference donors
    Bfull = ((R + 1.0) / (R + 1.0).sum(-1, keepdims=True)).mean(0)   # pseudocount: no zero baseline
    ref = Reference(B=Bfull, genes=list(counts.index), cell_types=list(TYPES), rna_content=np.ones(len(TYPES)),
                    n_cells=np.full(len(TYPES), len(ref_donors)), n_donors=np.full(len(TYPES), len(ref_donors)))
    prog = db._programs(pb, "progeny7")
    inp = pb.prepare_inputs(bulk, ref, prog, treatment=pd.Series(T, index=bulk.columns), max_genes=1000)
    gi = [list(counts.index).index(g) for g in inp.genes]
    cal = build_calibration_set(R[:, :, gi], np.array(ref_donors), inp.Gmat, n_folds=5, seed=0)
    S = sorted_counts(list(wb.donorid))[:, :, gi]                # test donors' sorted cells (truth)
    return inp, cal, T, S, list(wb.donorid), R, ref_donors


def truth(S, inp, T, level=0.9):
    """Per-donor activity of the sorted cells and the per-type difference of arm means with a Welch t
    interval (a bootstrap over 3-4 donors per arm is far too narrow)."""
    from scipy import stats
    from palm_bulk.core.scores import estimate_delta
    d = estimate_delta(S, np.maximum(inp.B, 1e-8), inp.Gmat)      # (N, C, P)
    a, b = d[T == 1], d[T == 0]
    beta = a.mean(0) - b.mean(0)
    va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
    df = (va + vb) ** 2 / np.maximum(va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1), 1e-30)
    h = stats.t.ppf(0.5 + level / 2, df) * np.sqrt(va + vb)
    return d, beta, beta - h, beta + h


def instaprism_activity(inp, R, ref_donors, bulk_samples, Y_full, counts):
    """InstaPrism with the reference donors' sorted samples as reference 'cells'; per-type expression ->
    activity (estimate_delta), and its fractions."""
    import scipy.io, scipy.sparse, subprocess
    from palm_bulk.compare.methods import safe_names
    from palm_bulk.core.scores import estimate_delta
    d = os.path.join(OUT, "work", "instaprism_" + "_".join(map(str, bulk_samples[:2])))
    os.makedirs(d, exist_ok=True)
    fwd, back = safe_names(list(TYPES), "ct")
    X = R.reshape(-1, R.shape[-1]).T                              # genes x (donor, type)
    labels = [fwd[t] for _ in ref_donors for t in TYPES]
    scipy.io.mmwrite(os.path.join(d, "sc_counts.mtx"), scipy.sparse.csr_matrix(X))
    pd.Series(list(counts.index)).to_csv(os.path.join(d, "sc_genes.txt"), index=False, header=False)
    pd.Series(labels).to_csv(os.path.join(d, "sc_labels.txt"), index=False, header=False)
    Y_full.to_csv(os.path.join(d, "bulk_counts.csv"))
    rs = os.environ.get("PALM_RSCRIPT", "C:/Users/duy.pham/AppData/Local/miniforge3/envs/palm-r/Lib/R/bin/x64/Rscript.exe")
    P = "C:/Users/duy.pham/AppData/Local/miniforge3/envs/palm-r"
    env = dict(os.environ, PATH=os.pathsep.join([P + "/Lib/R/bin/x64", P, P + "/Library/mingw-w64/bin",
                                                 P + "/Library/bin", os.environ["PATH"]]))
    subprocess.run([rs, os.path.join(ROOT, "src", "palm_bulk", "compare", "R", "run_instaprism.R"), d, "4"],
                   check=True, env=env, capture_output=True)
    theta = pd.read_csv(os.path.join(d, "theta.csv"), index_col=0).rename(columns=back)
    Z = np.stack([pd.read_csv(os.path.join(d, "Z_%s.csv" % fwd[t]), index_col=0)
                  .reindex(index=Y_full.columns, columns=inp.genes).fillna(0).values for t in TYPES], 1)
    return estimate_delta(Z, inp.B, inp.Gmat), theta.reindex(index=Y_full.columns, columns=TYPES).values


def score(eff, lo, hi, calls, beta, tlo, thi):
    tcall = (tlo > 0) | (thi < 0)
    null = ~tcall & (np.abs(beta) < 0.05)
    hit = calls & (np.sign(eff) == np.sign(beta))
    f = lambda m, x: float(x[m].mean()) if m.any() else np.nan
    return dict(r=float(np.corrcoef(eff.ravel(), beta.ravel())[0, 1]), neg_flagged=f(null, calls),
                pos_found=f(tcall, hit), cov=float(((beta >= lo) & (beta <= hi)).mean()) if lo is not None else np.nan,
                n_null=int(null.sum()), n_truth_calls=int(tcall.sum()), calls=int(calls.sum()))


def run(contrasts):
    import palm_bulk as pb
    import practice_vs_palm as pv
    from palm_bulk.core.realloc import reallocate
    from palm_bulk.compare.methods import nnls_fractions
    os.makedirs(OUT, exist_ok=True)
    counts, meta = load()
    for con in contrasts:
        t0 = time.time()
        inp, cal, T, S, donors, R, ref_donors = build(counts, meta, con)
        d_true, beta, tlo, thi = truth(S, inp, T)
        res = dict(contrast=con, arms=CONTRASTS[con], n=[int((T == 1).sum()), int((T == 0).sum())],
                   cell_types=TYPES, programs=list(inp.programs), beta=beta.tolist(), truth_lo=tlo.tolist(),
                   truth_hi=thi.tolist(), methods={})
        for name, kw in (("PALM", dict(estimator="auto", unmix_kw=dict(pev=True, misfit=True))),
                         ("PALM_abs", dict(estimator="auto", sim_widen="abs", unmix_kw=dict(pev=True, misfit=True)))):
            r = pb.program_deconvolution(inp, cal, engine="palm", seed=0, **kw)
            lo, hi = r.effects_interval
            if name == "PALM":                                    # design calculator on this design
                import identifiability as idf
                from palm_bulk.core.realloc import reallocate as _re
                dc = idf.design_calculator(r.fractions, inp.B, inp.Gmat, T, _re(inp.Y, r.fractions, inp.B, inp.Gmat))
                res["mde"] = dc["mde_full"].tolist()
                res["truth_detectable"] = int((np.abs(beta) >= dc["mde_full"]).sum())
            res["methods"][name] = dict(score(r.effects, lo, hi, r.calls, beta, tlo, thi), effects=r.effects.tolist(),
                                        lo=lo.tolist(), hi=hi.tolist(), call_matrix=r.calls.tolist(),
                                        fractions=r.fractions.mean(0).tolist())
        acts = {"NNLS + t-test": reallocate(inp.Y, nnls_fractions(inp.Y, inp.B), inp.B, inp.Gmat),
                "Bulk score + t-test": np.repeat(reallocate(inp.Y, np.full((len(T), len(TYPES)), 1 / len(TYPES)),
                                                            inp.B, inp.Gmat).mean(1, keepdims=True), len(TYPES), 1)}
        try:
            wb = meta[(meta.celltype == "Whole Blood") & meta.donorid.isin(donors)].index
            acts["InstaPrism + t-test"], fr_ip = instaprism_activity(inp, R, ref_donors, list(wb), counts[wb], counts)
        except Exception as e:
            print("InstaPrism failed:", repr(e)[:300])
        for name, act in acts.items():
            eff, lo, hi, p = pv.practice_test(act, T, np.arange(len(T)).astype(str))
            calls = pv.bh(p) < 0.1
            res["methods"][name] = dict(score(eff, lo, hi, calls, beta, tlo, thi), effects=eff.tolist(),
                                        lo=lo.tolist(), hi=hi.tolist(), call_matrix=calls.tolist())
        res["seconds"] = time.time() - t0
        with open(os.path.join(OUT, con + ".json"), "w") as fh:
            json.dump(res, fh, default=float)
        print("== %s (%d vs %d): truth calls %d of %d; identifiable (|true| >= MDE) %s; median MDE %.2f" % (
            con, *res["n"], int(((tlo > 0) | (thi < 0)).sum()), beta.size, res.get("truth_detectable"),
            np.median(res["mde"]) if "mde" in res else np.nan))
        for m, s in res["methods"].items():
            print("  %-22s r %5.2f  unchanged flagged %s  real found %s  coverage %s  calls %d" % (
                m, s["r"], _f(s["neg_flagged"]), _f(s["pos_found"]), _f(s["cov"]), s["calls"]), flush=True)


def _f(x):
    return "  - " if x is None or not np.isfinite(x) else "%.2f" % x


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2].split(",") if len(sys.argv) > 2 else list(CONTRASTS))
