"""Real bulk with single-cell truth from the same samples (COMBAT; build_combat.py, platform_gap.py).

Per split: make_benchmark on the COMBAT single-cell pseudobulk table (contrast sepsis vs healthy volunteers; test
donors' specimens; reference and calibration from every other donor's single-cell data; truth = the test
specimens' own single-cell per-type program change, from the held-out half). PALM (PALM-lite fractions, causal
default) then runs on three bulks for the same specimens and the same truth:
  sc           the single-cell pseudobulk (the usual assembled bulk)
  real         the real whole-blood total-RNA bulk (raw counts, genes shared with the single-cell data)
  real_offset  the real bulk divided by per-gene platform offsets 2^off_g, off_g = median log2(bulk CPM /
               pseudobulk CPM) over the paired samples of NON-test donors only (no leakage)
  real_selfoffset  the real bulk divided by offsets estimated from the study bulk itself (_self_offsets: no
               paired data needed, the situation of a new cohort)
The real bulk is whole blood, the single-cell reference PBMC: neutrophils and red-cell RNA are in the bulk only,
a missing-cell-type gap on top of the platform gap.
Scores as causal_loop.score (known=None). One json per (contrast, split, bulk) in benchmarks/dev/combat_real/.
Comparator (2026-11-03): InstaPrism on the same bulk and reference rows, its per-type expression scored with the
programs, then the usual Welch t-test per type x program (practice_vs_palm.practice_test; calls BH q < 0.1 and
p < 0.1): one json per (contrast, split, bulk) with suffix __instaprism.
Usage: python combat_real.py run SPLITS [CONTRAST]   (CONTRAST: sepsis (default) | severe)
       python combat_real.py summary
"""
import os, sys, json, glob, time
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
D = os.path.join(ROOT, "data", "combat")
OUT = os.path.join(ROOT, "benchmarks", "dev", "combat_real")
TYPES = ["B", "CD14 Mono", "CD16 Mono", "CD4 T", "CD8 T", "DC", "NK"]
# PALM effects: causal_loop estimator PALM_EST (environment; default auto_abs, the setting of the 2026-11-02 runs)
EST = cl.ESTIMATORS[os.environ.get("PALM_EST", "auto_abs")]
CONTRASTS = {"sepsis": (["Sepsis"], ["HV"], 20),
             "severe": (["COVID_SEV", "COVID_CRIT"], ["HV", "COVID_HCW_MILD", "COVID_MILD"], 40)}


def _cpm(M):
    return np.log2(M / M.sum(1, keepdims=True) * 1e6 + 1)


def _offsets(bulk, sc_table, specs, genes):
    """Per-gene log2 offset bulk vs single-cell pseudobulk over the paired specimens specs."""
    o = sc_table.obs
    X = sp.csr_matrix(sc_table.counts)
    gi = {g: i for i, g in enumerate(sc_table.genes)}
    cols = [gi[g] for g in genes]
    ps = np.vstack([np.asarray(X[np.flatnonzero((o["specimen"] == s).values)].sum(0)).ravel()[cols] for s in specs])
    B = bulk.loc[genes, specs].values.T.astype(float)
    off = np.median(_cpm(B) - _cpm(ps), 0)
    return pd.Series(off, index=genes)


def _self_offsets(Y, ref, iters=4, min_cpm=10):
    """Per-gene log2 offsets estimated from the study bulk alone (program2cell.platform_offsets)."""
    from program2cell import platform_offsets
    return platform_offsets(Y, ref, iters=iters, min_cpm=min_cpm)


def run(splits, contrast="sepsis"):
    import program2cell as pb
    import anndata as ad
    from program2cell.core import PseudobulkTable, make_benchmark
    os.makedirs(OUT, exist_ok=True)
    treated, control, n_test = CONTRASTS[contrast]
    bk = ad.read_h5ad(os.path.join(D, "combat_bulk.h5ad"))
    bk = bk[bk.obs["paired"].values & ~bk.obs_names.duplicated()]
    sc = ad.read_h5ad(os.path.join(D, "combat_sc_pseudobulk.h5ad"))
    arm = np.where(sc.obs["label"].isin(treated), "T", np.where(sc.obs["label"].isin(control), "C", "other"))
    sc.obs["arm"] = np.where(sc.obs["specimen"].isin(set(bk.obs_names)), arm, "other")   # test only paired samples
    path = os.path.join(D, "combat_sc_pseudobulk_%s.h5ad" % contrast)
    if not os.path.exists(path):
        sc.write_h5ad(path)
    table = PseudobulkTable.from_h5ad(path, {}, TYPES)
    bulk_all = pd.DataFrame(sp.csr_matrix(bk.X).toarray().T, index=bk.var_names, columns=bk.obs_names)
    prog = cl.db._programs(pb, "progeny7")
    for split in splits:
        b = make_benchmark(table, cell_types=TYPES, min_cells=20, seed=split, truth_split=True,
                           contrast=("arm", "T", "C"), n_test=n_test)
        specs = list(b.bulk.columns)
        genes = [g for g in b.bulk.index if g in bulk_all.index]
        test_donors = set(b.meta["donor_id"].astype(str))
        other = [s for s in bulk_all.columns if s.split("|")[0] not in test_donors]
        off = _offsets(bulk_all, table, other, genes)
        real = bulk_all.loc[genes, specs]
        corr = (real.T / np.exp2(off.clip(-8, 8)).values).T
        corr = corr / corr.sum(0) * real.sum(0)                                   # keep each sample's depth
        self_c = pb.correct_platform(real, b.reference)
        bulks = {"sc": b.bulk.loc[genes], "real": real, "real_offset": corr.round(), "real_selfoffset": self_c}
        for name, Y in bulks.items():
            if COMPARATOR:
                _instaprism(b, Y, prog, contrast, split, name, table, test_donors)
            f = os.path.join(OUT, "%s__s%d__%s.json" % (contrast, split, name))
            if os.path.exists(f):
                continue
            t0 = time.time()
            inp = pb.prepare_inputs(Y, b.reference, prog, treatment=b.treatment, max_genes=1000)
            T = np.asarray(inp.T)
            beta = pb.core.contrast(T)(b.truth(inp)["delta"])
            cal = pb.calibration_from_table(table, test_donors, inp.cell_types, inp.genes, inp.Gmat, seed=split)
            r = pb.program_deconvolution(inp, cal, engine="lite", treatment=T, seed=split, **EST)
            lo, hi = r.effects_interval
            s = cl.score(r.effects, lo, hi, r.calls, beta, list(inp.cell_types), list(inp.programs), None)
            tr = b.truth(inp)
            s.update(contrast=contrast, split=split, bulk=name, n=int(len(T)), n_treated=int(T.sum()),
                     seconds=time.time() - t0, unmixed=",".join(r.info["unmix"]["programs"]),
                     frac_r=float(np.corrcoef(np.asarray(r.fractions).ravel(), tr["pi"].ravel())[0, 1]),
                     arrays=dict(effects=r.effects.tolist(), lo=lo.tolist(), hi=hi.tolist(),
                                 calls=np.asarray(r.calls).tolist(), beta=beta.tolist(),
                                 cell_types=list(inp.cell_types), programs=list(inp.programs)))
            json.dump(s, open(f, "w"))
            print("[%s] combat %s s%d %-11s cov %.2f r %.2f fpr_null %.3f calls %d frac_r %.2f (%.0fs)" % (
                time.strftime("%H:%M"), contrast, split, name, s["cov"], s["r"], s["fpr_null"], s["calls"],
                s["frac_r"], s["seconds"]), flush=True)


COMPARATOR = os.environ.get("COMBAT_COMPARATOR", "1") == "1"


def _instaprism(b, Y, prog, contrast, split, name, table, test_donors):
    """InstaPrism + Welch t-test on bulk Y (same specimens, genes, truth as PALM's row)."""
    import program2cell as pb
    from program2cell import compare
    import practice_vs_palm as pvp
    f = os.path.join(OUT, "%s__s%d__%s__instaprism.json" % (contrast, split, name))
    if os.path.exists(f):
        return
    t0 = time.time()
    inp = pb.prepare_inputs(Y, b.reference, prog, treatment=b.treatment, max_genes=1000)
    T = np.asarray(inp.T)
    beta = pb.core.contrast(T)(b.truth(inp)["delta"])
    rr = b.reference_rows
    wd = os.path.join(ROOT, "benchmarks", "dev", "work", "combat_" + contrast, "s%d_%s" % (split, name))
    data = compare.CompareData(bulk=Y, sc_counts=rr["counts"], sc_genes=b.reference.genes, sc_cell_types=rr["cell_types"],
                               sc_donors=rr["donors"], inputs=inp, programs=prog, treatment=b.treatment, workdir=wd)
    res, err = compare.run_method("instaprism", data, n_cores=4, max_cells_per_type=1000)
    import shutil
    shutil.rmtree(wd, ignore_errors=True)
    if res is None:
        print("  instaprism FAILED %s s%d %s: %s" % (contrast, split, name, err), flush=True)
        return
    donors = np.asarray(Y.columns).astype(str)              # one specimen per test person: Welch t-test
    eff, lo, hi, pv = pvp.practice_test(np.asarray(res.activity), T, donors)
    rows = []
    for rule, calls in (("p10", pv < 0.1), ("bh10", pvp.bh(pv) < 0.1)):
        sc = cl.score(eff, lo, hi, np.nan_to_num(calls, nan=False).astype(bool), beta, list(inp.cell_types),
                      list(inp.programs), None)
        sc.update(contrast=contrast, split=split, bulk=name, method="instaprism_ttest", rule=rule,
                  seconds=time.time() - t0)
        rows.append(sc)
    json.dump(rows, open(f, "w"), default=float)
    print("[%s] combat %s s%d %-15s InstaPrism + t-test (BH): cov %.2f r %.2f fpr_null %.3f calls %d" % (
        time.strftime("%H:%M"), contrast, split, name, rows[1]["cov"], rows[1]["r"], rows[1]["fpr_null"],
        rows[1]["calls"]), flush=True)


def summary():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        r = json.load(open(f))
        for x in (r if isinstance(r, list) else [dict(r, method="palm", rule="palm")]):
            rows.append({k: v for k, v in x.items() if k != "arrays"})
    df = pd.DataFrame(rows)
    df["set"] = np.where(df.split >= 20, "held-out v2", np.where(df.split >= 10, "held-out v1", "development"))
    key = ["set", "contrast", "bulk", "method", "rule"]
    t = df.groupby(key)[[c for c in ("cov", "r", "rmse", "fpr_null", "calls", "frac_r") if c in df]].mean()
    t.insert(0, "n", df.groupby(key).size())
    print(t.round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(cl.db._splits(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "sepsis")
    else:
        summary()
