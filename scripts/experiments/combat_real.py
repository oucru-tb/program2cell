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
    """Per-gene log2 offsets estimated from the study bulk alone (no paired data): alternate PALM-lite fractions
    on the offset-adjusted reference and b_g += median over samples of log2(observed / predicted CPM), on genes
    expressed in the bulk. One offset per gene for all samples, so a treatment difference is not absorbed
    (it moves the median only through the treated share of samples)."""
    from palm_bulk.core.realloc import em_fractions
    ri = {g: i for i, g in enumerate(ref.genes)}
    genes = [g for g in Y.index if g in ri]
    Bm = ref.B[:, [ri[g] for g in genes]] + 1e-9
    V = Y.loc[genes].values.T.astype(float)                                  # N x G
    cpm = V / V.sum(1, keepdims=True) * 1e6
    expr = cpm.mean(0) > min_cpm
    b = np.zeros(len(genes))
    for _ in range(iters):
        Bb = Bm * np.exp2(b)[None]
        Bb = Bb / Bb.sum(1, keepdims=True)
        pi = em_fractions(V[:, expr], Bb[:, expr])
        q = pi @ Bb
        q = q / q.sum(1, keepdims=True) * 1e6
        step = np.median(np.log2(cpm + 1) - np.log2(q + 1), 0)
        b = np.where(expr, b + step, b)
    return pd.Series(b, index=genes)


def run(splits, contrast="sepsis"):
    import palm_bulk as pb
    import anndata as ad
    from palm_bulk.core import PseudobulkTable, make_benchmark
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
        so = _self_offsets(real, b.reference)
        self_c = (real.loc[so.index].T / np.exp2(so.clip(-8, 8)).values).T
        self_c = self_c / self_c.sum(0) * real.loc[so.index].sum(0)
        bulks = {"sc": b.bulk.loc[genes], "real": real, "real_offset": corr.round(), "real_selfoffset": self_c.round()}
        for name, Y in bulks.items():
            f = os.path.join(OUT, "%s__s%d__%s.json" % (contrast, split, name))
            if os.path.exists(f):
                continue
            t0 = time.time()
            inp = pb.prepare_inputs(Y, b.reference, prog, treatment=b.treatment, max_genes=1000)
            T = np.asarray(inp.T)
            beta = pb.core.contrast(T)(b.truth(inp)["delta"])
            cal = pb.calibration_from_table(table, test_donors, inp.cell_types, inp.genes, inp.Gmat, seed=split)
            r = pb.program_deconvolution(inp, cal, engine="lite", treatment=T, seed=split, estimator="auto",
                                         sim_widen="abs", unmix_kw=dict(pev=True, misfit=True))
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


def summary():
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "arrays"} for r in rows])
    print(df.groupby(["contrast", "bulk"])[["cov", "r", "rmse", "fpr_null", "calls", "frac_r"]].mean().round(3))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(cl.db._splits(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "sepsis")
    else:
        summary()
