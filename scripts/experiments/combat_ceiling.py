"""Reference numbers for scoring on real bulk (2026-11-06; docs/v4_plan.md section 12): what can a method reach on
COMBAT, before any deconvolution?

Per split (make_benchmark as combat_real.py; truth_split halves A = held-out truth, B = the pseudobulk input):
  truth reliability   per-type program change beta from half A vs half B (sc vs sc, same specimens): overall r over
                      types x programs, within-program r (programs with sd(beta_A) >= 0.05), slope of B on A.
                      A method fed half B that recovered its per-type effects exactly would score this r.
  platform transfer   no deconvolution. Bulk-level program change (estimate_delta on each whole sample, one
                      profile) and per-gene log2 fold change (modelled genes, mean log2 CPM treated - control) of
                        sc_B       the pseudobulk input half          (sc-to-sc ceiling)
                        real       real whole-blood bulk, raw
                        real_off   real bulk / non-test-donor platform offsets (combat_real._offsets)
                      each against the held-out half A summed over types: r, slope (x = half A).
Usage: python combat_ceiling.py SPLITS [CONTRAST]  -> benchmarks/dev/combat_ceiling/{contrast}__s{split}.json
       python combat_ceiling.py summary
"""
import glob, json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import combat_real as cr
import program2cell as pb
from program2cell.core.scores import estimate_delta

OUT = os.path.join(ROOT, "benchmarks", "dev", "combat_ceiling")


def _agree(x, y):
    x, y = np.ravel(x), np.ravel(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    return dict(r=float(np.corrcoef(x, y)[0, 1]), slope=float(np.polyfit(x, y, 1)[0]))


def _within(a, b):
    rs = [np.corrcoef(a[:, p], b[:, p])[0, 1] for p in range(a.shape[1]) if a[:, p].std() >= 0.05]
    return float(np.mean(rs)) if rs else None


def run(splits, contrast="sepsis"):
    import anndata as ad
    from program2cell.core import PseudobulkTable, make_benchmark
    os.makedirs(OUT, exist_ok=True)
    n_test = cr.CONTRASTS[contrast][2]
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
                           contrast=("arm", "T", "C"), n_test=n_test)
        specs = list(b.bulk.columns)
        genes = [g for g in b.bulk.index if g in bulk_all.index]
        test_donors = set(b.meta["donor_id"].astype(str))
        other = [s for s in bulk_all.columns if s.split("|")[0] not in test_donors]
        off = cr._offsets(bulk_all, table, other, genes)
        real = bulk_all.loc[genes, specs]
        corr = (real.T / np.exp2(off.clip(-8, 8)).values).T
        corr = corr / corr.sum(0) * real.sum(0)
        inp = pb.prepare_inputs(b.bulk.loc[genes], b.reference, prog, treatment=b.treatment, max_genes=1000)
        T = np.asarray(inp.T)
        gi = [b.reference.genes.index(g) for g in inp.genes]
        ci = [b.cell_types.index(c) for c in inp.cell_types]
        ZA, ZB = b.truth_counts[:, ci][:, :, gi], b.type_counts[:, ci][:, :, gi]
        B, G = np.asarray(inp.B, float), np.asarray(inp.Gmat, float)
        con = lambda d: d[T == 1].mean(0) - d[T == 0].mean(0)
        bA, bB = con(estimate_delta(ZA, B, G)), con(estimate_delta(ZB, B, G))
        out = dict(contrast=contrast, split=split, n=int(len(T)), programs=list(inp.programs),
                   types=list(inp.cell_types), beta_A=bA.tolist(), beta_B=bB.tolist(),
                   n_clear=int((np.abs(bA) >= 0.1).sum()), sd_beta=float(bA.std()))
        out["reliability"] = dict(_agree(bA, bB), within_r=_within(bA, bB))
        Bm = B.mean(0, keepdims=True)
        mg = list(inp.genes)
        samples = dict(sc_A=ZA.sum(1), sc_B=ZB.sum(1), real=real.loc[mg].values.T.astype(float),
                       real_off=corr.loc[mg].values.T.astype(float))
        prog_b = {k: con(estimate_delta(X[:, None], Bm, G))[0] for k, X in samples.items()}
        lfc = {k: con(np.log2(X / X.sum(1, keepdims=True) * 1e6 + 1)) for k, X in samples.items()}
        out["bulk_program_beta"] = {k: v.tolist() for k, v in prog_b.items()}
        out["transfer"] = {k: dict(program=_agree(prog_b["sc_A"], prog_b[k]), gene=_agree(lfc["sc_A"], lfc[k]))
                           for k in ("sc_B", "real", "real_off")}
        json.dump(out, open(f, "w"))
        t = out["transfer"]
        print("%s s%d clear %d sd %.3f | reliability r %.2f within %.2f slope %.2f | program-level r (slope): "
              "sc %.2f (%.2f) real %.2f (%.2f) off %.2f (%.2f) | gene-level r: sc %.2f real %.2f off %.2f" % (
                  contrast, split, out["n_clear"], out["sd_beta"], out["reliability"]["r"],
                  out["reliability"]["within_r"] or np.nan, out["reliability"]["slope"],
                  t["sc_B"]["program"]["r"], t["sc_B"]["program"]["slope"], t["real"]["program"]["r"],
                  t["real"]["program"]["slope"], t["real_off"]["program"]["r"], t["real_off"]["program"]["slope"],
                  t["sc_B"]["gene"]["r"], t["real"]["gene"]["r"], t["real_off"]["gene"]["r"]), flush=True)


def summary():
    for contrast in ("sepsis", "severe"):
        R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "%s__s*.json" % contrast)))]
        if not R:
            continue
        m = lambda fn: np.nanmean([np.nan if fn(o) is None else fn(o) for o in R])
        print("%s (n splits %d): clear cells %.1f / 49, sd(beta) %.3f" % (
            contrast, len(R), m(lambda o: o["n_clear"]), m(lambda o: o["sd_beta"])))
        print("  truth reliability  r %.2f  within %.2f  slope %.2f" % (
            m(lambda o: o["reliability"]["r"]), m(lambda o: o["reliability"]["within_r"]),
            m(lambda o: o["reliability"]["slope"])))
        for k in ("sc_B", "real", "real_off"):
            print("  transfer %-8s  program r %.2f slope %.2f | gene r %.2f slope %.2f" % (
                k, m(lambda o: o["transfer"][k]["program"]["r"]), m(lambda o: o["transfer"][k]["program"]["slope"]),
                m(lambda o: o["transfer"][k]["gene"]["r"]), m(lambda o: o["transfer"][k]["gene"]["slope"])))
        A = np.concatenate([np.ravel(o["beta_A"]) for o in R])
        Bv = np.concatenate([np.ravel(o["beta_B"]) for o in R])
        c = np.abs(A) >= 0.1
        print("  pooled clear cells (|beta_A| >= 0.1, n %d): r %.2f, sign agreement %.2f" % (
            c.sum(), np.corrcoef(A[c], Bv[c])[0, 1] if c.sum() > 2 else np.nan,
            np.mean(np.sign(A[c]) == np.sign(Bv[c])) if c.sum() else np.nan))


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(cl.db._splits(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "sepsis")
