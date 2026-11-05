"""Self-tests on COMBAT real bulk (v4 development, idea register 3 and 4; program2cell.selftest).

The assembled benchmarks barely move composition with treatment, so the marker-program leakage test is weak
there. COMBAT sepsis vs healthy (combat_real.py setup) is real whole-blood bulk whose composition does move
(neutrophils, absent from the PBMC reference). Per split and bulk (sc pseudobulk / real / real_selfoffset):
the real programs, 3 matched null-program sets and the marker programs, each with gene selection, calibration
and truth rebuilt; estimator v2 (PALM_EST, default auto_v2), PALM-lite fractions.
Usage: python combat_selftest.py SPLITS -> benchmarks/dev/selftest_combat/sepsis__s{split}__{bulk}.json
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import combat_real as cr
import program2cell as pb
from program2cell import selftest as st

OUT = os.path.join(ROOT, "benchmarks", "dev", "selftest_combat")
NAME = os.environ.get("PALM_EST", "auto_v2")


def _analyse(Y, b, table, prog, split, test_donors):
    inp = pb.prepare_inputs(Y, b.reference, prog, treatment=b.treatment, max_genes=1000)
    T = np.asarray(inp.T)
    cal = pb.calibration_from_table(table, test_donors, inp.cell_types, inp.genes, inp.Gmat, seed=split)
    r = pb.program_deconvolution(inp, cal, engine="lite", treatment=T, seed=split, **cl.ESTIMATORS[NAME])
    beta = np.asarray(pb.core.contrast(T)(b.truth(inp)["delta"]))
    lo, hi = (np.asarray(x) for x in r.effects_interval)
    pf = np.asarray(r.fractions)
    return dict(programs=list(inp.programs), effects=np.asarray(r.effects).tolist(), lo=lo.tolist(),
                hi=hi.tolist(), calls=np.asarray(r.calls).tolist(), beta=beta.tolist(),
                cov=float(((beta >= lo) & (beta <= hi)).mean()),
                frac_shift=(pf[T == 1].mean(0) - pf[T == 0].mean(0)).tolist()), inp


def run(splits, contrast="sepsis"):
    import anndata as ad
    from program2cell.core import PseudobulkTable, make_benchmark
    os.makedirs(OUT, exist_ok=True)
    treated, control, n_test = cr.CONTRASTS[contrast]
    bk = ad.read_h5ad(os.path.join(cr.D, "combat_bulk.h5ad"))
    bk = bk[bk.obs["paired"].values & ~bk.obs_names.duplicated()]
    table = PseudobulkTable.from_h5ad(os.path.join(cr.D, "combat_sc_pseudobulk_%s.h5ad" % contrast), {}, cr.TYPES)
    bulk_all = pd.DataFrame(sp.csr_matrix(bk.X).toarray().T, index=bk.var_names, columns=bk.obs_names)
    prog = cl.db._programs(pb, "progeny7")
    for split in splits:
        b = make_benchmark(table, cell_types=cr.TYPES, min_cells=20, seed=split, truth_split=True,
                           contrast=("arm", "T", "C"), n_test=n_test)
        specs = list(b.bulk.columns)
        genes = [g for g in b.bulk.index if g in bulk_all.index]
        test_donors = set(b.meta["donor_id"].astype(str))
        real = bulk_all.loc[genes, specs]
        bulks = {"sc": b.bulk.loc[genes], "real": real, "real_selfoffset": pb.correct_platform(real, b.reference)}
        for bname, Y in bulks.items():
            f = os.path.join(OUT, "%s__s%d__%s.json" % (contrast, split, bname))
            if os.path.exists(f):
                continue
            t0 = time.time()
            out = dict(contrast=contrast, split=split, bulk=bname, estimator=NAME, null=[])
            out["real"], inp = _analyse(Y, b, table, prog, split, test_donors)
            out["cell_types"] = list(inp.cell_types)
            for tab in st.null_programs(prog, Y, b.reference, n_sets=3, seed=split):
                s, _ = _analyse(Y, b, table, tab, split, test_donors)
                c, e, bt = (np.asarray(s[k], float) for k in ("calls", "effects", "beta"))
                s.update(st.null_summary(c.astype(bool), e), truth_mean_abs=float(np.abs(bt).mean()))
                out["null"].append(s)
                print("combat %s s%d %-15s null: calls %.3f floor %.3f |eff| %.3f truth |b| %.3f cov %.2f" % (
                    contrast, split, bname, s["null_call_rate"], (c.astype(bool) & (np.abs(e) >= 0.1)).mean(),
                    s["null_mean_abs"], s["truth_mean_abs"], s["cov"]), flush=True)
            s, inp_m = _analyse(Y, b, table, st.marker_programs(b.reference, Y), split, test_donors)
            owner = st.marker_owner(s["programs"], inp_m.cell_types)
            c, e, bt = (np.asarray(s[k], float) for k in ("calls", "effects", "beta"))
            s.update(st.marker_summary(c.astype(bool), e, owner), owner=owner.tolist())
            out["marker"] = s
            print("combat %s s%d %-15s marker: leak calls %.3f |eff| %.3f; own calls %.2f |eff| %.3f; cov %.2f; "
                  "real cov %.2f calls %d  %.0fs" % (contrast, split, bname, s["leak_call_rate"], s["leak_mean_abs"],
                                                    s["own_call_rate"], s["own_mean_abs"], s["cov"],
                                                    out["real"]["cov"], int(np.sum(out["real"]["calls"])),
                                                    time.time() - t0), flush=True)
            json.dump(out, open(f, "w"))


if __name__ == "__main__":
    run(cl.db._splits(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "sepsis")
