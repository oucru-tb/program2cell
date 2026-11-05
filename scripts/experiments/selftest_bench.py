"""Self-tests on benchmark splits (v4 development, idea register 3 and 4; program2cell.selftest).

Per dataset x split: the programs are replaced by 3 sets of matched null programs and by one marker program
per cell type; inputs (gene selection included), calibration and truth are rebuilt on each, and the
estimator is rerun with PALM-lite fractions. Scored both ways: in-study (no truth: call rates) and against
the truth (are the nulls null? is the leakage real?). A real-program run with the same settings is included
as the reference point.
Usage: python selftest_bench.py DATASETS SPLITS [ESTIMATOR=auto_v2]
       -> benchmarks/dev/selftest/{ds}__s{split}__{estimator}.json
"""
import json, os, sys, time
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import program2cell as pb
from program2cell import selftest as st

OUT = os.path.join(ROOT, "benchmarks", "dev", "selftest")


def _analyse(table, ds, b, prog, split, name):
    inp = pb.prepare_inputs(b.bulk, b.reference, prog, treatment=b.treatment, max_genes=1000)
    T = np.asarray(inp.T)
    cal = pb.calibration_from_table(table, set(b.meta["donor_id"].astype(str)), inp.cell_types, inp.genes,
                                    inp.Gmat, ref_filter=ds.get("ref_filter"),
                                    all_specimens=ds.get("all_test_specimens", False), seed=split)
    kw = dict(cl.ESTIMATORS[name])
    if kw.pop("paired", False):
        kw["groups"] = b.meta["donor_id"].astype(str).values
    r = pb.program_deconvolution(inp, cal, seed=split, engine="lite", **kw)
    truth = b.truth(inp)
    beta = np.asarray(pb.core.contrast(T)(truth["delta"]))
    lo, hi = (np.asarray(x) for x in r.effects_interval)
    pi = np.asarray(truth["pi"])
    return dict(programs=list(inp.programs), effects=np.asarray(r.effects).tolist(), lo=lo.tolist(),
                hi=hi.tolist(), calls=np.asarray(r.calls).tolist(), beta=beta.tolist(),
                cov=float(((beta >= lo) & (beta <= hi)).mean()),
                pi_shift=(pi[T == 1].mean(0) - pi[T == 0].mean(0)).tolist()), inp


def run(datasets, splits, name="auto_v2"):
    os.makedirs(OUT, exist_ok=True)
    for dsn in datasets:
        ds = cl.db.DATASETS[dsn]
        table = cl.db._table(ds)
        prog = cl.db._programs(pb, "progeny7")
        for split in splits:
            f = os.path.join(OUT, "%s__s%d__%s.json" % (dsn, split, name))
            if os.path.exists(f):
                continue
            t0 = time.time()
            b = cl.db._benchmark(table, ds, split)
            out = dict(dataset=dsn, split=split, estimator=name, null=[])
            out["real"], inp = _analyse(table, ds, b, prog, split, name)
            out["cell_types"] = list(inp.cell_types)
            for tab in st.null_programs(prog, b.bulk, b.reference, n_sets=3, seed=split):
                s, _ = _analyse(table, ds, b, tab, split, name)
                c, e, bt = (np.asarray(s[k], float) for k in ("calls", "effects", "beta"))
                s.update(st.null_summary(c.astype(bool), e), truth_mean_abs=float(np.abs(bt).mean()),
                         truth_share_big=float((np.abs(bt) >= 0.1).mean()),
                         calls_on_truth_null=float(c[np.abs(bt) < 0.05].mean()) if (np.abs(bt) < 0.05).any() else None)
                out["null"].append(s)
                print("%s s%d %s null: calls %.3f (truth-null pairs %.3f) truth |b| %.3f cov %.2f" % (
                    dsn, split, name, s["null_call_rate"], s["calls_on_truth_null"] or np.nan, s["truth_mean_abs"],
                    s["cov"]), flush=True)
            s, inp_m = _analyse(table, ds, b, st.marker_programs(b.reference, b.bulk), split, name)
            owner = st.marker_owner(s["programs"], inp_m.cell_types)
            c, e, bt = (np.asarray(s[k], float) for k in ("calls", "effects", "beta"))
            s.update(st.marker_summary(c.astype(bool), e, owner), owner=owner.tolist())
            tr = st.marker_summary(np.zeros_like(c, bool), bt, owner)
            s.update(truth_leak_mean_abs=tr["leak_mean_abs"], truth_own_mean_abs=tr["own_mean_abs"])
            out["marker"] = s
            print("%s s%d %s marker: leak calls %.3f |eff| %.3f (truth %.3f); own calls %.2f |eff| %.3f (truth %.3f); "
                  "cov %.2f  %.0fs" % (dsn, split, name, s["leak_call_rate"], s["leak_mean_abs"],
                                       s["truth_leak_mean_abs"], s["own_call_rate"], s["own_mean_abs"],
                                       s["truth_own_mean_abs"], s["cov"], time.time() - t0), flush=True)
            json.dump(out, open(f, "w"))


if __name__ == "__main__":
    run(sys.argv[1].split(","), cl.db._splits(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "auto_v2")
