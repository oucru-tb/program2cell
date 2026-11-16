"""Joint posterior vs cut, head to head on development splits (docs/methods_cut_model.md section 3).

Three effect estimators on the same datasets, splits, truth and scorer (causal_loop.score):
  joint      the joint PALM model's own beta: posterior median, 90% equal-tailed interval from its draws
             (saved NB fit benchmarks/dev/<ds>/s<k>__joint_nb.npz, key "beta")
  cut_raw    cut without calibration: treated-minus-control contrast of reallocation on each posterior draw
             of pi (key "realloc_draws"), median and 90% interval over draws
  cut_v2     cut + calibration + unmixing: freeze v2 (causal_loop estimator auto_v2, run with the same pi draws)
Calls: the 90% interval excludes 0, for all three.
Development splits only (0-2); no held-out split is read.
Usage: python cut_vs_joint.py run DATASETS SPLITS     (runs the missing auto_v2 jsons via causal_loop)
       python cut_vs_joint.py score DATASETS SPLITS   (writes benchmarks/dev/cut_vs_joint.csv and prints a summary)
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import causal_loop as cl

DEV = os.path.join(cl.ROOT, "benchmarks", "dev")


def _T(dsn, split):
    import program2cell as pb
    ds = cl.db.DATASETS[dsn]
    b = cl.db._benchmark(cl.db._table(ds), ds, split)
    inp = pb.prepare_inputs(b.bulk, b.reference, cl.db._programs(pb, "progeny7"), treatment=b.treatment,
                            max_genes=1000)
    return np.asarray(inp.T), list(inp.cell_types), list(inp.programs)


def _interval(draws, level=0.9):
    a = (1 - level) / 2
    lo, hi = np.quantile(draws, [a, 1 - a], axis=0)
    return np.median(draws, axis=0), lo, hi


def score(datasets, splits):
    from program2cell.core.calibrate import contrast
    rows = []
    for dsn in datasets:
        for split in splits:
            z_path = os.path.join(DEV, dsn, "s%d__joint_nb.npz" % split)
            v2_path = os.path.join(cl.OUT, "%s__s%d__auto_v2.json" % (dsn, split))
            if not (os.path.exists(z_path) and os.path.exists(v2_path)):
                print("skip %s s%d (missing %s)" % (dsn, split, "joint fit" if not os.path.exists(z_path) else "auto_v2"))
                continue
            v2 = json.load(open(v2_path))
            a = v2["arrays"]
            beta = np.asarray(a["beta"], float)
            T, types, progs = _T(dsn, split)
            assert types == a["cell_types"] and progs == a["programs"], (dsn, split)
            z = np.load(z_path)
            assert z["pi"].shape[1] == len(T), (dsn, split, z["pi"].shape, len(T))
            known = cl.KNOWN.get(dsn)
            ests = {"joint": _interval(z["beta"]),
                    "cut_raw": _interval(contrast(T)(z["realloc_draws"]))}
            ests["cut_v2"] = tuple(np.asarray(a[k], float) for k in ("effects", "lo", "hi"))
            for name, (e, lo, hi) in ests.items():
                calls = np.asarray(a["calls"], bool) if name == "cut_v2" else (lo > 0) | (hi < 0)
                s = cl.score(e, lo, hi, calls, beta, types, progs, known)
                # wrong calls: on a near-null pair (|beta| < 0.05) or with the wrong sign
                wrong = calls & ((np.abs(beta) < 0.05) | (np.sign(e) != np.sign(beta)))
                s.update(dataset=dsn, split=split, estimator=name, n_null=int((np.abs(beta) < 0.05).sum()),
                         wrong_calls=int(wrong.sum()))
                rows.append(s)
    df = pd.DataFrame(rows)
    out = os.path.join(DEV, "cut_vs_joint.csv")
    df.to_csv(out, index=False)
    cols = ["cov", "r", "width", "calls", "wrong_calls", "fpr_null", "tpr_t", "auroc"]
    cols = [c for c in cols if c in df]
    pd.set_option("display.width", 200)
    print(df.groupby(["dataset", "estimator"], sort=False)[cols].mean().round(3).to_string())
    print("\nall datasets (mean over dataset x split):")
    print(df.groupby("estimator", sort=False)[cols].mean().round(3).to_string())
    print("\nwritten", out)


if __name__ == "__main__":
    ds, sp = sys.argv[2].split(","), cl.db._splits(sys.argv[3])
    if sys.argv[1] == "run":
        cl.run(ds, sp, ["auto_v2"])
    else:
        score(ds, sp)
