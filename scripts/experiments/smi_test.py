"""Semi-modular inference (SMI) for PALM, development splits only (docs/palm_uni_plan.md, R2 side test).

The fractions are informed by two modules: composition only (delta = 0) and the full model (eta and delta).
smi_eta in [0, 1] sets how much the full model feeds back into the fractions (bayes.twostage._smi_summaries):
0 = cut with composition-only fractions, 1 = the two-step fit (delta given the full model's fractions).
Per (dataset, split, smi_eta) one PALM fit (BASE settings of dev_bench, NB), then three readouts on the same
truth and scorer as cut_vs_joint.py:
  smi_beta    the SMI model's own beta (posterior median, 90% equal-tailed interval)
  smi_v2      freeze v2 effects (estimator auto) with the SMI fraction draws
  fractions   RMSE of the posterior-median fractions vs the true fractions
Fits saved to benchmarks/dev/smi/<ds>__s<k>__eta<e>.npz (pi, beta draws); scores to benchmarks/dev/smi/*.json.
Usage: python smi_test.py run DATASETS SPLITS ETAS   |   python smi_test.py summary
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import causal_loop as cl

OUT = os.path.join(cl.ROOT, "benchmarks", "dev", "smi")


def _score_beta(e, lo, hi, calls, beta, types, progs, known):
    s = cl.score(e, lo, hi, calls, beta, types, progs, known)
    s["wrong_calls"] = int((calls & ((np.abs(beta) < 0.05) | (np.sign(e) != np.sign(beta)))).sum())
    return s


def run(datasets, splits, etas):
    from program2cell.bayes.twostage import run_palm_fast
    os.makedirs(OUT, exist_ok=True)
    for dsn in datasets:
        for split in splits:
            todo = [e for e in etas if not os.path.exists(os.path.join(OUT, "%s__s%d__eta%.2f.json" % (dsn, split, e)))]
            if not todo:
                continue
            pb, inp, T, beta, truth, _, _, cal, donors = cl._setup(dsn, split)
            types, progs, known = list(inp.cell_types), list(inp.programs), cl.KNOWN.get(dsn)
            pi_true = np.asarray(truth["pi"]) if "pi" in truth else None
            for e in todo:
                t0 = time.time()
                res = run_palm_fast(inp, **{**cl.db.BASE, "likelihood": "nb", "smi_eta": e, "seed": split})
                pi = np.asarray(res.draws["pi"])
                bd = np.asarray(res.draws["beta"])
                keep = np.linspace(0, len(pi) - 1, 50).astype(int)
                np.savez_compressed(os.path.join(OUT, "%s__s%d__eta%.2f.npz" % (dsn, split, e)),
                                    pi=pi[keep], beta=bd[np.linspace(0, len(bd) - 1, 100).astype(int)])
                t_fit = time.time() - t0
                rows = {}
                med, lo, hi = np.median(bd, 0), *np.quantile(bd, [0.05, 0.95], 0)
                rows["smi_beta"] = _score_beta(med, lo, hi, (lo > 0) | (hi < 0), beta, types, progs, known)
                r = pb.program_deconvolution(inp, cal, seed=split, fraction_draws=pi[keep], estimator="auto")
                lo2, hi2 = r.effects_interval
                rows["smi_v2"] = _score_beta(r.effects, lo2, hi2, np.asarray(r.calls, bool), beta, types, progs, known)
                out = dict(dataset=dsn, split=split, smi_eta=e, seconds_fit=t_fit, seconds=time.time() - t0,
                           frac_rmse=float(np.sqrt(((np.median(pi, 0) - pi_true) ** 2).mean()))
                           if pi_true is not None else None, rows=rows)
                with open(os.path.join(OUT, "%s__s%d__eta%.2f.json" % (dsn, split, e)), "w") as fh:
                    json.dump(out, fh, default=float)
                print("[%s] %s s%d eta %.2f  beta cov %.2f r %.3f null %.2f | v2 cov %.2f r %.3f null %.2f | frac %s (%.0fs)"
                      % (time.strftime("%H:%M"), dsn, split, e, rows["smi_beta"]["cov"], rows["smi_beta"]["r"],
                         rows["smi_beta"]["fpr_null"], rows["smi_v2"]["cov"], rows["smi_v2"]["r"],
                         rows["smi_v2"]["fpr_null"], out["frac_rmse"], out["seconds"]), flush=True)


def summary():
    recs = []
    for f in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        d = json.load(open(f))
        for name, s in d["rows"].items():
            recs.append(dict(dataset=d["dataset"], split=d["split"], smi_eta=d["smi_eta"], readout=name,
                             frac_rmse=d["frac_rmse"], **{k: s.get(k) for k in
                                                          ("cov", "r", "width", "calls", "wrong_calls", "fpr_null",
                                                           "tpr_t", "auroc")}))
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(cl.ROOT, "benchmarks", "dev", "smi_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    cols = ["frac_rmse", "cov", "r", "width", "calls", "wrong_calls", "fpr_null", "tpr_t", "auroc"]
    print(df.groupby(["dataset", "readout", "smi_eta"])[cols].mean().round(3).to_string())
    print("\nall datasets:")
    print(df.groupby(["readout", "smi_eta"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(sys.argv[2].split(","), cl.db._splits(sys.argv[3]), [float(x) for x in sys.argv[4].split(",")])
