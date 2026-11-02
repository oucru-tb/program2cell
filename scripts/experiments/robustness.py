"""Robustness of the benchmark conclusions (docs/benchmarking.md "Stress tests"), offline on saved arrays.

(2) A second answer key: the truth scored with decoupler-style ULM (dev_bench truth_ulm, a different activity
    model from PALM's), compared with scale-free metrics only: Spearman r with the estimates, sign agreement on
    the boxes the ULM truth calls clearly changed (top 20% |ULM|), and AUROC of |estimate| for those boxes vs the
    bottom 50%.
(7) Threshold curves: unchanged boxes flagged for "unchanged" = |true| < t0 (t0 = 0.02 ... 0.1), and real changes
    found (right sign) for "real" = |true| >= t1 (t1 = 0.05 ... 0.2).
Methods: PALM (causal_loop auto_m, auto_abs) and InstaPrism + paired t-test (BH q < 0.1, and q < 0.1 with
|change| >= 0.1).
Usage: python robustness.py [DATASETS] [SPLITS]  -> benchmarks/dev/robustness.json
"""
import json, os, sys
import numpy as np
from scipy import stats

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import practice_vs_palm as pv

T0 = (0.02, 0.03, 0.05, 0.075, 0.1)
T1 = (0.05, 0.075, 0.1, 0.15, 0.2)
OUT = os.path.join(ROOT, "benchmarks", "dev", "robustness.json")


def methods(dsn, split, T, donors):
    out = {}
    z = os.path.join(pv.DEV, dsn, "s%d__instaprism.npz" % split)
    if os.path.exists(z):
        e, lo, hi, p = pv.practice_test(np.load(z)["model__activity"], T, donors)
        q = pv.bh(p) < 0.1
        out["InstaPrism + t-test"] = (e, q)
        out["InstaPrism + t-test, cut-off"] = (e, q & (np.abs(e) >= 0.1))
    for name in ("auto_m", "auto_abs"):
        f = os.path.join(cl.OUT, "%s__s%d__%s.json" % (dsn, split, name))
        if os.path.exists(f):
            a = json.load(open(f))["arrays"]
            e, lo, hi = (np.array(a[k], float) for k in ("effects", "lo", "hi"))
            out["PALM" if name == "auto_m" else "PALM (abs)"] = (e, (lo > 0) | (hi < 0))
    return out


def main(datasets="mcf_ida,mcf_tram,kang", splits="0-2"):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for dsn in datasets.split(","):
        for split in cl.db._splits(splits):
            tf = os.path.join(pv.DEV, dsn, "s%d__truth.npz" % split)
            if not os.path.exists(tf):
                continue
            pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
            t = np.load(tf)
            Tt = t["T"]
            ulm = t["ulm"][Tt == 1].mean(0) - t["ulm"][Tt == 0].mean(0)
            hi_u = np.abs(ulm) >= np.quantile(np.abs(ulm), 0.8)
            lo_u = np.abs(ulm) <= np.quantile(np.abs(ulm), 0.5)
            for m, (e, calls) in methods(dsn, split, T, donors).items():
                from sklearn.metrics import roc_auc_score
                row = dict(spearman_ulm=float(stats.spearmanr(e.ravel(), ulm.ravel())[0]),
                           sign_ulm=float((np.sign(e[hi_u]) == np.sign(ulm[hi_u])).mean()),
                           auroc_ulm=float(roc_auc_score(np.r_[np.zeros(lo_u.sum()), np.ones(hi_u.sum())],
                                                         np.r_[np.abs(e[lo_u]), np.abs(e[hi_u])])),
                           spearman=float(stats.spearmanr(e.ravel(), beta.ravel())[0]),
                           unchanged_flagged={str(t0): float(calls[np.abs(beta) < t0].mean())
                                              if (np.abs(beta) < t0).any() else None for t0 in T0},
                           real_found={str(t1): float((calls & (np.sign(e) == np.sign(beta)))[np.abs(beta) >= t1].mean())
                                       if (np.abs(beta) >= t1).any() else None for t1 in T1})
                res.setdefault(dsn, {}).setdefault(str(split), {})[m] = row
            print(dsn, split, flush=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    summary(res)


def summary(res=None):
    import pandas as pd
    res = res or json.load(open(OUT))
    rows = []
    for dsn, sp in res.items():
        for split, ms in sp.items():
            for m, r in ms.items():
                d = dict(dataset=dsn, split=split, method=m, spearman=r["spearman"], spearman_ulm=r["spearman_ulm"],
                         sign_ulm=r["sign_ulm"], auroc_ulm=r["auroc_ulm"])
                d.update({"unch<%s" % k: v for k, v in r["unchanged_flagged"].items()})
                d.update({"real>=%s" % k: v for k, v in r["real_found"].items()})
                rows.append(d)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.groupby(["dataset", "method"]).mean(numeric_only=True).round(2).to_string())
    return df


if __name__ == "__main__":
    main(*sys.argv[1:3])
