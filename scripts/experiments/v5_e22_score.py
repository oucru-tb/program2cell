"""Score plan E22 (unmixed-route widening with realistic simulated responses) against freeze v2 and `sized`
(development, 2026-11-12; docs/weak_points_plan_2026-11-12.md).

Rows: causal_loop jsons <ds>__s<k>__<name>.json for auto_v5u0 (widening, exact simulated responses) and auto_v5u10
(widening, per-type weight jitter sd 1.0); freeze v2 and sized_fx from benchmarks/dev/v5_prior_offline/sims_20-29.csv
(v5_sims_offline.py; sized_fx = estimator auto_v5s exactly). Same splits for all.
Usage: python v5_e22_score.py [splits 20-24]
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from v5_prior_offline import DS, L, OUT, score

NEW = ["auto_v5u0", "auto_v5u10"]


def main():
    a0, a1 = map(int, (sys.argv[1] if len(sys.argv) > 1 else "20-24").split("-"))
    rows = []
    for d in DS:
        for k in range(a0, a1 + 1):
            for n in NEW:
                f = os.path.join(L, "%s__s%d__%s.json" % (d, k, n))
                if not os.path.exists(f):
                    continue
                a = json.load(open(f))["arrays"]
                b, e, lo, hi = (np.asarray(a[x], float) for x in ("beta", "effects", "lo", "hi"))
                rows.append(dict(dataset=d, split=k, method=n, **score(b, e, lo, hi, np.asarray(a["calls"], bool))))
    new = pd.DataFrame(rows)
    old = pd.read_csv(os.path.join(OUT, "sims_20-29.csv"))
    old = old[old.method.isin(["v2", "sized_fx"]) & old.split.between(a0, a1)]
    done = new.groupby(["dataset", "split"]).size()
    done = done[done == len(NEW)].index                     # only (dataset, split) pairs where every variant ran
    df = pd.concat([old, new])
    df = df.set_index(["dataset", "split"]).loc[lambda x: x.index.isin(done)].reset_index()
    order = ["v2", "sized_fx"] + NEW
    K = ["cov", "null_calls", "power", "fdp", "n_calls", "width"]
    m = df.groupby(["dataset", "method"])[K].mean()
    pd.set_option("display.width", 220)
    print("pairs scored: %d (datasets %d)" % (len(done), df.dataset.nunique()))
    print("\n== mean of dataset means")
    print(m.groupby(level=1).mean().reindex(order).round(3).to_string())
    g = m.reset_index().groupby("method")
    s1 = pd.DataFrame(dict(fails=g.apply(lambda x: int(((x["cov"] < 0.85) | (x["null_calls"] > 0.10)).sum())),
                           worst_cov=g["cov"].min(), worst_null=g["null_calls"].max(),
                           failing=g.apply(lambda x: ",".join(x.dataset[(x["cov"] < 0.85) | (x["null_calls"] > 0.10)]))))
    print("\n== S1"); print(s1.reindex(order).round(3).to_string())
    piv = m.reset_index().pivot(index="dataset", columns="method")
    rng = np.random.default_rng(0)
    print("\n== difference vs sized_fx [95% bootstrap CI over datasets]")
    for meth in NEW:
        line = []
        for kk in ("cov", "null_calls", "power", "fdp"):
            dd = (piv[(kk, meth)] - piv[(kk, "sized_fx")]).values
            bs = [rng.choice(dd, len(dd)).mean() for _ in range(4000)]
            line.append("%s %+.3f [%+.3f, %+.3f]" % (kk, dd.mean(), *np.quantile(bs, [0.025, 0.975])))
        print("%-10s %s" % (meth, "  ".join(line)))
    print("\n== per dataset: coverage / null calls / power / FDP")
    for meth in order:
        sub = m.xs(meth, level=1)
        print("%-10s %s" % (meth, "  ".join("%s %.2f/%.2f/%.2f/%.2f" % (d[:9], r["cov"], r["null_calls"], r["power"], r["fdp"])
                                             for d, r in sub.iterrows())))
    df.to_csv(os.path.join(OUT, "e22_%d-%d.csv" % (a0, a1)), index=False)


if __name__ == "__main__":
    main()
