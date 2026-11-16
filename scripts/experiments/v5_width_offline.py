"""Offline test: cross-study calibration of the interval width (development, 2026-11-11).

Finding behind it (v5_prior_offline.py, splits 20-29): freeze v2's 90% intervals cover 0.98 of the truths on
average (1.00 on Parse, Kang, Zhang), so they are wider than needed and width limits power.

Method: scale each interval about its estimate, lo' = e - k (e - lo), hi' = e + k (hi - e). k is chosen by
leave-one-study-out (5 studies: McFarland, Tahoe, Kang, Zhang, Parse): for the held-out study, k = the smallest
grid value whose pooled coverage on the OTHER studies' development runs is >= TARGET and whose null-call rate is
<= 0.10 on every other dataset. The held-out study is then scored at that k. Two variants:
  global    one k for every pair
  route     separate k for the unmixed and calibrated routes (j["unmixed"] lists the unmixed programs)
In deployment k would come from all benchmark studies; transport to a new study type is the assumption, which the
leave-one-study-out scores measure.
Usage: python v5_width_offline.py [splits 20-29] [target 0.90]
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from v5_prior_offline import DS, L, OUT, score

STUDY = {d: ("mcfarland" if d.startswith("mcf") else d.split("_")[0]) for d in DS}
GRID = np.round(np.arange(0.30, 1.001, 0.05), 2)


def load(splits):
    runs = []
    for d in DS:
        for k in splits:
            f = os.path.join(L, "%s__s%d__auto_v2.json" % (d, k))
            if os.path.exists(f):
                j = json.load(open(f))
                a = {x: np.array(j["arrays"][x], float) for x in ("effects", "lo", "hi", "beta")}
                um = set((j.get("unmixed") or "").split(","))
                a["umask"] = np.array([[p in um for p in j["arrays"]["programs"]]] * a["effects"].shape[0])
                runs.append((d, k, a))
    return runs


def scaled(a, k_un, k_cal):
    e, lo, hi = a["effects"], a["lo"], a["hi"]
    k = np.where(a["umask"], k_un, k_cal)
    return e - k * (e - lo), e + k * (hi - e)


def ds_scores(runs, k_un, k_cal):
    rows = []
    for d, s, a in runs:
        lo, hi = scaled(a, k_un, k_cal)
        rows.append(dict(dataset=d, split=s, **score(a["beta"], a["effects"], lo, hi, (lo > 0) | (hi < 0))))
    return pd.DataFrame(rows).groupby("dataset").mean(numeric_only=True)


def ok(m, target):
    return (m["cov"].mean() >= target) and (m["null_calls"].max() <= 0.10)


def main():
    a0, a1 = map(int, (sys.argv[1] if len(sys.argv) > 1 else "20-29").split("-"))
    target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
    runs = load(range(a0, a1 + 1))
    studies = sorted(set(STUDY.values()))
    # precompute dataset scores on the grid
    cache = {}
    for ku in GRID:
        for kc in GRID:
            cache[(ku, kc)] = ds_scores(runs, ku, kc)
    out, chosen = [], []
    for st in studies:
        others = [d for d in DS if STUDY[d] != st]
        held = [d for d in DS if STUDY[d] == st]
        # global k
        kg = next((k for k in GRID if ok(cache[(k, k)].loc[others], target)), 1.0)
        # route-specific: smallest sum, both must satisfy jointly
        cand = [(ku + kc, ku, kc) for ku in GRID for kc in GRID if ok(cache[(ku, kc)].loc[others], target)]
        _, ku, kc = min(cand) if cand else (2, 1.0, 1.0)
        chosen.append(dict(held_out=st, k_global=kg, k_unmixed=ku, k_calibrated=kc))
        for name, key in (("v2", (1.0, 1.0)), ("global", (kg, kg)), ("route", (ku, kc))):
            m = cache[key].loc[held]
            for d in held:
                out.append(dict(dataset=d, method=name, **m.loc[d].to_dict()))
    res = pd.DataFrame(out).drop(columns="split")
    pd.set_option("display.width", 200)
    print("k chosen on the other studies (target pooled coverage %.2f, null calls <= 0.10 per dataset):" % target)
    print(pd.DataFrame(chosen).to_string(index=False))
    K = ["cov", "null_calls", "power", "wrong_sign", "fdp", "n_calls", "r", "width"]
    t = res.set_index(["dataset", "method"])[K]
    print("\n== per dataset (held-out study scored at the k of the other studies)")
    print(t.round(3).to_string())
    print("\n== mean of dataset means")
    print(res.groupby("method")[K].mean().round(3).to_string())
    print("\n== S1 (every dataset coverage >= 0.85, null calls <= 0.10)")
    g = res.groupby("method")
    print(pd.DataFrame(dict(fails=g.apply(lambda x: int(((x["cov"] < 0.85) | (x.null_calls > 0.10)).sum())),
                            worst_cov=g["cov"].min(), worst_null=g.null_calls.max())).round(3).to_string())
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "width_loso_%d-%d.csv" % (a0, a1)), index=False)


if __name__ == "__main__":
    main()
