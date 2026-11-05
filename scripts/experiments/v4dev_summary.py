"""Summary of the v4 development runs on the idea register (2026-11-05): self-tests (selftest_bench), fraction
error (fraction_error), low-rank deviations (lowrank_dev). Means over the splits present.
Usage: python v4dev_summary.py
"""
import glob, json, os
from collections import defaultdict
import numpy as np

D = "C:/Users/duy.pham/Documents/dev/program2cell/benchmarks/dev"
M = lambda x: np.nanmean(np.array(x, float)) if len(x) else np.nan


def selftests(floor=0.1):
    print("== self-tests (v2 = auto_v2, PALM-lite fractions): null programs and marker leakage")
    print("   calls / calls with |estimate| >= %.2f (effect floor); truth-null = |truth| < 0.05" % floor)
    print("%-10s n  real:null-calls | null:calls  floor  truth-null  cov  |truth| | marker: leak  leak|eff|  own  own|eff|(truth)" % "dataset")
    R = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(D, "selftest", "*__auto_v2.json"))):
        o = json.load(open(f))
        r = R[o["dataset"]]
        c, b = np.array(o["real"]["calls"], bool), np.array(o["real"]["beta"])
        r["real_null"].append(c[np.abs(b) < 0.05].mean() if (np.abs(b) < 0.05).any() else np.nan)
        for n in o["null"]:
            c, e, b = np.array(n["calls"], bool), np.array(n["effects"]), np.array(n["beta"])
            r["calls"].append(c.mean()); r["floor"].append((c & (np.abs(e) >= floor)).mean())
            r["tnull"].append(c[np.abs(b) < 0.05].mean() if (np.abs(b) < 0.05).any() else np.nan)
            r["cov"].append(n["cov"]); r["tabs"].append(np.abs(b).mean())
        m = o["marker"]
        for k in ("leak_call_rate", "leak_mean_abs", "own_call_rate", "own_mean_abs", "truth_own_mean_abs"):
            r[k].append(m[k])
        r["n"].append(1)
    for d, r in R.items():
        print("%-10s %d  %.3f            | %.3f  %.3f  %.3f      %.2f %.3f  | %.3f  %.3f      %.2f %.3f (%.3f)" % (
            d, len(r["n"]), M(r["real_null"]), M(r["calls"]), M(r["floor"]), M(r["tnull"]), M(r["cov"]), M(r["tabs"]),
            M(r["leak_call_rate"]), M(r["leak_mean_abs"]), M(r["own_call_rate"]), M(r["own_mean_abs"]),
            M(r["truth_own_mean_abs"])))


def fraction_error():
    print("\n== fractions measured with error (split 20): estimator x fractions")
    print("%-10s %-4s %-5s frac_r  cov   r     within null  power" % ("dataset", "est", "frac"))
    for f in sorted(glob.glob(os.path.join(D, "fraction_error", "*.json"))):
        o = json.load(open(f))
        for est in ("v2", "gls"):
            for k in ("true", "lite", "rand", "sys"):
                s = o.get("%s_%s" % (est, k))
                if s:
                    print("%-10s %-4s %-5s %.2f   %.2f  %.2f  %5.2f %.2f  %.2f" % (
                        o["dataset"], est, k, o["frac_r"][k], s["cov"], s["r"], s["within_r"] if s["within_r"] is not None
                        else np.nan, s["null_calls"], s["power"]))


def lowrank():
    print("\n== low-rank deviations (GLS + off-program calibration; splits 20-22)")
    print("%-10s %-5s n  cov   r     within null  power width" % ("dataset", "rank"))
    R = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(D, "lowrank", "*.json"))):
        o = json.load(open(f))
        for k in ("full", "rank1", "rank2"):
            if k in o:
                R[(o["dataset"], k)]["s"].append([o[k][m] if o[k][m] is not None else np.nan
                                                 for m in ("cov", "r", "within_r", "null_calls", "power", "width")])
    for (d, k), r in R.items():
        print("%-10s %-5s %d  " % (d, k, len(r["s"])) + " ".join("%5.2f" % v for v in np.nanmean(np.array(r["s"]), 0)))


if __name__ == "__main__":
    selftests()
    fraction_error()
    lowrank()
