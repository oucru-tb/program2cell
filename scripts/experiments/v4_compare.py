"""v4 vs v2 on the development splits (2026-11-05; docs/v4_plan.md): causal_loop outputs auto_v2 and v4 /
v4_pair, same datasets and splits, scored with the v4 scoring (fraction_error.score: coverage, r, within-program
r on programs with truth sd >= 0.05, calls on true nulls |beta| < 0.05, power = |beta| >= 0.1 called with the
right sign, width). Calls are interval-excludes-zero; "floor" also requires |effect| >= 0.1.
Usage: python v4_compare.py [SPLITS=20-29]
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from fraction_error import score

D = os.path.join(ROOT, "benchmarks", "dev", "causal_loop")
V4 = os.environ.get("V4", "v4")
KEYS = ("cov", "r", "within_r", "null_calls", "power", "width")


def _load(ds, split, name):
    f = os.path.join(D, "%s__s%d__%s.json" % (ds, split, name))
    if not os.path.exists(f):
        return None
    a = json.load(open(f))["arrays"]
    return {k: np.asarray(a[k], float) for k in ("effects", "lo", "hi", "beta")}


def _score(a, floor=0.0):
    s = score(a["beta"], a["effects"], a["lo"], a["hi"])
    if floor:
        c = ((a["lo"] > 0) | (a["hi"] < 0)) & (np.abs(a["effects"]) >= floor)
        b = a["beta"]
        s["null_calls"] = float((c & (np.abs(b) < 0.05)).sum() / max((np.abs(b) < 0.05).sum(), 1))
        big = np.abs(b) >= 0.1
        s["power"] = float((c & big & (np.sign(a["effects"]) == np.sign(b))).sum() / max(big.sum(), 1))
    return [s[k] if s[k] is not None else np.nan for k in KEYS]


def main(splits):
    dss = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(D, "*__auto_v2.json"))})
    print("v2 = auto_v2, v4 = v4 (v4_pair on kang, zhang); means over splits present for both; floor = calls need |eff| >= 0.1")
    print("%-12s n  %-8s cov   r     within null  power width" % ("dataset", "est"))
    tot = defaultdict(list)
    for ds in dss:
        n4 = V4 + "_pair" if ds in ("kang", "zhang") else V4
        R = defaultdict(list)
        for s in splits:
            a2, a4 = _load(ds, s, "auto_v2"), _load(ds, s, n4)
            if a2 is None or a4 is None:
                continue
            R["v2"].append(_score(a2)); R[V4].append(_score(a4))
        if not R:
            continue
        for k, v in R.items():
            m = np.nanmean(np.array(v), 0)
            tot[k].append(m)
            print("%-12s %-2d %-8s " % (ds, len(v), k) + " ".join("%5.2f" % x for x in m))
    print("\nmean over datasets")
    for k, v in tot.items():
        print("%-12s %-2d %-8s " % ("ALL", len(v), k) + " ".join("%5.2f" % x for x in np.nanmean(np.array(v), 0)))


if __name__ == "__main__":
    import causal_loop as cl
    main(cl.db._splits(sys.argv[1] if len(sys.argv) > 1 else "20-29"))
