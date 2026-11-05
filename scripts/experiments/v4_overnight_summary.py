"""Summary of the overnight v4 run (overnight_v4.sh): stacked variants vs v2 on 13 datasets x splits 20-29,
coarse functionals, and CyTOF-anchored COMBAT. Usage: python v4_overnight_summary.py
"""
import glob, json, os, subprocess, sys
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
HERE = os.path.join(ROOT, "scripts", "experiments")
sys.path.insert(0, HERE)


def sh(cmd, env=None):
    e = dict(os.environ, **(env or {}))
    print(subprocess.run(cmd, cwd=HERE, env=e, capture_output=True, text=True).stdout, flush=True)


print("=" * 100)
for v in ("v4k", "v4kn", "v4kr", "v4knr"):
    print("\n### %s vs v2 (splits 20-29)" % v)
    sh([sys.executable, "v4_compare.py", "20-29"], dict(V4=v))
print("=" * 100)
print("\n### coarse functionals")
sh([sys.executable, "coarse_functionals.py", "20-29", "auto_v2,v4k,v4kn,v4kr,v4knr"])
print("=" * 100)
print("\n### COMBAT real bulk, CyTOF-anchored fractions (sepsis, splits 20-24)")
R = {}
for f in sorted(glob.glob(os.path.join(ROOT, "benchmarks", "dev", "combat_cytof", "sepsis__s*.json"))):
    o = json.load(open(f))
    for fr in ("lite", "cytof"):
        R.setdefault((fr, "frac"), []).append(o[fr]["frac_mean_r"])
        for est in ("v2", "v4k"):
            s = o[fr][est]
            R.setdefault((fr, est), []).append([s["cov"], s["r"], s["calls"], s["null_calls"], s["power"], s["width"]])
for fr in ("lite", "cytof"):
    if (fr, "frac") not in R:
        continue
    print("%-6s fraction r vs single-cell truth %.2f (n splits %d)" % (fr, np.mean(R[(fr, "frac")]), len(R[(fr, "frac")])))
    for est in ("v2", "v4k"):
        m = np.nanmean(np.array(R[(fr, est)], float), 0)
        print("   %-4s cov %.2f r %.2f calls %.1f null %.2f power %.2f width %.2f" % ((est,) + tuple(m)))
