"""Summary of the fraction-error propagation runs (frac_prop.py; anchors absolute and excess) and the COMBAT
real-bulk fraction-noise level (combat_fracnoise.py). Means over the splits present.
Usage: python frac_prop_summary.py
"""
import glob, json, os
from collections import defaultdict
import numpy as np

D = "C:/Users/duy.pham/Documents/dev/program2cell/benchmarks/dev"
KEYS = ("cov", "r", "within_r", "null_calls", "power", "width")


def propagation():
    for sub, anchor in (("frac_prop", "absolute"), ("frac_prop_excess", "excess")):
        print("\n== GLS + off-program calibration, fraction noise propagated (anchor %s)" % anchor)
        print("%-10s %-4s %-4s n  s_true s_used  cov   r     within null  power width" % ("dataset", "frac", "lvl"))
        R = defaultdict(lambda: defaultdict(list))
        for f in sorted(glob.glob(os.path.join(D, sub, "*.json"))):
            o = json.load(open(f))
            for k in ("true", "lite", "rand", "sys"):
                for lv in ("none", "est", "orac"):
                    s = o.get("%s_%s" % (k, lv))
                    if not s:
                        continue
                    r = R[(o["dataset"], k, lv)]
                    r["s"].append([s[m] if s[m] is not None else np.nan for m in KEYS])
                    r["true"].append(o["rand_sd"] if k == "rand" else 0.0)
                    r["used"].append({"none": 0.0, "est": o["frac_sd_" + k], "orac": o["rand_sd"]}[lv])
        for (d, k, lv), r in R.items():
            print("%-10s %-4s %-4s %d  %.2f   %.2f   " % (d, k, lv, len(r["s"]), np.mean(r["true"]), np.mean(r["used"]))
                  + " ".join("%5.2f" % v for v in np.nanmean(np.array(r["s"]), 0)))


def combat():
    print("\n== COMBAT sepsis: PALM-lite fractions vs single-cell truth, matching per-sample noise s")
    R = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(D, "combat_fracnoise", "*.json"))):
        o = json.load(open(f))
        for b in ("sc", "real", "real_selfoffset"):
            for m in ("frac_r", "mae", "s_match_r", "s_match_sd"):
                R[b][m].append(o[b][m])
    for b, r in R.items():
        print("%-15s n %d  frac r %.2f  mae %.3f  s (r) %.2f [%.2f-%.2f]  s (sd) %.2f" % (
            b, len(r["frac_r"]), np.mean(r["frac_r"]), np.mean(r["mae"]), np.mean(r["s_match_r"]),
            np.min(r["s_match_r"]), np.max(r["s_match_r"]), np.mean(r["s_match_sd"])))


if __name__ == "__main__":
    propagation()
    combat()
