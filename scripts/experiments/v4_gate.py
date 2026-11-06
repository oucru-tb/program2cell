"""v4gated gate, chosen on development splits 20-29 from saved results (2026-11-06; docs/v4_plan.md section 12).

Per study: v4 stacked (v4k; v4k_pair on Kang, Zhang) where core.gls.gate_statistics passes, else v2 (auto_v2).
Prints per-split heterogeneity / agreement, the mean scores over 13 datasets for a grid of thresholds, and the
per-dataset scores at the package thresholds (core.gls.GATE_HETEROGENEITY, GATE_AGREEMENT).
Usage: python v4_gate.py
"""
import json, os, sys
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from fraction_error import score
from program2cell.core.gls import gate_statistics, GATE_HETEROGENEITY, GATE_AGREEMENT

L = os.path.join(ROOT, "benchmarks", "dev", "causal_loop")
DS = ["kang", "zhang", "mcf_ida", "mcf_tram", "tahoe_enc", "tahoe_gem", "tahoe_rmc", "tahoe_tram",
      "parse_gmcsf", "parse_ifnb", "parse_ifng", "parse_il4", "parse_tnf"]
K = ["cov", "r", "within_r", "null_calls", "power", "width"]


def _arr(f):
    a = json.load(open(f))["arrays"]
    return {k: np.array(a[k], float) for k in ("effects", "lo", "hi", "beta")}


def _sc(x):
    s = score(x["beta"], x["effects"], x["lo"], x["hi"])
    return [np.nan if s[k] is None else s[k] for k in K]


data = {}
for d in DS:
    sfx = "_pair" if d in ("kang", "zhang") else ""
    for s in range(20, 30):
        a, b = _arr(os.path.join(L, "%s__s%d__v4k%s.json" % (d, s, sfx))), _arr(os.path.join(L, "%s__s%d__auto_v2.json" % (d, s)))
        g = gate_statistics(a["effects"], a["lo"], a["hi"], b["effects"])
        data.setdefault(d, []).append((g["heterogeneity"], g["agreement"], _sc(b), _sc(a)))
print("per split heterogeneity / agreement")
for d in DS:
    print("  %-12s %s" % (d, " ".join("%.2f/%.2f" % (q, g) for q, g, _, _ in data[d])))
print("\nmean over 13 datasets (%s)" % " ".join(K))
for thr, agt in ((np.inf, -1), (0, -1), (0.2, -1), (0.15, 0.4), (0.2, 0.4), (0.25, 0.4), (0.2, 0.3), (0.2, 0.5)):
    per = [np.nanmean([a if (q >= thr and g >= agt) else b for q, g, b, a in data[d]], 0) for d in DS]
    n = sum(q >= thr and g >= agt for d in DS for q, g, _, _ in data[d])
    lab = "all v2" if thr == np.inf else "all v4k" if thr == 0 and agt < 0 else "het>=%.2f agree>=%.1f" % (thr, agt)
    print("  %-22s v4 on %3d/130  %s" % (lab, n, " ".join("%5.3f" % v for v in np.nanmean(per, 0))))
print("\nper dataset at het >= %.2f, agree >= %.1f: v2 | gated" % (GATE_HETEROGENEITY, GATE_AGREEMENT))
for d in DS:
    use = [q >= GATE_HETEROGENEITY and g >= GATE_AGREEMENT for q, g, _, _ in data[d]]
    b = np.nanmean([x for _, _, x, _ in data[d]], 0)
    gd = np.nanmean([a if u else x for u, (_, _, x, a) in zip(use, data[d])], 0)
    print("  %-12s v4 %2d/10  %s | %s" % (d, sum(use), " ".join("%5.2f" % v for v in b), " ".join("%5.2f" % v for v in gd)))
