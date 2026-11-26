"""Fraction accuracy of every method on the held-out v5 splits 30-39 (docs/heldout_v5_results.md 4b), descriptive.

Reads the dev_bench rows (row 'model'): benchmarks/dev/<ds>/ for the v5 run (joint NB on McFarland, Kang, Zhang;
competitors everywhere) and benchmarks/dev/fraction_joint/<ds>/ for the joint NB fits added on Tahoe and Parse
(fraction_joint_chain.sh). Metrics: frac_rmse, frac_r_per_type (mean per-type Pearson over samples), frac_cov90
(coverage of the joint fit's 90% fraction intervals).

Usage: python fraction_joint_score.py   -> benchmarks/dev/fraction_joint_v5_heldout.txt
"""
import glob, json, os
import pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
DEV = os.path.join(ROOT, "benchmarks", "dev")
DS = ["mcf_ida", "mcf_tram", "tahoe_enc", "tahoe_gem", "tahoe_rmc", "tahoe_tram", "kang", "zhang",
      "parse_gmcsf", "parse_ifnb", "parse_ifng", "parse_il4", "parse_tnf"]
METHODS = {"joint_nb": "PALM joint NB", "instaprism": "InstaPrism", "cibersort": "CIBERSORT", "reallocation": "NNLS"}
SPLITS = range(30, 40)


def rows():
    out = []
    for ds in DS:
        for stem, name in METHODS.items():
            for k in SPLITS:
                f = os.path.join(DEV, ds, "s%d__%s.json" % (k, stem))
                if stem == "joint_nb" and not os.path.exists(f):
                    f = os.path.join(DEV, "fraction_joint", ds, "s%d__%s.json" % (k, stem))
                if not os.path.exists(f):
                    continue
                r = [x for x in json.load(open(f)) if x.get("row") == "model"]
                if r and "frac_rmse" in r[0]:
                    out.append(dict(ds=ds, method=name, split=k, rmse=r[0]["frac_rmse"],
                                    r=r[0].get("frac_r_per_type"), cov90=r[0].get("frac_cov90")))
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = rows()
    g = df.groupby(["ds", "method"])
    t = g[["rmse", "r", "cov90"]].mean().join(g.size().rename("n"))
    lines = ["Fraction accuracy, held-out splits 30-39 (row 'model'; joint NB on Tahoe/Parse from fraction_joint/)"]
    for m in ("rmse", "r", "n"):
        lines += ["", "== " + m, t[m].unstack().reindex(DS).round(3).to_string()]
    lines += ["", "== joint NB 90% fraction-interval coverage",
              t.xs("PALM joint NB", level="method")["cov90"].reindex(DS).round(3).to_string()]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(DEV, "fraction_joint_v5_heldout.txt"), "w").write(txt + "\n")
