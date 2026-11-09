"""Pre-registered comparison for method freeze v4 (docs/method_spec.md, "Method freeze v4"; written and committed
with the freeze, before any split 40-49 result exists). Do not edit the rule after the held-out run.

Gated v4 (causal_loop v4g; v4g_pair on Kang and Zhang) against freeze v2 (auto_v2) on the same held-out splits.
Per dataset x split: scores from fraction_error.score (cov, r, within_r, null_calls, power, width) on the saved
arrays; difference d = v4 - v2. Means: over splits within a dataset, then over the 13 datasets (equal weight).
95% CI: two-stage cluster bootstrap (datasets with replacement, then splits within each), 4000 draws, seed 0.

Decision rule:
  S1 every dataset, gated v4: coverage >= 0.85 and null-call rate <= 0.10 (dataset means).
  S2 mean difference: coverage lower CI >= -0.03; null-call rate upper CI <= +0.02.
  S3 COMBAT real bulk (raw and offset-corrected, combat_v4.py): gated v4 wrong-sign calls per split <= v2's + 0.2.
  E  primary, mean difference in r: "superior" if the lower CI > 0; "non-inferior" if the mean > 0 and the lower
     CI > -0.02; else "not shown".
  Outcome: S1-S3 pass and E superior or non-inferior -> gated v4 is the Paper 1 method, claim worded by E.
           Otherwise freeze v2 stays the Paper 1 method and gated v4 is reported as secondary. No retuning either way.
Secondary (reported, no decision): within_r, power, width; cell lines (McFarland, Tahoe) vs PBMC (Kang, Zhang,
Parse); gate choices per dataset against development (v4 expected on McFarland and Tahoe, never on Parse or Zhang);
on Kang and Zhang, paired freeze v2 (auto_v2_pair, the v2 arm inside v4g_pair) to separate the effect of pairing.

Usage: python heldout_v4_compare.py [SPLITS]     (default 40-49; 20-29 runs the same code on development splits)
"""
import glob, json, os, sys
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from fraction_error import score
import combat_v4 as cv

L = os.path.join(ROOT, "benchmarks", "dev", "causal_loop")
CELL = ["mcf_ida", "mcf_tram", "tahoe_enc", "tahoe_gem", "tahoe_rmc", "tahoe_tram"]
PBMC = ["kang", "zhang", "parse_gmcsf", "parse_ifnb", "parse_ifng", "parse_il4", "parse_tnf"]
DS = CELL + PBMC
K = ["cov", "r", "within_r", "null_calls", "power", "width"]
N_BOOT, SEED = 4000, 0


def _load(d, s, est):
    f = os.path.join(L, "%s__s%d__%s.json" % (d, s, est))
    if not os.path.exists(f):
        return None, None
    j = json.load(open(f))
    a = {k: np.array(j["arrays"][k], float) for k in ("effects", "lo", "hi", "beta")}
    sc = score(a["beta"], a["effects"], a["lo"], a["hi"])
    return np.array([np.nan if sc[k] is None else sc[k] for k in K]), j.get("gate")


def collect(splits):
    v4, v2, gates, missing = {}, {}, {}, []
    for d in DS:
        e4 = "v4g_pair" if d in ("kang", "zhang") else "v4g"
        for s in splits:
            a, g = _load(d, s, e4)
            b, _ = _load(d, s, "auto_v2")
            if a is None or b is None:
                missing.append("%s s%d" % (d, s))
                continue
            v4.setdefault(d, []).append(a)
            v2.setdefault(d, []).append(b)
            gates.setdefault(d, []).append(bool(g and g.get("use_v4")))
    return ({d: np.array(x) for d, x in v4.items()}, {d: np.array(x) for d, x in v2.items()}, gates, missing)


def mean_over(ds, x):
    return np.nanmean([np.nanmean(x[d], 0) for d in ds], 0)


def boot(ds, v4, v2, rng):
    """Two-stage cluster bootstrap of the mean difference v4 - v2 over datasets: (N_BOOT, len(K))."""
    diff = {d: v4[d] - v2[d] for d in ds}
    out = np.empty((N_BOOT, len(K)))
    for i in range(N_BOOT):
        pick = rng.choice(ds, len(ds), replace=True)
        out[i] = np.nanmean([np.nanmean(diff[d][rng.integers(0, len(diff[d]), len(diff[d]))], 0) for d in pick], 0)
    return out


def main(splits):
    v4, v2, gates, missing = collect(splits)
    ds = [d for d in DS if d in v4]
    print("Freeze v4 comparison, splits %d-%d: gated v4 (v4g / v4g_pair) vs freeze v2 (auto_v2)" % (min(splits), max(splits)))
    if missing:
        print("MISSING (%d): %s" % (len(missing), ", ".join(missing)))
    print("\nper dataset (dataset means)      %s" % "  ".join("%9s" % k for k in K))
    s1 = True
    for d in ds:
        a, b = np.nanmean(v4[d], 0), np.nanmean(v2[d], 0)
        ok = a[0] >= 0.85 and a[3] <= 0.10
        s1 &= ok
        print("  %-12s v2    %s" % (d, "  ".join("%9.3f" % x for x in b)))
        print("  %-12s v4g   %s   v4 on %d/%d%s" % ("", "  ".join("%9.3f" % x for x in a), sum(gates[d]), len(gates[d]),
                                                  "" if ok else "   S1 FAIL"))
    rng = np.random.default_rng(SEED)
    res = {}
    for lab, sub in (("all", ds), ("cell lines", [d for d in CELL if d in v4]), ("PBMC", [d for d in PBMC if d in v4])):
        if not sub:
            continue
        m = mean_over(sub, v4) - mean_over(sub, v2)
        bt = boot(sub, v4, v2, rng)
        lo, hi = np.nanpercentile(bt, 2.5, 0), np.nanpercentile(bt, 97.5, 0)
        res[lab] = (m, lo, hi)
        print("\nmean difference v4 - v2, %s (%d datasets) [95%% cluster bootstrap CI]" % (lab, len(sub)))
        for k, x, l, h in zip(K, m, lo, hi):
            print("  %-10s %+.3f [%+.3f, %+.3f]" % (k, x, l, h))
    m, lo, hi = res["all"]
    s2 = lo[0] >= -0.03 and hi[3] <= 0.02
    e = "superior" if lo[1] > 0 else "non-inferior" if (m[1] > 0 and lo[1] > -0.02) else "not shown"
    s3, s3txt = None, "COMBAT not run"
    fs = [f for f in glob.glob(os.path.join(cv.OUT, "sepsis__s*.json"))
          if int(os.path.basename(f).split("__s")[1].split(".")[0]) in splits]
    if fs:
        R = [json.load(open(f)) for f in fs]
        s3, parts = True, []
        for bulk in ("real", "real_offset"):
            w4 = np.mean([o[bulk]["v4gated"]["wrong_sign"] for o in R])
            w2 = np.mean([o[bulk]["v2"]["wrong_sign"] for o in R])
            s3 &= w4 <= w2 + 0.2
            parts.append("%s %.1f vs %.1f" % (bulk, w4, w2))
        s3txt = "wrong-sign calls per split, v4gated vs v2 (%d splits): %s" % (len(R), "; ".join(parts))
    print("\nsecondary: pairing on Kang and Zhang (dataset means; r, within_r, null_calls)")
    for d in ("kang", "zhang"):
        p = [x for x in (_load(d, s, "auto_v2_pair")[0] for s in splits) if x is not None]
        if p and d in v4:
            row = lambda x: "%.3f %.3f %.3f" % (x[1], x[2], x[3])
            print("  %-6s auto_v2 %s | auto_v2_pair %s | v4g_pair %s" % (
                d, row(np.nanmean(v2[d], 0)), row(np.nanmean(p, 0)), row(np.nanmean(v4[d], 0))))
    print("\nDECISION (pre-registered)")
    print("  S1 every dataset coverage >= 0.85, null calls <= 0.10: %s" % ("pass" if s1 else "FAIL"))
    print("  S2 coverage lower CI %+.3f >= -0.03, null-call upper CI %+.3f <= +0.02: %s" % (lo[0], hi[3], "pass" if s2 else "FAIL"))
    print("  S3 %s: %s" % (s3txt, {True: "pass", False: "FAIL", None: "PENDING"}[s3]))
    print("  E  r difference %+.3f [%+.3f, %+.3f]: %s" % (m[1], lo[1], hi[1], e))
    if missing or s3 is None:
        print("  OUTCOME: incomplete (missing results); no decision")
    elif s1 and s2 and s3 and e != "not shown":
        print("  OUTCOME: gated v4 is the Paper 1 method (%s in r)" % e)
    else:
        print("  OUTCOME: freeze v2 stays the Paper 1 method; gated v4 reported as secondary")


if __name__ == "__main__":
    main(cv.cl.db._splits(sys.argv[1]) if len(sys.argv) > 1 else list(range(40, 50)))
