"""Pre-registered comparison for method freeze v5 (docs/method_spec.md, "Method freeze v5"; written and committed
with the freeze, before any split 30-39 result exists). Do not edit the rule after the held-out run.

Freeze v5 (causal_loop "v5" = program_deconvolution(estimator="v5")) against freeze v2 ("auto_v2") on the same
held-out splits 30-39, 13 datasets. v5 changes only the reported calibrated interval (size-matched simulated error);
estimates and routes are those of v2, so r is identical by construction and the primary endpoint is power.
Per dataset x split: scores from fraction_error.score (cov, r, within_r, null_calls, power, width) on the saved
arrays; difference d = v5 - v2. Means: over splits within a dataset, then over the 13 datasets (equal weight).
95% CI: two-stage cluster bootstrap (datasets with replacement, then splits within each), 4000 draws, seed 0.

Decision rule:
  C0 identity check: v5 and v2 effects equal on every run (max |difference| <= 1e-8); otherwise the run is a
     protocol deviation and no decision is made.
  S1 every dataset, v5: coverage >= 0.85 and null-call rate <= 0.10 (dataset means).
  S2 mean difference: coverage lower CI >= -0.03; null-call rate upper CI <= +0.02.
  S3 COMBAT real bulk (raw and offset-corrected, combat_v5.py): v5 wrong-sign calls per split <= v2's + 0.2.
  E  primary, mean difference in power: "superior" if the lower CI > 0; else "not shown". (No non-inferiority
     claim: with equal estimates, v5 is only worth adopting if it calls more true effects.)
     Development value (splits 20-29, chosen on them; offline sized_fx, v5_sims_offline.py): +0.047 [+0.017, +0.081].
  Outcome: C0 and S1-S3 pass and E superior -> v5 is the Paper 1 method. Otherwise freeze v2 stays the Paper 1
           method (package default set back to "auto") and v5 is reported as secondary. No retuning either way.
Secondary (reported, no decision): r and within_r (equal by construction), width, FDP-type summaries via
causal_loop summary heldout5; cell lines (McFarland, Tahoe) vs PBMC (Kang, Zhang, Parse); study-level power
difference (5 studies, exact sign-flip p; the dataset-level CI is optimistic because datasets share donors within
a study). Transport check (descriptive): `python heldout_v5_compare.py transport` on 9 perturbations never used in
v5 development (Parse IFN-a, IL-10, IL-1b, IL-6; Tahoe afatinib, alpelisib, bortezomib, everolimus, paclitaxel).

Usage: python heldout_v5_compare.py [SPLITS]     (default 30-39)
       python heldout_v5_compare.py transport [SPLITS]
"""
import glob, json, os, sys
import itertools
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from fraction_error import score
import combat_v5 as cv

L = os.path.join(ROOT, "benchmarks", "dev", "causal_loop")
CELL = ["mcf_ida", "mcf_tram", "tahoe_enc", "tahoe_gem", "tahoe_rmc", "tahoe_tram"]
PBMC = ["kang", "zhang", "parse_gmcsf", "parse_ifnb", "parse_ifng", "parse_il4", "parse_tnf"]
DS = CELL + PBMC
TRANSPORT = ["parse_ifna", "parse_il10", "parse_il1b", "parse_il6",
             "tahoe_afa", "tahoe_alp", "tahoe_bor", "tahoe_evr", "tahoe_ptx"]
STUDY = {d: ("mcfarland" if d.startswith("mcf") else d.split("_")[0]) for d in DS + TRANSPORT}
K = ["cov", "r", "within_r", "null_calls", "power", "width"]
IP = K.index("power")
N_BOOT, SEED = 4000, 0
DEFAULT_SPLITS = list(range(30, 40))


def _load(d, s, est):
    f = os.path.join(L, "%s__s%d__%s.json" % (d, s, est))
    if not os.path.exists(f):
        return None, None
    j = json.load(open(f))
    a = {k: np.array(j["arrays"][k], float) for k in ("effects", "lo", "hi", "beta")}
    sc = score(a["beta"], a["effects"], a["lo"], a["hi"])
    return np.array([np.nan if sc[k] is None else sc[k] for k in K]), a["effects"]


def collect(splits, datasets=DS):
    v5, v2, missing, maxdiff = {}, {}, [], 0.0
    for d in datasets:
        for s in splits:
            a, ea = _load(d, s, "v5")
            b, eb = _load(d, s, "auto_v2")
            if a is None or b is None:
                missing.append("%s s%d" % (d, s))
                continue
            maxdiff = max(maxdiff, float(np.abs(ea - eb).max()))
            v5.setdefault(d, []).append(a)
            v2.setdefault(d, []).append(b)
    return {d: np.array(x) for d, x in v5.items()}, {d: np.array(x) for d, x in v2.items()}, missing, maxdiff


def mean_over(ds, x):
    return np.nanmean([np.nanmean(x[d], 0) for d in ds], 0)


def boot(ds, a, b, rng):
    """Two-stage cluster bootstrap of the mean difference a - b over datasets: (N_BOOT, len(K))."""
    diff = {d: a[d] - b[d] for d in ds}
    out = np.empty((N_BOOT, len(K)))
    for i in range(N_BOOT):
        pick = rng.choice(ds, len(ds), replace=True)
        out[i] = np.nanmean([np.nanmean(diff[d][rng.integers(0, len(diff[d]), len(diff[d]))], 0) for d in pick], 0)
    return out


def study_level(ds, a, b, j=IP, label="power"):
    per = {}
    for d in ds:
        per.setdefault(STUDY[d], []).append(np.nanmean(a[d][:, j] - b[d][:, j]))
    st = {k: float(np.mean(v)) for k, v in per.items()}
    x = np.array(list(st.values()))
    obs = x.mean()
    flips = [np.mean(x * np.array(s)) for s in itertools.product((1, -1), repeat=len(x))]
    p = np.mean([f >= obs - 1e-12 for f in flips])
    print("\nsensitivity: study level (%s difference; descriptive)" % label)
    print("  " + "  ".join("%s %+.3f (%d)" % (k, v, len(per[k])) for k, v in st.items()))
    print("  study-weighted mean %+.3f; exact one-sided sign-flip p %.3f over %d studies (%d nonzero)" % (
        obs, p, len(x), int((np.abs(x) > 1e-9).sum())))


def table(ds, a, b, flag_s1):
    print("\nper dataset (dataset means)      %s" % "  ".join("%9s" % k for k in K))
    ok_all = True
    for d in ds:
        x, y = np.nanmean(a[d], 0), np.nanmean(b[d], 0)
        ok = x[0] >= 0.85 and x[3] <= 0.10
        ok_all &= ok
        print("  %-12s v2    %s" % (d, "  ".join("%9.3f" % v for v in y)))
        print("  %-12s v5    %s%s" % ("", "  ".join("%9.3f" % v for v in x),
                                      "" if ok else ("   S1 FAIL" if flag_s1 else "   below S1 targets")))
    return ok_all


def diffs(label, sub, a, b, rng):
    m = mean_over(sub, a) - mean_over(sub, b)
    bt = boot(sub, a, b, rng)
    lo, hi = np.nanpercentile(bt, 2.5, 0), np.nanpercentile(bt, 97.5, 0)
    print("\nmean difference v5 - v2, %s (%d datasets) [95%% cluster bootstrap CI]" % (label, len(sub)))
    for k, x, l, h in zip(K, m, lo, hi):
        print("  %-10s %+.3f [%+.3f, %+.3f]" % (k, x, l, h))
    return m, lo, hi


def transport(splits):
    a, b, missing, maxdiff = collect(splits, TRANSPORT)
    ds = [d for d in TRANSPORT if d in a]
    print("Transport check (descriptive, no decision), splits %d-%d" % (min(splits), max(splits)))
    if missing:
        print("MISSING or skipped (%d): %s" % (len(missing), ", ".join(missing)))
    print("identity check: max |v5 - v2 effect| = %.2e" % maxdiff)
    table(ds, a, b, flag_s1=False)
    if ds:
        diffs("transport", ds, a, b, np.random.default_rng(SEED))
        study_level(ds, a, b)


def main(splits):
    a, b, missing, maxdiff = collect(splits)
    ds = [d for d in DS if d in a]
    print("Freeze v5 comparison, splits %d-%d: v5 vs freeze v2 (auto_v2)" % (min(splits), max(splits)))
    if missing:
        print("MISSING (%d): %s" % (len(missing), ", ".join(missing)))
    c0 = maxdiff <= 1e-8
    s1 = table(ds, a, b, flag_s1=True)
    rng = np.random.default_rng(SEED)
    res = {}
    for lab, sub in (("all", ds), ("cell lines", [d for d in CELL if d in a]), ("PBMC", [d for d in PBMC if d in a])):
        if sub:
            res[lab] = diffs(lab, sub, a, b, rng)
    study_level(ds, a, b)
    m, lo, hi = res["all"]
    s2 = lo[0] >= -0.03 and hi[3] <= 0.02
    e = "superior" if lo[IP] > 0 else "not shown"
    s3, s3txt = None, "COMBAT not run"
    fs = [f for f in glob.glob(os.path.join(cv.OUT, "sepsis__s*.json"))
          if int(os.path.basename(f).split("__s")[1].split(".")[0]) in splits]
    if fs:
        R = [json.load(open(f)) for f in fs]
        s3, parts = True, []
        for bulk in ("real", "real_offset"):
            w5 = np.mean([o[bulk]["v5"]["wrong_sign"] for o in R])
            w2 = np.mean([o[bulk]["v2"]["wrong_sign"] for o in R])
            s3 &= w5 <= w2 + 0.2
            parts.append("%s %.1f vs %.1f" % (bulk, w5, w2))
        s3txt = "wrong-sign calls per split, v5 vs v2 (%d splits): %s" % (len(R), "; ".join(parts))
    print("\nDECISION (pre-registered)")
    print("  C0 estimates identical (max |v5 - v2| = %.2e <= 1e-8): %s" % (maxdiff, "pass" if c0 else "FAIL (deviation)"))
    print("  S1 every dataset coverage >= 0.85, null calls <= 0.10: %s" % ("pass" if s1 else "FAIL"))
    print("  S2 coverage lower CI %+.3f >= -0.03, null-call upper CI %+.3f <= +0.02: %s" % (lo[0], hi[3], "pass" if s2 else "FAIL"))
    print("  S3 %s: %s" % (s3txt, {True: "pass", False: "FAIL", None: "PENDING"}[s3]))
    print("  E  power difference %+.3f [%+.3f, %+.3f]: %s" % (m[IP], lo[IP], hi[IP], e))
    if missing or s3 is None or not c0:
        print("  OUTCOME: incomplete or deviation; no decision")
    elif s1 and s2 and s3 and e == "superior":
        print("  OUTCOME: v5 is the Paper 1 method (superior power at equal safety)")
    else:
        print("  OUTCOME: freeze v2 stays the Paper 1 method; v5 reported as secondary")


if __name__ == "__main__":
    import causal_loop as cl
    if len(sys.argv) > 1 and sys.argv[1] == "transport":
        transport(cl.db._splits(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SPLITS)
    else:
        main(cl.db._splits(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SPLITS)
