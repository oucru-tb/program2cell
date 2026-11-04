"""Held-out v2 report (method freeze v2, splits 20-29; protocol in docs/method_spec.md, "Method freeze v2").

1. PALM (causal_loop estimator auto_v2) per dataset: coverage, r (where the truth has effects), calls on null pairs,
   responder AUROC and truth-based TPR, against the pre-set targets (PASS / which target failed).
2. Comparators per dataset (practice_vs_palm rows): deconvolution + t-test (BH q < 0.1), and PALM's module on
   each comparator's fractions.
3. Paired PALM minus comparator over dataset x split, bootstrap 95% CI (known_answer.paired_test).
4. COMBAT real bulk (combat_real rows, splits >= 20).
Writes benchmarks/dev/heldout_v2_summary.txt (and prints it). Usage: python heldout_v2_report.py
"""
import glob, io, json, os, sys
from contextlib import redirect_stdout
import numpy as np, pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl

EST = os.environ.get("PALM_EST", "auto_v2")
FIRST = int(os.environ.get("REPORT_FIRST", 20))          # 0 only to test the script on development rows
OUTF = os.path.join(ROOT, "benchmarks", "dev", "heldout_v2_summary.txt")
METRICS = ["cov", "r_t", "rmse", "width", "calls", "fpr_null", "auroc", "tpr_t"]


def palm_rows():
    rows = []
    for f in glob.glob(os.path.join(cl.OUT, "*__%s.json" % EST)):
        s = json.load(open(f))
        if s["split"] < FIRST:
            continue
        a = s.pop("arrays")
        s.update(cl.score(*(np.asarray(a[k], float) for k in ("effects", "lo", "hi")), np.asarray(a["calls"], bool),
                          np.asarray(a["beta"], float), a["cell_types"], a["programs"], cl.KNOWN.get(s["dataset"])))
        s.update(method="PALM", kind="palm", rule="palm")
        rows.append(s)
    return pd.DataFrame(rows)


def comparator_rows():
    rows = []
    for f in glob.glob(os.path.join(cl.ROOT, "benchmarks", "dev", "practice", "*.json")):
        for r in json.load(open(f)):
            if r["split"] >= FIRST:
                rows.append(r)
    return pd.DataFrame(rows)


def targets(t):
    fail = []
    for m, (op, v) in cl.TARGETS.items():
        m = "r_t" if m == "r" else m
        x = t.get(m, np.nan)
        if pd.isna(x):
            continue
        if (op == ">=" and x < v) or (op == "<=" and x > v):
            fail.append("%s %.2f" % (m, x))
    return "PASS" if not fail else "fail: " + ", ".join(fail)


def main():
    from program2cell.compare import known_answer as ka
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    P, C = palm_rows(), comparator_rows()
    print("Held-out v2 report (method-freeze-v2, splits >= %d, PALM estimator %s)\n" % (FIRST, EST))
    if P.empty:
        print("no PALM rows yet"); return
    print("== 1. PALM per dataset (mean over splits; targets: %s)" %
          ", ".join("%s %s %s" % (m, op, v) for m, (op, v) in cl.TARGETS.items()))
    t = P.groupby("dataset")[[m for m in METRICS if m in P]].mean()
    t.insert(0, "n", P.groupby("dataset").split.nunique())
    t["verdict"] = [targets(r) for _, r in t.iterrows()]
    print(t.round(3).to_string())
    if not C.empty:
        print("\n== 2. Comparators per dataset (BH q < 0.1 for deconvolution + t-test; PALM module = PALM's effects on that "
              "tool's fractions)")
        C2 = C[(C.rule == "bh10") | (C.kind == "palm_module")]
        t2 = C2.groupby(["dataset", "kind", "method"])[[m for m in METRICS if m in C2]].mean()
        t2.insert(0, "n", C2.groupby(["dataset", "kind", "method"]).split.nunique())
        print(t2.round(3).to_string())
        print("\n== 3. PALM minus comparator, paired over dataset x split [95% bootstrap CI]")
        palm = P.set_index(["dataset", "split"])
        out = {}
        for (kind, m), sub in C2.groupby(["kind", "method"]):
            sub = sub.set_index(["dataset", "split"])
            idx = sub.index.intersection(palm.index)
            res = {}
            for metric in ("cov", "fpr_null", "tpr_t", "r_t", "auroc"):
                a, b = palm.loc[idx, metric].values.astype(float), sub.loc[idx, metric].values.astype(float)
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() >= 2:
                    r = ka.paired_test(a[ok], b[ok])
                    res[metric] = "%+.3f [%+.3f, %+.3f] n=%d" % (r["diff"], r["lo"], r["hi"], r["n"])
            out["%s | %s" % (kind, m)] = res
        print(pd.DataFrame(out).T.to_string())
    print("\n== 4. COMBAT real bulk (splits >= %d)" % FIRST)
    rows = []
    for f in glob.glob(os.path.join(ROOT, "benchmarks", "dev", "combat_real", "*.json")):
        r = json.load(open(f))
        for x in (r if isinstance(r, list) else [dict(r, method="palm", rule="palm")]):
            if x["split"] >= FIRST and x.get("rule") in ("palm", "bh10"):
                rows.append({k: v for k, v in x.items() if k != "arrays"})
    if rows:
        d = pd.DataFrame(rows)
        t4 = d.groupby(["contrast", "bulk", "method"])[["cov", "r", "rmse", "fpr_null", "calls"]].mean()
        t4.insert(0, "n", d.groupby(["contrast", "bulk", "method"]).size())
        print(t4.round(3).to_string())
    else:
        print("no rows yet")
    breakdowns(P)


FAMILY = lambda d: "cell lines" if d.startswith(("mcf", "tahoe")) else "blood cells"


def breakdowns(P):
    """Added 2026-11-04 (reporting only, after the external critique; docs/v3_plan.md A2-A4): results by benchmark
    family, and pair-level coverage by route (unmixed vs calibrated program; the per-type guard is not recorded in
    v2 rows) and by response regime of the program in the truth (none / some types / all types respond)."""
    print("\n== 5a. By benchmark family (mean over dataset x split; read separately, do not average across)")
    t = P.assign(family=P.dataset.map(FAMILY)).groupby("family")[[m for m in METRICS if m in P]].mean()
    t.insert(0, "datasets", P.assign(family=P.dataset.map(FAMILY)).groupby("family").dataset.nunique())
    print(t.round(3).to_string())
    pairs = []
    for f in glob.glob(os.path.join(cl.OUT, "*__%s.json" % EST)):
        s = json.load(open(f))
        if s["split"] < FIRST:
            continue
        a = s["arrays"]
        eff, lo, hi, beta = (np.asarray(a[k], float) for k in ("effects", "lo", "hi", "beta"))
        progs = list(a["programs"])
        unm = set(filter(None, (s.get("unmixed") or "").split(",")))
        resp = (np.abs(beta) > cl.R_MIN_EFFECT).sum(0)                       # responding types per program
        C = beta.shape[0]
        for c in range(C):
            for p, prog in enumerate(progs):
                if not np.isfinite(beta[c, p]) or not np.isfinite(lo[c, p]):
                    continue
                pairs.append(dict(dataset=s["dataset"], family=FAMILY(s["dataset"]), split=s["split"],
                                  route="unmixed" if prog in unm else "calibrated",
                                  regime="none" if resp[p] == 0 else ("all types" if resp[p] == C else "some types"),
                                  covered=lo[c, p] <= beta[c, p] <= hi[c, p], width=hi[c, p] - lo[c, p],
                                  err=eff[c, p] - beta[c, p]))
    if not pairs:
        return
    d = pd.DataFrame(pairs)
    agg = dict(pairs=("covered", "size"), cov=("covered", "mean"), width=("width", "mean"),
               rmse=("err", lambda e: float(np.sqrt(np.mean(np.square(e))))))
    print("\n== 5b. Pair-level coverage by route (program level; guard firings not recorded in v2)")
    print(d.groupby(["family", "route"]).agg(**agg).round(3).to_string())
    print("\n== 5c. Pair-level coverage by response regime in the truth (|beta| > %.2f counts as responding)"
          % cl.R_MIN_EFFECT)
    print(d.groupby(["family", "regime"]).agg(**agg).round(3).to_string())
    print("\n   per dataset, 'all types' regime only:")
    sub = d[d.regime == "all types"]
    print(sub.groupby("dataset").agg(**agg).round(3).to_string() if len(sub) else "   none")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open(OUTF, "w").write(buf.getvalue())
    print(buf.getvalue())
