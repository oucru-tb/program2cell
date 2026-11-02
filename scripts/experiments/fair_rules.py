"""Fair comparison of calling rules (answers "is the t-test appropriate?").

Per dataset and split (0-2), for InstaPrism + paired t-test (current practice, practice_vs_palm.practice_test)
and PALM (causal_loop auto_m):
  q-only      BH q < 0.1 (the usual practice)
  q+cut-off   BH q < 0.1 and |estimated change| >= 0.1 (a relevant change)
  ranking     threshold-free AUROC of |estimate|: real boxes (|true| >= 0.1) vs unchanged (|true| < 0.05)
and the 90% range coverage, mean absolute error, the known program's responders found, and the known
non-responders flagged under q+cut-off (the attribution error that survives a cut-off).
Sample-size curve: InstaPrism + t-test, unchanged flagged (q-only) when the test uses m paired mixtures
(m = 3, 5, 10, 20; 30 random subsets each, split 0).
Writes benchmarks/dev/fair_rules.json. Usage: python fair_rules.py
"""
import json, os, sys
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import practice_vs_palm as pv

DATASETS = ("mcf_ida", "mcf_tram", "kang")
OUT = os.path.join(ROOT, "benchmarks", "dev", "fair_rules.json")


def auroc(score, beta):
    from sklearn.metrics import roc_auc_score
    null, real = np.abs(beta) < 0.05, np.abs(beta) >= 0.1
    if not real.any() or not null.any():
        return np.nan
    return roc_auc_score(np.r_[np.zeros(null.sum()), np.ones(real.sum())], np.r_[score[null], score[real]])


def main():
    out = dict(rows=[], curve={})
    for dsn in DATASETS:
        k = cl.KNOWN[dsn]
        for split in range(3):
            pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
            ct, pg = list(inp.cell_types), list(inp.programs)
            j = pg.index(k["program"])
            resp = np.abs(beta[:, j]) >= 0.1
            resp &= np.sign(beta[:, j]) == k["sign"]
            nonresp = ~np.isin(ct, k["types"]) if k["types"] else np.zeros(len(ct), bool)
            null = np.abs(beta) < 0.05
            act = np.load(os.path.join(pv.DEV, dsn, "s%d__instaprism.npz" % split))["model__activity"]
            e, lo, hi, p = pv.practice_test(act, T, donors)
            q = pv.bh(p) < 0.1
            a = json.load(open(os.path.join(cl.OUT, "%s__s%d__auto_m.json" % (dsn, split))))["arrays"]
            pe, pl, ph = (np.array(a[x], float) for x in ("effects", "lo", "hi"))
            pc = (pl > 0) | (ph < 0)
            for method, est, l_, h_, rules in (
                    ("InstaPrism + t-test", e, lo, hi, {"q-only": q, "q+cut-off": q & (np.abs(e) >= 0.1)}),
                    ("PALM", pe, pl, ph, {"q-only": pc, "q+cut-off": pc & (np.abs(pe) >= 0.1)})):
                for rule, calls in rules.items():
                    right = calls[:, j] & (np.sign(est[:, j]) == k["sign"])
                    out["rows"].append(dict(
                        dataset=dsn, split=split, method=method, rule=rule,
                        unchanged_flagged=float(calls[null].mean()),
                        responders_found=float(right[resp].mean()) if resp.any() else None,
                        nonresponders_flagged=float(calls[nonresp, j].mean()) if nonresp.any() else None,
                        coverage=float(((beta >= l_) & (beta <= h_)).mean()),
                        mae=float(np.abs(est - beta).mean()), auroc=float(auroc(np.abs(est), beta))))
            if split == 0:
                u = np.unique(donors)
                rng = np.random.default_rng(0)
                cur = {}
                for m in (3, 5, 10, 20):
                    f = []
                    for _ in range(30 if m < len(u) else 1):
                        pick = rng.choice(u, min(m, len(u)), replace=False)
                        sel = np.isin(donors, pick)
                        _, _, _, pp = pv.practice_test(act[sel], T[sel], donors[sel])
                        f.append(float(((pv.bh(pp) < 0.1) & null).sum() / null.sum()))
                    cur[m] = float(np.mean(f))
                out["curve"][dsn] = dict(instaprism=cur, palm_full=float(pc[null].mean()), n_full=int(len(u)))
            print(dsn, split, flush=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    import pandas as pd
    df = pd.DataFrame(out["rows"])
    pd.set_option("display.width", 200)
    print(df.groupby(["dataset", "method", "rule"]).mean(numeric_only=True).round(3).to_string())
    print(json.dumps(out["curve"], indent=1))


if __name__ == "__main__":
    main()
