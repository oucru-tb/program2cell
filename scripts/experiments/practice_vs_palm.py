"""C1 + C2 (research_plan 3.10): current practice vs PALM on the known-answer datasets.

C1, current practice: each competitor's per-sample per-type program activity (InstaPrism's per-type expression,
or its fractions with reallocation, scored with PROGENy), then one test per cell type and program, as a user
would run it:
  paired t-test over donors (treated mean - control mean within each donor that has both arms; Welch t-test
  when no donor has both); TOAST / TCA (cell-type-specific DE, gene log fold changes only): decoupler-style
  ULM of the gene log fold changes on the program weights, per type.
Two call rules: 'p10' (unadjusted p < 0.1, i.e. the 90% interval excludes 0, PALM's own rule) and 'bh10'
(Benjamini-Hochberg q < 0.1 over all type x program pairs, the usual practice).
C2: PALM's module with its default (causal) settings on the same competitor's fractions.
PALM itself: the auto_m rows of causal_loop (PALM posterior fractions).

Scores (causal_loop.score): coverage of the 90% interval (where one exists), r with the true effect, calls on
truly null pairs (fpr_null), responder AUROC and truth-based TPR on the known program.

Usage: python practice_vs_palm.py run DATASETS SPLITS [METHODS]   |   python practice_vs_palm.py summary [heldout]
One json per (dataset, split, method) in benchmarks/dev/practice/. Restart-safe.
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd
from scipy import stats

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl

OUT = os.path.join(ROOT, "benchmarks", "dev", "practice")
DEV = os.path.join(ROOT, "benchmarks", "dev")
# dev_bench rows with per-sample activity (file stem -> row prefix in the npz)
ACTIVITY = {"naive": "model", "reallocation": "model", "cibersort": "model", "cellgeometry": "model",
            "cellgeometry_cells": "model", "instaprism": "model", "instaprism_cells": "model", "blade": "model"}
CSDE = ("toast", "tca")


def practice_test(act, T, donors, level=0.9):
    """act (N, C, P) -> effect, lo, hi, p (C, P): paired t-test over donors with both arms, else Welch."""
    T = np.asarray(T).astype(bool)
    donors = np.asarray(donors).astype(str)
    both = [d for d in np.unique(donors) if T[donors == d].any() and (~T[donors == d]).any()]
    if len(both) >= 2:
        d = np.stack([act[(donors == u) & T].mean(0) - act[(donors == u) & ~T].mean(0) for u in both])
        eff, se, df = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(both)), len(both) - 1
    else:
        a, b = act[T], act[~T]
        va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
        eff, se = a.mean(0) - b.mean(0), np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    se = np.maximum(se, 1e-12)
    p = 2 * stats.t.sf(np.abs(eff / se), df)
    q = stats.t.ppf(0.5 + level / 2, df)
    return eff, eff - q * se, eff + q * se, p


def ulm_p(lfc, G):
    """lfc (C, Ggenes), G (P, Ggenes) -> p (C, P): t-test of the slope of lfc on the program weights."""
    C, P = lfc.shape[0], G.shape[0]
    p = np.full((C, P), np.nan)
    for c in range(C):
        y = lfc[c]
        ok = np.isfinite(y)
        for j in range(P):
            r = stats.linregress(G[j, ok], y[ok])
            p[c, j] = r.pvalue
    return p


def bh(p):
    p = np.asarray(p, float)
    flat = p.ravel()
    ok = np.flatnonzero(np.isfinite(flat))
    o = ok[np.argsort(flat[ok])]
    m = len(o)
    q = np.full_like(flat, np.nan)
    adj = np.minimum.accumulate((flat[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    q[o] = np.minimum(adj, 1)
    return q.reshape(p.shape)


def _methods(dsn, split):
    have = [os.path.basename(f)[len("s%d__" % split):-4] for f in glob.glob(os.path.join(DEV, dsn, "s%d__*.npz" % split))]
    return [m for m in have if m in ACTIVITY or m in CSDE]


def _save(dsn, split, name, rows):
    with open(os.path.join(OUT, "%s__s%d__%s.json" % (dsn, split, name)), "w") as fh:
        json.dump(rows, fh, default=float)


def run(datasets, splits, only=None):
    os.makedirs(OUT, exist_ok=True)
    for dsn in datasets:
        for split in splits:
            methods = [m for m in _methods(dsn, split) if only is None or m in only]
            todo = [m for m in methods for kind in ("practice", "palm_module")
                    if not os.path.exists(os.path.join(OUT, "%s__s%d__%s__%s.json" % (dsn, split, m, kind)))
                    and not (kind == "palm_module" and m in ("naive",))]
            if not todo:
                continue
            pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
            ct, pg, known = list(inp.cell_types), list(inp.programs), cl.KNOWN.get(dsn)

            def rows_for(eff, lo, hi, p, extra):
                out = []
                for rule, calls in (("p10", p < 0.1), ("bh10", bh(p) < 0.1)):
                    calls = np.nan_to_num(calls, nan=False).astype(bool)
                    lo_, hi_ = (lo, hi) if lo is not None else (np.full_like(eff, -np.inf), np.full_like(eff, np.inf))
                    s = cl.score(eff, lo_, hi_, calls, beta, ct, pg, known)
                    if lo is None:
                        s["cov"] = s["width"] = s["cov_known"] = np.nan
                    s.update(rule=rule, **extra)
                    out.append(s)
                return out

            for m in methods:
                z = np.load(os.path.join(DEV, dsn, "s%d__%s.npz" % (split, m)))
                f = os.path.join(OUT, "%s__s%d__%s__practice.json" % (dsn, split, m))
                if not os.path.exists(f):
                    t0 = time.time()
                    if m in CSDE:
                        eff = z["model__effects"]
                        lfc_path = os.path.join(DEV, "work", dsn, "s%d" % split, m, "logfc.csv")
                        lfc = pd.read_csv(lfc_path, index_col=0).reindex(index=list(inp.genes), columns=ct).values.T
                        p = ulm_p(lfc, np.asarray(inp.Gmat))
                        rows = rows_for(eff, None, None, p, {})
                    else:
                        eff, lo, hi, p = practice_test(z["%s__activity" % ACTIVITY[m]], T, donors)
                        rows = rows_for(eff, lo, hi, p, {})
                    for r in rows:
                        r.update(dataset=dsn, split=split, method=m, kind="practice", seconds=time.time() - t0)
                    _save(dsn, split, "%s__practice" % m, rows)
                    print("[%s] %s s%d %s practice: fpr_null %.2f tpr_t %s" % (
                        time.strftime("%H:%M"), dsn, split, m, rows[1]["fpr_null"], rows[1].get("tpr_t")), flush=True)
                f = os.path.join(OUT, "%s__s%d__%s__palm_module.json" % (dsn, split, m))
                if m != "naive" and not os.path.exists(f):
                    t0 = time.time()
                    pi = np.clip(z["model__fractions"], 1e-6, None)
                    pi = pi / pi.sum(1, keepdims=True)
                    r = pb.program_deconvolution(inp, cal, fractions=pi, seed=split)
                    lo, hi = r.effects_interval
                    s = cl.score(r.effects, lo, hi, r.calls, beta, ct, pg, known)
                    s.update(dataset=dsn, split=split, method=m, kind="palm_module", rule="palm",
                             seconds=time.time() - t0)
                    _save(dsn, split, "%s__palm_module" % m, [s])
                    print("[%s] %s s%d %s + PALM module: cov %.2f fpr_null %.2f tpr_t %s (%.0fs)" % (
                        time.strftime("%H:%M"), dsn, split, m, s["cov"], s["fpr_null"], s.get("tpr_t"),
                        s["seconds"]), flush=True)


def _palm_rows():
    rows = []
    for f in glob.glob(os.path.join(cl.OUT, "*__auto_m.json")):
        s = json.load(open(f))
        a = s.pop("arrays")
        s.update(cl.score(*(np.asarray(a[k], float) for k in ("effects", "lo", "hi")), np.asarray(a["calls"], bool),
                          np.asarray(a["beta"], float), a["cell_types"], a["programs"], cl.KNOWN.get(s["dataset"])))
        s.update(method="palm", kind="palm", rule="palm")
        rows.append(s)
    return rows


def summary(heldout=False):
    """Development splits (< 10) or, with heldout=True, the held-out splits only."""
    from palm_bulk.compare import known_answer as ka
    rows = [r for f in sorted(glob.glob(os.path.join(OUT, "*.json"))) for r in json.load(open(f))] + _palm_rows()
    df = pd.DataFrame(rows)
    df = df[(df.split >= cl.db.HELDOUT_FROM) == heldout]
    df = df[df.split.isin(sorted(set(df[df.kind != "palm"].split)))]
    cols = ["cov", "r_t", "rmse", "calls", "fpr_null", "auroc", "tpr_t"]
    t = df.groupby(["dataset", "kind", "method", "rule"])[cols].mean()
    t.insert(0, "n", df.groupby(["dataset", "kind", "method", "rule"]).size())
    pd.set_option("display.width", 250)
    print(t.round(3).to_string())
    # paired over (dataset, split): PALM minus each alternative, on fpr_null and tpr_t
    print("\n== PALM (auto_m) minus alternative, paired over dataset x split [95% bootstrap CI]")
    palm = df[df.kind == "palm"].set_index(["dataset", "split"])
    out = {}
    for (kind, m, rule), sub in df[df.kind != "palm"].groupby(["kind", "method", "rule"]):
        sub = sub.set_index(["dataset", "split"])
        idx = sub.index.intersection(palm.index)
        res = {}
        for metric in ("cov", "fpr_null", "tpr_t", "auroc"):
            r = ka.paired_test(palm.loc[idx, metric].values, sub.loc[idx, metric].values)
            if r["n"] >= 2:
                res[metric] = "%+.3f [%+.3f, %+.3f] n=%d" % (r["diff"], r["lo"], r["hi"], r["n"])
        out["%s | %s | %s" % (kind, m, rule)] = res
    print(pd.DataFrame(out).T.to_string())


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2].split(","), cl.db._splits(sys.argv[3]), sys.argv[4].split(",") if len(sys.argv) > 4 else None)
    else:
        summary(heldout=len(sys.argv) > 2 and sys.argv[2] == "heldout")
