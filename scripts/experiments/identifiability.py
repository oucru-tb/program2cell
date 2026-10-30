"""Identifiability of cell-type-specific effects from bulk: a design calculator, checked against the benchmarks.

For a design (fractions pi, reference B, programs G, treatment T), the treated-arm mean of the leakage operator
A_bar ((C P) x (C P), core.unmix.leakage_operator) maps true per-type shifts to reallocated activity. The
reallocated contrast has noise covariance Sigma (bootstrap contrast_se^2, diagonal). The best any estimator
can do without extra assumptions is to invert A_bar, so
    V = A_bar^-1 Sigma A_bar^-T,     MDE_cp = (z_0.95 + z_0.80) sqrt(V_cp,cp)
is the minimum detectable effect of type c, program p (90% two-sided, 80% power), and
    self-share_cp = A_bar[cp, cp] / sum_kq |A_bar[cp, kq]|
the share of the estimate that is the type's own signal. Both are computed from the design alone (no truth).
The check: across datasets, splits, types and programs, does MDE predict PALM's error, interval width and
which real effects get detected?

Usage: python identifiability.py run DATASETS SPLITS   |   python identifiability.py summary
Writes benchmarks/dev/identifiability/{ds}__s{split}.json
"""
import json, os, sys
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl

OUT = os.path.join(ROOT, "benchmarks", "dev", "identifiability")
Z = 1.645 + 0.842


def design_calculator(pi, B, G, T, raw, groups=None, seed=0):
    """Per (type, program): MDE (C, P), self-share (C, P), condition number of A_bar."""
    from palm_bulk.core import unmix as um
    T = np.asarray(T)
    A = um.leakage_operator(pi, B, G, diagonal=False)               # (N, C, P, C, P)
    N, C, P = A.shape[:3]
    Ab = A[T == 1].mean(0).reshape(C * P, C * P)
    se = um.contrast_se(raw, T, np.random.default_rng(seed), groups=groups).reshape(-1)
    Ai = np.linalg.pinv(Ab)
    V = Ai @ np.diag(se ** 2) @ Ai.T
    mde = Z * np.sqrt(np.clip(np.diag(V), 0, None)).reshape(C, P)
    share = (np.diag(Ab) / np.abs(Ab).sum(1)).reshape(C, P)
    # full design: every sample's own operator (composition varies across samples), per-(type, program)
    # intercepts; rows weighted by the within-arm noise sd of the reallocated activity. Generalised least
    # squares covariance of the effects = what unmixing can reach without shrinkage or calibration.
    y = np.asarray(raw, float).reshape(N, C * P)
    res = y.copy()
    for t in (0, 1):
        res[T == t] -= y[T == t].mean(0)
    sd = np.sqrt((res ** 2).sum(0) / max(N - 2, 1))                  # (C P,)
    An = A.reshape(N, C * P, C * P)
    rows = []
    for n in range(N):
        X = np.concatenate([np.eye(C * P), T[n] * An[n]], 1) / sd[:, None]
        rows.append(X)
    X = np.concatenate(rows, 0)
    M = np.linalg.pinv(X.T @ X)
    mde_full = Z * np.sqrt(np.clip(np.diag(M)[C * P:], 0, None)).reshape(C, P)
    return dict(mde=mde, mde_full=mde_full, self_share=share, cond=float(np.linalg.cond(Ab)),
                contrast_se=se.reshape(C, P), frac_mean=pi.mean(0),
                frac_cv=pi.std(0) / np.maximum(pi.mean(0), 1e-9))


def run(datasets, splits):
    from palm_bulk.core.realloc import reallocate
    os.makedirs(OUT, exist_ok=True)
    for dsn in datasets:
        for split in splits:
            f = os.path.join(OUT, "%s__s%d.json" % (dsn, split))
            if os.path.exists(f):
                continue
            pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
            pi = np.median(src["fraction_draws"], 0) if "fraction_draws" in src else None
            if pi is None:
                continue
            raw = reallocate(inp.Y, pi, inp.B, inp.Gmat)
            paired = dsn in ("kang", "zhang")
            d = design_calculator(pi, inp.B, inp.Gmat, T, raw, groups=donors if paired else None, seed=split)
            out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
            out.update(dataset=dsn, split=split, cell_types=list(inp.cell_types), programs=list(inp.programs),
                       beta=beta.tolist(), n=int(len(T)))
            for name in ("auto_m", "auto_abs"):
                p = os.path.join(cl.OUT, "%s__s%d__%s.json" % (dsn, split, name))
                if os.path.exists(p):
                    a = json.load(open(p))["arrays"]
                    out[name] = {k: a[k] for k in ("effects", "lo", "hi", "calls")}
            with open(f, "w") as fh:
                json.dump(out, fh)
            print("%s s%d cond %.0f  MDE median %.3f  self-share median %.2f" % (
                dsn, split, d["cond"], np.median(d["mde"]), np.median(d["self_share"])), flush=True)


def summary(name="auto_abs"):
    import pandas as pd
    from scipy import stats
    rows = []
    for f in sorted(os.listdir(OUT)):
        r = json.load(open(os.path.join(OUT, f)))
        est = r.get(name) or r.get("auto_m")
        if est is None:
            continue
        mde, share, beta = (np.array(r[k]) for k in (os.environ.get("MDE", "mde_full"), "self_share", "beta"))
        e, lo, hi, calls = (np.array(est[k], float) for k in ("effects", "lo", "hi", "calls"))
        for c, ct in enumerate(r["cell_types"]):
            for p, pg in enumerate(r["programs"]):
                rows.append(dict(dataset=r["dataset"], split=r["split"], cell_type=ct, program=pg,
                                 mde=mde[c, p], self_share=share[c, p], beta=beta[c, p],
                                 abs_err=abs(e[c, p] - beta[c, p]), half=(hi[c, p] - lo[c, p]) / 2,
                                 called=bool(calls[c, p]), frac=r["frac_mean"][c], cond=r["cond"]))
    df = pd.DataFrame(rows)
    df["detectable"] = np.abs(df.beta) >= df.mde
    print("units:", len(df), "datasets:", df.dataset.nunique())
    for y in ("abs_err", "half"):
        rho = stats.spearmanr(df.mde, df[y])[0]
        print("Spearman(MDE, %s) = %.2f   Spearman(self-share, %s) = %.2f" % (
            y, rho, y, stats.spearmanr(df.self_share, df[y])[0]))
    real = df[np.abs(df.beta) >= 0.1]
    print("\nReal effects (|true| >= 0.1): detection rate by predicted detectability")
    print(real.groupby("detectable").called.agg(["mean", "size"]).round(2))
    print("\nPer dataset: median MDE, median self-share, cond, share of real effects detectable / called")
    print(df.groupby("dataset").agg(mde=("mde", "median"), share=("self_share", "median"),
                                    cond=("cond", "median")).round(3))
    print(real.groupby("dataset").agg(detectable=("detectable", "mean"), called=("called", "mean")).round(2))
    print("\nPer cell type (median MDE):")
    print(df.groupby(["dataset", "cell_type"]).agg(mde=("mde", "median"), share=("self_share", "median"),
                                                   frac=("frac", "mean")).round(3).to_string())
    return df


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2].split(","), cl.db._splits(sys.argv[3]))
    else:
        summary(*(sys.argv[2:3]))
