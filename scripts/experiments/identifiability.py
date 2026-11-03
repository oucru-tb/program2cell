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

Formal statement (docs/method_spec.md, "Identifiability"): with reallocated activity
    r_n = a + T_n A_n beta + e_n,   e_n ~ (0, diag(s^2)),
every linear unbiased estimator of beta has covariance >= M = [(X' W X)^-1]_beta (Gauss-Markov; X stacks
[I, T_n A_n], W = diag(s^-2)), so no estimator without extra assumptions detects beta_cp below
MDE_cp = (z_0.95 + z_0.80) sqrt(M_cp,cp). Paired designs: per-donor intercepts, removed by centring the rows
within donor (Frisch-Waugh-Lovell). Fraction uncertainty: MDE over the posterior draws of pi.
power: MDE against arm size n (samples resampled per arm) and composition spread s (pi -> mean + s (pi - mean)).

Usage: python identifiability.py run DATASETS SPLITS   |   python identifiability.py summary
       python identifiability.py power DATASETS [SPLIT]   |   python identifiability.py powerfig
Writes benchmarks/dev/identifiability/{ds}__s{split}.json and power__{ds}.json
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
    from program2cell.core import unmix as um
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


def _sample_info(A, T, sd, groups=None):
    """Per-sample information blocks X_n' W X_n (N, 2CP, 2CP) of the design [I, T_n A_n] (rows / sd); with
    groups, rows centred within group (per-group intercepts) and the intercept block dropped."""
    N = A.shape[0]
    K = A.shape[1] * A.shape[2]
    An = A.reshape(N, K, K)
    X = np.concatenate([np.broadcast_to(np.eye(K), (N, K, K)), np.asarray(T, float)[:, None, None] * An], 2)
    X = X / sd[None, :, None]
    if groups is not None:
        g = np.asarray(groups).astype(str)
        X = X[:, :, K:].copy()
        for u in np.unique(g):
            m = g == u
            X[m] -= X[m].mean(0)
    return np.einsum("nkj,nkl->njl", X, X)


def _mde(info, K):
    M = np.linalg.pinv(info.sum(0))
    return Z * np.sqrt(np.clip(np.diag(M)[-K:], 0, None))


def _noise_sd(y, T, groups=None):
    res = np.asarray(y, float).copy()
    T = np.asarray(T)
    if groups is not None:                                       # within donor, after the arm effect
        for t in (0, 1):
            res[T == t] -= res[T == t].mean(0)
        g = np.asarray(groups).astype(str)
        for u in np.unique(g):
            res[g == u] -= res[g == u].mean(0)
        dof = len(res) - 1 - len(np.unique(g))
    else:
        for t in (0, 1):
            res[T == t] -= res[T == t].mean(0)
        dof = len(res) - 2
    return np.sqrt((res ** 2).sum(0) / max(dof, 1))


def power(dsn, split=0, sizes=(3, 5, 10, 20, 50, 100, 200), spreads=(0.25, 0.5, 1.0, 2.0), reps=20):
    """MDE (C, P) against arm size and composition spread, unpaired (paired designs: also paired); and under
    fraction uncertainty (posterior draws) at the observed design."""
    from program2cell.core import unmix as um
    from program2cell.core.realloc import reallocate
    pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
    if "fraction_draws" not in src:
        return
    draws = np.asarray(src["fraction_draws"])
    pi = np.median(draws, 0)
    raw = reallocate(inp.Y, pi, inp.B, inp.Gmat)
    N, C, P = raw.shape
    K = C * P
    y = raw.reshape(N, K)
    sd = _noise_sd(y, T)
    rng = np.random.default_rng(split)
    i1, i0 = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
    out = dict(dataset=dsn, split=split, cell_types=list(inp.cell_types), programs=list(inp.programs),
               beta=beta.tolist(), sizes=list(sizes), spreads=list(spreads), n_obs=[int(len(i0)), int(len(i1))],
               frac_mean=pi.mean(0).tolist(), mde=[])
    for s in spreads:
        ps = np.clip(pi.mean(0) + s * (pi - pi.mean(0)), 1e-4, None)
        ps = ps / ps.sum(1, keepdims=True)
        info = _sample_info(um.leakage_operator(ps, inp.B, inp.Gmat, diagonal=False), T, sd)
        row = []
        for m in sizes:
            v = [_mde(info[np.concatenate([rng.choice(i0, m), rng.choice(i1, m)])], K) for _ in range(reps)]
            row.append(np.median(v, 0).reshape(C, P).tolist())
        out["mde"].append(row)
    A = um.leakage_operator(pi, inp.B, inp.Gmat, diagonal=False)
    out["mde_observed"] = _mde(_sample_info(A, T, sd), K).reshape(C, P).tolist()
    sub = draws[np.linspace(0, len(draws) - 1, 10).astype(int)]
    md = [_mde(_sample_info(um.leakage_operator(z, inp.B, inp.Gmat, diagonal=False), T, sd), K) for z in sub]
    out["mde_draws_q90"] = np.quantile(md, 0.9, 0).reshape(C, P).tolist()
    if dsn in ("kang", "zhang"):                                   # paired: per-donor intercepts
        sdp = _noise_sd(y, T, donors)
        out["mde_paired_observed"] = _mde(_sample_info(A, T, sdp, donors), K).reshape(C, P).tolist()
        # arm size for paired designs: resample whole donors
        g = np.asarray(donors).astype(str)
        units = np.unique(g)
        infop = _sample_info(A, T, sdp, donors)
        row = []
        for m in sizes:
            v = []
            for _ in range(reps):
                pick = rng.choice(units, m)
                v.append(_mde(np.concatenate([infop[g == u] for u in pick]), K))
            row.append(np.median(v, 0).reshape(C, P).tolist())
        out["mde_paired"] = row
    with open(os.path.join(OUT, "power__%s.json" % dsn), "w") as fh:
        json.dump(out, fh)
    obs = np.array(out["mde_observed"])
    print("%s: observed median MDE %.3f; at n=20/arm %.3f; fraction-uncertainty q90 %.3f" % (
        dsn, np.median(obs), np.median(out["mde"][spreads.index(1.0)][sizes.index(20)]),
        np.median(out["mde_draws_q90"])), flush=True)


def run(datasets, splits):
    from program2cell.core.realloc import reallocate
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
    elif sys.argv[1] == "power":
        for d in sys.argv[2].split(","):
            power(d, int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        summary(*(sys.argv[2:3]))
