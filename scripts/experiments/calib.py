"""Calibrated reallocation (experiment on saved dev_bench fits).

Calibration set: reference donors (never test donors) with every cell type >= min_cells; one specimen per
donor; K-fold over donors. For fold k the reference profiles B_k come from the other reference donors
only, the specimen's counts are thinned in two halves (bulk / truth) exactly as the test set, and
  dhat = realloc(bulk, true pi, B_k)       (true pi: fraction error is added at test time from the draws)
  d    = estimate_delta(truth half, B_k)
Per (cell type c, program p) a ridge regression d_c,p ~ 1 + dhat_{.,p} (all types: data-driven de-mixing)
or ~ 1 + dhat_c,p (own type: affine recalibration) is fitted on the calibration set; out-of-fold
residuals give the noise level.

On the test set (the dev_bench split), with PALM's posterior fractions:
  activity  dtil = f(posterior realloc); 90% interval = split-conformal |residual| quantile per (c,p),
            widened in quadrature by the spread of f over the fraction draws
  effects   btil = design(dtil); variance = bootstrap over calibration donors (coefficient
            uncertainty) + per-(c,p) residual variance x (1/n1 + 1/n0) + fraction-draw variance
Rows: raw posterior realloc (its own intervals), calib own, calib all.
Usage: python calib.py DATASET CONFIG [n_splits] [max_cal]
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import norm

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dev_bench as db
import palm_bulk as pb
from palm_bulk.core import PseudobulkTable, make_benchmark
from palm_bulk.core.pseudobulk import _csr
from palm_bulk.core.scores import estimate_delta
from palm_bulk.compare.base import MethodResult

ds_name, config = sys.argv[1], sys.argv[2]
n_splits = int(sys.argv[3]) if len(sys.argv) > 3 else 10
MAX_CAL = int(sys.argv[4]) if len(sys.argv) > 4 else 300
K, NBOOT, LAM, LEVEL = 5, 200, 1e-3, 0.9
ds = db.DATASETS[ds_name]
table = PseudobulkTable.from_h5ad(ds["path"], {}, db.TYPES)
prog = pb.load_programs(os.path.join(ROOT, "data", "progeny_human_top500.csv")).loc[db.PROGS]
X = _csr(table.counts)
obs = table.obs
pos = {k: i for i, k in enumerate(obs.index)}


def profiles(donors, types, gidx):
    """B (C, G) on the given genes from the reference rows of `donors`, built as make_benchmark does."""
    ref = obs[obs["donor_id"].astype(str).isin(donors) & (obs["n_cells"] >= 10)]
    B = np.zeros((len(types), X.shape[1]))
    for k, c in enumerate(types):
        prof = []
        for _, rd in ref[ref["cell_type"] == c].groupby("donor_id"):
            s = np.asarray(X[[pos[i] for i in rd.index]].sum(0)).ravel()
            prof.append(s / s.sum())
        B[k] = np.mean(prof, 0)
    B = (B + 1e-6) / (B + 1e-6).sum(1, keepdims=True)
    B = B[:, gidx]
    return B / B.sum(1, keepdims=True)


def realloc(Y, pi, B, G):
    w = pi[:, :, None] * B[None]
    return estimate_delta(Y[:, None, :] * (w / w.sum(1, keepdims=True)), B, G)


def calibration_set(test_donors, types, gidx, G, seed):
    rng = np.random.default_rng(seed + 100)
    wide = obs.pivot_table(index="specimen", columns="cell_type", values="n_cells", aggfunc="sum",
                           fill_value=0).reindex(columns=types, fill_value=0)
    spec = obs.drop_duplicates("specimen").set_index("specimen")
    ok = (wide >= 20).all(1)
    elig = spec.loc[ok[ok].index]
    elig = elig[~elig["donor_id"].astype(str).isin(test_donors)]
    donors = rng.permutation(np.array(sorted(map(str, elig["donor_id"].unique())), dtype=object))[:MAX_CAL]
    all_ref = set(map(str, obs["donor_id"].unique())) - set(test_donors)
    folds = np.arange(len(donors)) % K
    out = dict(dhat=[], d=[], donor=[])
    for k in range(K):
        cal = donors[folds == k]
        B = profiles(all_ref - set(cal), types, gidx)
        tc = np.zeros((len(cal), len(types), len(gidx)))
        for i, dnr in enumerate(cal):
            s = rng.choice(np.array(elig.index[elig["donor_id"].astype(str) == dnr], dtype=object))
            rows = obs[obs["specimen"] == s]
            for j, c in enumerate(types):
                r = rows[rows["cell_type"] == c]
                tc[i, j] = np.asarray(X[[pos[q] for q in r.index]].sum(0)).ravel()[gidx]
        half = rng.binomial(tc.astype(np.int64), 0.5).astype(float)
        held = tc - half
        tot = half.sum(-1)
        pi = tot / tot.sum(1, keepdims=True)
        out["dhat"].append(realloc(half.sum(1), pi, B, G))
        out["d"].append(estimate_delta(held, B, G))
        out["donor"] += list(cal)
    return np.concatenate(out["dhat"]), np.concatenate(out["d"]), np.array(out["donor"])


def design_mat(dh, c, p, own):
    """(n, k) predictors for target (c, p)."""
    Xp = dh[:, c:c + 1, p] if own else dh[:, :, p]
    return np.column_stack([np.ones(len(dh)), Xp])


def fit(dh, d, own):
    N, C, P = d.shape
    W = {}
    for c in range(C):
        for p in range(P):
            Xm = design_mat(dh, c, p, own)
            R = LAM * np.eye(Xm.shape[1]); R[0, 0] = 0
            W[c, p] = np.linalg.solve(Xm.T @ Xm + R, Xm.T @ d[:, c, p])
    return W


def predict(W, dh, own):
    out = np.zeros(dh.shape)
    for (c, p), w in W.items():
        out[..., c, p] = (design_mat(dh.reshape(-1, *dh.shape[-2:]), c, p, own) @ w).reshape(dh.shape[:-2])
    return out


rows = []
for split in range(n_splits):
    f = os.path.join(ROOT, "benchmarks", "dev", ds_name, "s%d__%s.npz" % (split, db._slug(config)))
    if not os.path.exists(f):
        continue
    z = np.load(f)
    b = make_benchmark(table, min_cells=20, seed=split, truth_split=True, contrast=ds["contrast"], n_test=ds["n_test"])
    inp = pb.prepare_inputs(b.bulk, b.reference, prog, treatment=b.treatment, max_genes=1000)
    tz = np.load(os.path.join(ROOT, "benchmarks", "dev", ds_name, "s%d__truth.npz" % split))
    T = tz["T"]
    design = db._design(T)
    truth = dict(pi=tz["pi"], delta=tz["delta"], beta=design(tz["delta"]))
    truth_ulm = dict(pi=tz["pi"], delta=tz["ulm"])
    gidx = [table.genes.index(g) for g in inp.genes]
    test_donors = set(b.meta["donor_id"].astype(str))
    dh_cal, d_cal, donors = calibration_set(test_donors, inp.cell_types, gidx, inp.Gmat, split)
    n1, n0 = (T == 1).sum(), (T == 0).sum()
    pi_med = np.median(z["pi"], 0)
    r_test, r_draws = z["realloc"], z["realloc_draws"]                   # (N,C,P), (S,N,C,P)

    def add(name, act, act_iv, eff, eff_iv):
        res = MethodResult(name, fractions=pi_med, activity=act, effects=eff,
                           intervals=dict(activity=act_iv, effects=eff_iv))
        s = db._score(res, truth, truth_ulm)
        s.update(split=split, method=name, n_cal=len(d_cal))
        rows.append(s)

    from palm_bulk.bayes.diagnostics import interval
    add("posterior realloc", r_test, interval(r_draws), design(r_test), interval(design(r_draws)))
    for own in (True, False):
        name = "calib own" if own else "calib all"
        # out-of-fold residuals within the calibration set (folds by donor)
        ud = np.unique(donors)
        fold = {d_: i % K for i, d_ in enumerate(np.random.default_rng(split).permutation(ud))}
        fk = np.array([fold[d_] for d_ in donors])
        resid = np.zeros_like(d_cal)
        for k in range(K):
            W = fit(dh_cal[fk != k], d_cal[fk != k], own)
            resid[fk == k] = d_cal[fk == k] - predict(W, dh_cal[fk == k], own)
        nc = len(d_cal)
        qlev = min(1.0, np.ceil((nc + 1) * LEVEL) / nc)
        q = np.quantile(np.abs(resid), qlev, axis=0)                      # (C, P) conformal half-width
        sig2 = resid.var(0)
        W = fit(dh_cal, d_cal, own)
        act = predict(W, r_test, own)
        act_draws = predict(W, r_draws, own)
        h_draw = (np.quantile(act_draws, 0.95, 0) - np.quantile(act_draws, 0.05, 0)) / 2
        hw = np.sqrt(q[None] ** 2 + h_draw ** 2)
        eff = design(act)
        # effects: bootstrap over calibration donors for coefficient uncertainty
        rng = np.random.default_rng(split + 7)
        boot = []
        for _ in range(NBOOT):
            pick = rng.choice(ud, len(ud))
            idx = np.concatenate([np.flatnonzero(donors == d_) for d_ in pick])
            boot.append(design(predict(fit(dh_cal[idx], d_cal[idx], own), r_test, own)))
        v = np.var(boot, 0) + sig2 * (1 / n1 + 1 / n0) + np.var(design(act_draws), 0)
        zq = norm.ppf(0.5 + LEVEL / 2)
        add(name, act, (act - hw, act + hw), eff, (eff - zq * np.sqrt(v), eff + zq * np.sqrt(v)))
        if not own:
            # effects by inverse calibration: per program, dhat = a + M d + e learned on the calibration set
            # (M: gain on the diagonal, leakage off it); a contrast of dhat then estimates M beta, so
            # beta = M^-1 contrast(dhat). Averaging over samples removes most of e before inverting.
            def inv_fit(dh, d):
                Ms, Ss = [], []
                for p in range(d.shape[2]):
                    D1 = np.column_stack([np.ones(len(d)), d[:, :, p]])
                    coef = np.linalg.lstsq(D1, dh[:, :, p], rcond=None)[0]            # (C+1, C)
                    Ms.append(coef[1:].T)                                            # M[c, c']
                    Ss.append(np.cov((dh[:, :, p] - D1 @ coef).T))
                return Ms, Ss

            def inv_apply(Ms, bh):                                                   # bh (..., C, P)
                return np.stack([np.einsum("ij,...j->...i", np.linalg.inv(M), bh[..., p])
                                 for p, M in enumerate(Ms)], -1)

            Ms, Ss = inv_fit(dh_cal, d_cal)
            eff_i = inv_apply(Ms, design(r_test))
            v_res = np.stack([np.diag(np.linalg.inv(M) @ S @ np.linalg.inv(M).T) for M, S in zip(Ms, Ss)], -1)
            boot_i = []
            for _ in range(NBOOT):
                pick = rng.choice(ud, len(ud))
                idx = np.concatenate([np.flatnonzero(donors == d_) for d_ in pick])
                boot_i.append(inv_apply(inv_fit(dh_cal[idx], d_cal[idx])[0], design(r_test)))
            boot_i = np.array(boot_i)
            sd_b = (np.quantile(boot_i, 0.95, 0) - np.quantile(boot_i, 0.05, 0)) / (2 * zq)   # robust to outliers
            v_i = sd_b ** 2 + v_res * (1 / n1 + 1 / n0) + np.var(inv_apply(Ms, design(r_draws)), 0)
            add("calib inv", act, (act - hw, act + hw), eff_i, (eff_i - zq * np.sqrt(v_i), eff_i + zq * np.sqrt(v_i)))
    print("split %d: %d calibration donors" % (split, len(d_cal)), flush=True)

df = pd.DataFrame(rows)
out = os.path.join(ROOT, "benchmarks", "dev", "calib_%s_%s.json" % (ds_name, db._slug(config)))
df.to_json(out, orient="records", indent=1)
cols = dict(act_spearman_across_samples="rho", act_spearman_specific="spec", act_gain="gain", act_leakage="leak",
            act_mae="mae", act_cov90="cov", act_iscore="iscore", ulm_act_spearman_across_samples="ulm_rho",
            eff_pearson="eff_r", eff_mae="eff_mae", eff_cov90="eff_cov", eff_iscore="eff_is", eff_fsr="fsr", eff_fdr="fdr",
            eff_n_called="n_call", eff_localise="local")
t = df.groupby("method", sort=False)[[c for c in cols if c in df]].mean().rename(columns=cols)
pd.set_option("display.width", 250)
print("== %s %s, %d splits, ~%d calibration donors" % (ds_name, config, df.split.nunique(), df.n_cal.mean()))
print(t.round(3).to_string())
