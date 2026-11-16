"""Offline test of new effect priors and call rules on saved freeze v2 outputs (development, 2026-11-11).

Plan: docs/improvement_research_2026-11.md, Phase 1a (structured mixture prior, lfsr calls), 1c (lineage
patterns) and 3a (multiplicity). No PALM refit: each saved run (benchmarks/dev/causal_loop/<ds>__s<k>__auto_v2.json)
gives per-pair estimates e, 90% interval [lo, hi] and the truth beta (C types x P programs).

Observation model per study and program p: y_p = e[:, p] ~ N(beta_p, V_p), V_p = diag(s^2), s = (hi - lo) / (2 z_0.95)
(diagonal: the saved files hold no cross-type covariance; spillover errors are correlated, so this is the
weak form of the test). Prior: beta_p ~ sum_j w_j N(0, U_j), U_j from a FIXED set of patterns x scales:
  null        eps^2 I
  shared      sigma^2 11' + eps^2 I
  single k    sigma^2 e_k e_k' + eps^2 I          (one per type)
  lineage L   sigma^2 1_L 1_L' + eps^2 I          (PBMC types with a lineage; none for cell lines)
  indep       sigma^2 I
Weights w by EM over the P programs of the study (marginal y_p ~ N(0, U_j + V_p)), with a null-favouring Dirichlet
penalty (ashr style, lambda0 on null). Posterior: mixture of Gaussians, closed form; lfsr; 90% credible interval.
eps: full rank (eps = EPS_FRAC x median s) vs low rank (eps = 0, point-null/low-rank components) to test whether
low-rank priors inflate false sign calls (Xie 2025, arXiv 2502.16118).

Methods scored (dataset means over splits; score as fraction_error.score plus wrong-sign and FDP):
  v2            saved calls (interval excludes 0)
  bh            Benjamini-Hochberg q < 0.10 on z = e / s over all pairs of the study
  mix_full      mixture prior, eps > 0, calls lfsr < 0.05, credible intervals
  mix_low       same, eps = 0
  mix_full_lin  mix_full + lineage patterns
  mix_full_10   mix_full, calls lfsr < 0.10
Usage: python v5_prior_offline.py [splits, default 20-29] [estimator tag, default auto_v2]
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import norm

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
L = os.path.join(ROOT, "benchmarks", "dev", "causal_loop")
OUT = os.path.join(ROOT, "benchmarks", "dev", "v5_prior_offline")
DS = ["mcf_ida", "mcf_tram", "tahoe_enc", "tahoe_gem", "tahoe_rmc", "tahoe_tram",
      "kang", "zhang", "parse_gmcsf", "parse_ifnb", "parse_ifng", "parse_il4", "parse_tnf"]
Z90 = norm.ppf(0.95)
SCALES = np.array([0.01, 0.02, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0])
EPS_FRAC = 0.25
LAMBDA0 = float(os.environ.get("LAMBDA0", 1.0))
NDRAW = 4000


def lineage(ct):
    s = ct.lower()
    if any(k in s for k in ("cd4", "cd8", "t cell", "treg", "mait", "gdt", "t_", " t")) or s in ("t", "tcells"):
        return "T"
    if any(k in s for k in ("mono", "dc", "macro", "myel")):
        return "Myeloid"
    if s.startswith("b") or "b cell" in s or "plasma" in s:
        return "B"
    if "nk" in s:
        return "NK"
    return None


def patterns(C, cell_types, eps, use_lineage):
    """List of (name, U) over patterns x scales."""
    I = np.eye(C)
    out = [("null", eps ** 2 * I)]
    for sg in SCALES:
        s2 = sg ** 2
        out.append(("shared", s2 * np.ones((C, C)) + eps ** 2 * I))
        out.append(("indep", s2 * I + eps ** 2 * I))
        for k in range(C):
            e = np.zeros(C); e[k] = 1
            out.append(("single", s2 * np.outer(e, e) + eps ** 2 * I))
        if use_lineage:
            lin = [lineage(c) for c in cell_types]
            for g in sorted({x for x in lin if x}):
                v = np.array([x == g for x in lin], float)
                if 1 < v.sum() < C:
                    out.append(("lineage", s2 * np.outer(v, v) + eps ** 2 * I))
    return out


def _logmvn(y, S):
    S = S + 1e-12 * np.eye(len(y))
    sign, ld = np.linalg.slogdet(S)
    return -0.5 * (ld + y @ np.linalg.solve(S, y) + len(y) * np.log(2 * np.pi))


def fit_mixture(Y, Sd, pats, iters=300):
    """Y (C, P) estimates, Sd (C, P) se. EM for mixture weights pooled over programs."""
    C, P = Y.shape
    J = len(pats)
    ll = np.array([[_logmvn(Y[:, p], U + np.diag(Sd[:, p] ** 2)) for (_, U) in pats] for p in range(P)])  # P x J
    prior = np.ones(J); prior[0] = LAMBDA0                       # Dirichlet pseudo-counts, null favoured
    w = np.ones(J) / J
    for _ in range(iters):
        lr = ll + np.log(w + 1e-300)
        r = np.exp(lr - lr.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
        w_new = (r.sum(0) + prior - 1); w_new = np.clip(w_new, 0, None); w_new /= w_new.sum()
        if np.abs(w_new - w).max() < 1e-8:
            w = w_new; break
        w = w_new
    lr = ll + np.log(w + 1e-300)
    R = np.exp(lr - lr.max(1, keepdims=True)); R /= R.sum(1, keepdims=True)  # P x J responsibilities
    return w, R


def posterior(Y, Sd, pats, R, rng):
    """Per program: mixture posterior. Returns mean, lfsr, lo, hi (C, P)."""
    C, P = Y.shape
    mean, lfsr, lo, hi = (np.zeros((C, P)) for _ in range(4))
    for p in range(P):
        V = np.diag(Sd[:, p] ** 2)
        mus, sds, ws = [], [], []
        for j, (_, U) in enumerate(pats):
            if R[p, j] < 1e-10:
                continue
            K = U @ np.linalg.inv(U + V)
            mus.append(K @ Y[:, p]); sds.append(np.sqrt(np.clip(np.diag(U - K @ U), 0, None))); ws.append(R[p, j])
        mus, sds, ws = np.array(mus), np.array(sds), np.array(ws); ws /= ws.sum()
        mean[:, p] = ws @ mus
        # P(beta > 0) and P(beta < 0); zero-variance mass at a zero mean counts on both sides (ashr convention)
        with np.errstate(divide="ignore", invalid="ignore"):
            zpos = np.where(sds > 0, norm.sf(0, mus, np.where(sds > 0, sds, 1)), (mus >= 0).astype(float))
            zneg = np.where(sds > 0, norm.cdf(0, mus, np.where(sds > 0, sds, 1)), (mus <= 0).astype(float))
        lfsr[:, p] = np.minimum(ws @ zneg, ws @ zpos)
        comp = rng.choice(len(ws), NDRAW, p=ws)
        draws = mus[comp] + sds[comp] * rng.standard_normal((NDRAW, C))
        lo[:, p], hi[:, p] = np.quantile(draws, 0.05, 0), np.quantile(draws, 0.95, 0)
    return mean, lfsr, lo, hi


def score(b, e, lo, hi, calls):
    big, null, small = np.abs(b) >= 0.1, np.abs(b) < 0.05, np.abs(b) >= 0.05
    ws = [np.corrcoef(e[:, p], b[:, p])[0, 1] for p in range(b.shape[1]) if b[:, p].std() >= 0.05]
    n = max(calls.sum(), 1)
    return dict(cov=((b >= lo) & (b <= hi)).mean(), r=np.corrcoef(e.ravel(), b.ravel())[0, 1],
                within_r=np.mean(ws) if ws else np.nan,
                power=(calls & big & (np.sign(e) == np.sign(b))).sum() / max(big.sum(), 1),
                null_calls=(calls & null).sum() / max(null.sum(), 1),
                wrong_sign=(calls & small & (np.sign(e) != np.sign(b))).sum() / n,
                fdp=(calls & null).sum() / n if calls.any() else 0.0,
                n_calls=float(calls.sum()), width=(hi - lo).mean())


def bh(p, q=0.10):
    m = p.size; o = np.argsort(p.ravel()); thr = q * np.arange(1, m + 1) / m
    ok = p.ravel()[o] <= thr
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    c = np.zeros(m, bool); c[o[:k]] = True
    return c.reshape(p.shape)


def run_one(f, rng):
    j = json.load(open(f))
    a = {k: np.array(j["arrays"][k], float) for k in ("effects", "lo", "hi", "beta", "calls")}
    ct = j["arrays"]["cell_types"]
    e, lo, hi, b = a["effects"], a["lo"], a["hi"], a["beta"]
    s = np.clip((hi - lo) / (2 * Z90), 1e-4, None)
    C = e.shape[0]
    res = {"v2": score(b, e, lo, hi, (lo > 0) | (hi < 0))}
    p = 2 * norm.sf(np.abs(e) / s)
    res["bh"] = score(b, e, lo, hi, bh(p))
    eps_full = EPS_FRAC * np.median(s)
    weights = {}
    for name, eps, lin, thr in (("mix_full", eps_full, False, 0.05), ("mix_low", 0.0, False, 0.05),
                                ("mix_full_lin", eps_full, True, 0.05), ("mix_full_10", eps_full, False, 0.10)):
        pats = patterns(C, ct, eps, lin)
        w, R = fit_mixture(e, s, pats)
        m, lf, l2, h2 = posterior(e, s, pats, R, rng)
        res[name] = score(b, m, l2, h2, lf < thr)
        agg = {}
        for (pn, _), wj in zip(pats, w):
            agg[pn] = agg.get(pn, 0) + wj
        weights[name] = agg
    return res, weights


def main():
    sp = sys.argv[1] if len(sys.argv) > 1 else "20-29"
    a0, a1 = map(int, sp.split("-"))
    tag = sys.argv[2] if len(sys.argv) > 2 else "auto_v2"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    rows, wrows = [], []
    for d in DS:
        for k in range(a0, a1 + 1):
            f = os.path.join(L, "%s__s%d__%s.json" % (d, k, tag))
            if not os.path.exists(f):
                continue
            res, w = run_one(f, rng)
            for meth, sc in res.items():
                rows.append(dict(dataset=d, split=k, method=meth, **sc))
            for meth, ag in w.items():
                wrows.append(dict(dataset=d, split=k, method=meth, **ag))
    import pandas as pd
    df, wd = pd.DataFrame(rows), pd.DataFrame(wrows)
    df.to_csv(os.path.join(OUT, "scores_%s_%s.csv" % (tag, sp)), index=False)
    wd.to_csv(os.path.join(OUT, "weights_%s_%s.csv" % (tag, sp)), index=False)
    pd.set_option("display.width", 200)
    K = ["cov", "null_calls", "power", "wrong_sign", "fdp", "n_calls", "r", "within_r", "width"]
    m = df.groupby(["dataset", "method"])[K].mean()
    order = list(res.keys())
    for d in DS:
        if d in m.index.get_level_values(0):
            print("\n==", d); print(m.loc[d].reindex(order).round(3).to_string())
    print("\n== all datasets (mean of dataset means)")
    print(m.groupby(level=1).mean().reindex(order).round(3).to_string())
    s1 = m.reset_index().groupby("method").apply(
        lambda g: pd.Series(dict(datasets_fail=int(((g["cov"] < 0.85) | (g.null_calls > 0.10)).sum()),
                                 worst_null=g.null_calls.max(), worst_cov=g["cov"].min())))
    print("\n== S1 check (coverage >= 0.85 and null calls <= 0.10 in every dataset)")
    print(s1.reindex(order).round(3).to_string())
    print("\n== mean prior weights by pattern")
    print(wd.groupby(["method"]).mean(numeric_only=True).drop(columns="split").round(3).to_string())


if __name__ == "__main__":
    main()
