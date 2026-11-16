"""Offline variants of freeze v2 from its saved intermediate parts (development, 2026-11-11).

Input: benchmarks/dev/causal_loop/<ds>__s<k>__auto_v2_parts.npz (causal_loop.py run ... auto_v2_parts; freeze v2
unchanged, both routes' parts kept). Each variant rebuilds the router (choose="width_guard": per program, unmixed
where its mean half-width is below the calibrated one; guard: a type whose unmixed effect is outside its
calibrated interval keeps the calibrated result) from modified parts, then scores against the truth.

Findings that motivate the variants (v5_prior_offline.py, v5_width_offline.py, splits 20-29):
- freeze v2's 90% intervals cover 0.98 of the truths (1.00 on PBMC): too wide;
- one cross-study width factor fails McFarland (null calls 0.13-0.17), so the excess is study-specific;
- the calibrated half-width is max(calibrated interval, simulated error) (deconvolution.py, sim_widen="abs"),
  the unmixed one t_{n-1} x bootstrap se with few units.

Variants:
  v2          rebuilt from the parts (must equal the saved json: checked)
  simonly     calibrated half-width = simulated error alone (drop the max with the atlas interval)
  calonly     calibrated half-width = atlas interval alone (no simulated widening; reference, expected unsafe)
  modt        unmixed se moderated across the study's pairs (limma-style scaled-inverse-chi2 prior on se^2, df
              n_units - 1 per pair), quantile t_{d0 + d}
  modt_prog   modt with the prior fitted per program (types of one program share a scale)
  simonly_modt  both
  lfsr        freeze v2 estimates and intervals; calls by lfsr < 0.05 under the mixture prior of
              v5_prior_offline (full rank), with the FULL error covariance per program (bootstrap correlation of
              the unmixed effects, scaled to the final half-widths / quantile)
  bh          freeze v2; calls by Benjamini-Hochberg q < 0.10 over the study's pairs
Usage: python v5_parts_offline.py [splits 20-29]
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.special import digamma, polygamma
from scipy.optimize import brentq
from scipy.stats import t as tdist, norm

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from v5_prior_offline import DS, L, OUT, score, patterns, fit_mixture, posterior, bh, EPS_FRAC

LEVEL = 0.9


def router(eff_u, hu, eff_cal, h):
    use = np.broadcast_to((hu.mean(0) < h.mean(0))[None], eff_u.shape)
    use = use & (np.abs(eff_u - eff_cal) <= h)
    eff = np.where(use, eff_u, eff_cal)
    return eff, np.where(use, eff_u - hu, eff_cal - h), np.where(use, eff_u + hu, eff_cal + h), use


def _trigamma_inv(x):
    """Solve trigamma(y) = x (Smyth 2004, Newton)."""
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = polygamma(1, y)
        dif = tri * (1 - tri / x) / polygamma(2, y)
        y += dif
        if np.all(np.abs(dif / y) < 1e-8):
            break
    return y


def moderate(s2, d):
    """limma fitFDist: prior (d0, s0^2) for variances s2 with d df; returns posterior s2 and total df."""
    z = np.log(s2) - digamma(d / 2) + np.log(d / 2)
    v = np.var(z, ddof=1) - polygamma(1, d / 2)
    if v <= 0:
        d0, s0 = np.inf, np.exp(z.mean())
        return np.full_like(s2, s0), 1e6
    d0 = 2 * _trigamma_inv(v)
    s0 = np.exp(z.mean() + digamma(d0 / 2) - np.log(d0 / 2))
    return (d0 * s0 + d * s2) / (d0 + d), d0 + d


def variants(z):
    e_u, se_u, e_c, h, hs, hc = z["eff_u"], z["se_u"], z["eff_cal"], z["h"], z["h_sim"], z["h_cal"]
    q_u, n = float(z["q_u"]), float(z["n_units"])
    d = max(n - 1, 1)
    out = {}
    out["v2"] = router(e_u, q_u * se_u, e_c, h)
    out["simonly"] = router(e_u, q_u * se_u, e_c, hs)
    out["calonly"] = router(e_u, q_u * se_u, e_c, hc)
    s2m, df = moderate(se_u.ravel() ** 2, d)
    hu_m = tdist.ppf(0.5 + LEVEL / 2, df) * np.sqrt(s2m.reshape(se_u.shape))
    out["modt"] = router(e_u, hu_m, e_c, h)
    hu_p = np.zeros_like(se_u)
    for p in range(se_u.shape[1]):
        s2p, dfp = moderate(se_u[:, p] ** 2, d) if se_u.shape[0] > 2 else (se_u[:, p] ** 2, d)
        hu_p[:, p] = tdist.ppf(0.5 + LEVEL / 2, dfp) * np.sqrt(s2p)
    out["modt_prog"] = router(e_u, hu_p, e_c, h)
    out["simonly_modt"] = router(e_u, hu_m, e_c, hs)
    return out


def lfsr_calls(z, eff, lo, hi, use, ct, thr=0.05):
    """Mixture-prior lfsr with full per-program error covariance."""
    C, P = eff.shape
    q = norm.ppf(0.5 + LEVEL / 2)
    s = np.clip((hi - lo) / (2 * q), 1e-4, None)
    boot = z["boot"] if "boot" in z.files else None
    pats = patterns(C, ct, EPS_FRAC * np.median(s), False)
    # per program covariance: correlation from the unmixed bootstrap (unmixed pairs), diagonal otherwise
    Vs = []
    for p in range(P):
        R = np.eye(C)
        if boot is not None and use[:, p].all():
            R = np.corrcoef(boot[:, :, p], rowvar=False)
            R = np.nan_to_num(R) + 1e-6 * np.eye(C)
        Vs.append(np.outer(s[:, p], s[:, p]) * R)
    # EM over programs with full V
    J = len(pats)

    def logmvn(y, S):
        sign, ld = np.linalg.slogdet(S)
        return -0.5 * (ld + y @ np.linalg.solve(S, y) + len(y) * np.log(2 * np.pi))
    ll = np.array([[logmvn(eff[:, p], U + Vs[p]) for (_, U) in pats] for p in range(P)])
    w = np.ones(J) / J
    for _ in range(300):
        lr = ll + np.log(w + 1e-300)
        r = np.exp(lr - lr.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
        w = r.mean(0)
    lr = ll + np.log(w + 1e-300)
    Rr = np.exp(lr - lr.max(1, keepdims=True)); Rr /= Rr.sum(1, keepdims=True)
    lf = np.zeros((C, P))
    for p in range(P):
        pp, pn = 0.0, 0.0
        for j, (_, U) in enumerate(pats):
            K = U @ np.linalg.inv(U + Vs[p])
            mu, sd = K @ eff[:, p], np.sqrt(np.clip(np.diag(U - K @ U), 1e-16, None))
            pp = pp + Rr[p, j] * norm.sf(0, mu, sd)
            pn = pn + Rr[p, j] * norm.cdf(0, mu, sd)
        lf[:, p] = np.minimum(pp, pn)
    return lf < thr


def main():
    a0, a1 = map(int, (sys.argv[1] if len(sys.argv) > 1 else "20-29").split("-"))
    rows, maxdiff = [], 0.0
    for dname in DS:
        for k in range(a0, a1 + 1):
            f = os.path.join(L, "%s__s%d__auto_v2_parts.npz" % (dname, k))
            if not os.path.exists(f):
                continue
            z = np.load(f)
            j = json.load(open(f.replace(".npz", ".json")))
            ct = j["arrays"]["cell_types"]
            b = z["beta"]
            V = variants(z)
            e0, lo0, hi0, use0 = V["v2"]
            maxdiff = max(maxdiff, float(np.abs(e0 - np.array(j["arrays"]["effects"])).max()),
                          float(np.abs(lo0 - np.array(j["arrays"]["lo"])).max()))
            for name, (e, lo, hi, use) in V.items():
                rows.append(dict(dataset=dname, split=k, method=name, unmixed_share=float(use.mean()),
                                 **score(b, e, lo, hi, (lo > 0) | (hi < 0))))
            s = np.clip((hi0 - lo0) / (2 * norm.ppf(0.95)), 1e-4, None)
            rows.append(dict(dataset=dname, split=k, method="bh", unmixed_share=float(use0.mean()),
                             **score(b, e0, lo0, hi0, bh(2 * norm.sf(np.abs(e0) / s)))))
            rows.append(dict(dataset=dname, split=k, method="lfsr", unmixed_share=float(use0.mean()),
                             **score(b, e0, lo0, hi0, lfsr_calls(z, e0, lo0, hi0, use0, ct))))
    df = pd.DataFrame(rows)
    print("rebuild check: max |v2 rebuilt - saved| = %.2e (must be ~0)" % maxdiff)
    print("runs: %d" % df[df.method == "v2"].shape[0])
    order = ["v2", "simonly", "calonly", "modt", "modt_prog", "simonly_modt", "lfsr", "bh"]
    K = ["cov", "null_calls", "power", "wrong_sign", "fdp", "n_calls", "r", "within_r", "width", "unmixed_share"]
    m = df.groupby(["dataset", "method"])[K].mean()
    pd.set_option("display.width", 220)
    for dname in DS:
        if dname in m.index.get_level_values(0):
            print("\n==", dname); print(m.loc[dname].reindex(order).round(3).to_string())
    print("\n== mean of dataset means")
    print(m.groupby(level=1).mean().reindex(order).round(3).to_string())
    g = m.reset_index().groupby("method")
    s1 = pd.DataFrame(dict(fails=g.apply(lambda x: int(((x["cov"] < 0.85) | (x["null_calls"] > 0.10)).sum())),
                           worst_cov=g["cov"].min(), worst_null=g["null_calls"].max(),
                           failing=g.apply(lambda x: ",".join(x.dataset[(x["cov"] < 0.85) | (x["null_calls"] > 0.10)]))))
    print("\n== S1 (every dataset: coverage >= 0.85, null calls <= 0.10)")
    print(s1.reindex(order).round(3).to_string())
    # paired difference vs v2 with a cluster bootstrap over datasets (as heldout_v4_compare S2)
    piv = m.reset_index().pivot(index="dataset", columns="method")
    rng = np.random.default_rng(0)
    print("\n== difference vs v2, mean over datasets [95% bootstrap CI over datasets]")
    for meth in order[1:]:
        line = []
        for kk in ("cov", "null_calls", "power", "r"):
            dd = (piv[(kk, meth)] - piv[(kk, "v2")]).values
            bs = [rng.choice(dd, len(dd)).mean() for _ in range(4000)]
            line.append("%s %+.3f [%+.3f, %+.3f]" % (kk, dd.mean(), *np.quantile(bs, [0.025, 0.975])))
        print("%-13s %s" % (meth, "  ".join(line)))
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "parts_%d-%d.csv" % (a0, a1)), index=False)


if __name__ == "__main__":
    main()
