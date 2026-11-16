"""Exposure-based widening of the unmixed route (development, 2026-11-12; plan E23 in docs/weak_points_plan_2026-11-12.md).

Problem (W2, E20–E22): on pooled cell lines the unmixed route leaves a consistent bias on non-responding types next to
strong responders (McFarland idasanutlin TP53-mutant lines +0.08 even with the true fractions), because the real
responses are not exactly beta G and the leakage does not cancel; the design-check cohorts (exact beta G) never show it.
Widening every unmixed interval to the simulated error quantile cuts null calls but costs power everywhere (E22).

Here the widening is targeted: a pair's expected leaked error grows with its exposure
    x_cp = sum_{k != c} |leak[c, k, p]| * |effect_kp|,
how much of the other types' responses shows in type c's reallocated activity (leak = the study's within-program
leakage operator, mean over samples). Simulated cohorts with realistic responses (auto_v5j05_sims / auto_v5j10_sims:
per-type weight jitter sd 0.5 / 1.0, 20 cohorts) give pairs (x, |unmixed error|); per program
    |err| ~ b0 + b1 x  (least squares, b >= 0),  ratio quantile r_q = 90% quantile of |err| / (b0 + b1 x),
and the study's unmixed half-width becomes max(q_u se_u, r_q (b0 + b1 x_study)), with x_study from the study's own
unmixed estimates of the other types.

Everything else is sized_fx (v5_sims_offline.py): the estimates, the route (frozen as freeze v2) and the size-matched
calibrated width come from the freeze v2 parts (<ds>__s<k>__auto_v2_sims.npz); only the simulated unmixed errors come
from the jittered run. Variants per jitter level J in (05, 10):
  sized_fx       baseline (no unmixed widening)
  uwJ_sized      unmixed widened to the size-matched 90% quantile of the jittered errors (E22 form)
  expJ           exposure model per program
  expJ_pool      exposure model pooled over the run's programs (one b0, b1 in units of each program's error scale)
  expdJ(_pool)   exposure from the neighbours' DIFFERENCE to the type's own effect, sum |leak| |eff_k - eff_c| (a shared
                 response largely cancels in the leakage; first files showed expJ widening all trametinib responders)
Usage: python v5_exposure_offline.py [splits 20-29]
"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from v5_prior_offline import DS, L, OUT, score
from v5_sims_offline import variants as sims_variants

LEVEL = 0.9


def exposure(leak, eff, dev=False):
    """x (C, P): sum over k != c of |leak[c, k, p]| |eff[k, p]|; dev=True: |eff[k, p] - eff[c, p]| (a response
    shared by all types largely cancels in the leakage; only neighbours that respond differently bias type c)."""
    C = leak.shape[0]
    off = np.abs(leak) * (1 - np.eye(C))[:, :, None]
    if dev:
        return np.einsum("ckp,ckp->cp", off, np.abs(eff[None, :, :] - eff[:, None, :]))
    return np.einsum("ckp,kp->cp", off, np.abs(eff))


def fit_exposure(eu, bt, leak, pool=False, dev=False):
    """Per program (or pooled): b0, b1 >= 0 and the ratio quantile. eu, bt (S, C, P)."""
    S, C, P = eu.shape
    xs = np.stack([exposure(leak, bt[s], dev) for s in range(S)])     # (S, C, P)
    y = np.abs(eu)
    out = np.zeros((P, 3))

    def one(x, yy):
        X = np.c_[np.ones_like(x), x]
        b = np.linalg.lstsq(X, yy, rcond=None)[0]
        b = np.maximum(b, [np.quantile(yy, 0.25) * 0.5, 0.0])        # nonneg slope, floor on the intercept
        if b[1] == 0:
            b[0] = max(np.mean(yy), 1e-6)
        f = b[0] + b[1] * x
        return b[0], b[1], np.quantile(yy / np.maximum(f, 1e-9), LEVEL)

    if pool:
        # y / sc = b0 + b1 x / sc per program scale sc, i.e. y = b0 sc + b1 x: one slope, intercepts per program
        sc = np.maximum(np.sqrt((y ** 2).mean((0, 1))), 1e-9)          # (P,)
        b0, b1, rq = one((xs / sc).ravel(), (y / sc).ravel())
        for p in range(P):
            out[p] = b0 * sc[p], b1, rq
        return out
    for p in range(P):
        out[p] = one(xs[:, :, p].ravel(), y[:, :, p].ravel())
    return out


def main():
    a0, a1 = map(int, (sys.argv[1] if len(sys.argv) > 1 else "20-29").split("-"))
    rows = []
    for d in DS:
        for k in range(a0, a1 + 1):
            f0 = os.path.join(L, "%s__s%d__auto_v2_sims.npz" % (d, k))
            if not os.path.exists(f0):
                continue
            z = dict(np.load(f0))
            V = sims_variants(z, 1.645)
            e_fx, lo_fx, hi_fx, use0 = V["sized_fx"]
            b = z["beta"]
            rows.append(dict(dataset=d, split=k, method="sized_fx", **score(b, e_fx, lo_fx, hi_fx, (lo_fx > 0) | (hi_fx < 0))))
            e_u, hu = z["eff_u"], float(z["q_u"]) * z["se_u"]
            for J in ("05", "10"):
                fj = os.path.join(L, "%s__s%d__auto_v5j%s_sims.npz" % (d, k, J))
                if not os.path.exists(fj):
                    continue
                zj = np.load(fj)
                eu, bt, leak = zj["sim_eu"], zj["sim_beta"], zj["leak"]
                S, C, P = eu.shape
                e_c = z["eff_cal"]
                # E22 form: size-matched quantile of the jittered unmixed errors
                q = np.zeros(P)
                for p in range(P):
                    mm = np.abs(bt[:, :, p]).max(1)
                    keep = mm <= max(np.abs(e_c[:, p]).max(), 0.05)
                    if keep.sum() < 3:
                        keep = np.argsort(mm)[:3]
                    q[p] = np.quantile(np.abs(eu[keep, :, p]), LEVEL)
                hw = {"uw%s_sized" % J: np.maximum(hu, q[None])}
                for dev in (False, True):
                    x_st = exposure(leak, e_u, dev)
                    for pool in (False, True):
                        par = fit_exposure(eu, bt, leak, pool, dev)
                        pred = par[:, 0][None] + par[:, 1][None] * x_st
                        hw["exp%s%s%s" % ("d" if dev else "", J, "_pool" if pool else "")] =                             np.maximum(hu, par[:, 2][None] * pred)
                for nm, h in hw.items():
                    lo = np.where(use0, e_u - h, lo_fx)
                    hi = np.where(use0, e_u + h, hi_fx)
                    rows.append(dict(dataset=d, split=k, method=nm, **score(b, e_fx, lo, hi, (lo > 0) | (hi < 0))))
    df = pd.DataFrame(rows)
    print("runs with jittered sims: %s" % {m: int((df.method == m).sum()) for m in sorted(set(df.method))})
    order = ["sized_fx"] + [m for J in ("05", "10") for m in ("uw%s_sized" % J, "exp%s" % J, "exp%s_pool" % J,
                                                             "expd%s" % J, "expd%s_pool" % J)]
    order = [m for m in order if m in set(df.method)]
    # compare on the same (dataset, split) pairs: those with every method present
    cnt = df.groupby(["dataset", "split"]).method.nunique()
    keep = cnt[cnt == len(order)].index
    df = df.set_index(["dataset", "split"]).loc[lambda x: x.index.isin(keep)].reset_index()
    K = ["cov", "null_calls", "power", "fdp", "n_calls", "width"]
    m = df.groupby(["dataset", "method"])[K].mean()
    pd.set_option("display.width", 230)
    print("pairs compared: %d, datasets %d" % (len(keep), df.dataset.nunique()))
    print("\n== mean of dataset means"); print(m.groupby(level=1).mean().reindex(order).round(3).to_string())
    g = m.reset_index().groupby("method")
    s1 = pd.DataFrame(dict(fails=g.apply(lambda x: int(((x["cov"] < 0.85) | (x["null_calls"] > 0.10)).sum())),
                           worst_cov=g["cov"].min(), worst_null=g["null_calls"].max()))
    print("\n== S1"); print(s1.reindex(order).round(3).to_string())
    piv = m.reset_index().pivot(index="dataset", columns="method")
    rng = np.random.default_rng(0)
    print("\n== difference vs sized_fx [95% bootstrap CI over datasets]")
    for meth in order[1:]:
        line = []
        for kk in ("cov", "null_calls", "power", "fdp"):
            dd = (piv[(kk, meth)] - piv[(kk, "sized_fx")]).values
            bs = [rng.choice(dd, len(dd)).mean() for _ in range(4000)]
            line.append("%s %+.3f [%+.3f, %+.3f]" % (kk, dd.mean(), *np.quantile(bs, [0.025, 0.975])))
        print("%-12s %s" % (meth, "  ".join(line)))
    print("\n== per dataset: null calls / power")
    for meth in order:
        sub = m.xs(meth, level=1)
        print("%-12s %s" % (meth, "  ".join("%s %.2f/%.2f" % (dd[:9], r["null_calls"], r["power"]) for dd, r in sub.iterrows())))
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "exposure_%d-%d.csv" % (a0, a1)), index=False)


if __name__ == "__main__":
    main()
