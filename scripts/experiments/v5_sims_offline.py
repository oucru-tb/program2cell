"""Calibrated-route width variants from the saved simulated cohorts (development, 2026-11-12; plan E6 / E17 in
docs/weak_points_plan_2026-11-12.md).

Input: benchmarks/dev/causal_loop/<ds>__s<k>__auto_v2_sims.npz (causal_loop.py run ... auto_v2_sims: freeze v2
unchanged; both routes' parts plus design_check's per-cohort arrays sim_er (S, C, P) = rescaled-reallocation error
(slope * rr - beta), sim_beta planted effects, sim_rr, sim_rse = contrast_se of the simulated reallocation, se_r =
|slope| * contrast_se(study), q_abs_cp).

Every variant changes only the calibrated route's half-width h; the unmixed route (q_u * se_u), the router
(width_guard) and the guard are rebuilt as in freeze v2 (v5_parts_offline.router).

Variants (h for the calibrated route):
  v2        max(h_cal, sqrt(q_abs^2 + (q_t se_r)^2))             (freeze v2; rebuilt, checked against the json)
  nonoise   max(h_cal, q_abs)                                     E6b: the cohorts already carry sampling noise
  qabs      q_abs                                                 E6b without the max
  sized     as v2 with q_abs from the cohorts whose largest planted |beta| is at most the study's largest
            calibrated |effect| for that program (the simulation plants up to 2x)   E6c
  sized_nn  sized without the added study noise
  sized_fx  sized with the route frozen as chosen by v2 (isolates the width change from the routing change)
  uw_all    sized_fx + unmixed half-width max(q_u se_u, 90% quantile of |simulated unmixed error|) (plan E22, = auto_v5u0)
  uw_sized  the same with the quantile from the cohorts whose planted sizes match the study's (as sized)
  ppi       eff_cal - mean_s(er); half-width z sqrt(var_s(er)/S + (q_t se_r)^2)      E6a (PPI form; expected to
            under-cover: the planted patterns average the leakage bias out)
  stud_p    z_q(p) * se_r, z_q(p) = 90% quantile over (s, c) of |er| / (|slope| rse)   E17 (freeze v1 form), no max
  stud_p_mx max(h_cal, stud_p)
  stud_c    z_q(c) * se_r, quantile over (s, p) per type, no max
  lodo      z * se_r with z the 90% quantile of the real studentised calibrated-route errors |eff_cal - beta| /
            se_r pooled over the OTHER 12 datasets (baseline for the synthetic-powered version, E16)
Usage: python v5_sims_offline.py [splits 20-29]
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import t as tdist, norm

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
from v5_prior_offline import DS, L, OUT, score
from v5_parts_offline import router

LEVEL = 0.9


def load(a0, a1):
    runs = []
    for dname in DS:
        for k in range(a0, a1 + 1):
            f = os.path.join(L, "%s__s%d__auto_v2_sims.npz" % (dname, k))
            if os.path.exists(f):
                z = dict(np.load(f))
                z["json"] = json.load(open(f.replace(".npz", ".json")))["arrays"]
                z["dataset"], z["split"] = dname, k
                runs.append(z)
    return runs


def real_stud(z):
    """Real studentised calibrated-route errors of one run (C, P)."""
    return np.abs(z["eff_cal"] - z["beta"]) / np.maximum(z["se_r"], 1e-9)


def variants(z, lodo_z):
    e_u, se_u, e_c, h_cal, se_r, q_abs = z["eff_u"], z["se_u"], z["eff_cal"], z["h_cal"], z["se_r"], z["q_abs_cp"]
    er, bt, rr, rse = z["sim_er"], z["sim_beta"], z["sim_rr"], z["sim_rse"]
    S, C, P = er.shape
    q_u, n = float(z["q_u"]), float(z["n_units"])
    q_t = tdist.ppf(0.5 + LEVEL / 2, max(n - 1, 1))
    noise = (q_t * se_r) ** 2
    hu = q_u * se_u
    out = {}
    out["v2"] = router(e_u, hu, e_c, np.maximum(h_cal, np.sqrt(q_abs ** 2 + noise)))
    out["nonoise"] = router(e_u, hu, e_c, np.maximum(h_cal, q_abs))
    out["qabs"] = router(e_u, hu, e_c, q_abs)
    # sized: cohorts whose largest planted |beta| <= the study's largest calibrated |effect| per program
    q_abs_p = np.quantile(np.abs(er), LEVEL, axis=(0, 1))                       # (P,)
    type_factor = q_abs / np.maximum(q_abs_p[None], 1e-12)                      # the (C, P) cell model of design_check
    q_sized = np.zeros(P)
    for p in range(P):
        m = np.abs(bt[:, :, p]).max(1)                                          # (S,) largest planted size per cohort
        keep = m <= max(np.abs(e_c[:, p]).max(), 0.05)
        if keep.sum() < 3:
            keep = np.argsort(m)[:3]
        q_sized[p] = np.quantile(np.abs(er[keep, :, p]), LEVEL)
    q_sz = q_sized[None] * type_factor
    out["sized"] = router(e_u, hu, e_c, np.maximum(h_cal, np.sqrt(q_sz ** 2 + noise)))
    out["sized_nn"] = router(e_u, hu, e_c, np.maximum(h_cal, q_sz))
    # route frozen as in v2 (the narrower calibrated interval must not pull programs off the unmixed route)
    use0 = out["v2"][3]
    h_sz = np.maximum(h_cal, np.sqrt(q_sz ** 2 + noise))
    out["sized_fx"] = (np.where(use0, e_u, e_c), np.where(use0, e_u - hu, e_c - h_sz),
                       np.where(use0, e_u + hu, e_c + h_sz), use0)
    # unmixed-route widening from the simulated unmixed errors (plan E22), on top of sized_fx, route frozen as v2
    eu = z["sim_eu"]                                                            # (S, C, P) unmixed estimate - truth
    def cellf(e):
        m = (e ** 2).mean(0)
        return np.sqrt(m.mean(1, keepdims=True) * m.mean(0, keepdims=True) / max(m.mean(), 1e-24))
    rmse_u = np.sqrt((eu ** 2).mean((0, 1)))
    tf_u = cellf(eu) / np.maximum(rmse_u[None], 1e-12)                          # type factor (C, P)
    qu_all = np.quantile(np.abs(eu), LEVEL, axis=(0, 1))[None] * tf_u
    qu_sz = np.zeros(P)
    for p in range(P):
        mm = np.abs(bt[:, :, p]).max(1)
        keep = mm <= max(np.abs(e_c[:, p]).max(), 0.05)
        if keep.sum() < 3:
            keep = np.argsort(mm)[:3]
        qu_sz[p] = np.quantile(np.abs(eu[keep, :, p]), LEVEL)
    qu_sz = qu_sz[None] * tf_u
    e_fx, lo_fx, hi_fx, _ = out["sized_fx"]
    for nm, qq in (("uw_all", qu_all), ("uw_sized", qu_sz)):
        hw = np.maximum(hu, qq)
        out[nm] = (e_fx, np.where(use0, e_u - hw, lo_fx), np.where(use0, e_u + hw, hi_fx), use0)
    # ppi form
    bias, v = er.mean(0), er.var(0, ddof=1) / S
    zq = norm.ppf(0.5 + LEVEL / 2)
    out["ppi"] = router(e_u, hu, e_c - bias, zq * np.sqrt(v + noise))
    # studentised quantiles (freeze v1 form)
    slope = (rr * bt).sum((0, 1)) / np.maximum((rr ** 2).sum((0, 1)), 1e-12)   # (P,)
    zr = np.abs(er) / np.maximum(np.abs(slope)[None, None] * rse, 1e-9)         # (S, C, P)
    z_p = np.quantile(zr, LEVEL, axis=(0, 1))                                   # (P,)
    z_c = np.quantile(zr, LEVEL, axis=(0, 2))                                   # (C,)
    out["stud_p"] = router(e_u, hu, e_c, z_p[None] * se_r)
    out["stud_p_mx"] = router(e_u, hu, e_c, np.maximum(h_cal, z_p[None] * se_r))
    out["stud_c"] = router(e_u, hu, e_c, z_c[:, None] * se_r)
    out["lodo"] = router(e_u, hu, e_c, lodo_z * se_r)
    return out


def main():
    a0, a1 = map(int, (sys.argv[1] if len(sys.argv) > 1 else "20-29").split("-"))
    runs = load(a0, a1)
    stud = {d: np.concatenate([real_stud(z).ravel() for z in runs if z["dataset"] == d]) for d in DS
            if any(z["dataset"] == d for z in runs)}
    rows, maxdiff = [], 0.0
    for z in runs:
        others = np.concatenate([v for d, v in stud.items() if d != z["dataset"]])
        lodo_z = float(np.quantile(others, LEVEL))
        V = variants(z, lodo_z)
        e0, lo0, hi0, _ = V["v2"]
        j = z["json"]
        maxdiff = max(maxdiff, float(np.abs(e0 - np.array(j["effects"])).max()),
                      float(np.abs(lo0 - np.array(j["lo"])).max()))
        for name, (e, lo, hi, use) in V.items():
            rows.append(dict(dataset=z["dataset"], split=z["split"], method=name, unmixed_share=float(use.mean()),
                             **score(z["beta"], e, lo, hi, (lo > 0) | (hi < 0))))
    df = pd.DataFrame(rows)
    print("rebuild check: max |v2 rebuilt - saved| = %.2e (must be ~0)" % maxdiff)
    print("runs: %d  datasets: %d" % (df[df.method == "v2"].shape[0], df.dataset.nunique()))
    order = ["v2", "nonoise", "qabs", "sized", "sized_nn", "sized_fx", "uw_all", "uw_sized", "ppi", "stud_p", "stud_p_mx", "stud_c", "lodo"]
    K = ["cov", "null_calls", "power", "fdp", "n_calls", "r", "width", "unmixed_share"]
    m = df.groupby(["dataset", "method"])[K].mean()
    pd.set_option("display.width", 220)
    print("\n== mean of dataset means")
    print(m.groupby(level=1).mean().reindex(order).round(3).to_string())
    g = m.reset_index().groupby("method")
    s1 = pd.DataFrame(dict(fails=g.apply(lambda x: int(((x["cov"] < 0.85) | (x["null_calls"] > 0.10)).sum())),
                           worst_cov=g["cov"].min(), worst_null=g["null_calls"].max(),
                           failing=g.apply(lambda x: ",".join(x.dataset[(x["cov"] < 0.85) | (x["null_calls"] > 0.10)]))))
    print("\n== S1 (every dataset: coverage >= 0.85, null calls <= 0.10)")
    print(s1.reindex(order).round(3).to_string())
    piv = m.reset_index().pivot(index="dataset", columns="method")
    rng = np.random.default_rng(0)
    print("\n== difference vs v2, mean over datasets [95% bootstrap CI over datasets]")
    for meth in order[1:]:
        line = []
        for kk in ("cov", "null_calls", "power", "width"):
            dd = (piv[(kk, meth)] - piv[(kk, "v2")]).values
            bs = [rng.choice(dd, len(dd)).mean() for _ in range(4000)]
            line.append("%s %+.3f [%+.3f, %+.3f]" % (kk, dd.mean(), *np.quantile(bs, [0.025, 0.975])))
        print("%-10s %s" % (meth, "  ".join(line)))
    print("\n== per dataset: coverage / null calls / power")
    for meth in order:
        sub = m.xs(meth, level=1)
        print("%-10s %s" % (meth, "  ".join("%s %.2f/%.2f/%.2f" % (d[:9], r["cov"], r["null_calls"], r["power"])
                                             for d, r in sub.iterrows())))
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "sims_%d-%d.csv" % (a0, a1)), index=False)


if __name__ == "__main__":
    main()
