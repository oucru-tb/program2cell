"""Partial pooling of per-type effects (random effects across cell types, per program), offline on saved
causal_loop arrays.

Per program p, the C per-type effects y_c with standard errors s_c (half-width of the 90% interval / its
quantile) are treated as y_c ~ N(beta_c, s_c^2), beta_c ~ N(m, tau^2). tau^2: DerSimonian-Laird estimate, or,
to stay honest when few types agree by chance, its upper confidence bound (Q-profile, level `tau_level`).
Shrunken effect (BLUP): m + B_c (y_c - m), B_c = tau^2 / (tau^2 + s_c^2); variance
B_c s_c^2 + (1 - B_c)^2 var(m). Types that cannot be told apart (large s_c) borrow from the types that can, but
only as far as those agree: a type-specific response (large tau) leaves every type almost unchanged.

Usage: python pooling.py [ESTIMATOR] [SPLITS]     (default auto_abs, 0-9) -> prints scores vs unpooled
"""
import json, os, sys
import numpy as np
from scipy import stats, optimize

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl


def _q(y, s2, t2):
    w = 1 / (s2 + t2)
    m = (w * y).sum() / w.sum()
    return (w * (y - m) ** 2).sum(), m, 1 / w.sum()


def tau2(y, s2, level=None):
    """DerSimonian-Laird tau^2; with level, the Q-profile upper bound at that one-sided level."""
    k = len(y)
    Q, _, _ = _q(y, s2, 0.0)
    w = 1 / s2
    dl = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    if level is None:
        return dl
    crit = stats.chi2.ppf(1 - level, k - 1)                # Q(tau^2) decreases in tau^2: upper bound solves Q = crit
    if _q(y, s2, 0.0)[0] <= crit:
        return dl
    hi = max(dl, 1e-6) * 10 + 1.0
    while _q(y, s2, hi)[0] > crit:
        hi *= 10
    return optimize.brentq(lambda t: _q(y, s2, t)[0] - crit, 0.0, hi)


def pool(eff, lo, hi, level=0.9, tau_level=0.05):
    """Shrunken effects and intervals (C, P)."""
    q = stats.norm.ppf(0.5 + level / 2)
    s = np.maximum((hi - lo) / 2 / q, 1e-6)
    out_e, out_lo, out_hi = eff.copy(), lo.copy(), hi.copy()
    for p in range(eff.shape[1]):
        y, s2 = eff[:, p], s[:, p] ** 2
        t2 = tau2(y, s2, tau_level)
        _, m, vm = _q(y, s2, t2)
        B = t2 / (t2 + s2)
        e = m + B * (y - m)
        sd = np.sqrt(B * s2 + (1 - B) ** 2 * vm)
        out_e[:, p], out_lo[:, p], out_hi[:, p] = e, e - q * sd, e + q * sd
    return out_e, out_lo, out_hi


def main(name="auto_abs", splits="0-9", tau_level=0.05):
    rows = []
    for dsn in ("mcf_ida", "mcf_tram", "kang", "zhang"):
        for split in cl.db._splits(splits):
            f = os.path.join(cl.OUT, "%s__s%d__%s.json" % (dsn, split, name))
            if not os.path.exists(f):
                continue
            a = json.load(open(f))["arrays"]
            e, lo, hi = (np.array(a[k], float) for k in ("effects", "lo", "hi"))
            beta, ct, pg = np.array(a["beta"], float), a["cell_types"], a["programs"]
            k = cl.KNOWN.get(dsn)
            for tag, (ee, ll, hh) in (("unpooled", (e, lo, hi)), ("pooled", pool(e, lo, hi, tau_level=tau_level))):
                calls = (ll > 0) | (hh < 0)
                sc = cl.score(ee, ll, hh, calls, beta, ct, pg, k)
                rows.append(dict(dataset=dsn, split=split, est=tag, **{m: sc.get(m) for m in
                                                                       ("cov", "r", "fpr_null", "tpr_t", "width")}))
    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.groupby(["dataset", "est"])[["fpr_null", "tpr_t", "cov", "r", "width"]].mean().round(3).to_string())
    return df


if __name__ == "__main__":
    main(*(sys.argv[1:3]), **({"tau_level": float(sys.argv[3])} if len(sys.argv) > 3 else {}))
