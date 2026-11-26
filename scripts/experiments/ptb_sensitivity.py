"""PTB sensitivity analysis (2026-11-25) after the null PTB MDR vs DS result (freeze v5: 0 of 40 calls at level 0.90).

Post-hoc analysis: the frozen method and its default (level 0.90, interval calls) are unchanged.

  python ptb_sensitivity.py bulk      bulk program scores MDR vs DS (+ batch), no cell types: is there anything to split?
  python ptb_sensitivity.py width     PTB: reported half-widths vs the study's own noise (sample bootstrap) part
  python ptb_sensitivity.py rescale   held-out 30-39 freeze v5: null calls / power / coverage if the 0.90 intervals
                                      are shrunk to level 0.80 / 0.67 / 0.50 (normal approximation, fast preview)
  python ptb_sensitivity.py summary   exact reruns (causal_loop v5_l80 / v5_l67 on 30-39) + PTB reruns at lower levels
"""
import json, os, sys, pickle
import numpy as np, pandas as pd
from scipy.stats import norm

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
L = os.path.join(ROOT, "benchmarks", "dev", "causal_loop")
CELL = ["mcf_ida", "mcf_tram", "tahoe_enc", "tahoe_gem", "tahoe_rmc", "tahoe_tram"]
PBMC = ["kang", "zhang", "parse_gmcsf", "parse_ifnb", "parse_ifng", "parse_il4", "parse_tnf"]
DS = CELL + PBMC
SPLITS = list(range(30, 40))
PTB = os.path.join(ROOT, "results", "ptb")
BASE = "combat_v5_immune_reactome_innate5_anchored_mdr_vs_ds_batch"


def score(b, e, lo, hi):
    c = (lo > 0) | (hi < 0)
    big, null = np.abs(b) >= 0.1, np.abs(b) < 0.05
    return dict(cov=float(((b >= lo) & (b <= hi)).mean()),
                power=float((c & big & (np.sign(e) == np.sign(b))).sum() / max(big.sum(), 1)),
                null_calls=float((c & null).sum() / max(null.sum(), 1)),
                wrong_sign=float((c & big & (np.sign(e) != np.sign(b))).sum() / max(big.sum(), 1)),
                width=float((hi - lo).mean()))


def _arrays(d, s, est):
    f = os.path.join(L, "%s__s%d__%s.json" % (d, s, est))
    if not os.path.exists(f):
        return None
    a = json.load(open(f))["arrays"]
    return {k: np.array(a[k], float) for k in ("effects", "lo", "hi", "beta")}


def table(est, shrink=1.0):
    """Dataset means of the scores for estimator `est` on 30-39; shrink scales each half-interval (level preview)."""
    rows = []
    for d in DS:
        sc = []
        for s in SPLITS:
            a = _arrays(d, s, est)
            if a is None:
                continue
            e = a["effects"]
            lo, hi = e - shrink * (e - a["lo"]), e + shrink * (a["hi"] - e)
            sc.append(score(a["beta"], e, lo, hi))
        if sc:
            rows.append(dict(dataset=d, n=len(sc), **pd.DataFrame(sc).mean().to_dict()))
    return pd.DataFrame(rows)


def _summ(t, label):
    cell, pbmc = t[t.dataset.isin(CELL)], t[t.dataset.isin(PBMC)]
    return dict(setting=label, n_ds=len(t), null_calls=t.null_calls.mean(), null_max=t.null_calls.max(),
                null_max_ds=t.loc[t.null_calls.idxmax(), "dataset"] if len(t) else "", power=t.power.mean(),
                power_cell=cell.power.mean(), power_pbmc=pbmc.power.mean(), cov=t["cov"].mean(),
                wrong_sign=t.wrong_sign.mean(), width=t.width.mean())


def rescale():
    z90 = norm.ppf(0.95)
    out = []
    for lev in (0.90, 0.80, 0.67, 0.50):
        t = table("v5", norm.ppf(0.5 + lev / 2) / z90)
        out.append(_summ(t, "v5 rescaled %.2f" % lev))
    print(pd.DataFrame(out).round(3).to_string(index=False))


def bulk():
    """Program score per sample (mean log2 CPM of the program genes, Gmat weights) regressed on group + batch."""
    import anndata as ad, scipy.sparse as sp, statsmodels.api as sm
    r, inp = pickle.load(open(os.path.join(PTB, BASE, "result.pkl"), "rb"))
    a = ad.read_h5ad(os.path.join(ROOT, "data", "ptb", "ptb_bulk.h5ad"))
    a = a[a.obs["cmp2_MDR_vs_DS"].notna()].copy()
    X = sp.csr_matrix(a.X).toarray()
    cpm = np.log2(X / X.sum(1, keepdims=True) * 1e6 + 1)
    gi = {g: i for i, g in enumerate(a.var_names)}
    G = np.asarray(inp.Gmat)
    T = a.obs["cmp2_MDR_vs_DS"].astype(int).values
    D = pd.get_dummies(a.obs["batch"].astype(str), drop_first=True).astype(float).values
    Z = sm.add_constant(np.column_stack([T, D]))
    rows = []
    for j, p in enumerate(inp.programs):
        idx = [(gi[g], w) for g, w in zip(inp.genes, G[j]) if w != 0 and g in gi]
        w = np.array([x[1] for x in idx])
        sc = cpm[:, [x[0] for x in idx]] @ w / np.abs(w).sum()
        f = sm.OLS(sc, Z).fit(cov_type="HC3")
        rows.append(dict(program=p, n_genes=len(idx), diff_log2=f.params[1], lo90=f.conf_int(0.10)[1, 0],
                         hi90=f.conf_int(0.10)[1, 1], p=f.pvalues[1], sd_score=sc.std()))
    t = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    t["q_bh"] = multipletests(t.p, method="fdr_bh")[1]
    print(t.round(4).to_string(index=False))
    t.to_csv(os.path.join(PTB, BASE, "bulk_program_test.csv"), index=False)


def width():
    r, inp = pickle.load(open(os.path.join(PTB, BASE, "result.pkl"), "rb"))
    lo, hi = r.effects_interval
    hw = (np.asarray(hi) - np.asarray(lo)) / 2
    vp = r.info["effect_variance_parts"]
    noise_sd = float(np.sqrt(vp["boot"] + vp["residual"]))
    print("noise-only sd %.4f -> 90%% half-width %.4f" % (noise_sd, norm.ppf(0.95) * noise_sd))
    df = pd.DataFrame(hw, index=r.cell_types, columns=r.programs)
    print("reported 90% half-widths:\n", df.round(3).to_string())
    print("share of half-width beyond noise (median): %.2f" % float(np.median(1 - norm.ppf(0.95) * noise_sd / hw)))
    e = np.asarray(r.effects)
    print("min |effect| needed to call, median over pairs: %.3f; largest |effect|/half-width: %.2f" %
          (np.median(hw), np.max(np.abs(e) / hw)))


def summary():
    out = [_summ(table("v5"), "v5 level 0.90 (frozen)")]
    for est, lab in (("v5_l80", "v5 level 0.80"), ("v5_l67", "v5 level 0.67")):
        t = table(est)
        if len(t):
            out.append(_summ(t, lab + " (n=%d runs)" % int(t.n.sum())))
    print(pd.DataFrame(out).round(3).to_string(index=False))
    pr = os.path.join(ROOT, "benchmarks", "dev", "causal_loop")
    prac = [f for f in os.listdir(os.path.join(ROOT, "benchmarks", "dev")) if f.startswith("practice")]
    print("\nPTB runs:")
    for d in sorted(os.listdir(PTB)):
        if "mdr_vs_ds" not in d or not os.path.exists(os.path.join(PTB, d, "effects.csv")):
            continue
        t = pd.read_csv(os.path.join(PTB, d, "effects.csv"))
        lev = json.load(open(os.path.join(PTB, d, "info.json"))).get("level", 0.90)
        c = t[t.called]
        print("  %-75s level %.2f calls %d/%d %s" % (d, lev, len(c), len(t),
              "; ".join("%s %s %+.3f [%+.3f,%+.3f]" % (x.cell_type, x.program, x.effect, x.lo, x.hi)
                        for x in c.itertuples())))


if __name__ == "__main__":
    {"bulk": bulk, "width": width, "rescale": rescale, "summary": summary}[sys.argv[1]]()
