"""Open items O1-O4 of docs/method_spec.md: the frozen pipeline (fractions -> reallocation -> calibration ->
effects) run from several fraction sources, under any program prior set.

Fraction sources (--sources, comma separated):
  palm:<config>    PALM posterior draws of pi saved by dev_bench under the default prior (progeny7), moved to
                   this prior's genes by reference mass ("fit once, reallocate under any prior": O1)
  native:<config>  PALM draws fitted under this prior itself (dev_bench dir of the prior), if present
  lite             multinomial EM, with parametric-bootstrap draws (O4)
  lite0            multinomial EM point only (no fraction uncertainty)
  nnls             NNLS point only
  cellgeometry     cellGeometry fractions saved by dev_bench under progeny7, moved to this prior's genes
  oracle           true read fractions

Rows per source:
  raw                       posterior reallocation, intervals from the fraction draws only
  calib / calib all         ridge recalibration per (c, p) on own type / all types (as calib.py)
  ... conf                  effect interval: conformal on random pseudo-contrasts of calibration donors (O3)
  ... strat                 effect interval: conformal on pseudo-contrasts split by true activity (O3)
  calib fr (lite, nnls)     calibration set built with the engine's own estimated fractions, so fraction
                            error is inside the calibration (O4)

Restart-safe: one json per (split, source) in benchmarks/dev/openitems/<dataset>__<prior>/.
Usage: python openitems.py run DATASET PRIOR SPLITS [--sources ...]
       python openitems.py summary DATASET PRIOR
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import norm

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dev_bench as db

K, LAM, LEVEL, NREP = 5, 1e-3, 0.9, 400
ZQ = norm.ppf(0.5 + LEVEL / 2)


def em_fractions(Y, B, pi0=None, iters=500, tol=1e-8):
    """Multinomial ML read fractions, y_n ~ Mult(n_n, pi_n B), by EM."""
    B = B / B.sum(1, keepdims=True)
    n = Y.sum(1, keepdims=True)
    pi = np.full((Y.shape[0], B.shape[0]), 1 / B.shape[0]) if pi0 is None else pi0.copy()
    for _ in range(iters):
        new = pi * ((Y / (pi @ B)) @ B.T) / n
        if np.abs(new - pi).max() < tol:
            return new
        pi = new
    return pi


def lite_draws(Y, B, S, rng):
    """EM point and S parametric-bootstrap draws (Y* ~ Mult(n, pi B), refit warm-started)."""
    pi = em_fractions(Y, B)
    Bn = B / B.sum(1, keepdims=True)
    mu = pi @ Bn
    n = np.rint(Y.sum(1)).astype(np.int64)
    draws = [em_fractions(rng.multinomial(n, mu / mu.sum(1, keepdims=True)).astype(float), B, pi0=pi)
             for _ in range(S)]
    return pi, np.array(draws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("dataset")
    ap.add_argument("prior")
    ap.add_argument("splits", nargs="?", default="0-9")
    ap.add_argument("--sources", default="palm:joint nb,palm:joint quasi+sw,native:joint quasi+sw,lite,lite0,nnls,"
                                         "cellgeometry,oracle")
    ap.add_argument("--ndraws", type=int, default=50)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--max-cal", type=int, default=300)
    ap.add_argument("--outname", default="openitems", help="results folder under benchmarks/dev")
    a = ap.parse_args()
    odir = os.path.join(ROOT, "benchmarks", "dev", a.outname, "%s__%s" % (a.dataset, a.prior))
    os.makedirs(odir, exist_ok=True)
    if a.cmd == "summary":
        return summary(odir, a)
    run(odir, a)


def run(odir, a):
    import palm_bulk as pb
    from palm_bulk.core import PseudobulkTable, make_benchmark
    from palm_bulk.core.pseudobulk import _csr
    from palm_bulk.core.scores import estimate_delta
    from palm_bulk.compare.base import MethodResult
    from palm_bulk.compare.activity import activity_from_expression
    from palm_bulk.compare.methods import nnls_fractions
    from palm_bulk.bayes.diagnostics import interval

    ds = db.DATASETS[a.dataset]
    table = db._table(ds)
    prog = db._programs(pb, a.prior)
    prog7 = db._programs(pb, "progeny7")
    X = _csr(table.counts)
    obs = table.obs
    pos = {k: i for i, k in enumerate(obs.index)}
    dir7 = os.path.join(ROOT, "benchmarks", "dev", a.dataset)
    dirp = db._odir(argparse.Namespace(out=os.path.join(ROOT, "benchmarks", "dev"), dataset=a.dataset, prior=a.prior))
    sources = [s.strip() for s in a.sources.split(",") if s.strip()]

    rf = ds.get("ref_filter")
    obs_ref = obs if rf is None else obs[obs[rf[0]] == rf[1]]          # reference and calibration rows

    def profiles(donors, types, gidx):
        ref = obs_ref[obs_ref["donor_id"].astype(str).isin(donors) & (obs_ref["n_cells"] >= 10)]
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
        pi = np.clip(pi, 1e-4, None)                 # NNLS-type zeros would leave a type with no reads
        w = (pi / pi.sum(-1, keepdims=True))[:, :, None] * B[None]
        return estimate_delta(Y[:, None, :] * (w / w.sum(1, keepdims=True)), B, G)

    def calibration_set(test_donors, types, gidx, G, seed):
        """Folds of thinned reference specimens: synthetic bulk (half 1), its true pi, truth d (half 2), B_k."""
        rng = np.random.default_rng(seed + 100)
        wide = obs_ref.pivot_table(index="specimen", columns="cell_type", values="n_cells", aggfunc="sum",
                                   fill_value=0).reindex(columns=types, fill_value=0)
        spec = obs_ref.drop_duplicates("specimen").set_index("specimen")
        ok = (wide >= 20).all(1)
        elig = spec.loc[ok[ok].index]
        elig = elig[~elig["donor_id"].astype(str).isin(test_donors)]
        donors = rng.permutation(np.array(sorted(map(str, elig["donor_id"].unique())), dtype=object))[:a.max_cal]
        all_ref = set(map(str, obs["donor_id"].unique())) - set(test_donors)
        kf = min(K, len(donors))
        folds = np.arange(len(donors)) % kf
        multi = ds.get("all_test_specimens", False)      # small studies: every specimen of a calibration donor
        out = []
        for k in range(kf):
            cal = donors[folds == k]
            B = profiles(all_ref - set(cal), types, gidx)
            specs, owners = [], []
            for dnr in cal:
                own_ = np.array(elig.index[elig["donor_id"].astype(str) == dnr], dtype=object)
                pick_ = list(own_) if multi else [rng.choice(own_)]
                specs += pick_; owners += [dnr] * len(pick_)
            tc = np.zeros((len(specs), len(types), len(gidx)))
            for i, s in enumerate(specs):
                rows = obs[obs["specimen"] == s]
                for j, c in enumerate(types):
                    r = rows[rows["cell_type"] == c]
                    tc[i, j] = np.asarray(X[[pos[q] for q in r.index]].sum(0)).ravel()[gidx]
            half = rng.binomial(tc.astype(np.int64), 0.5).astype(float)
            tot = half.sum(-1)
            out.append(dict(Y=half.sum(1), pi=tot / tot.sum(1, keepdims=True), d=estimate_delta(tc - half, B, G),
                            B=B, donors=owners))
        return out

    def cal_dhat(cs, G, frac_fn=None):
        """(dhat, d, donors, fold) over the calibration set; frac_fn(Y, B) -> pi, else the true pi."""
        dh, d, dn, fk = [], [], [], []
        for k, f in enumerate(cs):
            pi = f["pi"] if frac_fn is None else frac_fn(f["Y"], f["B"])
            dh.append(realloc(f["Y"], pi, f["B"], G))
            d.append(f["d"]); dn += f["donors"]; fk += [k] * len(f["donors"])
        return np.concatenate(dh), np.concatenate(d), np.array(dn), np.array(fk)

    def design_mat(dh, c, p, own):
        Xp = dh[:, c:c + 1, p] if own else dh[:, :, p]
        return np.column_stack([np.ones(len(dh)), Xp])

    def fit(dh, d, own):
        W = {}
        for c in range(d.shape[1]):
            for p in range(d.shape[2]):
                Xm = design_mat(dh, c, p, own)
                R = LAM * np.eye(Xm.shape[1]); R[0, 0] = 0
                W[c, p] = np.linalg.solve(Xm.T @ Xm + R, Xm.T @ d[:, c, p])
        return W

    def predict(W, dh, own):
        out = np.zeros(dh.shape)
        flat = dh.reshape(-1, *dh.shape[-2:])
        for (c, p), w in W.items():
            out[..., c, p] = (design_mat(flat, c, p, own) @ w).reshape(dh.shape[:-2])
        return out

    def leak_matrix(resid, d):
        """(C, C, P): slope of the out-of-fold residual of (c, p) on the true d of type c' (same p),
        i.e. how much of a true shift in c' the calibrated estimate of c misses (c' = c) or picks up."""
        C, P = d.shape[1:]
        L = np.zeros((C, C, P))
        for p in range(P):
            D = np.column_stack([np.ones(len(d)), d[:, :, p] - d[:, :, p].mean(0)])
            R = 1e-3 * np.eye(C + 1); R[0, 0] = 0
            L[:, :, p] = np.linalg.solve(D.T @ D + R, D.T @ resid[:, :, p])[1:].T
        return L

    def contrast_q(resid, d, n1, n0, rng, strat):
        """(C, P) conformal half-width for an effect from pseudo-contrasts of out-of-fold residuals,
        rescaled from the pseudo group sizes to the test group sizes."""
        n = len(resid)
        m = n // 2
        scale = np.sqrt((1 / n1 + 1 / n0) / (2 / m))
        C, P = resid.shape[1:]
        vals = np.zeros((NREP, C, P))
        for r in range(NREP):
            if not strat:
                idx = rng.permutation(n)
                vals[r] = resid[idx[:m]].mean(0) - resid[idx[m:2 * m]].mean(0)
            else:
                # groups split by one type's true activity of each program (a real effect, with its leakage)
                cs = r % C
                sub = rng.choice(n, size=2 * (n // 3), replace=False)
                for p in range(P):
                    o = sub[np.argsort(d[sub, cs, p])]
                    h = len(o) // 2
                    vals[r, :, p] = resid[o[h:]].mean(0)[:, p] - resid[o[:h]].mean(0)[:, p]
        if strat:
            scale = np.sqrt((1 / n1 + 1 / n0) / (2 / (n // 3)))
        lev = min(1.0, np.ceil((NREP + 1) * LEVEL) / NREP)
        return np.quantile(np.abs(vals), lev, axis=0) * scale

    for split in db._splits(a.splits):
        todo = [s for s in sources if not os.path.exists(os.path.join(odir, "s%d__%s.json" % (split, db._slug(s))))]
        if not todo:
            continue
        t0 = time.time()
        b = db._benchmark(table, ds, split)
        inp = pb.prepare_inputs(b.bulk, b.reference, prog, treatment=b.treatment, max_genes=1000)
        inp7 = inp if a.prior == "progeny7" else pb.prepare_inputs(b.bulk, b.reference, prog7, treatment=b.treatment,
                                                                   max_genes=1000)
        assert list(inp7.cell_types) == list(inp.cell_types) and list(inp7.samples) == list(inp.samples)
        truth = b.truth(inp)
        gi = [b.reference.genes.index(g) for g in inp.genes]
        ci = [b.cell_types.index(c) for c in inp.cell_types]
        ulm, _ = activity_from_expression(b.truth_counts[:, ci][:, :, gi], inp, "ulm", prog)
        T = np.asarray(b.treatment)
        design = db._design(T)
        truth["beta"] = design(truth["delta"])
        truth_ulm = dict(pi=truth["pi"], delta=ulm)
        Y, B, G = inp.Y, inp.B, inp.Gmat
        n1, n0 = (T == 1).sum(), (T == 0).sum()
        rmass = lambda genes: b.reference.B[:, [b.reference.genes.index(g) for g in genes]].sum(1)
        m7, mp = rmass(inp7.genes), rmass(inp.genes)

        def to_prior(pi):                    # read fractions on progeny7 genes -> on this prior's genes
            w = pi * mp / m7
            return w / w.sum(-1, keepdims=True)

        rng = np.random.default_rng(split)
        cs = calibration_set(set(b.meta["donor_id"].astype(str)), inp.cell_types, gi, G, split)
        base = cal_dhat(cs, G)
        print("[%s] split %d: setup %.0fs, %d calibration donors" % (time.strftime("%H:%M"), split,
                                                                       time.time() - t0, len(base[1])), flush=True)

        for src in todo:
            t1 = time.time()
            kind, _, cfg = src.partition(":")
            draws, point, fr_fn = None, None, None
            if kind in ("palm", "native"):
                f = os.path.join(dir7 if kind == "palm" else dirp, "s%d__%s.npz" % (split, db._slug(cfg)))
                if not os.path.exists(f) or (kind == "native" and a.prior == "progeny7"):
                    continue
                z = np.load(f)["pi"]
                z = z[np.linspace(0, len(z) - 1, min(a.ndraws, len(z))).astype(int)]
                draws = to_prior(z) if kind == "palm" else z
                point = np.median(draws, 0)
            elif kind == "lite":
                point, draws = lite_draws(Y, B, a.ndraws, rng)
                fr_fn = em_fractions
            elif kind == "lite0":
                point, fr_fn = em_fractions(Y, B), em_fractions
            elif kind == "nnls":
                point, fr_fn = nnls_fractions(Y, B), nnls_fractions
            elif kind in ("cellgeometry", "comp"):
                f = os.path.join(dir7, "s%d__%s.npz" % (split, db._slug(cfg or kind)))
                if not os.path.exists(f):
                    continue
                point = to_prior(np.load(f)["model__fractions"])
            elif kind == "oracle":
                point = truth["pi"]
            else:
                sys.exit("unknown source " + src)
            ft = time.time() - t1
            r_point = realloc(Y, point, B, G)
            r_draws = None if draws is None else np.array([realloc(Y, p_, B, G) for p_ in draws])
            rows = []

            def add(row, act, act_iv, eff, eff_iv, **extra):
                res = MethodResult(src, fractions=point, activity=act, effects=eff,
                                   intervals=dict(activity=act_iv, effects=eff_iv) if act_iv is not None else {})
                s = db._score(res, truth, truth_ulm)
                s.update(split=split, method=src, row=row, frac_sec=ft, **extra)
                rows.append(s)

            add("raw", r_point, None if r_draws is None else interval(r_draws), design(r_point),
                None if r_draws is None else interval(design(r_draws)))

            cal_sets = [("calib", base)]
            if fr_fn is not None:
                cal_sets.append(("calib fr", cal_dhat(cs, G, fr_fn)))
            elif kind != "oracle":
                # proxy fraction error: PALM-lite's fractions on the calibration bulks (O2 of 2026-10-28)
                cal_sets.append(("calib px", cal_dhat(cs, G, em_fractions)))
            for tag, (dh_cal, d_cal, donors, fk) in cal_sets:
                ud = np.unique(donors)
                for own in (True, False):
                    name = tag + ("" if own else " all")
                    resid = np.zeros_like(d_cal)
                    for k in range(K):
                        W = fit(dh_cal[fk != k], d_cal[fk != k], own)
                        resid[fk == k] = d_cal[fk == k] - predict(W, dh_cal[fk == k], own)
                    nc = len(d_cal)
                    q = np.quantile(np.abs(resid), min(1.0, np.ceil((nc + 1) * LEVEL) / nc), axis=0)
                    W = fit(dh_cal, d_cal, own)
                    act = predict(W, r_point, own)
                    eff = design(act)
                    if r_draws is not None:
                        act_d = predict(W, r_draws, own)
                        h_draw = (np.quantile(act_d, 0.95, 0) - np.quantile(act_d, 0.05, 0)) / 2
                        v_draw = np.var(design(act_d), 0)
                    else:
                        h_draw, v_draw = 0.0, 0.0
                    hw = np.sqrt(q[None] ** 2 + h_draw ** 2)
                    boot = []
                    for _ in range(a.nboot):
                        pick = rng.choice(ud, len(ud))
                        idx = np.concatenate([np.flatnonzero(donors == d_) for d_ in pick])
                        boot.append(design(predict(fit(dh_cal[idx], d_cal[idx], own), r_point, own)))
                    v_boot = np.var(boot, 0)
                    v = v_boot + resid.var(0) * (1 / n1 + 1 / n0) + v_draw
                    act_iv = (act - hw, act + hw)
                    add(name, act, act_iv, eff, (eff - ZQ * np.sqrt(v), eff + ZQ * np.sqrt(v)))
                    for vtag, strat in (("conf", False), ("strat", True)):
                        if vtag == "conf" and not own:
                            continue
                        qe = contrast_q(resid, d_cal, n1, n0, rng, strat)
                        he = np.sqrt(qe ** 2 + ZQ ** 2 * (v_boot + v_draw))
                        add(name + " " + vtag, act, act_iv, eff, (eff - he, eff + he))
                        if vtag == "conf":
                            # leakage-aware: random-split width (no effect) + the error a real effect of the
                            # estimated size would leave, sum_c' |L_cc'p| |beta_c'p| (L from the calibration set)
                            L = leak_matrix(resid, d_cal)
                            for ltag, bsize in (("lk", np.abs(eff)), ("lk+", np.abs(eff) + ZQ * np.sqrt(v))):
                                hl = he + np.einsum("ckp,kp->cp", np.abs(L), bsize)
                                add(name + " " + ltag, act, act_iv, eff, (eff - hl, eff + hl))
            with open(os.path.join(odir, "s%d__%s.json" % (split, db._slug(src))), "w") as fh:
                json.dump(rows, fh, indent=1, default=float)
            print("[%s] split %d: %s done (fractions %.0fs, total %.0fs)" % (time.strftime("%H:%M"), split, src, ft,
                                                                           time.time() - t1), flush=True)


COLS = dict(frac_rmse="frac", act_spearman_across_samples="rho", act_spearman_specific="spec",
            ulm_act_spearman_across_samples="ulm_rho", act_cov90="cov", act_iscore="iscore", eff_pearson="eff_r",
            eff_localise="local", eff_cov90="eff_cov", eff_iscore="eff_is", eff_fsr="fsr", eff_n_called="n_call",
            frac_sec="frac_s")


def summary(odir, a):
    import glob
    rows = [r for f in sorted(glob.glob(os.path.join(odir, "s*__*.json"))) for r in json.load(open(f))]
    if not rows:
        print("no results in", odir)
        return
    df = pd.DataFrame(rows)
    df["method_row"] = df["method"] + " | " + df["row"]
    t = df.groupby("method_row", sort=False)[[c for c in COLS if c in df]].mean().rename(columns=COLS)
    t.insert(0, "n", df.groupby("method_row", sort=False).split.nunique())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print("== %s, prior %s" % (a.dataset, a.prior))
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
