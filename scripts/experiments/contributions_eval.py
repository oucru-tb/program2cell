"""Contributions (program2cell.contributions) against the answer key.

Truth, from the held-out half of each test specimen's cells: per sample and cell type, the program signal
s_ncp = sum_g G_pg x_ncg (x = that type's truth-half counts normalised over the modelled genes) and the true read
fractions pi_nc; the Shapley split of each program's bulk change with the arms' means of pi and s.
Estimates, through the same formula:
  PALM        PALM posterior fractions (median of draws; draws for the intervals) + PALM effects (auto_m)
  InstaPrism  InstaPrism fractions + InstaPrism per-type effects (paired t-test difference), no intervals
Scores per program whose true total is clearly non-zero (|true total| above the 50% quantile of the dataset's
|totals|): share error (mean |estimated - true share| over cell types), top contributor right, coverage of the
true share by PALM's 90% share interval; and r of contributions over all (type, program) pairs.
Usage: python contributions_eval.py [DATASETS] [SPLITS]   -> benchmarks/dev/contributions.json
"""
import json, os, sys
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import causal_loop as cl
import practice_vs_palm as pv
import program2cell as pb
from program2cell.contributions import contributions
from program2cell.core.equations import effect_decomposition  # noqa: F401  (same formula, kept for reference)

OUT = os.path.join(ROOT, "benchmarks", "dev", "contributions.json")


def truth_split(inp, b, T):
    gi = [b.reference.genes.index(g) for g in inp.genes]
    ci = [b.cell_types.index(c) for c in inp.cell_types]
    X = np.asarray(b.truth_counts[:, ci][:, :, gi], float)                    # (N, C, G)
    pi = X.sum(-1) / X.sum((1, 2))[:, None]
    x = X / np.maximum(X.sum(-1, keepdims=True), 1e-12)
    s = np.einsum("pg,ncg->ncp", np.asarray(inp.Gmat, float), x)
    T = np.asarray(T).astype(bool)
    pi0, pi1, s0, s1 = pi[~T].mean(0), pi[T].mean(0), s[~T].mean(0), s[T].mean(0)
    I = 0.5 * (pi0 + pi1)[:, None] * (s1 - s0)
    K = 0.5 * (pi1 - pi0)[:, None] * (s0 + s1)
    tot = (I + K).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = (I + K) / tot[None]
    return dict(intrinsic=I, composition=K, contribution=I + K, total=tot, share=share)


def main(datasets="mcf_ida,mcf_tram,kang", splits="0-2"):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for dsn in datasets.split(","):
        ds = cl.db.DATASETS[dsn]
        table = cl.db._table(ds)
        prog = cl.db._programs(pb, "progeny7")
        for split in cl.db._splits(splits):
            f = os.path.join(cl.OUT, "%s__s%d__auto_m.json" % (dsn, split))
            z7 = os.path.join(cl.ROOT, "benchmarks", "dev", dsn, "s%d__joint_nb.npz" % split)
            if not (os.path.exists(f) and os.path.exists(z7)):
                continue
            b = cl.db._benchmark(table, ds, split)
            inp = pb.prepare_inputs(b.bulk, b.reference, prog, treatment=b.treatment, max_genes=1000)
            T = np.asarray(inp.T)
            tr = truth_split(inp, b, T)
            a = json.load(open(f))["arrays"]
            draws = np.load(z7)["pi"]
            est = {"PALM": contributions(inp.B, inp.Gmat, np.median(draws, 0), T, np.array(a["effects"]),
                                         np.array(a["lo"]), np.array(a["hi"]),
                                         fraction_draws=draws[np.linspace(0, len(draws) - 1, 50).astype(int)],
                                         seed=split)}
            zi = os.path.join(pv.DEV, dsn, "s%d__instaprism.npz" % split)
            if os.path.exists(zi):
                z = np.load(zi)
                donors = b.meta["donor_id"].astype(str).values
                e, _, _, _ = pv.practice_test(z["model__activity"], T, donors)
                est["InstaPrism"] = contributions(inp.B, inp.Gmat, z["model__fractions"], T, e, n_draws=1)
            big = np.abs(tr["total"]) >= np.quantile(np.abs(tr["total"]), 0.5)
            row = dict(programs=list(inp.programs), cell_types=list(inp.cell_types),
                       truth={k: np.asarray(v).tolist() for k, v in tr.items()})
            for m, r in est.items():
                sh, tsh = r["share"][:, big], tr["share"][:, big]
                s = dict(share_mae=float(np.nanmean(np.abs(sh - tsh))),
                         top_right=float(np.mean(np.argmax(np.abs(r["contribution"][:, big]), 0)
                                                 == np.argmax(np.abs(tr["contribution"][:, big]), 0))),
                         r_contribution=float(np.corrcoef(r["contribution"].ravel(), tr["contribution"].ravel())[0, 1]),
                         r_total=float(np.corrcoef(r["total"], tr["total"])[0, 1]),
                         composition_abs=float(np.abs(r["composition"]).sum() / np.abs(r["contribution"]).sum()),
                         true_composition_abs=float(np.abs(tr["composition"]).sum() / np.abs(tr["contribution"]).sum()))
                if m == "PALM":
                    s["share_cov"] = float(np.nanmean((tsh >= r["share_lo"][:, big]) & (tsh <= r["share_hi"][:, big])))
                row[m] = dict(scores=s, **{k: np.asarray(v).tolist() for k, v in r.items()
                                           if k in ("share", "share_lo", "share_hi", "composition", "intrinsic",
                                                    "total")})
            res.setdefault(dsn, {})[str(split)] = row
            print(dsn, split, {m: {k: round(v, 3) for k, v in row[m]["scores"].items()}
                               for m in est}, flush=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, default=float)


if __name__ == "__main__":
    main(*sys.argv[1:3])
