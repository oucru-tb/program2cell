"""The lab's pathway workflow vs PALM on the known-answer datasets.

Lab workflow: InstaPrism per-type expression Z (samples x genes per cell type) -> log1p(CPM) -> GSVA per cell type
(decoupler, gene sets = the up-weighted genes of each PROGENy pathway, so the truth and the sets agree) -> Spearman
correlation of the score with treatment, one test per cell type x pathway; calls at p < 0.1 ('p10') and BH q < 0.1
('bh10'). The effect is Spearman's rho, so coverage and rmse are not defined; r_t, calls on null pairs (fpr_null),
responder AUROC and TPR are.

Usage: python lab_workflow.py run DATASETS SPLITS [instaprism]
Writes benchmarks/dev/practice/{ds}__s{split}__{method}__lab_gsva.json (read by practice_vs_palm.py summary).
"""
import json, os, sys, time
import numpy as np, pandas as pd
from scipy import stats

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import practice_vs_palm as pvp


def gene_sets(inp):
    G = np.asarray(inp.Gmat)
    return pd.DataFrame([(p, g) for j, p in enumerate(inp.programs) for g, w in zip(inp.genes, G[j]) if w > 0],
                        columns=["source", "target"])


def lab_scores(dsn, split, inp, method="instaprism"):
    """(N, C, P) GSVA scores from the per-type expression of InstaPrism (or another method with Z files)."""
    import decoupler as dc
    from palm_bulk.compare.methods import safe_names
    d = os.path.join(cl.ROOT, "benchmarks", "dev", "work", dsn, "s%d" % split, method)
    fwd, _ = safe_names(inp.cell_types, "ct")
    net = gene_sets(inp)
    out = np.full((len(inp.samples), len(inp.cell_types), len(inp.programs)), np.nan)
    for c, ct in enumerate(inp.cell_types):
        z = pd.read_csv(os.path.join(d, "Z_%s.csv" % fwd[ct]), index_col=0).reindex(list(inp.samples))
        x = np.log1p(z.div(z.sum(1).clip(lower=1), axis=0) * 1e6)
        x = x.loc[:, x.var() > 0]
        es = dc.mt.gsva(x, net, tmin=5, verbose=False)
        es = es[0] if isinstance(es, tuple) else es
        out[:, c, :] = es.reindex(columns=list(inp.programs)).values
    return out


def run(datasets, splits, method="instaprism"):
    for dsn in datasets:
        for split in splits:
            f = os.path.join(pvp.OUT, "%s__s%d__%s__lab_gsva.json" % (dsn, split, method))
            if os.path.exists(f) or not os.path.isdir(os.path.join(cl.ROOT, "benchmarks", "dev", "work", dsn,
                                                                    "s%d" % split, method)):
                continue
            if not os.path.exists(os.path.join(cl.ROOT, "benchmarks", "dev", dsn, "s%d__%s.json" % (split, method))):
                continue                                                  # InstaPrism fit missing or failed
            t0 = time.time()
            pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
            ct, pg = list(inp.cell_types), list(inp.programs)
            act = lab_scores(dsn, split, inp, method)
            rho, p = np.full(beta.shape, np.nan), np.full(beta.shape, np.nan)
            for c in range(len(ct)):
                for j in range(len(pg)):
                    if np.isfinite(act[:, c, j]).all() and act[:, c, j].std() > 0:
                        rho[c, j], p[c, j] = stats.spearmanr(act[:, c, j], T.astype(float))
            rows = []
            for rule, calls in (("p10", p < 0.1), ("bh10", pvp.bh(p) < 0.1)):
                calls = np.nan_to_num(calls, nan=False).astype(bool)
                inf = np.full(beta.shape, np.inf)
                s = cl.score(np.nan_to_num(rho), -inf, inf, calls, beta, ct, pg, cl.KNOWN.get(dsn))
                s["cov"] = s["width"] = s["cov_known"] = s["rmse"] = np.nan
                s.update(rule=rule, dataset=dsn, split=split, method=method, kind="lab_gsva",
                         seconds=time.time() - t0)
                rows.append(s)
            with open(f, "w") as fh:
                json.dump(rows, fh, default=float)
            print("[%s] %s s%d %s lab GSVA: fpr_null p10 %.2f bh10 %.2f  tpr_t %s  auroc %s" % (
                time.strftime("%H:%M"), dsn, split, method, rows[0]["fpr_null"], rows[1]["fpr_null"],
                rows[1].get("tpr_t"), rows[1].get("auroc")), flush=True)


if __name__ == "__main__":
    run(sys.argv[2].split(","), cl.db._splits(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "instaprism")
