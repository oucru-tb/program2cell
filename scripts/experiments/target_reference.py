"""Target-revised reference on COMBAT real bulk (2026-11-05; modelling review: EPIC-unmix-style second
empirical-Bayes step, plus a learned profile for the missing type, cf. F6).

Fractions on the modelled genes, scored against the same specimens' single-cell read fractions (per-type r,
per-type log bias) and, for the extra column, against the CyTOF granulocyte frequency.
  ref       PALM-lite on the single-cell reference (current)
  target    reference revised by the cohort: per gene, non-negative regression of the bulk's CPM on the
            current fractions across samples gives target profiles; B <- (1 - lam) B_ref + lam B_target
            (both normalised); refit fractions; 3 rounds
  extra     ref plus one extra profile, initialised from the mean residual of the bulk over the reference fit
            and updated like the target profiles (the other profiles stay the reference); its fraction is
            the missing-type estimate
  extra+target  both
Usage: python target_reference.py SPLITS -> benchmarks/dev/target_reference/sepsis__s{split}.json
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.optimize import nnls
from scipy.stats import spearmanr

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import combat_real as cr
import program2cell as pb
from program2cell.core.realloc import em_fractions
from missing_type_score import cytof_freqs

COHORT = os.environ.get("COHORT", "0") == "1"      # learn profiles on all paired bulk specimens
OUT = os.path.join(ROOT, "benchmarks", "dev", "target_reference" + ("_cohort" if COHORT else ""))


def _norm(B):
    return B / B.sum(1, keepdims=True)


def target_profiles(Y, pi):
    """Per gene NNLS of bulk CPM (N,) on fractions (N, C) -> (C, G)."""
    P = Y / Y.sum(1, keepdims=True)
    return np.array([nnls(pi, P[:, g])[0] for g in range(P.shape[1])]).T + 1e-9


def fit(Y, B, extra=False, target=False, lam=0.5, rounds=3):
    Bref = _norm(B)
    Bc = Bref.copy()
    if extra:
        pi0 = em_fractions(Y, Bref)
        res = np.clip(Y / Y.sum(1, keepdims=True) - pi0 @ Bref, 0, None).mean(0) + 1e-9
        Bc = np.vstack([Bref, res / res.sum()])
    for _ in range(rounds if (extra or target) else 0):
        pi = em_fractions(Y, Bc)
        Bt = _norm(target_profiles(Y, pi))
        if target:
            Bc[:len(Bref)] = _norm((1 - lam) * Bref + lam * Bt[:len(Bref)])
        if extra:
            Bc[-1] = Bt[-1]
    return em_fractions(Y, Bc)


def score(ph, pt):
    r = [np.corrcoef(ph[:, c], pt[:, c])[0, 1] for c in range(pt.shape[1])]
    bias = (np.log(np.clip(ph, 1e-4, None)) - np.log(np.clip(pt, 1e-4, None))).mean(0)
    return np.array(r), bias


def run(splits):
    import anndata as ad
    from program2cell.core import PseudobulkTable, make_benchmark
    os.makedirs(OUT, exist_ok=True)
    bk = ad.read_h5ad(os.path.join(cr.D, "combat_bulk.h5ad"))
    bk = bk[bk.obs["paired"].values & ~bk.obs_names.duplicated()]
    table = PseudobulkTable.from_h5ad(os.path.join(cr.D, "combat_sc_pseudobulk_sepsis.h5ad"), {}, cr.TYPES)
    bulk_all = pd.DataFrame(sp.csr_matrix(bk.X).toarray().T, index=bk.var_names, columns=bk.obs_names)
    prog = cl.db._programs(pb, "progeny7")
    gran = cytof_freqs()
    for split in splits:
        b = make_benchmark(table, cell_types=cr.TYPES, min_cells=20, seed=split, truth_split=True,
                           contrast=("arm", "T", "C"), n_test=20)
        genes = [g for g in b.bulk.index if g in bulk_all.index]
        real = bulk_all.loc[genes, list(b.bulk.columns)]
        inp = pb.prepare_inputs(real, b.reference, prog, treatment=b.treatment, max_genes=1000)
        Y, B = np.asarray(inp.Y, float), np.asarray(inp.B, float)
        pt = np.asarray(b.truth(inp)["pi"], float)
        # cohort-wide fit (all paired bulk specimens learn the profiles), scored on the test specimens
        Yall = bulk_all.loc[list(inp.genes)].values.T.astype(float)
        test = [list(bulk_all.columns).index(s) for s in real.columns]
        specs = list(real.columns)
        g = np.array([gran.get(s, np.nan) for s in specs])
        out = dict(split=split, types=list(inp.cell_types))
        for name, kw in (("ref", {}), ("target", dict(target=True)), ("extra", dict(extra=True)),
                         ("extra+target", dict(extra=True, target=True))):
            ph = fit(Yall, B, **kw)[test] if COHORT else fit(Y, B, **kw)
            C = pt.shape[1]
            phc = ph[:, :C] / ph[:, :C].sum(1, keepdims=True)              # renormalise over the reference types
            r, bias = score(phc, pt)
            o = dict(r=r.tolist(), mean_r=float(np.nanmean(r)), bias=bias.tolist(),
                     bias_rms=float(np.sqrt((bias ** 2).mean())))
            if ph.shape[1] > C:
                ok = ~np.isnan(g)
                o.update(extra_mean=float(ph[:, -1].mean()), extra_rho_gran=float(spearmanr(ph[ok, -1], g[ok])[0]))
            out[name] = o
            print("s%d %-12s mean r %.2f  r %s  bias rms %.2f  bias %s%s" % (
                split, name, o["mean_r"], np.round(r, 2), o["bias_rms"], np.round(bias, 1),
                "  extra fraction %.2f, Spearman with CyTOF granulocytes %.2f" % (o["extra_mean"], o["extra_rho_gran"])
                if "extra_mean" in o else ""), flush=True)
        json.dump(out, open(os.path.join(OUT, "sepsis__s%d.json" % split), "w"))


if __name__ == "__main__":
    run(cl.db._splits(sys.argv[1]))
