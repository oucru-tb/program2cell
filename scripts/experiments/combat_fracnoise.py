"""Real-bulk fraction-error level (v4 development, 2026-11-05): the per-sample fraction noise s
(gls_headroom.perturb_fractions: softmax(log pi + N(0, s^2))) that makes the truth as wrong as PALM-lite is on
COMBAT real bulk. The bulk cannot show this error itself (estimate_frac_sd), so real-bulk analyses take s from
here. Per split and bulk (sc / real / real_selfoffset): PALM-lite fractions vs the single-cell read fractions
of the same specimens; per-type correlation, MAE, and the matching s (mean per-type r over 20 perturbations;
also matched on log-ratio sd: sd of log(pi_hat / pi) after removing each type's mean, the per-sample part).
Usage: python combat_fracnoise.py SPLITS -> benchmarks/dev/combat_fracnoise/sepsis__s{split}.json
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import combat_real as cr
import gls_headroom as gh
import program2cell as pb
from program2cell.core.realloc import em_fractions

OUT = os.path.join(ROOT, "benchmarks", "dev", "combat_fracnoise")
GRID = np.round(np.arange(0, 3.01, 0.05), 2)


def _stats(ph, pt):
    r = np.nanmean([np.corrcoef(ph[:, c], pt[:, c])[0, 1] for c in range(pt.shape[1])])
    lr = np.log(np.clip(ph, 1e-4, None)) - np.log(np.clip(pt, 1e-4, None))
    return float(r), float(np.abs(ph - pt).mean()), float((lr - lr.mean(0)).std())


def match_s(pt, r_obs, sd_obs, n=20, seed=0):
    rng = np.random.default_rng(seed)
    R, S = [], []
    for s in GRID:
        st = [_stats(gh.perturb_fractions(pt, s, rng), pt) for _ in range(n)]
        R.append(np.mean([x[0] for x in st])); S.append(np.mean([x[2] for x in st]))
    R, S = np.array(R), np.array(S)
    return float(GRID[np.argmin(np.abs(R - r_obs))]), float(GRID[np.argmin(np.abs(S - sd_obs))])


def run(splits, contrast="sepsis"):
    import anndata as ad
    from program2cell.core import PseudobulkTable, make_benchmark
    os.makedirs(OUT, exist_ok=True)
    treated, control, n_test = cr.CONTRASTS[contrast]
    bk = ad.read_h5ad(os.path.join(cr.D, "combat_bulk.h5ad"))
    bk = bk[bk.obs["paired"].values & ~bk.obs_names.duplicated()]
    table = PseudobulkTable.from_h5ad(os.path.join(cr.D, "combat_sc_pseudobulk_%s.h5ad" % contrast), {}, cr.TYPES)
    bulk_all = pd.DataFrame(sp.csr_matrix(bk.X).toarray().T, index=bk.var_names, columns=bk.obs_names)
    prog = cl.db._programs(pb, "progeny7")
    for split in splits:
        f = os.path.join(OUT, "%s__s%d.json" % (contrast, split))
        if os.path.exists(f):
            continue
        b = make_benchmark(table, cell_types=cr.TYPES, min_cells=20, seed=split, truth_split=True,
                           contrast=("arm", "T", "C"), n_test=n_test)
        genes = [g for g in b.bulk.index if g in bulk_all.index]
        real = bulk_all.loc[genes, list(b.bulk.columns)]
        bulks = {"sc": b.bulk.loc[genes], "real": real, "real_selfoffset": pb.correct_platform(real, b.reference)}
        out = dict(contrast=contrast, split=split)
        for name, Y in bulks.items():
            inp = pb.prepare_inputs(Y, b.reference, prog, treatment=b.treatment, max_genes=1000)
            pt = np.asarray(b.truth(inp)["pi"], float)
            ph = em_fractions(np.asarray(inp.Y, float), np.asarray(inp.B, float))
            r, mae, sd = _stats(ph, pt)
            s_r, s_sd = match_s(pt, r, sd, seed=split)
            out[name] = dict(frac_r=r, mae=mae, logratio_sd=sd, s_match_r=s_r, s_match_sd=s_sd,
                             bias=(np.log(np.clip(ph, 1e-4, None)) - np.log(np.clip(pt, 1e-4, None))).mean(0).tolist())
            print("combat s%d %-15s frac r %.2f mae %.3f logratio sd %.2f -> s (r) %.2f s (sd) %.2f" % (
                split, name, r, mae, sd, s_r, s_sd), flush=True)
        json.dump(out, open(f, "w"))


if __name__ == "__main__":
    run(cl.db._splits(sys.argv[1]))
