"""Spike-in power check on the real PTB MDR vs DS bulk (2026-11-26; docs/ptb_sensitivity_2026-11-25.md section 8).

Plants SCALE x the PTB vs healthy effect matrix (results/ptb/combat_v5_immune_reactome_innate5_anchored/effects.csv)
into the MDR samples and reruns freeze v5 (level 0.90, same recipe as run_ptb.py CMP=cmp2 COV=batch OFFSET=anchored
REF=combat PROG=immune_reactome_innate5). Planting follows the model's forward equation (core.equations.cell_profiles):
for MDR sample n and gene g, counts are multiplied by sum_c pi_nc B_cg exp(sum_p d_cp G_pg) / sum_c pi_nc B_cg, with
pi = the base run's PALM-lite fractions, then Poisson-resampled for the added part. Fractions are not changed.
Usage: SCALE=0.5 python ptb_spike.py   -> results/ptb/spike/scale<SCALE>.csv
       SINGLE="CD14 Mono:IFNg:0.3" python ptb_spike.py -> one planted pair only (no cross-type leakage of a full
       matrix, and the size-matched interval sized to one effect) -> results/ptb/spike/single_<type>_<prog>_<d>.csv
"""
import os, sys, time, json
import numpy as np, pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
os.environ.setdefault("REF", "combat"); os.environ.setdefault("CMP", "cmp2"); os.environ.setdefault("COV", "batch")
os.environ.setdefault("OFFSET", "anchored"); os.environ.setdefault("PROG", "immune_reactome_innate5")
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import run_ptb as rp                                    # reuses its settings and reference builder
import program2cell as pb
import anndata as ad
from program2cell.core import PseudobulkTable

SCALE = float(os.environ.get("SCALE", "1.0"))
SINGLE = os.environ.get("SINGLE", "")
BASE = os.path.join(ROOT, "results", "ptb", "combat_v5_immune_reactome_innate5_anchored_mdr_vs_ds_batch")
YARD = os.path.join(ROOT, "results", "ptb", "combat_v5_immune_reactome_innate5_anchored", "effects.csv")
OUT = os.path.join(ROOT, "results", "ptb", "spike")


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    table = PseudobulkTable.from_h5ad(os.path.join(ROOT, "data", "pbmc_ref", rp.REF_FILE), None, rp.TYPES)
    ref = rp.reference_from_table(table, rp.TYPES)
    a = ad.read_h5ad(os.path.join(ROOT, "data", "ptb", "ptb_bulk.h5ad"))
    a = a[a.obs[rp.CMP].notna()].copy()
    genes = [g for g in ref.genes if g in set(a.var_names)]
    Y = rp._frame(a).loc[genes]
    T = a.obs[rp.CMP].astype(int).values
    X = pd.get_dummies(a.obs[rp.COV].astype(str), drop_first=True).astype(float).values
    prog = pb.load_programs(os.path.join(ROOT, "data", rp.PROG + ".csv"))
    # planted effects d (C, P) on the program weights of the base run's prepared inputs
    import pickle
    r0, inp0 = pickle.load(open(os.path.join(BASE, "result.pkl"), "rb"))
    yard = pd.read_csv(YARD).pivot(index="cell_type", columns="program", values="effect")
    d = SCALE * yard.loc[list(inp0.cell_types), list(inp0.programs)].values
    tag = "scale%s" % SCALE
    if SINGLE:
        sc, sp_, sd = SINGLE.split(":")
        d = np.zeros_like(d)
        d[list(inp0.cell_types).index(sc), list(inp0.programs).index(sp_)] = float(sd)
        tag = "single_%s_%s_%s" % (sc.replace(" ", ""), sp_, sd)
    pi = pd.read_csv(os.path.join(BASE, "fractions.csv"), index_col=0)[list(inp0.cell_types)].reindex(Y.columns).values
    gi = {g: i for i, g in enumerate(ref.genes)}
    pg = [g for g in inp0.genes if g in set(Y.index)]
    Bp = ref.B[:, [gi[g] for g in pg]]                                     # (C, Gp)
    Gp = pd.DataFrame(np.asarray(inp0.Gmat), columns=list(inp0.genes))[pg].values  # (P, Gp)
    shift = np.exp(d @ Gp)                                                  # (C, Gp)
    rng = np.random.default_rng(int(SCALE * 1000))
    Yv = Y.loc[pg].values.astype(float)
    for n in np.where(T == 1)[0]:
        f = (pi[n] @ (Bp * shift)) / (pi[n] @ Bp)
        add = Yv[:, n] * (f - 1)
        Yv[:, n] = np.maximum(Yv[:, n] + np.sign(add) * rng.poisson(np.abs(add)), 0)
    Y.loc[pg] = Yv
    off = pd.read_csv(os.path.join(ROOT, "data", "ptb", "ptb_anchor_offsets.csv"), index_col=0).iloc[:, 0]
    Yc = pb.correct_platform(Y, ref, offsets=off.reindex(genes).fillna(0.0))
    inp = pb.prepare_inputs(Yc, ref, prog, treatment=pd.Series(T, index=Yc.columns), max_genes=1000)
    cal = pb.calibration_from_table(table, [], inp.cell_types, inp.genes, inp.Gmat, all_specimens=True, n_folds=3,
                                    min_cells=rp.MIN_CELLS, seed=0)
    r = pb.program_deconvolution(inp, cal, engine="lite", treatment=np.asarray(inp.T), estimator="v5", seed=0,
                                 covariates=X)
    tab = r.table("effects")
    base = pd.read_csv(os.path.join(BASE, "effects.csv"))[["cell_type", "program", "effect"]].rename(
        columns={"effect": "effect_base"})
    tab = tab.merge(base, on=["cell_type", "program"])
    pl = pd.DataFrame(d, index=list(inp0.cell_types), columns=list(inp0.programs)).stack().rename("planted")
    tab = tab.merge(pl.reset_index().rename(columns={"level_0": "cell_type", "level_1": "program"}),
                    on=["cell_type", "program"])
    tab["scale"] = SCALE
    tab.to_csv(os.path.join(OUT, tag + ".csv"), index=False)
    big = tab.planted.abs() >= 0.1
    hit = tab.called & (np.sign(tab.effect) == np.sign(tab.planted))
    if SINGLE:
        x = tab[tab.planted != 0].iloc[0]
        print("%s: effect %+.3f [%+.3f, %+.3f] base %+.3f called %s; other calls %d; %.0fs" % (
            tag, x.effect, x.lo, x.hi, x.effect_base, x.called, int(tab.called.sum() - x.called), time.time() - t0),
            flush=True)
        return
    print("scale %.2f: planted |d|>=0.1 %d, called right sign %d (%.2f); calls on |d|<0.05 %d of %d; "
          "slope(effect-base ~ planted) %.2f; %.0fs" % (
              SCALE, big.sum(), (hit & big).sum(), (hit & big).sum() / max(big.sum(), 1),
              (tab.called & (tab.planted.abs() < 0.05)).sum(), (tab.planted.abs() < 0.05).sum(),
              np.polyfit(tab.planted, tab.effect - tab.effect_base, 1)[0], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
