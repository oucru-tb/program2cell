"""PALM on the in-house PTB whole-blood bulk: pulmonary TB (DS_PTB + MDR_PTB, 296) vs healthy (43TB + HV, 30).
Application, not a benchmark: there is no truth.

Inputs (build_ptb.py; reference: build_pbmc_ref.py or build_combat_neutro_ref.py):
  data/ptb/ptb_bulk.h5ad            raw counts, obs cmp1_PTB_vs_Healthy (1 PTB, 0 healthy)
  data/pbmc_ref/<REF_FILE>          reference table (see REF below)

No run correction (user decision 2026-11-10): every PTB sample is sequencing Run1 and every healthy sample Run2,
so the PTB - Healthy effects include any Run1 vs Run2 technical difference. State this with every result.

Recipe (combat_v4.py real_selfoffset): study platform correction (correct_platform), PALM-lite fractions, freeze v2
(estimator "auto"; the held-out v4 decision 2026-11-10 kept v2 as the method; EST=v4gated runs gated v4 with
frac_sd = FRAC_SD_REAL_BULK["platform_corrected"]), decompose=True (cells changed vs more cells). Reference and
calibration from all reference donors (none of them is in the bulk).
OFFSET=anchored (2026-11-11, ptb_anchor.py; REF=combat only): platform offsets anchored on the healthy group having
the mean whole-blood composition of COMBAT's healthy volunteers (CyTOF), in place of the study self-offsets.
Output: results/ptb/<REF>[_<EST>][_<PROG>][_anchored]/ effects.csv, fractions.csv, info.json, effects_heatmap.png,
effects_intervals.png
Usage: [REF=l2|combat|broad] [EST=auto|v4gated] [PROG=progeny7|reactome_immune6|immune_response6|immune_response7]
       [OFFSET=self|anchored] python run_ptb.py
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import program2cell as pb
from program2cell.core import PseudobulkTable
from program2cell.core.data import Reference
from program2cell.core.gls import FRAC_SD_REAL_BULK

# REF=l2 (default, user decision 2026-11-10): celltype.l2 labels (17 types), specimen = donor, min_cells 1;
# REF=broad: 8 broad types, specimen = donor::library; REF=combat: COMBAT healthy PBMC + GSE188288 neutrophils
REF = os.environ.get("REF", "l2")
EST_TAG = "" if os.environ.get("EST", "auto") == "auto" else "_" + os.environ["EST"]
# PROG=progeny7 (default) or a program CSV in data/ (e.g. reactome_immune6: TLR, NLR, IFNb, TNF, BCR, TCR from
# Reactome; build_reactome_programs.py; user choice 2026-11-10, not benchmarked)
PROG = os.environ.get("PROG", "progeny7")
PROG_TAG = "" if PROG == "progeny7" else "_" + PROG
OFFSET = os.environ.get("OFFSET", "self")
OUT = os.path.join(ROOT, "results", "ptb", "%s%s%s%s" % (REF, EST_TAG, PROG_TAG, "" if OFFSET == "self" else "_" + OFFSET))
if REF == "l2":
    TYPES = ["B naive", "B memory", "Plasmablast", "CD4 Naive", "CD4 TM", "Treg", "CD4 CTL", "CD8 Naive", "CD8 TM",
             "MAIT", "OtherT", "NK", "CD14 Mono", "CD16 Mono", "cDC", "pDC", "Neutro"]
    REF_FILE, MIN_CELLS = "pbmc_neutro_l2_pseudobulk.h5ad", 1
elif REF == "combat5":
    # the 5 CyTOF groups (user choice 2026-11-11, PTB power; build_combat_neutro_ref.py --coarse)
    TYPES = ["B", "T", "NK", "Myeloid", "Neutro"]
    REF_FILE, MIN_CELLS = "combat_hv_neutro_coarse_pseudobulk.h5ad", 20
elif REF == "combat":
    # COMBAT healthy-volunteer PBMC (10 donors) + GSE188288 neutrophils (build_combat_neutro_ref.py)
    TYPES = ["B", "CD4 T", "CD8 T", "NK", "CD14 Mono", "CD16 Mono", "DC", "Neutro"]
    REF_FILE, MIN_CELLS = "combat_hv_neutro_pseudobulk.h5ad", 20
else:
    TYPES = ["B", "CD4 T", "CD8 T", "NK", "CD14 Mono", "CD16 Mono", "DC", "Neutro"]
    REF_FILE, MIN_CELLS = "pbmc_neutro_pseudobulk.h5ad", 20
CMP = "cmp1_PTB_vs_Healthy"


def _frame(a):
    return pd.DataFrame(sp.csr_matrix(a.X).toarray().T, index=a.var_names, columns=a.obs_names)


def reference_from_table(table, types, min_rows=1):
    """As make_benchmark's reference, with every donor: per donor and type sum the rows, normalise, average."""
    from program2cell.core.pseudobulk import _csr
    obs = table.obs[table.obs["n_cells"] >= min_rows]
    X = _csr(table.counts)
    pos = {k: i for i, k in enumerate(table.obs.index)}
    B = np.zeros((len(types), X.shape[1]))
    rna, ncell, ndon = np.zeros(len(types)), np.zeros(len(types), int), np.zeros(len(types), int)
    for k, c in enumerate(types):
        rc = obs[obs["cell_type"] == c]
        prof = [np.asarray(X[[pos[i] for i in rd.index]].sum(0)).ravel() for _, rd in rc.groupby("donor_id")]
        B[k] = np.mean([p / p.sum() for p in prof], 0)
        rna[k] = X[[pos[i] for i in rc.index]].sum() / rc["n_cells"].sum()
        ncell[k], ndon[k] = int(rc["n_cells"].sum()), len(prof)
    B = (B + 1e-6) / (B + 1e-6).sum(1, keepdims=True)
    return Reference(B=B, genes=list(table.genes), cell_types=list(types), rna_content=rna, n_cells=ncell,
                     n_donors=ndon)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    table = PseudobulkTable.from_h5ad(os.path.join(ROOT, "data", "pbmc_ref", REF_FILE),
                                      None, TYPES)
    ref = reference_from_table(table, TYPES)
    a = ad.read_h5ad(os.path.join(ROOT, "data", "ptb", "ptb_bulk.h5ad"))
    a = a[a.obs[CMP].notna()].copy()
    genes = [g for g in ref.genes if g in set(a.var_names)]
    Y = _frame(a).loc[genes]
    T = a.obs[CMP].astype(int).values
    info = dict(run_correction=None, offsets=OFFSET, reference=REF, programs=PROG, cell_types=TYPES, n_ptb=int(T.sum()), n_healthy=int((T == 0).sum()), n_genes_shared=len(genes),
                runs=pd.crosstab(a.obs[CMP].astype(int), a.obs.run).to_dict())
    if OFFSET == "anchored":
        assert REF in ("combat", "combat5"), "anchored offsets are fitted for the COMBAT references"
        off = pd.read_csv(os.path.join(ROOT, "data", "ptb", "ptb_anchor_offsets%s.csv" % (
            "_coarse" if REF == "combat5" else "")), index_col=0).iloc[:, 0]
        Yc = pb.correct_platform(Y, ref, offsets=off.reindex(genes).fillna(0.0))
    else:
        Yc = pb.correct_platform(Y, ref)                                   # study platform correction (frozen recipe)
    if PROG == "progeny7":
        import dev_bench as db
        prog = db._programs(pb, "progeny7")                                # PROGENy, 7 programs (as benchmarked)
    else:
        prog = pb.load_programs(os.path.join(ROOT, "data", PROG + ".csv"))
    inp = pb.prepare_inputs(Yc, ref, prog, treatment=pd.Series(T, index=Yc.columns), max_genes=1000)
    cal = pb.calibration_from_table(table, [], inp.cell_types, inp.genes, inp.Gmat, all_specimens=True, n_folds=3, min_cells=MIN_CELLS,
                                    seed=0)
    # held-out v4 decision (2026-11-10, docs/heldout_v4_results.md): freeze v2 (estimator="auto") is the method;
    # EST=v4gated reruns the secondary estimator
    est = os.environ.get("EST", "auto")
    kw = dict(v4_kw=dict(frac_sd=FRAC_SD_REAL_BULK["platform_corrected"])) if est == "v4gated" else {}
    r = pb.program_deconvolution(inp, cal, engine="lite", treatment=np.asarray(inp.T), estimator=est,
                                 decompose=True, seed=0, **kw)
    tab = r.table("effects")
    tab.to_csv(os.path.join(OUT, "effects.csv"), index=False)
    fr = pd.DataFrame(r.fractions, index=Yc.columns, columns=r.cell_types)
    fr.insert(0, "group", np.where(np.asarray(inp.T) == 1, "PTB", "Healthy"))
    fr.to_csv(os.path.join(OUT, "fractions.csv"))
    g = r.info.get("gate", {})
    info.update(gate={k: (v if isinstance(v, (str, bool, int, float)) else float(np.asarray(v)))
                      for k, v in g.items() if np.ndim(v) == 0},
                n_genes_model=len(inp.genes),
                program_genes={p: int((np.asarray(inp.Gmat)[j] != 0).sum()) for j, p in enumerate(inp.programs)},
                program_genes_list={p: [g for g, w in zip(inp.genes, np.asarray(inp.Gmat)[j]) if w != 0]
                                    for j, p in enumerate(inp.programs)},
                seconds=time.time() - t0,
                mean_fractions=fr.groupby("group").mean(numeric_only=True).round(3).to_dict())
    json.dump(info, open(os.path.join(OUT, "info.json"), "w"), indent=1, default=str)
    import matplotlib
    matplotlib.use("Agg")
    from program2cell import plots
    plots.plot_effect_heatmap(tab).figure.savefig(os.path.join(OUT, "effects_heatmap.png"), dpi=200,
                                                  bbox_inches="tight")
    plots.plot_effect_intervals(tab, ncols=4 if REF == "l2" else 4).savefig(os.path.join(OUT, "effects_intervals.png"), dpi=200,
                                                      bbox_inches="tight")
    print("done %s in %.0fs; gate -> %s; calls %d of %d" % (os.path.basename(OUT), time.time() - t0, g.get("chosen"),
                                                             int(tab.called.sum()), len(tab)))


if __name__ == "__main__":
    main()
