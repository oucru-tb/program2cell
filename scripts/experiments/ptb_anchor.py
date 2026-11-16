"""Composition-anchored platform offsets for whole-blood bulk without blood counts (PTB application, 2026-11-11).

Problem: on whole-blood bulk the study-estimated offsets (correct_platform) keep the mean composition the
uncorrected fit implies (platform.py docstring); on PTB that is T cells ~0.01-0.05 and NK 0.25-0.35 of reads
(the same failure as COMBAT real bulk: B and CD8 T driven to ~0, combat_cytof.py).
Anchor: per-gene log2 offset = median over the study's HEALTHY samples of log2(bulk CPM / expected CPM), expected
CPM = (healthy read fractions) @ reference profiles, with the healthy read fractions taken from measured
whole-blood composition of healthy adults (COMBAT whole-blood CyTOF, healthy volunteers; granulocytes other than
neutrophils, pDC, plasmablasts and unclassified dropped; myeloid split CD14 / CD16 Mono / DC and T split CD4 / CD8
by the COMBAT healthy single-cell proportions; cells -> reads with each type's reads per cell in the reference
table). Assumption: the study's healthy group has, on average, the composition of COMBAT's healthy volunteers.
The case - control difference is then estimated relative to that anchor (one offset per gene for every sample).

validate: the same procedure on COMBAT real bulk: anchor on the COMBAT healthy volunteers with CyTOF, estimate
PALM-lite fractions of every other paired specimen (sepsis, COVID), compare with their own CyTOF at group level
(B, T, Myeloid, NK, Neutro read fractions): Pearson r per group and mean log2 bias; against no correction and the
study self-offsets (correct_platform).
ptb: the anchored offsets for the PTB bulk -> data/ptb/ptb_anchor_offsets.csv and PALM-lite fractions summary.
Reference: data/pbmc_ref/combat_hv_neutro_pseudobulk.h5ad (8 types; build_combat_neutro_ref.py).
REFSET=coarse (2026-11-11): the 5-type reference (B, T, NK, Myeloid, Neutro = the CyTOF groups, no within-group
split; build_combat_neutro_ref.py --coarse) -> data/ptb/ptb_anchor_offsets_coarse.csv.
Usage: [REFSET=fine|coarse] python ptb_anchor.py validate | ptb
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments")); sys.path.insert(0, os.path.join(ROOT, "src"))
import program2cell as pb
from program2cell.core import PseudobulkTable
from program2cell.core.realloc import em_fractions

TYPES = ["B", "CD4 T", "CD8 T", "NK", "CD14 Mono", "CD16 Mono", "DC", "Neutro"]
GROUP = {"B": "B.cell", "CD4 T": "T.cell", "CD8 T": "T.cell", "NK": "NK", "CD14 Mono": "Myeloid",
         "CD16 Mono": "Myeloid", "DC": "Myeloid", "Neutro": "Neutrophil"}
REF_FILE = os.path.join(ROOT, "data", "pbmc_ref", "combat_hv_neutro_pseudobulk.h5ad")
OFF_FILE = os.path.join(ROOT, "data", "ptb", "ptb_anchor_offsets.csv")
COARSE = os.environ.get("REFSET", "fine") == "coarse"
if COARSE:
    TYPES = ["B", "T", "NK", "Myeloid", "Neutro"]
    GROUP = {"B": "B.cell", "T": "T.cell", "NK": "NK", "Myeloid": "Myeloid", "Neutro": "Neutrophil"}
    REF_FILE = os.path.join(ROOT, "data", "pbmc_ref", "combat_hv_neutro_coarse_pseudobulk.h5ad")
    OFF_FILE = os.path.join(ROOT, "data", "ptb", "ptb_anchor_offsets_coarse.csv")


def load_ref():
    os.environ.setdefault("REF", "combat5" if COARSE else "combat")
    import run_ptb as rp
    table = PseudobulkTable.from_h5ad(REF_FILE, None, TYPES)
    ref = rp.reference_from_table(table, TYPES)
    o = table.obs
    rpc = np.array([o.loc[o.cell_type == t, "n_counts"].sum() / o.loc[o.cell_type == t, "n_cells"].sum()
                    for t in TYPES])
    # within-group cell proportions from the healthy single-cell data (PBMC types only)
    cells = o[o.cell_type != "Neutro"].groupby("cell_type")["n_cells"].sum()
    within = {t: cells[t] / sum(cells[u] for u in TYPES if GROUP[u] == GROUP[t]) if t in cells else 1.0
              for t in TYPES}
    return table, ref, rpc, within


def cytof_cells(cyt_row, within):
    """CyTOF group % -> cell fractions over the 8 types (dropped groups renormalised away)."""
    c = np.array([float(cyt_row[GROUP[t]]) * within[t] for t in TYPES])
    return c / c.sum()


def cells_to_reads(c, rpc):
    r = c * rpc
    return r / r.sum()


def anchor_offsets(bulk, ref, read_frac, min_cpm=10, clip=8.0):
    """bulk: genes x healthy samples (raw counts); read_frac: (K,) mean healthy read fractions. log2 offsets per gene
    (bulk / expected), 0 for genes below min_cpm mean CPM."""
    ri = {g: i for i, g in enumerate(ref.genes)}
    genes = [g for g in bulk.index if g in ri]
    Bm = np.asarray(ref.B, float)[:, [ri[g] for g in genes]]
    V = bulk.loc[genes].values.T.astype(float)
    cpm = V / V.sum(1, keepdims=True) * 1e6
    q = read_frac @ (Bm / Bm.sum(1, keepdims=True))
    q = q / q.sum() * 1e6
    off = np.median(np.log2(cpm + 1) - np.log2(q + 1)[None], 0)
    return pd.Series(np.where(cpm.mean(0) > min_cpm, np.clip(off, -clip, clip), 0.0), index=genes)


def lite(Y, ref, prog, n_genes=1000):
    inp = pb.prepare_inputs(Y, ref, prog, treatment=pd.Series(np.arange(Y.shape[1]) % 2, index=Y.columns),
                            max_genes=n_genes)
    return em_fractions(np.asarray(inp.Y, float), np.asarray(inp.B, float)), list(inp.cell_types)


def _frame(a):
    return pd.DataFrame(sp.csr_matrix(a.X).toarray().T, index=a.var_names, columns=a.obs_names)


def validate():
    import combat_cytof as cc
    table, ref, rpc, within = load_ref()
    prog = pb.load_programs(os.path.join(ROOT, "data", "hallmark_immune.csv"))
    bk = ad.read_h5ad(os.path.join(cc.cr.D, "combat_bulk.h5ad"))
    bk = bk[bk.obs["paired"].values & ~bk.obs_names.duplicated()]
    cyt = cc.cytof_table()
    bk = bk[bk.obs_names.isin(cyt.index)]
    genes = [g for g in ref.genes if g in set(bk.var_names)]
    Y = _frame(bk).loc[genes]
    hv = bk.obs_names[bk.obs.label == "HV"]
    truth = pd.DataFrame([cells_to_reads(cytof_cells(cyt.loc[s], within), rpc) for s in bk.obs_names],
                         index=bk.obs_names, columns=TYPES)
    off = anchor_offsets(Y[hv], ref, truth.loc[hv].mean().values)
    test = [s for s in bk.obs_names if s not in set(hv)]
    print("COMBAT: anchor on %d healthy with CyTOF, test %d other specimens" % (len(hv), len(test)))
    groups = ["B.cell", "T.cell", "Myeloid", "NK", "Neutrophil"]
    tg = pd.DataFrame({g: truth[[t for t in TYPES if GROUP[t] == g]].sum(1) for g in groups})
    for name, Yc in (("none", Y), ("self", pb.correct_platform(Y, ref)),
                     ("anchored", pb.correct_platform(Y, ref, offsets=off.reindex(genes).fillna(0)))):
        pi, ct = lite(Yc, ref, prog)
        P = pd.DataFrame(pi, index=Yc.columns, columns=ct)
        pg = pd.DataFrame({g: P[[t for t in ct if GROUP[t] == g]].sum(1) for g in groups})
        rows = []
        for g in groups:
            a, b = pg.loc[test, g], tg.loc[test, g]
            rows.append(dict(group=g, r=np.corrcoef(a, b)[0, 1], est_mean=a.mean(), cytof_mean=b.mean(),
                             log2_bias=np.mean(np.log2(np.clip(a, 1e-3, None) / np.clip(b, 1e-3, None)))))
        print("\n== %s offsets (test specimens; read fractions)" % name)
        print(pd.DataFrame(rows).round(3).to_string(index=False))


def ptb():
    import combat_cytof as cc
    table, ref, rpc, within = load_ref()
    prog = pb.load_programs(os.path.join(ROOT, "data", "hallmark_immune.csv"))
    bk = ad.read_h5ad(os.path.join(cc.cr.D, "combat_bulk.h5ad"))
    cyt = cc.cytof_table()
    hv = [s for s in bk.obs_names[bk.obs.label == "HV"] if s in cyt.index]
    anchor = np.mean([cells_to_reads(cytof_cells(cyt.loc[s], within), rpc) for s in hv], 0)
    print("anchor (COMBAT healthy, %d with CyTOF) read fractions:" % len(hv), dict(zip(TYPES, anchor.round(3))))
    a = ad.read_h5ad(os.path.join(ROOT, "data", "ptb", "ptb_bulk.h5ad"))
    a = a[a.obs["cmp1_PTB_vs_Healthy"].notna()].copy()
    genes = [g for g in ref.genes if g in set(a.var_names)]
    Y = _frame(a).loc[genes]
    healthy = a.obs_names[a.obs.cmp1_PTB_vs_Healthy == 0]
    off = anchor_offsets(Y[healthy], ref, anchor)
    off.rename("log2_offset").to_csv(OFF_FILE)
    print("wrote", OFF_FILE, "(%d genes, %d non-zero)" % (len(off), int((off != 0).sum())))
    grp = np.where(a.obs.cmp1_PTB_vs_Healthy.values == 1, "PTB", "Healthy")
    runs = [] if COARSE else [("self", pb.correct_platform(Y, ref))]       # self offsets are slow; fine set only
    runs.append(("anchored", pb.correct_platform(Y, ref, offsets=off.reindex(genes).fillna(0))))
    for name, Yc in runs:
        pi, ct = lite(Yc, ref, prog)
        P = pd.DataFrame(pi, index=Yc.columns, columns=ct)
        print("\n== PTB, %s offsets: mean read fractions" % name)
        print(P.groupby(grp).mean().T.round(3).to_string())


if __name__ == "__main__":
    {"validate": validate, "ptb": ptb}[sys.argv[1]]()
