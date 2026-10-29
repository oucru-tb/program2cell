"""McFarland et al. 2020 (MIX-Seq; via pertpy) -> semi-synthetic pooled-cell-line mixtures with REAL drug responses,
in the format of PseudobulkTable.from_h5ad. Known answer: idasanutlin (MDM2 inhibitor) activates p53 only in
TP53-wild-type lines; trametinib (MEK inhibitor) switches MAPK off in every line.

Cell types = 6 cancer cell lines (24 h, 'normal' quality cells): 3 p53 responders and 3 non-responders, chosen by
the measured induction of CDKN1A + MDM2 under idasanutlin (LNCaP 4.6, RCC10RGB 3.3, SNU1079 2.8 log2 FC vs
0.5-0.7 for COLO680N, BICR31, TEN).

Each line's cells per arm (control, idasanutlin, trametinib) are split 50/50 into a reference pool and a test
pool (disjoint cells):
  - test mixtures, N_TEST per split set: random line proportions (Dirichlet), ~CELLS cells; the same proportions
    and donor id for the control, idasanutlin and trametinib mixture (a pooled experiment, paired by donor);
    cells drawn without replacement within a mixture, with replacement across mixtures (semi-synthetic)
  - reference mixtures, N_REF: control cells of the reference pool only (label 'refctrl'), for the reference
    profiles and the calibration set
obs: donor_id (mixture id), label (ctrl / ida / tram / refctrl), specimen, cell_type (line), n_cells.
Responder truth for the metrics is the measured per-line effect (benchmark truth), with the TP53 grouping above.
Usage: python build_mcfarland.py"""
import os
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
LINES = ["LNCAPCLONEFGC", "RCC10RGB", "SNU1079", "COLO680N", "BICR31", "TEN"]
P53_RESPONDERS = ["LNCAPCLONEFGC", "RCC10RGB", "SNU1079"]
ARMS = {"control": "ctrl", "Idasanutlin": "ida", "Trametinib": "tram"}
N_TEST, N_REF, CELLS, MIN_PER_LINE = 40, 30, 240, 20

a = ad.read_h5ad(os.path.join(ROOT, "data", "pert", "mcfarland_2020.h5ad"), backed="r")
o = a.obs
m = ((o["cell_quality"] == "normal") & (o["time"].astype(str) == "24") & o["perturbation"].astype(str).isin(ARMS)
     & o["cell_line"].astype(str).isin(LINES)).values
sub = a[np.flatnonzero(m)].to_memory()
X = sp.csr_matrix(sub.X)
genes = list(sub.var_names)
line = sub.obs["cell_line"].astype(str).values
arm = sub.obs["perturbation"].astype(str).map(ARMS).values
rng = np.random.default_rng(0)
pool = {}
for ln in LINES:
    for ar in ARMS.values():
        idx = rng.permutation(np.flatnonzero((line == ln) & (arm == ar)))
        h = len(idx) // 2
        pool[ln, ar, "ref"], pool[ln, ar, "test"] = idx[:h], idx[h:]
        print("%-14s %-4s ref %3d test %3d" % (ln, ar, h, len(idx) - h))

rows, mats = [], []


def mixture(donor, label, arm_key, which, props):
    counts = np.maximum(np.rint(props * CELLS).astype(int), MIN_PER_LINE)
    for ln, n in zip(LINES, counts):
        cand = pool[ln, arm_key, which]
        pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
        mats.append(sp.csr_matrix(X[pick].sum(0)))
        rows.append(dict(donor_id=donor, label=label, specimen="%s|%s" % (donor, label), cell_type=ln,
                         n_cells=len(pick), p53_responder=ln in P53_RESPONDERS))


for k in range(N_TEST):
    props = rng.dirichlet(np.full(len(LINES), 2.0))
    for ar in ARMS.values():
        mixture("mix%02d" % k, ar, ar, "test", props)
for k in range(N_REF):
    mixture("ref%02d" % k, "refctrl", "ctrl", "ref", rng.dirichlet(np.full(len(LINES), 2.0)))

obs = pd.DataFrame(rows)
obs.index = obs["specimen"] + "::" + obs["cell_type"]
obs["project"] = "mcfarland2020"
P = sp.vstack(mats).tocsr()
obs["n_counts"] = np.asarray(P.sum(1)).ravel()
res = ad.AnnData(P.astype(np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(genes)))
res.var_names_make_unique()
dst = os.path.join(ROOT, "data", "pert", "mcfarland_mix.h5ad")
res.write_h5ad(dst)
print("%d specimens (%d test mixtures x 3 arms, %d reference) -> %s" % (obs["specimen"].nunique(), N_TEST, N_REF, dst))
