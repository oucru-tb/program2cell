"""McFarland et al. 2020 (MIX-Seq; via pertpy) -> semi-synthetic pooled-cell-line mixtures with REAL drug responses,
in the format of PseudobulkTable.from_h5ad. Known answer: idasanutlin (MDM2 inhibitor) activates p53 only in
TP53-wild-type lines; trametinib (MEK inhibitor) switches MAPK off in every line.

Cell types = 6 cancer cell lines (24 h, 'normal' quality cells): 3 p53 responders and 3 non-responders, chosen by
the measured induction of CDKN1A + MDM2 under idasanutlin (LNCaP 4.6, RCC10RGB 3.3, SNU1079 2.8 log2 FC vs
0.5-0.7 for COLO680N, BICR31, TEN). Genetics agree (Cellosaurus, checked 2026-11-03): COLO 680N TP53 p.R248W, BICR 31
p.V173_R174insG, TEN p.R342* (all homozygous); no TP53 mutation listed for LNCaP clone FGC, RCC10RGB, SNU-1079.

Each line's cells per arm (control, idasanutlin, trametinib) are split 50/50 into a reference pool and a test
pool (disjoint cells):
  - test mixtures, N_TEST per split set: random line proportions (Dirichlet), ~CELLS cells; the same proportions
    and donor id for the control, idasanutlin and trametinib mixture (a pooled experiment, paired by donor);
    cells drawn without replacement within a mixture, with replacement across mixtures (semi-synthetic)
  - reference mixtures, N_REF: control cells of the reference pool only (label 'refctrl'), for the reference
    profiles and the calibration set
obs: donor_id (mixture id), label (ctrl / ida / tram / refctrl), specimen, cell_type (line), n_cells.
Responder truth for the metrics is the measured per-line effect (benchmark truth), with the TP53 grouping above.
Variants (robustness, docs/benchmarking.md "Stress tests"):
  MIX_VARIANT=shift  the treated arms' line proportions differ from the control's: the p53 responders x1.6 in
                     the idasanutlin arm (composition change correlated with the response, the hardest case),
                     LNCaP (the MAPK non-responder) x1.6 in the trametinib arm -> mcfarland_mix_shift.h5ad
  MIX_VARIANT=tight  less composition spread across mixtures (Dirichlet concentration 10 instead of 2, like
                     blood cohorts) -> mcfarland_mix_tight.h5ad
  MIX_VARIANT=gradient  single-group cohort (benchmarking.md 0.8): each test mixture is one "patient" with a
                     clinical value z ~ U(0, 1); every line contributes a share z of drug-treated cells and 1 - z
                     control cells, so a responder line's true activity rises with z. Labels grad_ida and
                     grad_tram (same z and proportions for both drugs), obs column z; reference as usual
                     (refctrl) -> mcfarland_mix_gradient.h5ad
Usage: [MIX_VARIANT=shift|tight] python build_mcfarland.py"""
import os
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
LINES = ["LNCAPCLONEFGC", "RCC10RGB", "SNU1079", "COLO680N", "BICR31", "TEN"]
P53_RESPONDERS = ["LNCAPCLONEFGC", "RCC10RGB", "SNU1079"]
ARMS = {"control": "ctrl", "Idasanutlin": "ida", "Trametinib": "tram"}
N_TEST, N_REF, CELLS, MIN_PER_LINE = 40, 30, 240, 20
VARIANT = os.environ.get("MIX_VARIANT", "")
ALPHA = 10.0 if VARIANT == "tight" else 2.0
SHIFT = {"ida": dict.fromkeys(P53_RESPONDERS, 1.6), "tram": {"LNCAPCLONEFGC": 1.6}} if VARIANT == "shift" else {}


def shifted(props, arm):
    f = np.array([SHIFT.get(arm, {}).get(ln, 1.0) for ln in LINES])
    q = props * f
    return q / q.sum()

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


def mixture(donor, label, arm_key, which, props, z=None):
    counts = np.maximum(np.rint(props * CELLS).astype(int), MIN_PER_LINE)
    for ln, n in zip(LINES, counts):
        if z is None:
            cand = pool[ln, arm_key, which]
            pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
        else:                                     # share z treated, 1 - z control (gradient variant)
            nt = int(round(z * n))
            pt, pc = pool[ln, arm_key, which], pool[ln, "ctrl", which]
            pick = np.concatenate([rng.choice(pt, size=min(nt, len(pt)), replace=False),
                                   rng.choice(pc, size=min(n - nt, len(pc)), replace=False)]).astype(int)
        mats.append(sp.csr_matrix(X[pick].sum(0)))
        rows.append(dict(donor_id=donor, label=label, specimen="%s|%s" % (donor, label), cell_type=ln,
                         n_cells=len(pick), p53_responder=ln in P53_RESPONDERS,
                         z=np.nan if z is None else float(z)))


for k in range(N_TEST):
    props = rng.dirichlet(np.full(len(LINES), ALPHA))
    if VARIANT == "gradient":
        z = rng.uniform()
        for ar in ("ida", "tram"):
            mixture("mix%02d" % k, "grad_" + ar, ar, "test", props, z=z)
        continue
    for ar in ARMS.values():
        mixture("mix%02d" % k, ar, ar, "test", shifted(props, ar))
for k in range(N_REF):
    mixture("ref%02d" % k, "refctrl", "ctrl", "ref", rng.dirichlet(np.full(len(LINES), ALPHA)))

obs = pd.DataFrame(rows)
obs.index = obs["specimen"] + "::" + obs["cell_type"]
obs["project"] = "mcfarland2020"
P = sp.vstack(mats).tocsr()
obs["n_counts"] = np.asarray(P.sum(1)).ravel()
res = ad.AnnData(P.astype(np.float32), obs=obs, var=pd.DataFrame(index=pd.Index(genes)))
res.var_names_make_unique()
dst = os.path.join(ROOT, "data", "pert", "mcfarland_mix%s.h5ad" % ("_" + VARIANT if VARIANT else ""))
res.write_h5ad(dst)
print("%d specimens (%d test mixtures x 3 arms, %d reference) -> %s" % (obs["specimen"].nunique(), N_TEST, N_REF, dst))
