"""Build a donor x cell-type pseudo-bulk table (and a cell subsample) from a single-cell h5ad, e.g. OneK1K
from CELLxGENE. The output has the same format as scripts/build_pbmc_pseudobulk.py, so
palm_bulk.core.PseudobulkTable / make_benchmark and scripts/run_compare.py --pseudobulk read it directly.

Usage:
  python scripts/build_pseudobulk_from_cells.py onek1k.h5ad onek1k_50k [--n-cells 50000 | --n-donors N | --all]
         [--label predicted.celltype.l2] [--split-t] [--min-score 0] [--keep-doublets]
         [--cells-per-type 200] [--seed 0] [--chunk 20000]

What it does
  1. Picks donors at random (seed) until about --n-cells cells, and keeps ALL cells of those donors
     (random cells across all donors would leave too few cells per donor and cell type).
  2. Finds raw counts: CELLxGENE keeps normalised values in X and raw counts in raw.X; whichever holds
     integers is used. Genes are mapped from Ensembl IDs to symbols (var 'feature_name'); duplicated
     symbols are summed.
  3. Groups Azimuth level-2 labels into broad types (B, Monocyte, NK, T or CD4 T / CD8 T with --split-t,
     DC, Other); 'Doublet' cells are dropped unless --keep-doublets; cells with label score below
     --min-score are dropped. Unknown labels go to 'Other' and are listed.
  4. Reads only the selected cells, in chunks, and sums them per donor x broad type and per
     donor x level-2 type (two tables). Works the same for 50k cells or the full 1.25M.
  5. Writes
       OUT_broad.h5ad   rows = donor x broad type   (use this for benchmarks first)
       OUT_fine.h5ad    rows = donor x level-2 type
       OUT_cells.h5ad   raw counts of up to --cells-per-type cells per donor and broad type
                        (cell-level reference for BayesPrism / CIBERSORTx / BLADE and HVG scores)
       OUT_report.txt   donors, cells per donor and type, label grouping, what was dropped
"""
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import anndata as ad

p = argparse.ArgumentParser()
p.add_argument("inp")
p.add_argument("out", help="output prefix")
sel = p.add_mutually_exclusive_group()
sel.add_argument("--n-cells", type=int, default=50000, help="approximate number of cells (whole donors)")
sel.add_argument("--n-donors", type=int)
sel.add_argument("--all", action="store_true", help="all donors")
p.add_argument("--label", default="predicted.celltype.l2")
p.add_argument("--score", default=None, help="label score column (default: <label>.score if present)")
p.add_argument("--min-score", type=float, default=0.0)
p.add_argument("--split-t", action="store_true", help="CD4 T and CD8 T instead of one T type")
p.add_argument("--keep-doublets", action="store_true")
p.add_argument("--cells-per-type", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--chunk", type=int, default=20000)
args = p.parse_args()
rng = np.random.default_rng(args.seed)
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


# Azimuth PBMC level 2 -> broad type. Plasmablasts differ strongly from B cells, so they go to Other.
BROAD = {
    "B naive": "B cell", "B intermediate": "B cell", "B memory": "B cell",
    "CD14 Mono": "Monocyte", "CD16 Mono": "Monocyte",
    "NK": "NK cell", "NK Proliferating": "NK cell", "NK_CD56bright": "NK cell",
    "CD4 Naive": "CD4 T", "CD4 TCM": "CD4 T", "CD4 TEM": "CD4 T", "CD4 CTL": "CD4 T",
    "CD4 Proliferating": "CD4 T", "Treg": "CD4 T",
    "CD8 Naive": "CD8 T", "CD8 TCM": "CD8 T", "CD8 TEM": "CD8 T", "CD8 Proliferating": "CD8 T",
    "MAIT": "other T", "gdT": "other T", "dnT": "other T",
    "cDC1": "DC", "cDC2": "DC", "pDC": "DC", "ASDC": "DC",
    "Plasmablast": "Other", "Platelet": "Other", "Eryth": "Other", "HSPC": "Other", "ILC": "Other",
    "Doublet": "Doublet",
}

# ---- metadata -------------------------------------------------------------------------------
a = ad.read_h5ad(args.inp, backed="r")
obs = a.obs
say("input", a.shape)
lab = obs[args.label].astype(str)
unknown = sorted(set(lab.unique()) - set(BROAD))
if unknown:
    say("labels not in the grouping (-> Other):", unknown)
broad = lab.map(BROAD).fillna("Other")
if not args.split_t:
    broad = broad.replace({"CD4 T": "T cell", "CD8 T": "T cell", "other T": "T cell"})
else:
    broad = broad.replace({"other T": "Other"})
keep = np.ones(len(obs), bool)
if not args.keep_doublets:
    d = (broad == "Doublet").values
    say("drop doublets: %d cells" % (keep & d).sum())
    keep &= ~d
score_col = args.score or (args.label + ".score" if args.label + ".score" in obs else None)
if args.min_score > 0 and score_col:
    lo = (obs[score_col].astype(float) < args.min_score).values
    say("drop label score < %.2f: %d cells" % (args.min_score, (keep & lo).sum()))
    keep &= ~lo

donor = obs["donor_id"].astype(str).values
per_donor = pd.Series(donor[keep]).value_counts()
order = rng.permutation(per_donor.index.values)
if args.all:
    chosen = list(order)
elif args.n_donors:
    chosen = list(order[:args.n_donors])
else:
    cum = per_donor.loc[order].cumsum()
    chosen = list(order[:int(np.searchsorted(cum.values, args.n_cells)) + 1])
keep &= np.isin(donor, chosen)
cells = np.flatnonzero(keep)
say("donors chosen: %d of %d; cells: %d" % (len(chosen), len(per_donor), len(cells)))

# ---- find raw counts and gene symbols ------------------------------------------------------------
f = h5py.File(args.inp, "r")


def matrix_group(name):
    if name not in f:
        return None
    return f[name]["X"] if name == "raw" else f[name]


def sample_values(g):
    if isinstance(g, h5py.Group):
        return g["data"][:100000]
    return g[:200].ravel()


src = None
for name in ["raw", "X"]:
    g = matrix_group(name)
    if g is not None:
        v = sample_values(g)
        if v.size and np.allclose(v, np.round(v)) and v.min() >= 0:
            src = name
            break
if src is None:
    raise SystemExit("no raw integer counts found in raw.X or X")
G = matrix_group(src)
def read_var_index(group):
    """Gene IDs of a stored var dataframe, read directly (backed mode may not expose raw.var)."""
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
    return list(read_elem(f[group]).index)


gene_ids = read_var_index("raw/var") if src == "raw" else list(a.var.index)
sym_map = a.var["feature_name"].astype(str) if "feature_name" in a.var else pd.Series(gene_ids, index=gene_ids)
symbols = pd.Index([sym_map.get(g, g) for g in gene_ids])
gcodes, gnames = pd.factorize(symbols)
Gmap = sp.csr_matrix((np.ones(len(symbols)), (np.arange(len(symbols)), gcodes)), shape=(len(symbols), len(gnames)))
say("raw counts from %s; %d genes -> %d symbols (%d duplicated symbols summed)"
    % ("raw.X" if src == "raw" else "X", len(symbols), len(gnames), len(symbols) - len(gnames)))
sparse = isinstance(G, h5py.Group)
if sparse and G.attrs.get("encoding-type", "csr_matrix") != "csr_matrix":
    raise SystemExit("matrix is %s; this script reads CSR (cells x genes)" % G.attrs.get("encoding-type"))
indptr = G["indptr"][:] if sparse else None


def read_rows(rows):
    """CSR block of the given increasing cell indices, read run by run of consecutive cells."""
    if not sparse:
        return sp.csr_matrix(G[rows, :])
    runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
    blocks = []
    for r in runs:
        s, e = indptr[r[0]], indptr[r[-1] + 1]
        ptr = indptr[r[0]:r[-1] + 2] - s
        blocks.append(sp.csr_matrix((G["data"][s:e], G["indices"][s:e], ptr), shape=(len(r), len(symbols))))
    return sp.vstack(blocks).tocsr()


# ---- aggregate --------------------------------------------------------------------------------
fine = lab.values
groups = {}
for level, labels in [("broad", broad.values), ("fine", fine)]:
    key = pd.Series(donor[cells]).astype(str) + "::" + pd.Series(labels[cells]).astype(str)
    codes, uniq = pd.factorize(key)
    groups[level] = [codes, uniq, sp.csr_matrix((len(uniq), len(gnames)))]       # sparse running sums

sub_take = []                                          # cells for the cell-level subsample
bkey = pd.Series(donor[cells]).astype(str) + "::" + pd.Series(broad.values[cells]).astype(str)
for k, idx in bkey.groupby(bkey).groups.items():
    idx = np.asarray(idx)
    sub_take.append(idx if len(idx) <= args.cells_per_type else rng.choice(idx, args.cells_per_type, replace=False))
sub_take = np.sort(np.concatenate(sub_take))
sub_mask = np.zeros(len(cells), bool)
sub_mask[sub_take] = True
sub_blocks = []

for k in range(0, len(cells), args.chunk):
    rows = cells[k:k + args.chunk]
    X = read_rows(rows) @ Gmap                                                  # cells x symbols
    for level in groups:
        codes, uniq, acc = groups[level]
        c = codes[k:k + len(rows)]
        M = sp.csr_matrix((np.ones(len(c)), (c, np.arange(len(c)))), shape=(len(uniq), len(c)))
        groups[level][2] = acc + M @ X
    m = sub_mask[k:k + len(rows)]
    if m.any():
        sub_blocks.append(X[np.flatnonzero(m)])
    print("  read %d / %d cells" % (min(k + args.chunk, len(cells)), len(cells)), end="\r", flush=True)
print()


def meta_for(uniq, labels_name):
    d = pd.DataFrame([u.split("::", 1) for u in uniq], columns=["donor_id", "cell_type"], index=uniq)
    first = obs.iloc[cells].assign(_d=donor[cells]).groupby("_d", observed=True).first()
    d["specimen"] = d["donor_id"]
    d["sample_id"] = d["donor_id"]
    d["project"] = "OneK1K"
    for c in ["disease", "sex", "age", "development_stage", "pool_number"]:
        if c in first:
            d[c] = first.loc[d["donor_id"], c].astype(str).values
    return d


report = {}
for level, (codes, uniq, acc) in groups.items():
    meta = meta_for(uniq, level)
    meta["n_cells"] = np.bincount(codes, minlength=len(uniq))
    meta["n_source_rows"] = meta["n_cells"]
    X = acc.tocsr()
    X.data = np.rint(X.data)
    X = X.astype(np.int64)
    meta["n_counts"] = np.asarray(X.sum(1)).ravel()
    out = ad.AnnData(X=X, obs=meta, var=pd.DataFrame(index=pd.Index(gnames.astype(str), name="gene")))
    out.uns["build"] = dict(source=args.inp, label=args.label, level=level, split_t=args.split_t,
                            min_score=args.min_score, donors=len(chosen), seed=args.seed)
    out.write_h5ad("%s_%s.h5ad" % (args.out, level), compression="gzip")
    report[level] = meta

cells_obs = obs.iloc[cells[sub_take]][[c for c in ["donor_id", args.label, "sex", "age"] if c in obs]].copy()
cells_obs["cell_type"] = broad.values[cells[sub_take]]
cells_obs["cell_type_fine"] = fine[cells[sub_take]]
cells_obs = cells_obs.astype({c: str for c in cells_obs.columns if c not in ("age",)})
sub = ad.AnnData(X=sp.vstack(sub_blocks).tocsr().astype(np.int32), obs=cells_obs,
                 var=pd.DataFrame(index=pd.Index(gnames.astype(str), name="gene")))
sub.write_h5ad("%s_cells.h5ad" % args.out, compression="gzip")

# ---- report -------------------------------------------------------------------------------------
b = report["broad"]
wide = b.pivot_table(index="donor_id", columns="cell_type", values="n_cells", fill_value=0)
say("\nbroad types, cells per donor (median / 10% / 90%):")
say(wide.quantile([0.5, 0.1, 0.9]).T.round(0).astype(int).rename(columns={0.5: "median", 0.1: "10%", 0.9: "90%"}).to_string())
say("share of cells:", (wide.sum() / wide.values.sum()).round(3).to_dict())
main = [c for c in ["T cell", "CD4 T", "CD8 T", "NK cell", "B cell", "Monocyte"] if c in wide]
for m in [20, 30, 50, 100]:
    say("donors with >= %3d cells in each of %s: %d" % (m, main, int((wide[main] >= m).all(axis=1).sum())))
say("\nlevel-2 labels -> broad type:", {k: v for k, v in sorted(BROAD.items())})
say("wrote %s_broad.h5ad (%d rows), %s_fine.h5ad (%d rows), %s_cells.h5ad (%d cells)"
    % (args.out, len(report["broad"]), args.out, len(report["fine"]), args.out, sub.n_obs))
open("%s_report.txt" % args.out, "w").write("\n".join(log) + "\n")
