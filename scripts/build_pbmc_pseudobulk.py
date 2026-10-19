"""Turn the PBMCpedia pseudo-bulk file into a clean specimen x cell-type count table for benchmarks.

Usage:
  python scripts/build_pbmc_pseudobulk.py IN.h5ad OUT.h5ad [--label AIFI_L1_name] [--min-cells 50]
         [--keep-pooled] [--keep-sorted] [--keep-non-gex]

Steps
  1. Decode strings. The file stores every string with '_' between its characters ('P_B_M_C' = 'PBMC');
     taking every other character restores it, also for names with real underscores.
  2. Filter source rows: GEX libraries only (VDJ/TCR/BCR rows carry few cells); no sorted T/B samples
     (sorting distorts composition); no pooled samples (number_of_pooled_individuals > 1), whose rows
     may mix several people.
  3. Sum the remaining rows per specimen (donor x sample) and cell type. A specimen is one bulk sample
     to be; donors are the unit for cross-fitting (never put one donor in both reference and test).
  4. Report: cell-type shares, marker genes per cell type (to judge whether 'DC' rows are really
     monocytes), and how many specimens have enough cells in the main types.
  5. Write OUT.h5ad: X = raw counts (rows = specimen x cell type, sparse int), obs = specimen, donor_id,
     sample_id, project, disease, sex, age, cell_type, n_cells, n_counts, n_source_rows; var = genes
     (duplicated symbols summed). Also OUT_markers.csv and OUT_report.txt.

Nothing is merged or dropped by cell type here: those are analysis choices made when loading.
"""
import argparse
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

p = argparse.ArgumentParser()
p.add_argument("inp")
p.add_argument("out")
p.add_argument("--label", default="AIFI_L1_name")
p.add_argument("--min-cells", type=int, default=50)
p.add_argument("--keep-pooled", action="store_true")
p.add_argument("--keep-sorted", action="store_true")
p.add_argument("--keep-non-gex", action="store_true")
p.add_argument("--chunk", type=int, default=500, help="rows read at a time")
args = p.parse_args()

MARKERS = {
    "classical mono": ["CD14", "LYZ", "S100A8", "S100A9", "VCAN", "FCN1"],
    "non-classical mono": ["FCGR3A", "MS4A7", "CDKN1C"],
    "cDC": ["FCER1A", "CLEC10A", "CD1C", "CLEC9A"],
    "pDC": ["LILRA4", "IL3RA", "CLEC4C"],
    "T": ["CD3E", "CD3D", "IL7R"],
    "NK": ["NKG7", "GNLY", "KLRD1"],
    "B": ["MS4A1", "CD79A"],
    "platelet / RBC": ["PPBP", "PF4", "HBB"],
}
META = ["donor_id", "sample_id", "project", "disease", "sex", "age"]
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


def decode(s):
    """'P_B_M_C' -> 'PBMC'; strings without the pattern are returned unchanged."""
    if not isinstance(s, str) or len(s) < 3 or len(s) % 2 == 0 or s[1::2] != "_" * (len(s) // 2):
        return s
    return s[::2]


def decode_series(x):
    """Decode a text column (categorical or object); numeric columns are returned unchanged."""
    if not (isinstance(x.dtype, pd.CategoricalDtype) or x.dtype == object or pd.api.types.is_string_dtype(x)):
        return x                                   # (pandas >= 3 uses a 'str' dtype, not object)
    x = x.astype(object).map(lambda v: decode(v) if isinstance(v, str) else v)
    return x.replace({"nan": np.nan, "None": np.nan, "": np.nan})


# ---- 1. read and decode ------------------------------------------------------------------
a = ad.read_h5ad(args.inp, backed="r")
say("input", a.shape)
obs = a.obs.copy()
obs = obs.apply(decode_series)
genes = pd.Index([decode(g) for g in a.var_names])
say("decoded; example:", obs[["donor_id", "project", args.label]].head(3).to_dict("records"))

# ---- 2. filter source rows -----------------------------------------------------------------
keep = pd.Series(True, index=obs.index)


def drop(mask, why):
    global keep
    n = int((keep & mask).sum())
    keep &= ~mask
    say("  drop %-40s %6d rows" % (why, n))


say("filters:")
if not args.keep_non_gex and "library_type" in obs:
    drop(obs["library_type"].fillna("GEX") != "GEX", "non-GEX library (VDJ/TCR/BCR)")
sort_col = next((c for c in obs.columns if "sorted" in c.lower()), None)    # 'unsorted_or_sorted_t/b_cells'
if not args.keep_sorted and sort_col:
    drop(obs[sort_col].isin(["Sorted", "Sorted and Unsorted"]), "sorted T/B samples")
if not args.keep_pooled and "number_of_pooled_individuals" in obs:
    drop(pd.to_numeric(obs["number_of_pooled_individuals"], errors="coerce").fillna(1) > 1, "pooled samples")
drop(obs[args.label].isna(), "no cell-type label")
drop(obs["donor_id"].isna(), "no donor id")
say("kept %d of %d source rows" % (keep.sum(), len(obs)))

# ---- 3. aggregate per specimen x cell type -----------------------------------------------------
o = obs[keep].copy()
o["specimen"] = o["donor_id"].astype(str) + "|" + o["sample_id"].astype(str)
o["cell_type"] = o[args.label].astype(str)
grp = o["specimen"] + "||" + o["cell_type"]
codes, uniq = pd.factorize(grp)
rows = np.flatnonzero(keep.values)
G = sp.csr_matrix((np.ones(len(rows)), (codes, np.arange(len(rows)))), shape=(len(uniq), len(rows)))

# genes: sum duplicated symbols
gcodes, gnames = pd.factorize(genes)
Gg = sp.csr_matrix((np.ones(len(genes)), (np.arange(len(genes)), gcodes)), shape=(len(genes), len(gnames)))
if len(gnames) < len(genes):
    say("summed %d duplicated gene symbols" % (len(genes) - len(gnames)))

parts = []
for k in range(0, len(rows), args.chunk):
    r = rows[k:k + args.chunk]
    X = a.X[r]                                     # backed read; r is increasing, as h5py requires
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
    parts.append(G[:, k:k + len(r)] @ (X @ Gg))
    print("  read %d / %d rows" % (min(k + args.chunk, len(rows)), len(rows)), end="\r", flush=True)
counts = sum(parts).tocsr()
counts.data = np.rint(counts.data)
if not np.allclose(counts.data, np.round(counts.data)):
    sys.exit("counts are not integers: is X raw counts?")
counts = counts.astype(np.int64)

first = o.groupby(grp, sort=False)
meta = pd.DataFrame(index=uniq)
for c in META:
    if c in o:
        vals = first[c].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        conflict = first[c].nunique() > 1
        if conflict.any():
            say("  note: %s differs within %d groups (first value kept)" % (c, int(conflict.sum())))
        meta[c] = vals.reindex(uniq).values
meta["specimen"] = first["specimen"].first().reindex(uniq).values
meta["cell_type"] = first["cell_type"].first().reindex(uniq).values
meta["n_cells"] = first["psbulk_cells"].sum().reindex(uniq).values.astype(int)
meta["n_source_rows"] = first.size().reindex(uniq).values
meta["n_counts"] = np.asarray(counts.sum(1)).ravel().astype(np.int64)
meta.index = [s.replace("||", "::") for s in uniq]
say("aggregated to %d specimen x cell-type rows; %d specimens; %d donors"
    % (len(meta), meta.specimen.nunique(), meta.donor_id.nunique()))

# ---- 4. report ------------------------------------------------------------------------------
share = meta.groupby("cell_type")["n_cells"].sum().sort_values(ascending=False)
say("\ncell-type share of cells after filtering:")
say((share / share.sum()).round(4).to_string())

cpm = sp.diags(1e6 / np.maximum(meta["n_counts"].values, 1)) @ counts
gi = {g: i for i, g in enumerate(gnames)}
big = meta["n_cells"].values >= args.min_cells
rows_m = []
for c in share.index:
    sel = big & (meta["cell_type"].values == c)
    if not sel.any():
        continue
    row = {"cell_type": c, "rows": int(sel.sum())}
    for grp_name, mk in MARKERS.items():
        for g in mk:
            if g in gi:
                row[g] = float(np.median(np.log1p(cpm[sel][:, gi[g]].toarray().ravel())))
    rows_m.append(row)
mk = pd.DataFrame(rows_m).set_index("cell_type")
say("\nmedian log1p(CPM) of marker genes, rows with >= %d cells (compare 'DC' with 'Monocyte'):" % args.min_cells)
say(mk.round(1).to_string())
mk.to_csv(args.out.replace(".h5ad", "_markers.csv"))

wide = meta.pivot_table(index="specimen", columns="cell_type", values="n_cells", aggfunc="sum", fill_value=0)
say("\nspecimens: %d; cells per specimen median %d" % (len(wide), wide.sum(axis=1).median()))
main = [c for c in ["T cell", "NK cell", "B cell", "Monocyte", "DC"] if c in wide]
say("specimens with >= %d cells in each of %s: %d" % (args.min_cells, main, int((wide[main] >= args.min_cells).all(axis=1).sum())))
if {"Monocyte", "DC"} <= set(wide):
    myel = wide["Monocyte"] + wide["DC"]
    other = [c for c in ["T cell", "NK cell", "B cell"] if c in wide]
    ok = (wide[other] >= args.min_cells).all(axis=1) & (myel >= args.min_cells)
    say("... with Monocyte + DC merged into one myeloid type: %d" % int(ok.sum()))
spec_meta = meta.groupby("specimen")[[c for c in META if c in meta]].first()
say("\nspecimens by project x disease (all specimens):")
say(spec_meta.groupby(["project", "disease"], dropna=False).size().unstack(fill_value=0).to_string())

# ---- 5. write -------------------------------------------------------------------------------
for c in meta.columns:
    if meta[c].dtype == object:
        meta[c] = meta[c].astype(str).replace("nan", "NA")
out = ad.AnnData(X=counts, obs=meta, var=pd.DataFrame(index=pd.Index(gnames.astype(str), name="gene")))
out.uns["build"] = dict(source=args.inp, label=args.label, min_cells_report=args.min_cells,
                        keep_pooled=args.keep_pooled, keep_sorted=args.keep_sorted, keep_non_gex=args.keep_non_gex)
out.write_h5ad(args.out, compression="gzip")
open(args.out.replace(".h5ad", "_report.txt"), "w").write("\n".join(log) + "\n")
say("\nwrote", args.out)
