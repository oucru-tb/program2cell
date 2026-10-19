"""Sanity checks of a donor x cell-type pseudo-bulk file before using it for benchmarks.

Usage: python scripts/check_pseudobulk.py PSEUDOBULK.h5ad [MIN_CELLS]
Prints: what the rows are keyed on, raw-count check, cell-type proportions pooled over donors (are the
labels plausible for PBMC?), rows and donors usable at MIN_CELLS (default 50) cells per row, sorted /
non-GEX / pooled samples to exclude, and disease x project counts for cross-fitting.
"""
import sys
import numpy as np
import pandas as pd
import anndata as ad

path = sys.argv[1]
min_cells = int(sys.argv[2]) if len(sys.argv) > 2 else 50
a = ad.read_h5ad(path, backed="r")
obs = a.obs.copy()
pd.set_option("display.width", 200, "display.max_rows", 100)

print("== shape", a.shape)
print("\n== example row keys\n", obs[["patient_celltype", "sample_id", "donor_id", "cell_type", "AIFI_L1_name"]].head(8))

# which column defines the cell type of a row: the one constant within patient_celltype
key = obs["patient_celltype"].astype(str)
for col in ["AIFI_L0", "AIFI_L1_name", "AIFI_L2_name", "AIFI_L3_name", "cell_type", "leiden_res_0.5"]:
    if col in obs:
        n = obs[col].astype(str).nunique()
        const = (obs.groupby(key, observed=True)[col].nunique() <= 1).all()
        print("%-16s levels %4d  constant within a row key: %s" % (col, n, const))

X = a.X[:200]
X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
print("\n== counts: integer-valued %s, min %g, max %g" % (np.allclose(X, np.round(X)), X.min(), X.max()))

lab = "AIFI_L1_name"
for cand in ["AIFI_L1_name", "AIFI_L2_name", "AIFI_L3_name"]:
    if cand in obs and (obs.groupby(key, observed=True)[cand].nunique() <= 1).all():
        lab = cand
        break
print("\n== label used for rows (finest constant AIFI level found): %s" % lab)
cells = obs.groupby(lab, observed=True)["psbulk_cells"].sum().sort_values(ascending=False)
print("cell-type share of all cells (PBMC expectation: T 45-70%, Mono 10-30%, B 5-15%, NK 5-15%):")
print((cells / cells.sum()).round(4).to_string())
if lab != "AIFI_L1_name":
    l1 = obs.groupby("AIFI_L1_name", observed=True)["psbulk_cells"].sum().sort_values(ascending=False)
    print("\nAIFI_L1_name share of all cells:")
    print((l1 / l1.sum()).round(4).to_string())

print("\n== rows with >= %d cells: %d of %d" % (min_cells, (obs.psbulk_cells >= min_cells).sum(), len(obs)))
per_donor = obs[obs.psbulk_cells >= min_cells].groupby("donor_id", observed=True)[lab].nunique()
print("donors: %d; donors with >= %d cell types at >= %d cells: %d"
      % (obs.donor_id.nunique(), max(3, int(0.6 * obs[lab].nunique())), min_cells,
         (per_donor >= max(3, int(0.6 * obs[lab].nunique()))).sum()))
print("cells per row: median %d, 10%% %d, 90%% %d" % tuple(np.percentile(obs.psbulk_cells, [50, 10, 90])))

print("\n== to exclude")
for col in ["unsorted_or_sorted_t/b_cells", "library_type", "number_of_pooled_individuals"]:
    if col in obs:
        print(col, obs.groupby(col, observed=True, dropna=False)["donor_id"].nunique().to_dict())

print("\n== disease x project (donors)")
tab = obs.groupby(["project", "disease"], observed=True)["donor_id"].nunique().unstack(fill_value=0)
print(tab.loc[tab.sum(1).sort_values(ascending=False).index].head(15).to_string())
if "psbulk_props" in a.layers:
    pr = a.layers["psbulk_props"][:3]
    pr = pr.toarray() if hasattr(pr, "toarray") else np.asarray(pr)
    print("\nlayer psbulk_props (first 3 rows): range %.3g-%.3g, row sums %s" % (pr.min(), pr.max(), pr.sum(1).round(2)))
