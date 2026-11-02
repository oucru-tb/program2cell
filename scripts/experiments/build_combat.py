"""COMBAT COVID-19 blood atlas (Cell 2022; Zenodo 6120249) -> real bulk paired with single-cell truth.

Inputs (data/combat/): the consortium's raw single-cell pseudobulks (CBD-KEY-CITESEQ-GEX-PSEUDOBULKS/raw_counts/
major_subset.tsv.gz: summed UMI per sample x major subset), per-cell metadata read from the CITE-seq h5ad
(citeseq_obs_subset.csv.gz: sample, major subset, source) for cell numbers, and whole-blood total-RNA bulk raw
counts (CBD-KEY-RNASEQ-WB/Raw_count_data_143_60683.txt). Bulk and single-cell samples are paired on participant +
timepoint (bulk S00045-Ja005T-TRGa <-> single-cell S00045-Ja005E-PBCa).

Major subsets are merged into the broad PBMC set used for Parse (MAP; HSC, platelets, reticulocytes, mast cells
dropped). Note the built-in gap: the bulk is whole blood (neutrophils, red-cell RNA), the single-cell data PBMC.

Outputs:
  data/combat/combat_sc_pseudobulk.h5ad  PseudobulkTable format (one row per sample x cell type; donor_id =
      participant, label = source group, specimen = participant|timepoint)
  data/combat/combat_bulk.h5ad           real bulk counts (samples x genes), obs: specimen, donor_id, label,
      paired (has single-cell data)
Usage: python build_combat.py
"""
import os, re
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
D = os.path.join(ROOT, "data", "combat")
MAP = {"CD4": "CD4 T", "DP": "CD4 T", "CD8": "CD8 T", "MAIT": "CD8 T", "GDT": "CD8 T", "DN": "CD8 T",
       "iNKT": "CD8 T", "cMono": "CD14 Mono", "ncMono": "CD16 Mono", "NK": "NK", "B": "B", "PB": "B", "DC": "DC"}


def _key(s):
    """participant|timepoint from either sample id style."""
    m = re.match(r"^([A-Z]\d+)-([A-Za-z]+\d+)", str(s))
    return "%s|%s" % (m.group(1), m.group(2)) if m else None


def _symbols(ids):
    """Ensembl ids -> gene symbols (CITE-seq var, citeseq_var.csv); keeps the first id of a duplicated symbol."""
    v = pd.read_csv(os.path.join(D, "citeseq_var.csv"))
    v = v[v["feature_types"] == "Gene Expression"].drop_duplicates("_index")
    m = dict(zip(v["gene_ids"], v["_index"]))
    sym = pd.Series([m.get(i) for i in ids], index=ids)
    return sym[sym.notna() & ~sym.duplicated()]


def main():
    obs = pd.read_csv(os.path.join(D, "citeseq_obs_subset.csv.gz"),
                      usecols=["scRNASeq_sample_ID", "Annotation_major_subset", "Source"])
    ncell = obs.groupby(["scRNASeq_sample_ID", "Annotation_major_subset"]).size()
    source = obs.groupby("scRNASeq_sample_ID")["Source"].first()
    pb = pd.read_csv(os.path.join(D, "CBD-KEY-CITESEQ-GEX-PSEUDOBULKS", "raw_counts", "major_subset.tsv.gz"),
                     sep="\t", index_col=0)
    sym = _symbols(pb.index)                                     # programs use gene symbols
    pb = pb.loc[sym.index]
    pb.index = sym.values
    genes = list(pb.index)
    rows, cols = [], []
    groups = {}
    for c in pb.columns:
        sub, smp = c.split("_", 1)
        if sub in MAP:
            groups.setdefault((smp, MAP[sub]), []).append((c, sub))
    for (smp, ct), parts in sorted(groups.items()):
        cols.append(pb[[c for c, _ in parts]].sum(1).values)
        k = _key(smp)
        rows.append(dict(donor_id=k.split("|")[0], label=source.get(smp, "NA"), specimen=k, cell_type=ct,
                         n_cells=int(sum(ncell.get((smp, s), 0) for _, s in parts)), sample_sc=smp,
                         project="combat"))
    o = pd.DataFrame(rows)
    o.index = o["specimen"] + "::" + o["cell_type"]
    P = sp.csr_matrix(np.vstack(cols).astype(np.float32))
    o["n_counts"] = np.asarray(P.sum(1)).ravel()
    sc = ad.AnnData(P, obs=o, var=pd.DataFrame(index=pd.Index(genes)))
    sc.write_h5ad(os.path.join(D, "combat_sc_pseudobulk.h5ad"), compression="gzip")
    print(sc)
    print(o.groupby("cell_type").n_cells.sum().to_string())

    bk = pd.read_csv(os.path.join(D, "rnaseq_wb", "CBD-KEY-RNASEQ-WB", "Raw_count_data_143_60683.txt"), sep="\t",
                     index_col=0)
    sym = _symbols(bk.index)
    bk = bk.loc[sym.index]
    bk.index = sym.values
    spec = [_key(c) for c in bk.columns]
    lab = o.drop_duplicates("specimen").set_index("specimen")["label"]
    bo = pd.DataFrame(dict(specimen=spec, donor_id=[s.split("|")[0] for s in spec], sample_bulk=list(bk.columns),
                           label=[lab.get(s, "NA") for s in spec], paired=[s in lab.index for s in spec]),
                      index=pd.Index(spec))
    bulk = ad.AnnData(sp.csr_matrix(bk.values.T.astype(np.float32)), obs=bo, var=pd.DataFrame(index=bk.index))
    bulk.write_h5ad(os.path.join(D, "combat_bulk.h5ad"), compression="gzip")
    print(bulk)
    print("paired bulk samples:", int(bo.paired.sum()), bo[bo.paired].label.value_counts().to_dict())
    print("genes shared with single-cell:", len(set(bk.index) & set(genes)))


if __name__ == "__main__":
    main()
