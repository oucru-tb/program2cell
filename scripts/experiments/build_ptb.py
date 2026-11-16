"""Pulmonary TB (PTB) whole-blood bulk RNA-seq -> data/ptb/ptb_bulk.h5ad, in the format of data/combat/combat_bulk.h5ad.

Source: data/ptb/Bulk_whole_blood/Data_RNAseq_RUN1andRUN2_15Feb2023.RData (count_all: featureCounts table,
Geneid/Chr/Start/End/Strand/Length then one column per sample (LIMS_ID); Cli_All: LIMS_ID, Timepoint,
trial_number, Run, ID_code, seq_Batch; anno: Ensembl gene annotation; meta_RNAseq: + RNA_RIN).

Kept: trial_number DS_PTB (29TB, drug-sensitive) and MDR_PTB (28TB, multidrug-resistant); AFBneg_PTB
(smear-negative) dropped (user 2026-11-10). All are D0, one sample per patient.
Batch: batch = run + seq_batch (seq_Batch is numbered within each sequencing run), used as a covariate.
PTB is all Run1 and healthy all Run2, so in PTB vs Healthy run is confounded with condition (no run
correction is applied: user decision 2026-11-10).
Healthy controls: trial_number HV or 43TB (user rule 2026-11-10; Readme: Study "HV" was recoded "43TB").
Also writes data/ptb/ptb_metadata.csv: one row per sample with condition (PTB / Healthy) and one column per
planned comparison (1 = case, 0 = reference, empty = not in that comparison), for checking before any run.
Genes: Ensembl ids summed per gene symbol (as in COMBAT), genes without a symbol dropped.

Output: X raw counts (samples x genes, float32 CSR); obs: specimen (LIMS_ID), donor_id (ID_code), label
(trial_number), timepoint, seq_batch, run, rin; var: gene_ids (';'-joined Ensembl ids), biotype.
Usage: python build_ptb.py
"""
import os, subprocess, tempfile
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
SRC = os.path.join(ROOT, "data", "ptb", "Bulk_whole_blood", "Data_RNAseq_RUN1andRUN2_15Feb2023.RData")
OUT = os.path.join(ROOT, "data", "ptb", "ptb_bulk.h5ad")
GROUPS = ["DS_PTB", "MDR_PTB", "HV", "43TB"]          # AFBneg_PTB dropped (user 2026-11-10)
HEALTHY = ("HV", "43TB")
META = os.path.join(ROOT, "data", "ptb", "ptb_metadata.csv")
RENV = os.environ.get("PALM_R_ENV", "C:/Users/duy.pham/AppData/Local/miniforge3/envs/palm-r")

R = r"""
load("%(src)s")
keep <- Cli_All[Cli_All$trial_number %%in%% c(%(groups)s), ]
X <- count_all[, keep$LIMS_ID, drop = FALSE]
stopifnot(all(colnames(X) == keep$LIMS_ID))
write.csv(data.frame(gene_id = count_all$Geneid, X, check.names = FALSE), gzfile("%(tmp)s/counts.csv.gz"), row.names = FALSE)
m <- meta_RNAseq[match(keep$LIMS_ID, meta_RNAseq$LIMS_ID), c("LIMS_ID", "RNA_RIN")]
write.csv(merge(keep, m, by = "LIMS_ID", all.x = TRUE, sort = FALSE), "%(tmp)s/obs.csv", row.names = FALSE)
write.csv(anno[, c("gene_id", "gene_symbol", "class_new")], "%(tmp)s/anno.csv", row.names = FALSE)
"""


def _symbols(counts, anno):
    """Ensembl rows -> gene-symbol rows (summed), genes without a symbol dropped."""
    sym = anno["gene_symbol"].reindex(counts.index)
    ok = sym.notna() & (sym.astype(str).str.len() > 0)
    return counts[ok].groupby(sym[ok].astype(str).values).sum()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = tmp.replace("\\", "/")
        script = os.path.join(tmp, "export.R")
        open(script, "w").write(R % dict(src=SRC.replace("\\", "/"), tmp=tmp, groups=",".join('"%s"' % g for g in GROUPS)))
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([os.path.join(RENV, "Lib", "R", "bin", "x64"), RENV,
                                       os.path.join(RENV, "Library", "mingw-w64", "bin"),
                                       os.path.join(RENV, "Library", "bin"), env["PATH"]])
        subprocess.run([os.path.join(RENV, "Lib", "R", "bin", "x64", "Rscript.exe"), script], check=True, env=env)
        counts = pd.read_csv(os.path.join(tmp, "counts.csv.gz"), index_col=0)
        obs = pd.read_csv(os.path.join(tmp, "obs.csv")).set_index("LIMS_ID").loc[counts.columns]
        anno = pd.read_csv(os.path.join(tmp, "anno.csv")).set_index("gene_id")
    sym = anno["gene_symbol"].reindex(counts.index)
    ok = sym.notna() & (sym.astype(str).str.len() > 0)
    return counts[ok].groupby(sym[ok].astype(str).values).sum()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = tmp.replace("\\", "/")
        script = os.path.join(tmp, "export.R")
        open(script, "w").write(R % dict(src=SRC.replace("\\", "/"), tmp=tmp, groups=",".join('"%s"' % g for g in GROUPS)))
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([os.path.join(RENV, "Lib", "R", "bin", "x64"), RENV,
                                       os.path.join(RENV, "Library", "mingw-w64", "bin"),
                                       os.path.join(RENV, "Library", "bin"), env["PATH"]])
        subprocess.run([os.path.join(RENV, "Lib", "R", "bin", "x64", "Rscript.exe"), script], check=True, env=env)
        counts = pd.read_csv(os.path.join(tmp, "counts.csv.gz"), index_col=0)
        obs = pd.read_csv(os.path.join(tmp, "obs.csv")).set_index("LIMS_ID").loc[counts.columns]
        anno = pd.read_csv(os.path.join(tmp, "anno.csv")).set_index("gene_id")

    sym = anno["gene_symbol"].reindex(counts.index)
    ok = sym.notna() & (sym.astype(str).str.len() > 0)
    counts, sym = counts[ok], sym[ok].astype(str)
    summed = counts.groupby(sym.values).sum()                       # genes (symbols) x samples
    ids = pd.Series(counts.index, index=sym.values).groupby(level=0).agg(";".join)
    bio = anno["class_new"].reindex(counts.index)
    bio.index = sym.values
    bio = bio.groupby(level=0).first()

    o = pd.DataFrame({"specimen": counts.columns, "donor_id": obs["ID_code"].astype(str).values,
                      "label": obs["trial_number"].astype(str).values,
                      "timepoint": obs["Timepoint"].astype(str).values,
                      "seq_batch": obs["seq_Batch"].astype(str).values, "run": obs["Run"].astype(str).values,
                      "rin": pd.to_numeric(obs["RNA_RIN"], errors="coerce").values}, index=counts.columns)
    v = pd.DataFrame({"gene_ids": ids.reindex(summed.index).values, "biotype": bio.reindex(summed.index).values},
                     index=summed.index)
    a = ad.AnnData(sp.csr_matrix(summed.T.values.astype(np.float32)), obs=o, var=v)
    a.obs_names, a.var_names = list(map(str, o.index)), list(map(str, v.index))
    a.obs["condition"] = np.where(a.obs.label.isin(HEALTHY), "Healthy", "PTB")
    lab = a.obs.label
    # planned comparisons: 1 = case, 0 = reference, NaN = sample not in the comparison
    cmp = {"cmp1_PTB_vs_Healthy": (lab.isin(["DS_PTB", "MDR_PTB"]), lab.isin(HEALTHY)),
           "cmp2_MDR_vs_DS": (lab == "MDR_PTB", lab == "DS_PTB")}
    for k, (case, ref) in cmp.items():
        a.obs[k] = np.where(case, 1.0, np.where(ref, 0.0, np.nan))
    a.obs["batch"] = a.obs.run.str.replace("Run", "R") + "_b" + a.obs.seq_batch.astype(str)
    a.obs["donor_repeated"] = a.obs.donor_id.duplicated(keep=False)
    a.obs["lib_size_M"] = np.round(np.asarray(a.X.sum(1)).ravel() / 1e6, 2)
    a.write_h5ad(OUT, compression="gzip")
    a.obs.to_csv(META, index=False)
    print("wrote", META)
    for k in cmp:
        sub = a.obs[a.obs[k].notna()]
        print("\n==", k, "(case 1 / reference 0)")
        print(pd.crosstab(sub[k].astype(int), sub.batch, margins=True).to_string())
        print(sub.groupby(k)[["rin", "lib_size_M"]].median().round(2).to_string())
    print("wrote", OUT, a.shape)
    print(a.obs.label.value_counts().to_string())
    print(pd.crosstab(a.obs.label, a.obs.seq_batch).to_string())
    lib = np.asarray(a.X.sum(1)).ravel() / 1e6
    print("library size (M): median %.1f, range %.1f-%.1f" % (np.median(lib), lib.min(), lib.max()))
    print("RIN: median %.1f, missing %d" % (np.nanmedian(a.obs.rin), a.obs.rin.isna().sum()))


if __name__ == "__main__":
    main()
