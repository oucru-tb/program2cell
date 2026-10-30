# InstaPrism (fast BayesPrism, fixed-point instead of Gibbs) on files written by
# palm_bulk.compare.methods.instaprism.
# Usage: Rscript run_instaprism.R WORKDIR N_CORES
# In WORKDIR: sc_counts.mtx (genes x cells), sc_genes.txt, sc_labels.txt (one label per cell),
#             bulk_counts.csv (genes x samples, first column gene).
# Writes:     theta.csv (samples x cell types), Z_<celltype>.csv (samples x genes), genes_used.txt.
suppressPackageStartupMessages({
  library(Matrix)
  library(InstaPrism)
})
args <- commandArgs(trailingOnly = TRUE)
wd <- args[1]; n_cores <- as.integer(args[2])

sc <- readMM(file.path(wd, "sc_counts.mtx"))
genes <- readLines(file.path(wd, "sc_genes.txt"))
labels <- readLines(file.path(wd, "sc_labels.txt"))
sc <- as.matrix(sc)                          # genes x cells (refPrepare wants genes in rows)
rownames(sc) <- genes
colnames(sc) <- paste0("cell", seq_len(ncol(sc)))
bulk <- as.matrix(read.csv(file.path(wd, "bulk_counts.csv"), row.names = 1, check.names = FALSE))

common <- intersect(rownames(sc), rownames(bulk))
sc <- sc[common, , drop = FALSE]; bulk <- bulk[common, , drop = FALSE]

ref <- refPrepare(sc_Expr = sc, cell.type.labels = labels, cell.state.labels = labels)
res <- InstaPrism(bulk_Expr = bulk, refPhi_cs = ref, n.core = n_cores)

theta <- t(res@Post.ini.ct@theta)            # samples x cell types
write.csv(theta, file.path(wd, "theta.csv"))
Z <- get_Z_array(res, resolution = "ct", n.core = n_cores)   # samples x genes x cell types
for (ct in dimnames(Z)[[3]]) write.csv(Z[, , ct], file.path(wd, paste0("Z_", ct, ".csv")))
writeLines(dimnames(Z)[[2]], file.path(wd, "genes_used.txt"))
