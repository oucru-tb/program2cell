# cellGeometry (Lau et al., Nat Commun 2026) on files written by palm_bulk.compare.methods.cellgeometry.
# Usage: Rscript run_cellgeometry.R WORKDIR ARITH_MEAN(0/1) USE_FILTER(0/1) NPASS NSUBCLASS
# In WORKDIR: sc_counts.mtx (genes x cells), sc_genes.txt, sc_labels.txt (one label per cell),
#             bulk_counts.csv (genes x samples, first column gene).
# Writes:     percent.csv and output.csv (samples x cell types: cell percentages and cell amounts),
#             signature.csv (genes x cell types: the signature used, per-cell mean counts),
#             se.csv (samples x cell types, var.e standard errors of the amounts) when available.
suppressPackageStartupMessages({
  library(Matrix)
  library(cellGeometry)
})
args <- commandArgs(trailingOnly = TRUE)
wd <- args[1]; arith <- args[2] == "1"; use_filter <- args[3] == "1"
npass <- as.integer(args[4]); nsub <- as.integer(args[5])

sc <- as(readMM(file.path(wd, "sc_counts.mtx")), "CsparseMatrix")       # genes x cells
rownames(sc) <- readLines(file.path(wd, "sc_genes.txt"))
colnames(sc) <- paste0("cell", seq_len(ncol(sc)))
labels <- readLines(file.path(wd, "sc_labels.txt"))
bulk <- as.matrix(read.csv(file.path(wd, "bulk_counts.csv"), row.names = 1, check.names = FALSE))

mk <- cellMarkers(sc, bulkdata = bulk, subclass = labels, cellgroup = NULL, nsubclass = nsub,
                  dual_mean = arith, verbose = FALSE)
fit <- deconvolute(mk, bulk, use_filter = use_filter, arith_mean = arith, npass = npass, verbose = FALSE)

write.csv(fit$subclass$percent, file.path(wd, "percent.csv"))
write.csv(fit$subclass$output, file.path(wd, "output.csv"))
write.csv(fit$subclass$X, file.path(wd, "signature.csv"))
s <- tryCatch(se(fit), error = function(e) NULL)
if (!is.null(s)) write.csv(s, file.path(wd, "se.csv"))
