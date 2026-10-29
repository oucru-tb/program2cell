# CIBERSORT (Newman et al. 2015; R script v1.02, CIBERSORT.R next to this file) on files written by
# palm_bulk.compare.methods.cibersort.
# Usage: Rscript run_cibersort.R WORKDIR CIBERSORT_R QN(0/1)
# In WORKDIR: signature.txt (genes x cell types, tab separated, first column gene),
#             mixture.txt (genes x samples, same format; CPM).
# Writes:     fractions.csv (samples x cell types) and fit.csv (samples x Correlation, RMSE).
args <- commandArgs(trailingOnly = TRUE)
wd <- args[1]; src <- args[2]; qn <- args[3] == "1"
suppressPackageStartupMessages(source(src))
setwd(wd)                                                 # CIBERSORT writes CIBERSORT-Results.txt here
res <- CIBERSORT("signature.txt", "mixture.txt", perm = 0, QN = qn)
k <- ncol(res) - 3
write.csv(res[, seq_len(k), drop = FALSE], "fractions.csv")
write.csv(res[, c("Correlation", "RMSE"), drop = FALSE], "fit.csv")
