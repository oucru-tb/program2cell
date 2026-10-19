# Cell-type-specific differential expression with TOAST or TCA, on files written by
# palm_bulk.compare.methods.toast / .tca.
# Usage: Rscript run_csde.R WORKDIR toast|tca
# In WORKDIR: bulk_cpm.csv (genes x samples), fractions.csv (samples x cell types), design.csv (sample, T).
# Writes:     logfc.csv (genes x cell types): natural-log fold change, treated vs control, per cell type.
args <- commandArgs(trailingOnly = TRUE)
wd <- args[1]; method <- args[2]

cpm <- as.matrix(read.csv(file.path(wd, "bulk_cpm.csv"), row.names = 1, check.names = FALSE))
W <- as.matrix(read.csv(file.path(wd, "fractions.csv"), row.names = 1, check.names = FALSE))
design <- read.csv(file.path(wd, "design.csv"), row.names = 1, check.names = FALSE)
W <- W[colnames(cpm), , drop = FALSE]
Tr <- design[colnames(cpm), "T"]
types <- colnames(W)

if (method == "toast") {
  suppressPackageStartupMessages(library(TOAST))
  d <- makeDesign(data.frame(T = factor(Tr, levels = c(0, 1))), W)
  fm <- fitModel(d, cpm)
  res <- csTest(fm, coef = "T", cell_type = NULL, verbose = FALSE)
  out <- sapply(types, function(ct) {
    r <- res[[ct]][rownames(cpm), ]
    log1p(pmax(r$effect_size, -0.99))       # effect_size = beta / mu: relative change in that cell type
  })
} else if (method == "tca") {
  suppressPackageStartupMessages(library(TCA))
  X <- log2(1 + cpm)
  C1 <- matrix(Tr, ncol = 1, dimnames = list(colnames(cpm), "T"))
  mdl <- tca(X = X, W = W, C1 = C1, verbose = FALSE, parallel = FALSE)
  g <- mdl$gammas_hat                        # genes x (cell type . covariate)
  out <- sapply(types, function(ct) g[, paste0(ct, ".T")] * log(2))
} else {
  stop("method must be toast or tca")
}
rownames(out) <- rownames(cpm)
write.csv(out, file.path(wd, "logfc.csv"))
