# BayesPrism on files written by palm_bulk.compare.methods.bayesprism.
# Usage: Rscript run_bayesprism.R WORKDIR N_CORES SPECIES CLEANUP(0/1)
# In WORKDIR: sc_counts.mtx (genes x cells), sc_genes.txt, sc_labels.txt (one label per cell),
#             bulk_counts.csv (genes x samples, first column gene).
# Writes:     theta.csv (samples x cell types), Z_<celltype>.csv (samples x genes), genes_used.txt.
suppressPackageStartupMessages({
  library(Matrix)
  library(BayesPrism)
})
args <- commandArgs(trailingOnly = TRUE)
wd <- args[1]; n_cores <- as.integer(args[2]); species <- args[3]; cleanup <- args[4] == "1"

sc <- readMM(file.path(wd, "sc_counts.mtx"))
genes <- readLines(file.path(wd, "sc_genes.txt"))
labels <- readLines(file.path(wd, "sc_labels.txt"))
sc <- t(as.matrix(sc))                       # cells x genes (BayesPrism wants dense)
colnames(sc) <- genes
rownames(sc) <- paste0("cell", seq_len(nrow(sc)))
bulk <- read.csv(file.path(wd, "bulk_counts.csv"), row.names = 1, check.names = FALSE)
bulk <- t(as.matrix(bulk))                   # samples x genes

if (cleanup) {
  sc <- cleanup.genes(input = sc, input.type = "count.matrix", species = species,
                      gene.group = c("Rb", "Mrp", "other_Rb", "chrM", "MALAT1", "chrX", "chrY"),
                      exp.cells = 5)
}
common <- intersect(colnames(sc), colnames(bulk))
sc <- sc[, common]; bulk <- bulk[, common, drop = FALSE]

prism <- new.prism(reference = sc, mixture = bulk, input.type = "count.matrix",
                   cell.type.labels = labels, cell.state.labels = labels, key = NULL,
                   outlier.cut = 0.01, outlier.fraction = 0.1)
bp <- run.prism(prism = prism, n.cores = n_cores)

theta <- get.fraction(bp = bp, which.theta = "final", state.or.type = "type")
write.csv(theta, file.path(wd, "theta.csv"))
for (ct in colnames(theta)) {
  z <- get.exp(bp = bp, state.or.type = "type", cell.name = ct)   # samples x genes
  write.csv(z, file.path(wd, paste0("Z_", ct, ".csv")))
}
writeLines(common, file.path(wd, "genes_used.txt"))
