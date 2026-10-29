# Experimental design: PALM-bulk vs cellGeometry (Lau et al., Nat Commun 2026)

Goal: a benchmark that matches cellGeometry's own design point for point, so a reviewer cannot say we
avoided their ground, and extends it where PALM-bulk does more (calibrated uncertainty, program
activity, treatment effects).

## 1. What cellGeometry is

- **Method:** non-negative geometric deconvolution (NGD).
  - Signature genes are chosen by vector angle (cell-type specificity) on log2-mean expression.
  - Genes are scaled to the unit sphere, so each gene has equal weight.
  - Abundance is a projection A = YᵀGΨ, corrected by a "compensation" matrix S⁻¹. The compensation is
    shrunk per cell type, with a weight w in [0, 1] chosen as the largest that keeps the type's counts
    ≥ 0.
  - Equivalent to a generalised Tikhonov regression whose penalty is set heuristically, not by
    cross-validation.
- **Uncertainty:** a sandwich-type SE ("var.e") per cell type. It uses each gene's residual variance
  pooled across all bulk samples.
  - Validated only by correlation with the actual error (ρ 0.90–0.95, CCC 0.67–0.83).
  - **Not validated by interval coverage.**
- **Diagnostics:** Studentized residuals and Cook's distance per gene, plus multi-pass removal of
  outlier genes (|Z| > 4 on the cross-sample residual variance).
- **Reference merging:** signatures from different atlases are merged by quantile–quantile mapping.
- **Output:** cell counts or fractions only. No per-sample expression states, no program activity, no
  treatment effects, no posterior.

## 2. Their evaluation design

| Axis | Their design |
|---|---|
| References | AMP RA synovium (18 subclasses, 10k cells); CellTypist blood (33, 28k); Tabula Sapiens (180, 1.1M); Human Brain Cell Atlas (116, 3.4M) |
| Simulated bulk | Random cell counts per subclass (up to 3–5 × the mean), cells drawn from **the same reference dataset** with Dirichlet resampling; N = 25–50 per replicate, 3–5 replicates |
| Metrics | RMSE, R², Lin's CCC on cell-subclass percentages |
| Speed | N = 100 / 500 / 5000 (RA); signature build vs deconvolution only; 1 vs 8 cores |
| Noise | Gaussian, log, square-root, "shift" (50% of genes × 2^N(0, sd), the same factor in every sample = chemistry difference) |
| Missing types | 0–50% of subclasses set to zero per sample |
| SE quality | Correlation of SE with the actual error, for OLS / OLS.v2 / HC0–3 / var.e |
| Real data | PEAC whole-blood RNA-seq (n = 67) vs Coulter neutrophil and lymphocyte %; R4RA (183), PEAC (160) and STRAP (273) synovium vs semi-quantitative histology (CD3, CD20, CD68, CD138); pathotype reclassification |
| Comparators | MuSiC, DWLS, LinDeconSeq; also Bisque, InstaPrism, NNLS, glmnet, NMF |

## 3. Where their design is weak, and how we exploit it

1. **The simulated bulk comes from the same cells as the signature.** The authors note that with
   arithmetic means "the gene signature is already perfect". This rewards projection methods and hides
   reference mismatch. We run their protocol *and* a stricter cross-donor split, where reference and
   test donors are disjoint (our `make_benchmark`). We report the drop between the two for every
   method.
2. **SE is judged by correlation, not coverage.** An SE can be well ranked and still badly scaled.
   We report 90% interval coverage per cell type and per sample, plus their ρ and CCC for parity.
3. **Fractions only.** Their own real-blood analysis flags interferon genes (IFIT2, IFIT3, OAS2) as
   residual "outliers" and removes them. In PALM-bulk that signal is *modelled*: it is the JAK-STAT
   program activity δ, per cell type. Their residuals are our estimand. This is the clearest
   difference in scope (E7, E8).
4. **Missing cell types are only tested as zero counts.** Types absent from the *reference* (their
   neutrophil case) are handled by hand-merging signatures. We also test removing a type from the
   reference while keeping it in the bulk.
5. **The compensation weight w is heuristic and gives no uncertainty for shrinking.** Our
   priors/likelihood regularise in a principled way, and the posterior carries the resulting
   uncertainty.

## 4. Where we will probably *not* win (state this up front)

- **Raw speed for fractions at large N.** cellGeometry deconvolves 5,000 samples in 42 s.
  - palm_fast takes minutes per ~150 samples.
  - Our fair fast competitor is the **stage-1 mode alone** (per-sample Fisher scoring, seconds).
  - We present speed as a two-tier result: stage-1 point fractions (compete on speed), then
    palm_fast posterior (adds calibration and activity at minutes).
- **Very fine resolution (180 subclasses).** PALM-bulk is tested at 4–5 broad types.
  - With C = 180 and P = 7, the per-sample latent dimension is 1,440, which is too large for the
    current dense summaries.
  - Plan: fractions at subclass level, and program activity at lineage level (the `groups` option
    already exists). That reduces the activity dimension to (number of lineages) × P.

## 5. Experiments

Common settings for all experiments:

- Methods:
  - cellGeometry (CRAN; installed in the `palm-r` env);
  - PALM stage-1 mode;
  - palm_fast (common activity, quasi + sandwich, NUTS 500 + 500 at target accept 0.95);
  - full PALM on small cases only;
  - NNLS, MuSiC, DWLS, BayesPrism/InstaPrism, reallocation, naive.
- Seeds: 5 replicates per setting.
- **Truth split:** the truth always uses `truth_split=True` for activity, so the bulk and the δ truth
  come from independent halves of the counts.

### E1. Fraction accuracy at their scale

- References: CellTypist blood (33) and AMP RA synovium (18) first; Tabula Sapiens (180) after the
  scaling work in §6.
- Two protocols:
  - (a) theirs, bulk drawn from the reference cells;
  - (b) cross-donor, with disjoint reference and test donors.
- Metrics: RMSE, R², CCC (theirs), plus read- vs cell-fraction consistency (we model read fractions,
  so convert using the reference RNA content).
- **Success:** at least ties cellGeometry under (a), and wins under (b).

### E2. Speed

- N = 100, 500, 5,000 on CellTypist.
- Time signature building and deconvolution separately.
- Run 1 core and 8 cores.
- Report: stage-1 mode, palm_fast end to end, cellGeometry.
- **Success:** stage 1 within 10× of cellGeometry at N = 5,000; palm_fast ≤ 30 min at N = 5,000.

### E3. Noise and platform robustness

- Their four noise types at their SD grids.
- Shift noise is a per-gene platform effect, the case the gene scale m_g was built for. Run
  PALM-bulk with and without `use_m`, and with `sigma_m` fixed vs auto.
- **Success:** RMSE rises no faster than cellGeometry's under log and shift noise, and m_g reduces
  the shift-noise penalty. This would also settle the open m_g question for real bulk.

### E4. Missing cell types

- (a) Theirs: 0, 5, 10, 20, 50% of subclasses set to zero in the bulk.
- (b) Ours: one or two types dropped from the **reference** but kept in the bulk (e.g. remove
  monocytes).
  - Add an "unexplained" component, or flag samples by stage-1 φ (the per-sample overdispersion
    diagnostic) and by outlier genes.
- **Success:** no false positives above 1% for absent types in (a); in (b), the affected samples are
  flagged.

### E5. Uncertainty quality (the headline differentiator)

- For each method with uncertainty (cellGeometry var.e as ±1.645 SE; palm_fast; full PALM;
  BayesPrism if available):
  - 90% coverage of fractions, per cell type and overall;
  - interval width;
  - their metric (ρ and CCC between SE and actual error).
- Under protocol (b), with and without reference-uncertainty propagation (P3 in our plan: multiple
  imputation over reference donors).
- **Success:** palm_fast coverage within 0.85–0.95; cellGeometry SE intervals shown to under-cover
  under (b).

### E6. Program activity and effects (outside their scope)

- Simulated program shifts (`palm_bulk.sim`) and real pseudobulk contrasts (COVID PBMC; OneK1K
  sex/age).
- Metrics:
  - activity ρ across samples and across cell types;
  - effect r, MAE, coverage;
  - the intrinsic-vs-composition split.
- Baselines: naive, reallocation, BayesPrism+ULM, TOAST, TCA. Also "cellGeometry + ULM": its
  fractions, then a per-type expression proxy, then programs. This gives cellGeometry a fair route
  to activity.

### E7. Real bulk validation (the gap PALM-bulk has today)

- PEAC whole blood (E-MTAB-6141, n = 67 with Coulter counts):
  - reference CellTypist, merged with Tabula Sapiens neutrophils and basophils, as they did;
  - correlation of neutrophil and lymphocyte % with Coulter;
  - **and** palm_fast's intervals: does the Coulter value fall inside them?
- R4RA, PEAC and STRAP synovium (E-MTAB-11611, E-MTAB-13733) with the AMP RA reference:
  - Spearman correlation with CD3, CD20, CD68, CD138 histology scores;
  - pathotype separation.
- **Success:** correlations ≥ cellGeometry's (blood neutrophil r 0.62, lymphocyte r 0.53; synovium
  r 0.50–0.70).

### E8. "Their residuals are our signal"

- On PEAC blood, run cellGeometry multi-pass and list the genes it removes.
- Show that PALM-bulk's JAK-STAT (interferon) activity per cell type:
  - explains those genes' residuals;
  - varies across patients;
  - associates with clinical variables where available (e.g. disease activity or treatment response
    in R4RA).
- **Success:** a per-sample interferon activity with intervals, which cellGeometry discards as noise.

## 6. Engineering needed before E1 (Tabula Sapiens), E2 and E7

1. **Large-reference signature building:** stream means from backed h5ad (chunked sparse sums), with
   an optional log-mean for gene selection, as they do.
2. **Fine resolution:** subclass fractions plus lineage-level activity (`groups`). Check that stage-1
   memory and time scale as O(N · D²) with D = C + L·P.
3. **Gene selection at C ≫ 5:** compare Fisher D-optimal with their angle method. The angle method
   is cheap and can pre-filter before Fisher.
4. **Outlier-gene pass:** remove genes with |Z| > 4 on the cross-sample residual variance of stage-1
   Pearson residuals (one cheap refit). This matches their multi-pass step.
5. **Runner for the R comparator:** `compare/methods.py::cellgeometry` via `PALM_RSCRIPT` (palm-r
   env).
6. **Dataset downloads:** CellTypist, AMP phase 1 (ImmPort SDY998), Tabula Sapiens blood subset, and
   PEAC/R4RA/STRAP from ArrayExpress.

## 7. When: after method development (decided 2026-10-26)

The full real-data benchmark (E7, E8, the competitor runs on real bulk, Tabula Sapiens) runs only
after the method is frozen, so expensive runs are not repeated each time the model changes. Method
development uses simulated and pseudobulk data (COVID PBMC, OneK1K), in this order:

1. Truth split (bulk and δ truth from independent halves): is the activity gap real or a
   measurement artefact?
2. Reference-uncertainty propagation (multiple imputation over reference donors), then calibration
   of the learning rate ω: fraction and effect coverage.
3. Prior comparison for β (horseshoe vs hierarchical Normal vs Student-t).
4. E3 shift noise and E4 missing types on pseudobulk: settle m_g and the outlier-gene pass. These
   count as development, not benchmarking.
5. One sanity check on real bulk before freezing: ~20 PEAC blood samples vs Coulter, PALM-bulk only
   (checks m_g and the outlier pass under a real platform shift).
6. Freeze the method, then run E1–E8 below.

### 7.1 Further development ideas (beyond cellGeometry), by weak point

| Priority | Idea | Weak point targeted | Cost |
|---|---|---|---|
| 1 | Design-based effect: β_design = mean δ̂(treated) − mean δ̂(control) from the per-sample posteriors, with uncertainty from the draws plus the between-sample variance (reported next to the shrunk model β) | Effect instability (r 0.02–0.71) and coverage; does not depend on the horseshoe | Hours |
| 2 | Pooled gene-block noise: a sandwich V shared across samples, scaled per sample by φ_i (a better version of cellGeometry's var.e) | Coverage, stability of Λ_i | Hours |
| 3 | m_g as a function of gene covariates (log m_g = f(length, GC, 3′-UTR length), ~5 coefficients), tested in E3 with covariate-driven shift noise | Platform shift; the regression from free m_g | ~1 day |
| 4 | Fit stage 2 on a subsample (~300), then draw latents for all samples exactly | Speed at N = 5,000 | ~½ day |
| 5 | Program weights per cell type, G_c, learned from a cytokine-stimulated scRNA atlas (e.g. Parse PBMC cytokines) | Separating activity between cell types (ρ 0.05–0.25) | Days (data pipeline) |
| 6 | Resolution chosen by the data: merge types whose δ cannot be told apart into lineage groups | Over-claiming per-type activity | Small |
| 7 | Correlation between programs (LKJ on the shared factor, or orthogonalised weights) | Confounding between NF-κB, TNFα and MAPK | Small |
| 8 | An "unknown" component (one learned profile) and a robust NB with gene × sample outlier indicators | Missing cell types, gene shifts outside the programs | ~1 day |
| 9 | Sorted-cell ground truth (Monaco GSE107011; FlowSorted blood) as the pre-freeze real-bulk check | Real-data sanity | Small |

## 8. Order and budget (after the freeze)

| Step | Content | Compute (this machine) |
|---|---|---|
| 1 | Runner + CellTypist/AMP references + E1 (a, b) at 18–33 types | ~1 day |
| 2 | E3 noise + E4 missing types + E5 coverage | ~1 day |
| 3 | E7 PEAC blood + R4RA synovium (real data) | ~1 day after download |
| 4 | E2 speed at N = 5,000; E6 activity; E8 interferon story | ~1 day |
| 5 | Tabula Sapiens 180 types (after §6.2) | stretch |

## 9. What we claim, if the results hold

PALM-bulk matches the best fast deconvolution on fractions. It is more honest under reference
mismatch, because its intervals cover. And it recovers what fraction-only methods throw away:
per-cell-type program activity and treatment effects, with calibrated uncertainty.
