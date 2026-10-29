# How PALM-bulk is benchmarked

A living document: what we compare, against what truth, with which metrics, and why. Update the
**Changelog** and **Current results** sections whenever the protocol or the numbers change.
Last updated: 2026-10-27.

---

## 1. What the benchmark has to show

PALM-bulk claims three things from bulk RNA-seq plus a single-cell reference:

1. **Composition**: read fractions π per sample and cell type, with honest intervals.
2. **Gene program deconvolution**: activity δ of known programs (PROGENy pathways) *per sample and per
   cell type*, not one bulk score per sample.
3. **Cell-type-specific effects**: the treatment shift β of each program in each cell type, with
   calibrated uncertainty.

Claim 2 is the one no standard deconvolution method makes, and the one a "bulk score copied to every
cell type" baseline can fake on some metrics. The metrics below are chosen so that faking it is
visible.

---

## 2. Data, splits and truth

### Datasets (development)

| Name | File | Contrast (β truth) | Test samples |
|---|---|---|---|
| `pbmc` | `pbmc_broad_clean.h5ad` | disease: COVID-19 vs healthy | all eligible test donors (about 38) |
| `o1k` | OneK1K, `onek1k_full_broad.h5ad` | sex: male vs female | 150 |

- **Cell types:** B cell, Monocyte, NK cell, T cell.
- **Programs:** 7 PROGENy programs: JAK-STAT, NFkB, TNFa, TGFb, Hypoxia, p53, MAPK.
- **Genes:** 1000 genes chosen by Fisher selection (`prepare_inputs(max_genes=1000)`).

### Splits

`make_benchmark(seed=split)` splits *donors*:

- **Test donors** give one specimen each, and that specimen becomes a pseudo-bulk sample (the sum of
  its cell types).
- **Reference donors** provide the reference profiles B. For competitors, they also provide the
  single-cell reference.

A donor is never in both. Each test specimen needs at least 20 cells of every type.

- **Splits 0–9: development.** These are used freely while we change the method.
- **Splits 10–19: held-out.** The script refuses them without `--heldout`. Run them **once**, at the
  method freeze, so the final numbers are not tuned to.

### Truth split (binomial thinning)

Each test specimen's per-cell-type counts are split into two independent halves:

- **Bulk half:** summed over cell types, this is what methods see.
- **Truth half:** used only to compute the truth.

Without this, a method is scored against the same counts it was given, and the truth's sampling noise
correlates with the method's input. In practice the effect was negligible, but the split is now
always on.

### Truth definitions

| Quantity | Definition |
|---|---|
| π (N×C) | Share of each cell type's reads on the modelled genes (read fractions, not cell fractions). |
| δ truth (N×C×P) | `estimate_delta` applied to each cell type's truth-half counts: the maximum-likelihood program shift of that cell type's own profile relative to the reference profile B_c. This is the model's estimand. |
| ULM truth (N×C×P) | decoupler ULM scores on the same truth-half counts. This is a model-free check: a method cannot win just by sharing the δ truth's estimator. It is a different scale, so only rank metrics are used against it. |
| β truth (C×P) | Mean δ truth of treated minus mean of controls (a design-based difference). |

**Why two truths.** The δ truth uses the same multinomial estimator as reallocation and as the model's
own definition of activity. A method built on that estimator has a home advantage. ULM weights genes
differently. When the two truths disagree (section 7), the disagreement tells us which gene weighting a
method favours, and the paper must state which estimand it targets.

---

## 3. Methods and rows

### Baselines

| Name | Fractions | Activity |
|---|---|---|
| `naive` | NNLS | One bulk program shift per sample, copied to every cell type. |
| `reallocation` | NNLS | Bulk counts split by π·B (BayesPrism's expectation step with a fixed reference), then `estimate_delta` per type. |
| `oracle realloc` | true π | Reallocation with perfect fractions: the ceiling for any reallocation estimator. |

### Published competitors (`--competitors`)

Each runs on the same split, genes and truth as the rows above.

| Name | What it gives | Activity used |
|---|---|---|
| `cellgeometry` | Cell amounts, converted to read fractions with each type's mean library size in the reference. | Reallocation with its fractions. |
| `bayesprism` | θ fractions, and per-type expression Z from Gibbs sampling. | `model` row: δ from its own Z. `realloc` row: its fractions with plain reallocation. |

- **Reference:** by default, the reference donors' pseudobulk rows (the same information PALM's B uses).
  With `--cells`, the reference donors' real cells are used instead (OneK1K only). The row name then
  gets a " cells" suffix.
- **Planned after the method freeze:** BLADE, CIBERSORTx (needs Docker and a token), TOAST and TCA
  (effects only), and BayesPrism embedding learning with PROGENy as `eta_prior` (section 6).

### PALM rows (per configuration)

| Row | Activity / effects |
|---|---|
| `model` | Posterior median δ; the model's β with its intervals. |
| `design beta` | Posterior δ; β = treated − control mean of δ per draw. |
| `stage-1 mode` | Each sample's own fit, with no pooling (a diagnostic). |
| `posterior realloc` | Bulk split by posterior π × B, then `estimate_delta` per type. Intervals come from π draws. |
| `realloc + alloc` | The same, with a multinomial split per draw (adds allocation noise). |
| `EM realloc`, `iter realloc` | The split uses program-shifted profiles. Both were worse; kept for the record. |

The main configurations are:

- `2step quasi`: NB fractions, then quasi δ given those fractions.
- `joint quasi+sw`: quasi likelihood with a sandwich covariance.
- `joint nb`
- `2step quasi+sw`
- `2step quasi indep`

---

## 4. Metrics, one by one

All metrics are averaged over splits. Comparisons are **paired by split**: the mean difference, with a
bootstrap 95% CI over splits (`summary --ref`). Within a split, cell types and programs share the same
donors, so the split is the unit of replication.

### 4.1 Fractions

| Metric | Definition | Reading |
|---|---|---|
| `frac_rmse` | √mean over (i,c) of (π̂ − π)² | Overall accuracy, including systematic bias between types. |
| `frac_pearson` | Pearson over all (i,c) pooled | Dominated by between-type differences; easy to make high. |
| `frac_r_per_type` | Pearson across samples within each type, averaged | Whether a method tracks variation between samples. This is what downstream association studies need. A method can have a poor RMSE (biased scale) but a good r. |
| `frac_cov90` | Share of the true π inside the 90% interval | Calibration. The target is 0.90; below that means overconfident. |

### 4.2 Program activity (the core of gene program deconvolution)

Let δ̂ be N×C×P estimates and δ the truth.

| Metric | Definition | What it shows |
|---|---|---|
| `act_spearman_across_samples` (**ρ**) | Spearman across samples for each (c,p), averaged over (c,p) with non-trivial truth variance. | "Within cell type c, which samples have high program p?" This is the paired-cohort question. It is the headline metric, but it **does not require cell-type resolution**: a bulk score copied to every type does well whenever samples differ mostly in a shared way. |
| `act_spearman_shared` | The same, on the mean over cell types (per sample and program). | The part any bulk method can get. |
| `act_spearman_specific` | The same, on deviations from that mean (δ_ic − mean over c of δ_ic). | **The part only true deconvolution can get.** The naive baseline is constant here, so it scores nan. This is the metric that shows program deconvolution. |
| `act_specific_share` | The truth's variance share that is type-specific. | How much the specific metric matters. It is about 0.49 on PBMC: half the signal is type-specific. |
| `act_spearman_across_types` | Spearman across cell types within (sample, program), averaged. | "In this sample, which cell type carries program p?" The naive baseline cannot answer this (nan). |
| `act_mae`, `act_pearson` | Absolute error and Pearson on the δ scale. | Scale accuracy. Only for methods on the δ scale. |
| `act_cov90` | Coverage of 90% intervals. | Calibration of the activity intervals. |
| `ulm_*` | The same rank metrics against the ULM truth. | Robustness to the estimand. |

### 4.3 Effects (cell-type-specific differential program activity)

| Metric | Definition | Reading |
|---|---|---|
| `eff_pearson` | Pearson of β̂ with β over all (c,p). | Whether the effect pattern is recovered. |
| `eff_auroc` | AUROC of \|β̂\| for separating \|β\| > 0.1 from the rest. | "Which (cell type, program) pairs respond?" It is nan when no true effect exceeds 0.1, which is the case for OneK1K's sex contrast. |
| `eff_sign_agree` | Share of true effects (\|β\| > 0.1) with the correct sign. | |
| `eff_mae` | Absolute error on the δ scale. | |
| `eff_cov90` | Coverage of 90% intervals for β. | The inferential claim: intervals we can test with. |

### 4.4 Cost and fit quality

- `time`: wall-clock seconds.
- `rhat`, `div`: convergence checks for NUTS. The target is R-hat below 1.01 and 0 divergences.

---

## 5. Leakage and inference metrics for gene program deconvolution

These are implemented in `compare/scoring.py` as of 2026-10-27, and `dev_bench rescore` adds them to
saved fits.

| Column | Metric |
|---|---|
| `act_gain` | Own-type gain |
| `act_leakage` | Leakage (also `ulm_act_leakage`) |
| `eff_localise` | Localisation accuracy |
| `act_coupling` | Spurious fraction–activity coupling |
| `act_iscore`, `eff_iscore` | Interval score |
| `eff_fsr`, `eff_n_called` | False sign rate, and the number of effects called |
| `eff_fdr` | Share of called effects that are not meaningful (wrong sign, or true \|β\| ≤ 0.1) |
| `act_prog_leakage` | Leakage across programs within a cell type (see 5b) |

`eff_fsr` used to combine wrong signs and too-small effects. It now means the wrong sign only (the
Stephens definition), and `eff_fdr` is the combined rate.

## 5b. Program prior sets, and what Tripso adds (2026-10-27)

**Source.** Moullet, Isobe, Vahidi, Leonardi et al., *Self-supervised learning for a gene
program-centric view of cell states* (Tripso), bioRxiv 10.64898/2026.03.24.713961.

- **Setting:** single-cell, not bulk. Tripso learns one transformer embedding per gene program,
  starting from known programs.
- **Benchmarks:**
  - Perturb-seq of 6 cancer cell lines stimulated with cytokines (Jiang et al., *Nat Cell Biol* 2025),
    scored on TGFβ and TNFα.
  - The human endometrium atlas (WNT, TGFβ).
- **Compared against:** Spectra, expiMap, Scanpy `score_genes`, and log-expression baselines.

### What we take from it

**1. Program priors beyond PROGENy-7.** These are now available through `dev_bench --prior`:

| `--prior` | File | Programs | Construction |
|---|---|---|---|
| `progeny7` (default) | `progeny_human_top500.csv` | 7 | The current set |
| `progeny14` | `progeny_human_top200.csv` | 14 | All PROGENy pathways, top 200 genes |
| `collectri` | `collectri_immune.csv` | 26 | CollecTRI regulons of immune TFs, 25–250 targets, signed |
| `hallmark` | `hallmark_immune.csv` | 15 | MSigDB Hallmark immune and state sets, unsigned (weight 1) |

Details of the construction:

- **PROGENy-14:** Tripso takes the 200 genes with the smallest p-values. Our file has only weights, so
  we take the top 200 by \|weight\|.
- **CollecTRI:** Tripso's rule keeps TFs with 25–250 targets and prunes overlapping regulons. Ours
  prunes at Jaccard > 0.5, which removed none.
  - Of 40 candidate TFs, 26 pass the size rule.
  - The rule drops the hubs STAT1, STAT3, NFKB1, RELA, SPI1, CEBPB, MYC, JUN, FOS, HIF1A and TP53
    (over 250 targets). A variant keeping them is worth testing, since they are central in blood.
- **Hallmark:** IFN-α and IFN-γ overlap with Jaccard 0.33, which `act_prog_leakage` should expose.
- **Downloads** (decoupler 2.2 `op.collectri`, `op.hallmark`): raw files `data/collectri_human.csv`
  and `data/hallmark_human.csv`.

Each prior writes to `benchmarks/dev/<dataset>__<prior>/`, because the truth δ is defined on that
program matrix.

The queue `priors.log` runs baselines, cellGeometry and PALM `joint quasi+sw` on 10 splits per prior,
PBMC first.

**First look, PBMC split 0 (baselines only):**

| Prior | naive ρ | NNLS realloc ρ | oracle realloc ρ | Program leakage (NNLS / oracle) |
|---|---|---|---|---|
| PROGENy-14 | 0.75 | 0.81 | 0.84 | 0.23 / 0.18 |
| CollecTRI | 0.59 | 0.73 | 0.81 | 0.32 / 0.23 |
| Hallmark | 0.66 | 0.67 | 0.84 | 0.35 / 0.15 |

- With TF regulons and Hallmark sets, the naive baseline falls far behind.
- The value of accurate fractions also grows: the gap from NNLS to the oracle is 0.08 to 0.16, against
  0.06 for PROGENy-7.
- These richer priors are therefore the more discriminating test of program deconvolution.

**2. Cross-program leakage** (`act_prog_leakage`). This mirrors cross-type leakage.

- **Definition:** regress δ̂_c,p on the true δ_c,· of all programs (same type, across samples). The
  metric is R²(all) − R²(own).
- **Requirement:** at least P + 5 samples; for CollecTRI (P = 26) that means OneK1K-sized cohorts,
  otherwise nan.

**3. Label-based truth: a stimulation benchmark** (proposed; not built).

Tripso scores programs against *known stimulation labels* instead of an estimated truth. That removes
the estimator-dependence problem we see between the δ and ULM truths. The bulk version:

- **Data:** PBMC with an in-vitro stimulus, for example Kang et al. 2018 (8 donors, IFN-β vs control;
  available through pertpy, not decoupler).
  - Or the Jiang 2025 Perturb-seq, using cell lines as "cell types" (cytokines: IFNβ, IFNγ, TNFα,
    TGFβ, insulin).
- **Mixtures:** per pseudo-sample, each cell type is drawn independently from stimulated or control
  cells of a test donor. Each (sample, cell type) then has a known 0/1 label.
  - For a graded truth, mix a share α ∈ {0, 0.1, …, 1} of stimulated cells within a type.
  - The reference comes from control cells of other donors.
- **Metrics:**
  - `stim_auroc`: AUROC of δ̂_{i,c,p*} for stimulated vs control (i,c), where p* is the matched program
    (PROGENy JAK-STAT; Hallmark IFN-α; CollecTRI STAT2, IRF7 and IRF1). Computed within each type and
    pooled.
  - `stim_specificity`: AUROC of p* minus the largest AUROC among unmatched programs, plus the full
    programs × stimuli AUROC matrix (Tripso's "TGFβ from the TGFβ embedding only").
  - `stim_offtype`: AUROC of δ̂_{i,c′,p*} in *unstimulated* types c′ when another type in the same sample
    is stimulated. The target is 0.5; above that is cross-type false activation, the failure
    naive-style methods cannot avoid.
  - `dose_spearman`: Spearman of δ̂ with α. A monotone, estimator-free check of graded activity.
  - Difficulty curve: the metrics against effect strength, like Tripso's knockout panels ranked by
    Mixscale. Here, against α and against the stimulated type's read fraction.
  - Linear probe (Tripso's protocol): a cross-validated logistic F1 or AUROC predicting the label from
    all programs of a type.

**4. Not adopted:**
- GP embeddings: we estimate scalar shifts by design.
- scIB batch metrics.
- The condition dominance and specificity summaries: descriptive, with no truth.
- The Jaccard novelty and reconstruction of discovered programs: we do not discover programs.

## 5c. Calibrated reallocation (in progress, 2026-10-27)

**Method.** Reference donors, never test donors, form a calibration set.
- 5-fold cross-fitting: the profiles B come from the other reference donors.
- The same binomial thinning as the test set.
- True π for the split; fraction uncertainty is added at test time from the posterior draws.
- Per (c,p), a ridge regression maps the reallocated δ̂ of all types to the truth.
- Out-of-fold residuals give split-conformal half-widths for activity.
- Effects get bootstrap (over calibration donors) plus residual variance.
- Script: scratchpad `calib.py`; results in `benchmarks/dev/calib_<dataset>_<config>.json`.

**PBMC joint NB, splits 0–2 (86 calibration donors):**

| Row | Activity cov90 | Activity iscore | Effect cov90 | Effect iscore | Effect FSR (calls) |
|---|---|---|---|---|---|
| posterior realloc | 0.17 | 0.61 | 0.21 | 0.33 | 0.25 (24) |
| calibrated, own type | 0.88 | 0.28 | 0.77 | 0.13 | 0.09 (10) |
| calibrated, all types | 0.87 | 0.29 | 0.77 | 0.14 | 0.07 (8) |

- Activity intervals are fixed, and point accuracy holds: ρ 0.83–0.86 against 0.86.
- Effect intervals are much better but still under 0.90.
- Inverse calibration of effects (invert the learned gain and leakage matrix): effect r falls to 0.03.
  Ruled out.

The metrics above say *how well* activity is ranked. They do not say *why* a method fails. The failure
that matters for program deconvolution is **leakage**: one cell type's program shift showing up in
another type's estimate.

The de-mixing diagnostic (2026-10-27) found that reallocation recovers only about 15% of a B or NK
cell's own shift, while the shifts leaking in from the other types add up to about 1 (PBMC split 0).
The proposed metrics make that visible for every method.

| Metric | Definition | Ideal | What it exposes |
|---|---|---|---|
| **Own-type gain** | For each (c,p): slope of δ̂_c on δ_c in the regression of δ̂_c on all types' true δ_c' (across samples). Averaged. | 1 | Attenuation. Reallocation shrinks the activity of minor types towards 0. |
| **Leakage** | Same regression: the share of δ̂_c variance explained by the other types' truth δ_c' (c'≠c), beyond δ_c itself. | 0 | Contamination. The naive baseline has maximal leakage by construction. |
| **Localisation accuracy** | For programs with a true effect: the share of (program, split) where the cell type with the largest \|β̂\| is the one with the largest \|β\|. | 1 | "Did the method put the effect in the right cell type?" This is the question users actually ask. |
| **Spurious fraction–activity coupling** | Corr(δ̂_c, π_c) − Corr(δ_c, π_c) across samples, averaged in absolute value. | 0 | Activity estimates that move with composition. This matters for the BayesPrism-style analysis in section 6. |
| **Interval score** (Gneiting–Raftery) | Width + (2/α) × miss distance, for 90% intervals. | Low | Calibration and sharpness in one number. Coverage alone rewards infinitely wide intervals. |
| **False sign rate** | Among effects whose 90% interval excludes 0: the share with the wrong sign (true \|β\| > 0.1), plus the share where the true \|β\| ≤ 0.1. | ≤ 0.1 | The error rate a user acting on "significant" effects would experience. |

Own-type gain and leakage come from the same per-(c,p) regression, so they are cheap to compute.
Localisation accuracy needs data with real effects, so it uses PBMC and the planned simulation
(experiment E3), not the OneK1K sex contrast.

**Implementation details:**

- **Gain and leakage:** for each (c,p), δ̂_c,p is regressed across samples on the true δ_·,p of all
  C types plus an intercept, over finite samples only.
  - `act_gain` is the coefficient on its own type. It is computed only for methods on the δ scale.
  - `act_leakage` is R²(all types) − R²(own type only), clipped at 0.
  - A (c,p) needs at least C + 3 samples and non-trivial truth variance.
- **Coupling:** Spearman correlation against the *true* π, so the metric measures the estimate's
  dependence on composition, not the fraction method's errors.
- **Localisation:** when the estimates tie across types (as with naive), each tied type gets 1/C of
  the credit. That gives naive exactly chance level.
- **False sign rate:** a call is "wrong" when the true \|β\| ≤ 0.1, or when its sign differs.

**Sanity values, PBMC split 0 (smoke-test fit):**

| | Gain | Leakage | Localisation |
|---|---|---|---|
| Naive | 0.22 | 0.29 | 0.25 (chance with 4 types) |
| Posterior realloc | 0.45 | 0.11 | 1.0 |

For posterior realloc, 19 of its 20 called effects were wrong. Its intervals are far too narrow, which
is what the false sign rate is meant to show.

---

## 6. BayesPrism and gene programs: what it does, and how ours differs

Checked against the installed package (BayesPrism 2.2.3: `learn.embedding`, `learn.embedding.nmf`) and
the paper (Chu et al., *Nature Cancer* 2022).

**What BayesPrism does:**

1. **Deconvolution.** Fractions θ and per-type expression Z, by Gibbs sampling with the scRNA-seq
   reference as a prior. A second pass pools non-malignant expression across samples.
2. **Embedding learning, for the malignant type only.** This needs `key` = the tumor cell type.
   - NMF on the deconvolved malignant expression gives K programs ("malignant bases" η). K is chosen
     by cophenetic score: 5 in GBM, 4 in HNSCC, 7 in SKCM.
   - An EM then refines η, with the non-malignant parts fixed, and gives each bulk sample program
     weights ω.
   - The weights are normalised to sum to 1 for analysis.
   - A user can pass their own programs as `eta_prior` instead of NMF.
3. **Downstream analysis.** Spearman correlation of
   - program weights ω with non-malignant cell fractions (paper Fig. 4e–g), and
   - deconvolved malignant gene expression with macrophage and T-cell fractions (Fig. 3, then GSEA on
     the correlations).

   The authors warn that missing reference cell states can create spurious correlations, and add
   filters to reduce false positives.

So the user's recollection is right: BayesPrism's program analysis is NMF factors (or user programs) in
the malignant compartment, then correlated with fractions estimated from the same bulk.

**How PALM-bulk differs, and how to show it:**

| | BayesPrism | PALM-bulk | How the benchmark shows it |
|---|---|---|---|
| Which cell types have programs | Malignant only | Every cell type | `act_spearman_specific`, across-types ρ, localisation accuracy. |
| What a program is | NMF factors, data-specific and unlabelled (or a user prior) | Fixed, named programs (PROGENy), the same across studies | Interpretability and comparability; not a metric. |
| Scale of activity | Weights summing to 1 within a sample (relative; one program up forces another down) | An absolute log-scale shift of each program relative to the reference | Effects are interpretable per program; ω cannot express "all programs up". |
| Association with a design | Post-hoc Spearman of estimated ω with estimated θ | β estimated jointly with π, with covariates and intervals | `eff_cov90`, interval score, false sign rate. |
| Uncertainty in the association | Not propagated: two plug-in estimates are correlated | Propagated from counts to β | Coverage, false sign rate. |
| Confounding | ω and θ come from the same bulk, so their estimation errors can correlate and create associations | δ and π are estimated jointly, and the coupling is part of the posterior | **Spurious fraction–activity coupling metric, plus a null simulation** (below). |

**The decisive experiment: a null for fraction–program coupling.**

- Simulate bulk where programs are independent of composition. Then do the same with a known true
  coupling.
- Run each method's full pipeline:
  - BayesPrism: deconvolution, embedding learning with `eta_prior` = PROGENy (so the programs match),
    then Spearman ω ~ θ.
  - Reallocation: activity ~ fractions.
  - PALM: posterior of the activity–fraction association.
- **Report** the false-positive rate at nominal 5% under the null, and the power under the alternative.
- **Expectation:** the plug-in correlations inflate false positives because of shared estimation
  error, and PALM's joint posterior does not.
- **Caution:** this is an expectation, not a result. Run it before claiming it.

**Running BayesPrism embedding on PBMC:** there is no malignant type, so declare one type as `key` (for
example Monocyte) and pass `eta_prior` = PROGENy weights on the modelled genes. That gives the one
cell type's program weights, which can be scored with the activity metrics above as a direct
competitor. Planned, not yet run.

**Honest status:**

- Today, PALM's own model δ ranks activity *worse* than naive. Our activity gains come from posterior
  reallocation (PALM's fractions plus BayesPrism's expectation step).
- The cell-type-specific claim therefore currently rests on reallocation plus PALM's fractions. The
  inference claim rests on β intervals, which are calibrated for the model but not for reallocation.
- De-mixing (section 7) and conformal intervals are the planned fixes.

---

## 7. Current results (dev splits 0–9)

Snapshot of 2026-10-27, morning. PBMC is complete; OneK1K has 6–10 splits per row.
Commands: `python scripts/dev_bench.py summary --dataset pbmc --ref "naive|model"`.

### Activity ρ across samples, δ truth (paired difference vs naive, 95% CI)

| | PBMC | OneK1K |
|---|---|---|
| naive | 0.783 | 0.639 |
| NNLS reallocation | 0.791 (+0.01) | 0.665 (+0.03) |
| posterior realloc (joint quasi+sw) | 0.847 (+0.064 [0.06, 0.07]) | 0.743 (+0.104 [0.10, 0.11]) |
| oracle realloc | 0.869 | 0.750 |
| PALM model δ | 0.52–0.60 | 0.43–0.49 |

**Type-specific ρ:**
- Posterior reallocation: 0.45–0.48 on PBMC, 0.50–0.52 on OneK1K.
- NNLS reallocation: 0.23 and 0.36.
- Naive: nan, since it is constant here.

**Effects r:**
- Posterior reallocation: 0.63–0.65 on PBMC, 0.63–0.71 on OneK1K.
- Naive: 0.43 on both.

**Against the ULM truth:**
- Posterior reallocation ties naive on PBMC and is slightly worse on OneK1K (−0.01 to −0.035). The
  oracle reallocation is also −0.03 there, so this is a limit of reallocation.
- The joint NB model's δ is +0.10 against naive on both datasets.

**Coverage:**
- Reallocation intervals: 0.17–0.26 for activity, 0.18–0.35 for effects. They are not usable as they
  are.
- Model intervals: 0.72–0.80 for activity, 0.82–0.84 for effects on PBMC.

### Competitors

Pending; the queue is logged in `benchmarks/dev/competitors.log`.

First check, PBMC split 0, cellGeometry after the unit fix:
- fraction RMSE 0.093
- activity ρ 0.85
- type-specific ρ 0.44
- effects r 0.63

That is at the level of posterior reallocation on activity, and worse on fractions (PALM is about
0.058).

**PALM posterior realloc vs cellGeometry + realloc** (paired by split, 10 splits each, 2026-10-27).
The reference is donor pseudobulk rows. Cells give the difference [95% CI]:

| Metric | PBMC, joint NB | PBMC, joint quasi+sw | OneK1K, joint NB | OneK1K, joint quasi+sw |
|---|---|---|---|---|
| Fraction RMSE (lower is better) | **−0.034** [−0.042, −0.027] | **−0.033** [−0.040, −0.026] | +0.001 [−0.003, +0.007] | +0.001 [−0.001, +0.003] |
| Activity ρ | **+0.017** [+0.011, +0.024] | **+0.015** [+0.006, +0.024] | **+0.027** [+0.019, +0.032] | **+0.030** [+0.024, +0.035] |
| Specific ρ | **+0.052** [+0.032, +0.074] | +0.028 [−0.008, +0.061] | +0.005 [−0.009, +0.018] | +0.003 [−0.007, +0.014] |
| Effect r | **+0.065** [+0.007, +0.116] | +0.067 [−0.001, +0.124] | −0.026 [−0.070, +0.016] | −0.010 [−0.043, +0.025] |
| ULM ρ | +0.003 [−0.002, +0.008] | +0.003 [−0.002, +0.008] | **+0.018** [+0.014, +0.021] | **+0.036** [+0.030, +0.041] |

What the table says:

- **PALM wins activity ρ everywhere.** The margin is small: +0.015 to +0.030.
- **Fractions:** PALM wins clearly on PBMC (0.062 vs 0.096). On OneK1K they are level: cellGeometry
  scores 0.048 and PALM's joint fits 0.049.
- **Specific ρ and effects:** PALM wins on PBMC. They are level on OneK1K, where the sex contrast has
  few real effects.
- **Localisation** (PBMC; OneK1K has no true effects): PALM 0.83–0.87, cellGeometry 0.76.
- **Speed:** cellGeometry takes about 1 minute per split; PALM takes about 2–10 minutes.
- **Intervals:** cellGeometry gives none for activity or effects. PALM's reallocation intervals are
  overconfident:
  - PBMC: 94% of called effects are wrong.
  - OneK1K: every called effect is wrong, because no true effect exceeds 0.1.

  The model's β intervals are conservative instead: PBMC coverage 0.79–0.84; OneK1K
  `joint quasi+sw` coverage 0.73, with 0 calls.

**BayesPrism, PBMC split 0** (the reference is donor pseudobulk rows, not single cells):

| Estimate | Fraction RMSE | Activity ρ | Specific ρ | Effect r |
|---|---|---|---|---|
| Its own expression Z | 0.17 | 0.69 | 0.36 | 0.34 |
| Its fractions with plain reallocation | 0.17 | 0.81 | 0.42 | 0.46 |

- **Runtime:** 26 minutes, against 103 s for the PALM fit.
- **Comparison:** both are behind cellGeometry and PALM on this split.
- **Caveat:** BayesPrism expects single cells. The OneK1K runs with real cells (`--cells`) are its
  fair test.

### De-mixing: ruled out (2026-10-27)

**Idea.** For each sample, compute the matrix A_i that maps the true δ to the expected reallocation
estimate under the model (A_i holds the attenuation and the leakage between types), then invert it
with a ridge. Script: scratchpad `demix.py` / `demix_r.py`; results in
`benchmarks/dev/demix*_pbmc_joint_nb.json`.

**What A_i looks like** (PBMC split 0):
- Own-type gain: about 0.15 for B and NK cells, about 0.5 for Monocytes and T cells.
- Summed leakage from the other types: about 0.6–1.1.

**Ridge towards zero** (10 splits). Every strength is worse than reallocation, also with the true π:

| Ridge | ρ | Specific ρ |
|---|---|---|
| none | 0.31 (blows up; MAE 0.95) | 0.31 |
| 0.01 | 0.52 | 0.40 |
| 1 | 0.77 | 0.10 |
| reallocation | 0.849 | 0.476 |

**Ridge towards the reallocation estimate** (5 splits). Results improve steadily as the ridge grows,
and the best only ties reallocation: ρ −0.006, specific +0.001.

**Conclusion.** The model-based leakage matrix does not describe reallocation's real errors: profiles
also differ from B outside the program genes, and inverting amplifies that. Reallocation's attenuation
acts as useful shrinkage for ranking. The remaining activity headroom is the oracle gap
(0.02 ρ, 0.05 specific ρ). Reallocation stays the activity estimator. Work moves to its intervals.

---

## 8. How to run

```
python scripts/dev_bench.py run --dataset pbmc --splits 0-9 --configs night        # PALM presets
python scripts/dev_bench.py run --dataset pbmc --splits 0-9 --configs {} --competitors cellgeometry,bayesprism
python scripts/dev_bench.py run --dataset o1k  --splits 0-2 --configs {} --competitors bayesprism \
       --cells C:/Users/duy.pham/Downloads/onek1k_full_cells.h5ad
python scripts/dev_bench.py summary --dataset pbmc --ref "joint quasi+sw|posterior realloc"
python scripts/dev_bench.py rescore --dataset pbmc      # after changing metrics: recompute from saved arrays
```

Every row file has a matching `.npz`:

- **PALM:** thinned draws, 100 of each.
- **Baselines and competitors:** `<row>__fractions`, `<row>__activity` and `<row>__effects`.

From the morning of 2026-10-27, metrics come from the 100 thinned draws, in both `run` and `rescore`.
Rows from before then have no `.npz` and keep their old metrics: the PBMC configurations
`2step quasi` and `joint quasi+sw` from phase 1.

- **Outputs:** one JSON per (split, configuration) in `benchmarks/dev/<dataset>/`. Fits also write
  `.npz` arrays, so new estimators can be scored without refitting.
- **Restarting:** runs are restart-safe; existing files are skipped.
- **R methods:** need `PALM_RSCRIPT`.

---

## Changelog

- **2026-10-27**
  - Added program prior sets (PROGENy-14, CollecTRI immune, Hallmark immune), the `--prior` option and
    `act_prog_leakage`, following Tripso (section 5b). Designed a stimulation benchmark with label
    truth (not built).
  - `eff_fsr` is now the wrong-sign rate only; added `eff_fdr`.
  - Calibrated reallocation: first results (section 5c).
  - De-mixed reallocation tested and ruled out (section 7).
  - Implemented the leakage, gain, coupling, localisation, interval-score and false-sign-rate metrics,
    with tests.
  - Added the `rescore` command, and `.npz` arrays for every row.
  - Added the competitors (cellGeometry, BayesPrism) and the `--cells` reference to `dev_bench`.
  - Fixed cellGeometry's conversion from cells to reads: it now uses each type's mean library size,
    not signature column sums. Fraction RMSE went from 0.229 to 0.093.
  - Proposed the leakage, localisation, coupling and interval-score metrics.
  - Started the de-mixing diagnostic.
- **2026-10-26**
  - Added the oracle, EM and iterated reallocation rows, and `.npz` saving.
  - Ran the night2 preset.
  - Introduced the shared/specific activity metrics and the ULM truth.
  - Introduced 10 dev splits with a reserved held-out set, and the truth split on by default.
- **2026-10-28**
  - Open-items run (`scripts/experiments/openitems.py`): fraction sources × 4 prior sets × 10 splits through the calibrated pipeline.
  - Fitting PALM once and transferring its fractions to other prior sets is valid.
  - Stratified conformal effect intervals reach PBMC coverage 0.88.
  - `calib fr` gives nominal activity coverage for engines without draws.
  - OneK1K competitors are complete.
  - Results and decisions are in `method_spec.md` section 7.
  - **Bug fix** in `estimate_delta`: the capped Newton step cycled for strong responses with large program weights (raw PROGENy).
    - Now uses backtracking line search.
    - Kang PROGENy truth was wrong; PBMC PROGENy-7 had 1.2% of entries wrong; Hallmark and OneK1K were unaffected.
    - PBMC PROGENy results are being redone (old files in `old_estimator/`).
  - New dataset **Kang IFN-β** (`--dataset kang`):
    - 8 donors, paired stim/ctrl pseudo-replicates.
    - Reference and calibration from control cells of the non-test donors.
    - `make_benchmark(ref_filter=..., all_test_specimens=...)`.
  - New competitor **CIBERSORT** (`--competitors cibersort`): R script v1.02 with an LM22-style marker signature; activity by reallocation.
  - New pipeline variants: leakage-aware effect width (`lk`), proxy calibration (`calib px`), `comp:<name>` sources.
  - Details are in `method_spec.md` section 7b.
