# PALM-bulk research plan: two papers

*Status as of 2026-10-29. Owner: Duy Pham.*

## 1. Summary

PALM-bulk estimates, from bulk RNA-seq and a single-cell reference, four things per sample, with calibrated uncertainty:

1. cell-type fractions;
2. activity of known gene programs (PROGENy, MSigDB Hallmark, CollecTRI) **inside each cell type**;
3. attribution: which cell types carry each program's bulk signal;
4. for treated vs control designs, the program effect split into **intrinsic** (cells respond) and **composition** (cell mix changes) parts.

Strategy: **two papers sharing one core library.**

| | Paper 1 | Paper 2 |
|---|---|---|
| What | The Bayesian method (PALM-bulk) | A fast model trained on real data (BulkFormer-based), plus a public atlas and a large-scale finding |
| Role | Defines and validates the quantities; gold standard | Makes them available for every public bulk sample in milliseconds |
| Status | Nearly done (Phase 0) | New project (Phases 1–4) |
| Target | Genome Biology / Nature Communications | Nature Methods / Nature Biotechnology (Genome Biology fallback) |
| Time | ~2–4 months to submission | ~9–13 months |

Why two papers: earlier output, lower risk (Paper 1 stands even if Paper 2 fails), and Paper 2 needs a published, validated reference to show its fast answers are right.

---

## 2. Shared core library (both papers)

One code base, three layers:

| Layer | Contents | Used by |
|---|---|---|
| **Core** | model equations (mixture of cell-type profiles, program shift `exp(delta · G)`, negative-binomial noise, platform term), program definitions, reference building, gene selection, covariates, simulators | both |
| **Bayesian inference** | NUTS sampler, VI warm start, calibration harness | Paper 1; gold standard for Paper 2 |
| **Fast inference** | BulkFormer encoder + probabilistic output head, training loops | Paper 2 |

Done (2026-10-19): `palm_bulk.core` (equations on numpy/JAX/torch, `estimate_delta`, data prep, simulators), `palm_bulk.bayes` (model, NUTS, diagnostics), `palm_bulk.compare` (competitors, scoring). The fast-inference layer will be `palm_bulk.fast`; it can call the core forward model on torch tensors directly (parity-tested against JAX and numpy).

---

## 3. Paper 1: the Bayesian method

### 3.1 Done (in the repository)

- Joint model: fractions, per-cell-type program shifts, attribution, Shapley intrinsic/composition split, platform term.
- **Gene selection** = union of program genes (45%), cell-type markers (30%), HVGs from the reference (10%), anchor genes (15%); filters for reference expression and bulk detection.
- **Covariates** (batch, sex, age; one-hot coding), **gene-wise dispersion**, **VI warm start**, optional **zero-sum reparameterisation**.
- **Level-1 / level-2 support:** composition-only mode (`programs=None`), lineage-shared program shifts (`program_groups="lineage"`), `lineage_fractions()`, a warning above 20 cell types.
- **Compute:** CPU/GPU, float64 option, parallel chains, sparsity speed-up (exact, ~1.4x), runtime estimator, batched attribution.
- Tutorial notebook (new realistic demo, recovery checks, benchmark section), user guide (docs/user_guide.md), tests.

### 3.2 Key evidence so far

| Finding | Numbers | Source |
|---|---|---|
| Baseline is well calibrated | 90% intervals cover 0.92–0.96 | benchmarks/calibration |
| Covariates matter under confounding | effect coverage 0.25–0.44 → 0.81–1.00; errors 2–3x lower | calibration |
| Gene-wise dispersion helps on noisy genes | coverage 0.88 → 0.94; fraction error −15%; +13% time | calibration |
| VI warm start | ~2.5x more efficient per gradient; same accuracy | calibration |
| New defaults combined (VI + gene dispersion) | coverage 0.91–0.94; ~40% faster per fit | calibration |
| float32 breaks at real-data scale | step size 330x smaller; fraction RMSE 0.018 vs 0.0105 in float64 | benchmarks/precision.json |
| Zero-sum reparameterisation | mixed: fractions 1.8x more efficient, worst effect 2x worse; kept opt-in | calibration, sampler.json |
| Fixed per-gene dispersion (PBMC) | 4x faster than sampled; min ESS 16 → 176 at R-hat 1.013; effect coverage 0.89 → 0.82 | sampler_speed_pbmc.json |
| Dense mass for globals / tree depth 8 (PBMC) | cheaper iterations but chains disagree (R-hat 1.26–1.36); not used | sampler_speed_pbmc.json |
| Direct NB log density | same values and gradients as numpyro's; gradient 9–12 ms → 3.5–4.3 ms | test_core, profiling |
| Demo recovery | fraction RMSE 0.012; per-cell-type response coverage 95% | tutorial |

### 3.3 Defaults (decided)

`init="svi"`, `num_warmup=200`, `dispersion="fixed"` (per-gene NB dispersion estimated once from the bulk; `"gene"` samples it), `precision="float64"` for real data (float32 only for small problems), covariates whenever known, `reparam=False`, `gene_selection="fisher"` (gene-budget study). The dispersion and reparam choices come from benchmarks/sampler_speed_pbmc.json (2026-10-20): fixed dispersion gave 15–25x more effective draws per minute and R-hat 1.013 at 1000+1000 draws; reparam, a dense mass block and a tree-depth cap did not help. Cost: effect-interval coverage 0.89 → 0.82, so final analyses may use `dispersion="gene"`.
Annotation level: up to ~20 cell types → full model. More than 20 → two stages (composition-only fine fractions, then full model on a level-1 reference) or `program_groups="lineage"`.

### 3.4 Remaining work for Paper 1

| # | Task | Notes |
|---|---|---|
| 1 | Finish MAX_GENES study | scripts/run_gene_budget.py; 300 and 600 done, 1000 and 1500 running |
| 2 | Collect benchmarks into `benchmarks/palm_benchmarks.json`; run the tutorial in place | scripts/collect_benchmarks.py |
| 3 | ~~Restructure into core + inference layers~~ | done 2026-10-19, see section 2 |
| 4 | **E2:** PBMC pseudo-bulk benchmark with cross-fitting (held-out donors) | exact truth; level-1 vs level-2; runtime scaling |
| 5 | **E3:** known stimulation response (e.g. IFN-β-stimulated PBMC, Kang et al. 2018) | intrinsic vs composition split and attribution |
| 6 | **E4:** paired tissue bulk + scRNA-seq (in-house) | real bulk; compare relative patterns across samples |
| 7 | Competitor comparison | InstaPrism (BayesPrism's model) → PROGENy (strongest), CIBERSORTx high-res → PROGENy, BLADE → PROGENy, TOAST/TCA csDE → enrichment, naive two-step. Harness in `palm_bulk.compare` + `scripts/run_compare.py` (2026-10-19); R/Docker/BLADE wrappers still to be run against the real tools |
| 8 | Modest application on the in-house unpaired cohort | optional; group-level validation |
| 9 | Write and submit | |

### 3.5 Ground-truth design (for E2–E4)

- **Pseudo-bulk from held-out donors:** exact truth, no platform difference. Build the reference from the other donors (cross-fitting; needs ~6–10 donors). Add realism step by step: platform factor estimated from the paired data, real-looking noise. Stress tests: remove a cell type, vary sample size and depth.
- **Paired real bulk:** realistic but noisy truth (scRNA-seq capture bias). Compare **across-sample correlations** per cell type, which cancel a fixed capture bias, plus lineage-level fractions and effect directions. Flag cell types known to be lost in scRNA-seq (neutrophils, plasma cells).
- Truth tables per sample: cell fractions; read fractions (cells × UMIs per cell); per-type pseudobulk → program scores (rank comparison); attribution shares; effects if conditions exist.
- Require ~50–100 cells per cell type per sample for per-type program truth.
- **Never** build the reference from the test donors (leakage).

### 3.6 Positioning and claims (revised 2026-10-28, after the literature check)

**Positioning.** Cell-type-specific program activity is already inferred from bulk data, but only
indirectly, never against a ground truth, and without uncertainty. PALM is the first framework that
measures how wrong those estimates are, corrects them, and gives valid intervals.

**One-sentence claim:**
> PALM estimates the activity of prior-defined gene programs (pathways, TF regulons) in each cell type of
> each bulk sample, with calibrated uncertainty, and comes with a ground-truth benchmark showing the
> errors of current practice and how to correct them.

**Contributions, strongest first.** Numbers are from the dev splits; final numbers come from the held-out
splits 10+.

1. **The first ground-truth benchmark for program deconvolution, and the failure modes it reveals.**
   - **Current practice is unchecked.** People deconvolve and then score pathways (CIBERSORTx, BayesPrism,
     BLADE), or correlate NMF programs with fractions. Nobody has tested these against a truth.
   - **Our truth:** held-out halves of real cells (binomial thinning) on PBMC/COVID, OneK1K and Kang IFN-β,
     a real stimulation.
   - **Attenuation:** a cell type's own shift is recovered at about 0.4× even with perfect fractions.
   - **Leakage:** other cell types' shifts contaminate minor types.
   - **Localisation:** bulk-level scoring finds the right cell type only at chance (0.25 with 4 types);
     reallocation reaches 0.7–1.0.
2. **Calibrated uncertainty at program level, for the first time.**
   - Uncorrected intervals cover 15–22%, and 17–25% of "significant" effects have the wrong sign.
   - Calibration on the reference donors reaches activity coverage of about 0.87–0.90 (PBMC, OneK1K).
   - The leakage-aware effect interval reaches 0.88–0.96 coverage, with a PBMC false sign rate of about
     0.05.
   - It works with any fraction source: posterior draws, a rerunnable method, or a point estimate (with a
     PALM-lite proxy for its error).
3. **Fractions are the lever, and PALM's are the best.**
   - **Kang:** fraction RMSE 0.025–0.031 vs 0.057–0.082 for cellGeometry and CIBERSORT, so 2–3× lower.
     Activity reaches the true-fraction ceiling.
   - **Activity ρ:** never worse than any competitor; +0.02 to +0.03 over cellGeometry and CIBERSORT, up to
     +0.20 over NNLS with rich priors.
   - **Type-specific ρ:** +0.06 to +0.09 over PALM-lite, which is what the Bayesian fit buys.
4. **Fit once, use any prior set.** Fractions fitted with PROGENy-7 and reused under Hallmark or CollecTRI
   match a native fit (ρ 0.789 vs 0.787) at about 1/15 of the cost. The program set becomes an analysis
   choice.
5. **A module for any deconvolution output.** `program_deconvolution` turns PALM, PALM-lite (under 1 s) or
   any method's fractions into calibrated program activity and effects. It answers "why not CIBERSORTx?"
   by improving it.

**Framing against prior work:**
- **CIBERSORTx, BayesPrism, BLADE, CODEFACS, EPIC-unmix, BLUE:** recover per-type expression, then score
  pathways. We show this attenuates and leaks, and we calibrate it. Our reallocation step belongs to this
  family, done with better fractions.
- **BayesPrism's embedding module, EcoTyper, Statescope:** unsupervised NMF programs or states. Correlating
  them with fractions can't say which cell type carries a named program, or how strongly. We give named
  programs per cell type, with intervals.
- **Cell-type-specific differential expression (TOAST, TCA, CARseq, CellDMC; Brief Bioinform 2023
  benchmark):** gene-level, with no program structure and no calibrated program effects.
- **Single-cell or spatial methods with programs or states (Spectra, Tripso, DestVI):** a different data
  type. They supply program definitions and motivation.

**Claims to avoid:**
- "First cell-type-specific pathway activity from bulk": BLADE and the IPF preprint (2024) claim it.
- "Much more accurate": the gains over good competitors are +0.02 to +0.03.
- Anything resting on the joint model's own δ: reallocation beats it.

**Required before writing:**
- A systematic search (PubMed, bioRxiv, Scholar, 2023–2026). Read the final Statescope paper (Nat Commun
  2026), the IPF preprint's methods and the latest BLADE.
- **Two explicit baselines:**
  1. InstaPrism's (BayesPrism model) per-type expression scored with PROGENy/decoupleR (the `instaprism` row).
  2. NMF programs correlated with fractions, as in BayesPrism's embedding module.
- PBMC numbers recomputed with the fixed `estimate_delta` (v2 run).
- Kang activity coverage (0.70–0.83) is either fixed or reported as the limit of calibrating from a resting
  reference.
- Final numbers from the held-out splits 10+.

### 3.7 Paper 1 outline and venue

**Two arms:**
- **Quantitative:** the benchmark above.
- **Qualitative:** in-house disease cohorts from the lab, PALM vs one strong existing pipeline (e.g.
  InstaPrism or CIBERSORTx expression + PROGENy, or cellGeometry + reallocation).

**Outline:**
1. **Introduction.** Bulk cohorts are large, and the question is "which cells change which programs". Current
   answers are indirect and unvalidated, and nobody knows how wrong they are.
2. **Results**
   - **Framework:** fractions (PALM, PALM-lite or any method) → reallocation → calibration → effects.
     Fit once, use any prior.
   - **Ground truth and failure modes:** attenuation, leakage and localisation (PBMC, OneK1K), with
     simulations for identifiability.
   - **Head-to-head accuracy:** fractions, activity and effects vs NNLS, CIBERSORT, cellGeometry,
     InstaPrism and NMF-correlation, across 4 prior sets.
   - **Calibration:** coverage, false sign rate and interval score; uncorrected vs calibrated; behaviour
     under null contrasts (OneK1K sex).
   - **Real perturbation:** Kang IFN-β, a known answer (interferon programs in all types, strongest in
     monocytes).
   - **Disease applications (in-house):**
     - Show where PALM and the existing pipeline disagree, and which is right. Use orthogonal evidence:
       matched scRNA-seq or flow, sorted-cell data, histology, or a known biology anchor.
     - Report effects with intervals, and how many effects the existing pipeline calls with the wrong sign
       or in the wrong cell type.
   - **Practical guidance:** when fractions matter, the gene budget, the choice of prior set, the
     calibration set size (the Kang lesson), and runtime.
3. **Discussion.** What bulk can and cannot identify (attenuation ceiling), missing cell types, a resting
   reference vs perturbed samples, and causal limits.
4. **Methods; code and data:** package, tutorial, benchmark scripts, Zenodo; the Kang and OneK1K builds
   reproducible from public data.

**The disease arm must be falsifiable, not anecdotal.**

**Available data (2026-10-28):** in-house bulk RNA-seq and scRNA-seq of the same disease, from different
patients (unpaired).
- The scRNA-seq has both disease and control samples (confirmed), so all steps below use in-house data
  only.

**Validation design with unpaired data, in order of strength:**
1. **In-domain ground truth.**
   - Run the dev_bench design on the in-house scRNA-seq: per-donor pseudo-bulk with thinned held-out
     halves, and reference donors separate from test donors.
   - This shows that accuracy and calibration hold in this disease's cell types.
2. **Cross-cohort replication.** This is the main quantitative-to-qualitative bridge.
   - **scRNA-seq side:** true per-cell-type program shifts, disease vs control, computed with
     `estimate_delta` on per-type pseudo-bulk.
   - **Bulk side:** effects from PALM and from the existing pipeline, disease vs control.
   - **Scores:** sign agreement, localisation (the same cell type carries each program), and correlation
     over cell type × program.
   - Both methods face the same cohort difference, so the comparison is fair.
   - The larger bulk cohort can show effects with intervals that the scRNA-seq is underpowered to detect.
3. **In-silico spike-ins in disease tissue.** Add a known program shift to one cell type of the scRNA-seq,
   mix, and test localisation and calibration in this tissue.
4. **Clinical utility.**
   - Per-cell-type activity (PALM vs the existing pipeline vs plain bulk program scores) predicting
     severity or outcome on held-out bulk samples.
   - This shows value, not truth.
5. **Optional targeted wet lab.** Confirm one or two calls where the methods disagree (IHC, flow, or qPCR
   on sorted cells from archived samples). This is what makes the reach tier plausible.

**Practical points:**
- Use the scRNA-seq **control** donors as the PALM reference and calibration set; that's the realistic
  resting reference.
- Expect the Kang issue if disease effects are strong: activity coverage may fall short. Measure it in
  step 1 and report it.

**Tier by design:**
- Steps 1 + 2 + 3: Genome Biology / Nature Communications.
- Plus step 5: Nature Methods becomes plausible.
- Without step 2: the arm is descriptive, which means the Cell Reports Methods / NAR tier.

**Venue (decide after the held-out splits and the first disease cohort):**

| Tier | Journals | What it needs |
|---|---|---|
| Reach | Nature Methods, Nature Biotechnology | A striking disease finding confirmed orthogonally; broad benchmark including spatial or more tissues; clear "current practice misleads" message |
| **Main target** | **Nature Communications**, **Genome Biology** | The benchmark + calibration + module story with 3 public datasets + 1–2 disease cohorts with validation. Genome Biology is the natural home for methods with benchmarks (EPIC-unmix appeared there). Nature Communications published BLADE and Statescope. |
| Solid | Cell Reports Methods, Cell Genomics, NAR, Briefings in Bioinformatics | The methods and benchmark alone, or disease arm without orthogonal validation |
| Methods-only fallback | Bioinformatics, PLoS Computational Biology, NAR Genomics and Bioinformatics | If the disease arm is delayed |

**Recommendation:** write for **Genome Biology** or **Nature Communications**. Present a reach submission
to Nature Methods only if the disease arm yields a finding that the existing pipeline gets wrong (e.g.
wrong cell type or wrong sign) and it is confirmed independently. The benchmark + calibration story is
strong but incremental on accuracy (+0.02 to +0.03), so the "current practice is miscalibrated and
mislocalises effects" message and the disease validation carry the impact.

### 3.8 PALM as a program-deconvolution module on any method's fractions (design, 2026-10-27)

**Idea.** Separate *composition* from *program deconvolution*. Any deconvolution method (cellGeometry,
InstaPrism, CIBERSORTx, MuSiC, PALM itself) supplies the fractions. The PALM module turns bulk counts
plus those fractions into per-sample, per-cell-type program activity δ, and design effects β, with
calibrated intervals.

This repositions PALM: it complements the fraction tools instead of competing with them. It also
answers the reviewer question "isn't this BayesPrism + PROGENy?", because the module can be run on
the fractions of InstaPrism (BayesPrism's model).

**Feasibility: already shown on the dev benchmark** (PBMC, 10 splits, PROGENy-7; posterior
reallocation on each method's fractions):

| Fractions from | Fraction RMSE | Activity ρ | Type-specific ρ | Effect r |
|---|---|---|---|---|
| NNLS | 0.117 | 0.791 | 0.23 | 0.44 |
| cellGeometry | 0.096 | 0.832 | 0.43 | 0.58 |
| multinomial EM ("PALM-lite", 5–15 s per split) | 0.076 | 0.830 | 0.40 | 0.62 |
| PALM | 0.062 | 0.849 | 0.48 | 0.65 |
| truth (oracle) | 0 | 0.869 | 0.53 | 0.62 |

- The module works with any fractions, and its activity accuracy rises with fraction accuracy.
- PALM's own fractions are the best plug-in so far.

**Pipeline.** Every step is closed form or a vectorised Newton step, so it runs in seconds.

1. **Unit harmonisation.** Methods report different quantities:
   - read fractions on their own gene set (InstaPrism θ);
   - cell fractions (CIBERSORTx, MuSiC);
   - cell amounts (cellGeometry).

   All are converted to read fractions on the modelled genes, using the reference's RNA per cell and
   each type's read mass on the gene sets (`compare.methods.to_modelled_read_fractions`, plus the
   library-size conversion used for cellGeometry). A wrong unit was the cellGeometry bug: fraction
   RMSE 0.229 against 0.093 after the fix. The module must therefore require a declared
   `fraction_kind`.
2. **Reallocation.** Bulk counts are split by π_ic B_cg / Σ_c π_ic B_cg.
3. **Per-type program activity.** `estimate_delta` (the multinomial program shift relative to B_c) on
   each type's share. ULM, or another decoupler method, is offered as an alternative scale.
4. **Calibration on reference donors** (section 5c of `docs/benchmarking.md`).
   - Cross-fitted pseudobulks from reference donors give an empirical map from reallocated δ̂ to true δ.
   - A per (c,p) ridge corrects attenuation and leakage, and split-conformal residuals give activity
     intervals.
   - First result (calibration on true π): activity coverage 0.17 → 0.88, effect coverage 0.21 → 0.77.
   - **Fraction-method error has to enter the calibration.** Three options, best first:
     - (a) The user gives a callable `fraction_fn(bulk, reference)`, which the module runs on the
       calibration pseudobulks. This captures that method's bias; cellGeometry takes about 1 minute,
       InstaPrism 10–25 seconds.
     - (b) The method's own uncertainty (draws or standard errors, such as cellGeometry `se()`) is
       propagated by resampling π.
     - (c) Neither is available: calibrate on true π and report that the intervals ignore fraction
       error.
5. **Effects.** Regress calibrated δ on the design and covariates per (c,p), with OLS and sandwich
   errors (or the calibrated variance), plus the coefficient uncertainty bootstrapped over calibration
   donors.
6. **Diagnostics,** estimated on the calibration set and reported to the user:
   - own-type gain and leakage per (c,p);
   - the effective detection limit per cell type: minor types such as B and NK cells have gain ≈ 0.15,
     against 0.5 for monocytes and T cells.

**API sketch:**

```python
res = pb.program_deconvolution(bulk, reference, programs, fractions,
                               fraction_kind="cell" | "read" | "amount",
                               fraction_genes=None, fraction_fn=None, fraction_draws=None,
                               design=None, covariates=None, calibrate="reference_donors")
res.activity, res.activity_interval, res.effects, res.effects_interval, res.diagnostics
```

`compare` would get a matching wrapper for each method ("X + PALM module"), so the benchmark becomes
a matrix: fraction method × activity module. The activity modules are naive, InstaPrism Z,
reallocation + ULM, and the PALM module.

**Requirements and limits:**

- The reference has to cover the same cell types as the fractions, and type names have to match.
  Fractions for types missing from the reference cannot be used.
- Calibration needs per-donor, per-type counts for several reference donors (pseudobulk by donor).
  With a single-donor reference, the module falls back to cell-level bootstraps, and the intervals are
  then weaker.
- Systematic fraction bias passes into δ. Only
  option (a) captures it.
- Activity is relative to the reference profile B_c, so δ = 0 means "as in the reference donors".

**Relation to full PALM.** The Bayesian model stays the joint route, for fractions with uncertainty
and the stage-2 hierarchy. External fractions can also enter it as an informative prior on π, centred
at the external estimate with its standard error. That is a second way to use other methods' output
and is untested.

**Next steps:**
1. Implement `program_deconvolution` with calibration options (a)–(c).
2. Wrappers in `compare` for cellGeometry, InstaPrism and NNLS fractions.
3. Dev-bench rows "X + module" on 10 splits, both datasets, all four prior sets.

---

### 3.9 Perturbation benchmarks, causal module and known-answer metrics (2026-10-28)

**Why perturbation data.** Held-out thinning gives a truth for activity, but only interventions give a known
*answer*: which cell type responds, with which program, and with what causal effect. They test our weak
spots: strong shifts outside the calibration range, leakage under shared responses, and localisation.

**Datasets (`pertpy`; `data/pert/`).** Each answers one question that the others cannot.

| Dataset | Design | Question | Metric |
|---|---|---|---|
| Kang 2018 (done) | PBMC, IFN-β vs control, 8 donors, paired | Strong shared response; causal under randomisation | Activity / effect accuracy, coverage |
| McFarland 2020 (MIX-Seq) | Pooled cancer lines, idasanutlin etc.; lines as "cell types", replicates by subsampling | **Localisation with a known answer:** p53 only in TP53-wild-type lines | `responder_auroc`, `leakage_fpr`, `responder_tpr` |
| Zhang 2021 | TNBC patients' immune cells before and after paclitaxel ± atezolizumab (22 patients) | **Real clinical intervention, in vivo,** paired pre/post; weaker effects | Paired estimator (`groups=patient`), coverage, calls |
| ~~Dong 2023 (CINEMA-OT)~~ | Dropped: 773 log-normalised genes, one donor | — | — |
| Zhao 2021 (optional) | Drug-treated slices of resected tumours | Intervention in solid tissue | As Zhang |

**Known-answer datasets to add (decided 2026-10-28; do not forget).** Selection rule: biology fixes which cell
type should move which program, and in which direction. Programs are never derived from the same dataset.

| Priority | Dataset | Design | Known answer | Notes |
|---|---|---|---|---|
| 1 | **sci-Plex 3** (Srivatsan 2020; `pt.data.srivatsan_2020_sciplex3`) | A549, K562, MCF7; 188 compounds × 4 doses | Line-specific responses by genotype or receptor (check which compounds: TP53-null K562 vs wild-type A549/MCF7; glucocorticoid receptor); HDAC inhibitors in all lines | Build like `build_mcfarland.py` (lines = "cell types"). Doses give the dose-response metric. |
| 2 | **Monaco 2019** (GSE107011) | Real PBMC bulk + 29 sorted types + flow, same donors (4, plus an extended cohort of 13) | Fractions measured by flow in **real bulk**; activity truth from the sorted bulk of the same donor | Closes the "pseudobulk only" gap. One real-bulk set is enough for fractions. |
| 3 | **Parse 10M PBMC cytokines** (2024) | 12 donors × 90 cytokines + PBS, 24 h, 18 cell types | Receptor-defined responders: IL-4 → STAT6 (B, monocytes), IL-2 → T/NK, GM-CSF → monocytes, IFN-γ → JAK-STAT (mostly myeloid) | Primary immune cells: the closest to the disease arm. **Check the licence first** (listed both as an open release and as a commercial product). |
| 4 | **OP3 / NeurIPS PBMC** (Szałata 2024) | 144–146 compounds, 3 donors, PBMC | Compound targets (weaker answers) | Optional |
| 4 | **1M-scBloodNL** (Oelen 2022) | 120 donors × 3 pathogens × 3 h / 24 h | Monocytes respond most; IFN at 3 h in monocytes | Realistic n for calibration and power; check raw-data access (likely EGA) |
| 4 | **Immune Dictionary** (Cui 2024, Nature) | Mouse lymph node in vivo, 86 cytokines × 17 types | Strongly cell-type-specific responses | Mouse; never use its own signatures (IREA) as programs (circular) |
| 4 | **Haber 2017** (`pt.data.haber_2017_regions`) | Mouse gut, Salmonella / H. polygyrus | Known composition shift (tuft and goblet expansion) | Tests the intrinsic / composition decomposition |
| later | Tahoe-100M (2025, CC0) | 50 pooled cancer lines × ~1,100 drugs × 3 doses | Mutation-defined responders (TP53, KRAS/BRAF/EGFR) | Parked: very large. Would be McFarland at scale. |

Not suitable: Papalexi, Frangieh, Norman, Replogle (knockouts within one line, no distinct cell types).

**New metric for the dose datasets:** dose response. In responder types, the effect rises with dose (Spearman
with dose); in non-responder types it stays flat. To add to `compare.known_answer`.

**Causal module (`palm_bulk.causal`, wired into `program_deconvolution`):**

| Piece | What | Assumption for a causal reading |
|---|---|---|
| `estimator_weights` | Every estimator is linear in the outcome (β = Σ w_n activity_n), so calibration and `lk` intervals carry over. Options: `diff`; `ols` (covariates and/or `groups` = donor ids → paired / within-donor); `ipw` (Hajek); `aipw` (doubly robust) | `diff`: randomisation. `ols`/`ipw`/`aipw`: conditional ignorability given the covariates |
| `decompose` | Shapley split of each bulk program effect into **intrinsic** (cells change state) and **composition** (the mix changes), per cell type, with bootstrap intervals | Natural direct/indirect effects under the usual mediation assumptions |
| `robustness_value` | Cinelli & Hazlett (2020): how strong an unmeasured confounder must be to remove an effect (or its significance) | Sensitivity analysis for observational cohorts |

Tests in `tests/test_causal.py`:
- Under confounding, the difference of means is biased (+0.9), while `ols`/`ipw`/`aipw` are within 0.15 of the true effect.
- `aipw` stays right when the outcome model is wrong.
- The within-donor estimator removes donor effects in unbalanced designs.
- The decomposition is exact (I + K = arm contrast) and attributes pure composition and pure intrinsic
  effects correctly.

**Comparing methods (`compare.known_answer`):**
- `paired_test`: per dataset, the difference over splits with a bootstrap CI, Wilcoxon and t p-values, and
  the win rate.
- `holm`: Holm correction across metrics.
- `random_effects`: DerSimonian–Laird pooling of the per-dataset differences, giving one pooled effect with
  CI and p, plus heterogeneity (I²).

This turns "+0.02–0.03 per dataset" into one significance statement across 5–6 datasets.

**What the paper can claim causally:**
- **Kang, McFarland, sci-Plex, Parse:** randomised in vitro, so effects are causal.
- **Zhang:** a randomised trial with paired samples, so effects are causal in patients.
- **In-house disease cohort:** adjusted associations with robustness values, plus the intrinsic/composition
  decomposition. The decomposition is causal only under the mediation assumptions; it should be stated as
  such.
- **Not claimed:** cell–cell causal structure, or individual-level effects.

**Decomposition truth for benchmarks:** `causal.decompose` on the true read fractions and the true activity
(thinned half) gives true I and K; `decomposition_accuracy` scores the estimate.

**Role of the causal part (decided 2026-10-28): supporting, not the theme of Paper 1.**
- **Where causality comes from:** the designs (randomisation), not the model. The estimators and the
  sensitivity analysis are standard.
- **PALM's contribution is measurement:** calibration is regression calibration for an outcome measured with
  error, which makes cell-type causal estimands estimable from bulk with valid intervals.
- **Paper 1 Results section** "From measurement to causal estimands":
  - randomised effects (Kang, Zhang);
  - adjusted effects with robustness values (in-house cohort);
  - the decomposition, stated with its mediation assumptions.
- **Wording:** "causal effects under randomisation" and "adjusted associations with sensitivity analysis";
  never "a causal model".
- **A causal-first paper (Paper 3, optional) would need:**
  - measurement-error theory for estimated mediators (attenuation and leakage biasing the natural
    direct/indirect effects);
  - defensible mediation assumptions;
  - causal validation;
  - a clinical outcome.
  - Natural vehicle: Mendelian randomisation with cell-type program QTLs on OneK1K genotypes.

### 3.10 Beating the competitors: where and how (2026-10-29)

**Where we stand.**
- **Causal intervals pass on three of four perturbation datasets** (method_spec 4.5b). The setting is now the
  default: `estimator="auto"` with `pev` and `misfit`.
- **Fractions:** PALM's are 2–3× more accurate (Kang).
- **Effect point estimates are a tie.** On Kang the correlation with the true effect is 0.976 for PALM and
  0.979 for CIBERSORT, and plain reallocation reaches 0.980. Effect accuracy cannot be the headline.
- **Our edge is valid inference:**
  - which cell type responds;
  - calls that are right and intervals that cover;
  - competitors give no intervals at all.

  That edge has not been measured against what users do today.

**Experiments, in order:**

| # | Experiment | What it shows | Status |
|---|---|---|---|
| C1 | **Current practice with standard tests on the known-answer datasets.** Competitors: InstaPrism, CIBERSORT(x), cellGeometry and BLADE (each scored per type with PROGENy, then a t-test or limma per type and program, BH-corrected), TOAST/TCA (cell-type-specific DE, then program enrichment), and bulk PROGENy with no deconvolution. Datasets: McFarland ida/tram, Kang, Zhang (and sci-Plex once built). | The headline. We expect their calls to leak to non-responder types (the same pairs every split), so false calls on null pairs and wrong localisation rise. Metrics: `fpr_null`, `tpr_t`, responder AUROC, localisation, coverage where an interval exists. | InstaPrism, cellGeometry and CIBERSORT fractions exist for Kang; InstaPrism and CIBERSORT for McFarland and Zhang (InstaPrism: splits 0–2); no tests yet |
| C2 | **The PALM module on each competitor's fractions** (`fractions=`, default settings) | "PALM improves any method": the same fractions turned from invalid calls into valid ones | Wiring exists; not yet run |
| C3 | **Pooled significance:** `known_answer.paired_test`, Holm, random-effects pooling over datasets × splits | One statement across datasets instead of +0.02 per dataset | Code exists |
| C4 | **Held-out splits 10+** for every final number | No tuning on the reported splits | Pending |
| C5 | **Power vs design:** `design_check` over the number of samples, the spread of composition and the cell type's read share | Explains Kang (B 4% of reads, little spread) and tells users how many samples they need. No competitor can say this. | New |
| C6 | **New known answers:** sci-Plex 3 (dose response), Monaco (real bulk with flow fractions), Parse cytokines | Generality; real bulk; primary immune cells | Planned (3.9) |
| C7 | **Speed of the default:** about 5 min per study, mainly `misfit_sd` Monte Carlo and the design check | Fewer draws, or vectorise across cohorts; keep `diff` as the fast setting | Optional |

**Claims C1 and C2 would support:**
- Pathway scoring after deconvolution makes confident calls in cell types that did not respond.
- PALM's calls are right at the stated rate, on any method's fractions.
- PALM reports which cell types the bulk cannot resolve, instead of guessing.

**Open method questions** (keep all engines; add, don't remove):
- Power for minor, low-variance types (Kang B/NK). Options:
  - a prior that shares the response across types (hierarchical over types);
  - gene-level information, with a `misfit` term for the direct engine (direct has power but invalid
    intervals).
- The disease setting: the response varies between patients, so check the coverage of `misfit` when
  effects are heterogeneous (simulate it, then run on the in-house cohort).

### 3.11 Making Paper 1 a principle paper: four pillars (plan, 2026-10-30)

**Working title:** *Which cells changed? Identifiable, causal cell-type attribution from bulk transcriptomes.*
The method alone ("better deconvolution") is incremental. The paper needs a principle, real truth at scale
and a discovery.

| Pillar | Claim | Work | Status |
|---|---|---|---|
| P1 Principle: identifiability | Attribution from bulk is possible only when the design allows it; a design calculator says which cell types are identifiable, before sequencing | Design-based minimum detectable effect (MDE) per type × program; validate as a bound on all benchmarks; turn into a tool (sample size, composition spread) | **Started** (below) |
| P1 Principle: causal split | Every bulk change = composition (more cells) + cell-intrinsic change + spillover bias; estimands on a causal diagram, PALM estimates each with a range | Formalise `decompose=True` as mediation (natural direct / indirect effects, composition as mediator); shared vs type-specific readout (`shared_response`) | Code exists; formalise; 2 fixes pending (§3.10 note) |
| P2 Scale: real perturbations | Across thousands of real perturbations the standard workflow puts responses on the wrong cell types | Turn perturbation atlases into known-answer bulk experiments (pooled cells per well / donor), same truth split as §3.9 | Plan; needs compute (below) |
| P3 Real truth in disease | In real patients, PALM on bulk recovers what purified cells show | Bulk + sorted-cell datasets from the same people, ideally TB (candidates being checked) | Search running |
| P4 Discovery | In TB, which cells change, with in-cohort controls and a prospective test | PTB cohort: PALM vs InstaPrism; positive control (interferon-inducible neutrophil signature, Berry 2010), negative controls (size-matched random sets, label permutation); pre-registered predictions checked by sorted-cell or scRNA-seq on a subset; treatment cohort for within-person contrasts | Next week (talk 2.5) |

**Extra causal tools for P4:** within-person contrasts (before vs during treatment), negative-control outcomes
(pathways that cannot respond) to detect confounding and spillover in real data, and robustness values for
observational contrasts.

**Optional, high impact if careful:** an audit of published cell-type claims from blood bulk cohorts (TB,
sepsis, COVID, lupus): what share is consistent with spillover alone.

#### P1 first result: the design calculator is a bound (2026-10-30)
`scripts/experiments/identifiability.py`, outputs `benchmarks/dev/identifiability/`. For each design, the
generalised least-squares covariance of the effects under the full leakage operator (each sample's own
composition, per-(type, program) intercepts, rows weighted by the within-arm noise) gives
MDE = (z_0.95 + z_0.80) × sd: the smallest effect any estimator can attribute to that type without extra
assumptions. (A first version that inverted only the average operator ignored composition variation and was
far too pessimistic on McFarland.)

Development splits 0–2, four datasets, 420 type × program units, PALM = `auto_abs`:

| Real effects (\|true\| ≥ 0.1) | n | Called by PALM |
|---|---|---|
| Calculator: not identifiable (\|true\| < MDE) | 28 | **0%** |
| Calculator: identifiable | 73 | 58% |

- The calculator is never violated: PALM never calls what the design cannot support.
- The gap is PALM's headroom: on McFarland every real effect is identifiable but 56–81% are called; on Kang
  the calculator says T cells (MDE 0.14, true JAK-STAT 0.28) are identifiable, PALM doesn't call them.
- Kang per type (median MDE): monocytes 0.09, T 0.14, NK 0.33, B 0.63; own-signal share 56%, 38%, 14%, 11%.

**Next for P1:** (1) add fraction uncertainty and donor pairing to the calculator; (2) power curves: MDE vs
number of samples and composition spread (simulated designs), for the design tool; (3) close the gap: use
the MDE to set where the widening (`sim_widen`) is needed; (4) a formal statement (identifiability holds iff
the stacked operator has full column rank; MDE from its information matrix).

#### P3 first result: GSE60424, whole blood + sorted cells from the same donors (2026-10-30)
`scripts/experiments/real_truth_gse60424.py`, outputs `benchmarks/real/gse60424/`. Design in the script
docstring: whole blood = bulk; the same donors' sorted neutrophils, monocytes, B, CD4, CD8 = truth (Welch
90% interval per type × program); reference and calibration from the other donors' sorted cells; PROGENy 7.

| Contrast (n) | Truth calls / 35 | Identifiable (calculator) | PALM r / unchanged flagged / real found / coverage | InstaPrism + t-test | Bulk score + t-test |
|---|---|---|---|---|---|
| MS after vs before IFN-β (3 v 3) | 17 | 2 | 0.85 / 0% / 29% / 0.94 | 0.88 / 0% / 82% / 0.57 | 0.85 / 0% / 82% / 0.40 |
| Sepsis vs healthy (3 v 4) | 13 | 2 | 0.13 / 0% / 0% / 1.00 | 0.04 / 24% / 38% / 0.43 | 0.04 / 76% / 15% / 0.23 |
| T1D vs healthy (4 v 4), negative control | 9 | 0 | 0.55 / 0% / 0% / 0.97 | 0.44 / 0% / 0% / 0.77 | 0.37 / 0% / 0% / 0.57 |
| ALS vs healthy (3 v 4), negative control | 8 | 0 | 0.59 / 0% / 0% / 1.00 | 0.53 / 0% / 0% / 0.74 | 0.57 / 0% / 0% / 0.71 |

What it says:
- **Interferon in the body (MS):** a shared response (JAK-STAT +0.16–0.19 in every sorted type), like Kang.
  Copying the bulk signal is right here; PALM's estimates are as accurate (r 0.85) with honest ranges
  (coverage 0.94 vs 0.57) but less power (29% vs 82% found). Same picture as the Kang benchmark.
- **Sepsis:** no method tracks the sorted-cell truth (r ≈ 0). Every method's estimates follow a bulk-wide
  change (NF-κB down, TNF up in all types) that no sorted type shows. Likely causes: (a) whole blood is fixed
  at once (Tempus) while sorted cells spend hours ex vivo, which shifts immediate-early / NF-κB / TNF / MAPK
  genes, so sorted cells are not a clean truth for those pathways; (b) unmodelled cells (NK, eosinophils,
  immature granulocytes in sepsis). InstaPrism + t-test flags 24% of unchanged pairs, bulk score 76%, PALM 0%.
- **Power:** the design calculator says only 0–2 of 35 true changes are identifiable at 3–4 donors per arm,
  consistent with PALM's few calls. GSE60424 is too small for attribution claims; it is useful as a
  real-data negative control (no false flags by PALM in any contrast) and to expose the sorting artefact.
- **Next for P3:** a larger same-donor dataset (Berry GSE19443/19444 TB, microarray, 7 vs 4; Bloom GSE42832);
  restrict the truth to pathways not driven by ex vivo handling, or model a sorting offset estimated from
  healthy donors; or build the truth from single-cell data of the same donors (no sorting step).

**Compute note (P2):** the atlases are large (tens of millions of cells). Pseudobulking is streaming
work, but the per-perturbation PALM runs (minutes each, thousands of perturbations) need a cluster or many
cores; ask before starting.

## 4. Paper 2: fast, real-data-trained model

### 4.1 Idea

Train a network once so it outputs, for any bulk sample, per-cell-type fractions and program activity **with uncertainty**, in milliseconds. This enables analysis of all public bulk data (~1 million samples: GTEx, TCGA, ARCHS4).

### 4.2 Components

| Part | Job |
|---|---|
| **BulkFormer** (pretrained on 500k+ bulk samples, MIT licence, Hugging Face) | Encoder: reads a bulk sample; already knows real bulk |
| **Output head** (new) | Probabilistic outputs per cell type and program |
| **Core equations** (shared with Paper 1) | "Rebuilder" for Teacher 2; defines what outputs mean |
| **Paper 1 Bayesian model** | Gold standard for checking the network |
| **Public program databases** | PROGENy, MSigDB Hallmark, curated CollecTRI subset |
| **Public single-cell atlases** | CELLxGENE, HCA, Tabula Sapiens: references and Teacher 1 data |

### 4.3 Two teachers, both real data

- **Teacher 1, real pseudo-bulk with exact answers:** pool real cells from public atlases; true fractions known; true program activity measured on the pooled cells with the same definition as the core model. Teaches correct answers.
- **Teacher 2, real bulk learned by rebuilding:** the network guesses fractions and activity; the core equations rebuild the counts; the network is rewarded when the rebuilt sample matches the real one. Teaches what real bulk looks like. Keep Teacher 1 in training so the meaning stays anchored.

### 4.4 Phases and decision points

| Phase | Duration | Work | Success test / decision |
|---|---|---|---|
| 0 | 1–2 months | Finish Paper 1 core (section 3.4) | Paper 1 is the fallback |
| 1 | 2 months | **PBMC pilot**: ~500k pseudo-bulk from public PBMC (held-out donors for testing); BulkFormer + output head trained on Teacher 1 | About as accurate as the Bayesian model; 90% intervals cover ~90%; >1000x faster. **Go / stop** |
| 2 | 2–3 months | Add Teacher 2 (ARCHS4, GTEx blood); test on real bulk with truth: flow-cytometry PBMC (e.g. Monaco et al. 2019), IFN-stimulated vs control, our paired tissue cohort | Beats CIBERSORTx and InstaPrism + PROGENy on real bulk; honest intervals |
| 3 | 3–4 months | More tissues (lung, gut, skin, tumour); run on GTEx, TCGA, ARCHS4; build the public atlas; one validated discovery | A finding confirmed independently |
| 4 | 1–2 months | Write the paper | Story: method → accurate and honest → scale → new biology |

### 4.5 Applications (Phase 3)

- **Atlas:** cell-type program activity for every public bulk sample.
- **Genetics:** cell-type-specific program QTLs in GTEx (variant → program activity in one cell type).
- **Cancer:** cell-type program activity vs survival in TCGA.
- **Treatment response:** before/after drug tissue bulk → which cell types respond.
- **In-house cohort:** disease-linked cell-type program, validated with in-house scRNA-seq and flow cytometry or IHC.

### 4.6 Risks and backups

| Risk | Backup |
|---|---|
| Pooled cells differ from real bulk | Teacher 2; keep the platform term |
| Rebuilding alone allows several explanations | Teacher 1 anchors meaning; check against the Bayesian model |
| Network overconfident | calibrate on held-out pseudo-bulk and paired data; report coverage |
| No reference for some tissues | start with blood, lung, gut |
| No strong discovery | atlas + tool as a resource paper |
| Causal attribution between cell types | not identifiable from bulk; do not claim it |

### 4.7 Resources

- GPU: available. One A100-class GPU for Phases 1–2; more for Phase 3.
- Data: public atlases and bulk repositories; in-house paired tissue cohort; in-house unpaired bulk and scRNA-seq.

---

## 5. Data inventory

| Data | Use |
|---|---|
| Public PBMC single-cell (many donors; e.g. OneK1K, lupus and COVID atlases) | Paper 1 E2; Paper 2 Teacher 1 pilot |
| IFN-β-stimulated PBMC (Kang et al. 2018) | Paper 1 E3; Paper 2 validation |
| PBMC bulk with flow-cytometry fractions (Monaco et al. 2019) | real-bulk fraction truth |
| In-house **paired** tissue bulk + scRNA-seq | Paper 1 E4; Paper 2 Phase 2 validation; platform-factor estimate |
| In-house **unpaired** bulk + scRNA-seq | application (Paper 1 modest, Paper 2 main finding) |
| GTEx, TCGA, ARCHS4/recount3 | Paper 2 Teacher 2 and applications |
| `pbmcpedia-v20250915-full.h5ad` (in repo folder) | candidate PBMC reference |

Check access and licences of each public dataset before use.

---

## 6. Practical notes

- Runtime (Bayesian model, CPU): 500 samples × 10 types × 14 programs × 1500 genes ≈ 1–2 days per chain in float64. 1000 samples × 40 types × 2000 genes ≈ 1–2 weeks per chain on CPU; ~5–20 h on an A100 (not measured). Always run `pb.estimate_runtime(inp)` first.
- A reference of 200k–500k cells needs ~16–32 GB RAM to build; the cell count does not affect fit time.
- Experiments on this machine: limit each process to one CPU thread (`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`) when running several at once.

## 7. Ideas parked for later

- Cell-type-specific program dictionaries `G[c, p, g]` (supported by single-cell evidence that most programs are lineage-specific).
- Reference-informed priors on program activity (typical range and co-variation of programs per cell type).
- Semi-supervised discovery of new programs from large bulk cohorts.
- Gene-level attribution within programs ("leading genes").
- InstaPrism extension module: InstaPrism (BayesPrism's model) for fractions and per-type reads, then the program model per cell type. A fast mode and an ablation.
- Hybrid Gibbs sampler (Pólya–Gamma augmentation): only if speed remains the bottleneck.

### 7.1 Spatial niches and microdissection (idea, 2026-10-29; nothing tested)

**Unit of analysis = the niche, not the spot or cell.** A niche (granuloma core, rim, lymphoid aggregate,
normal lung) is a pooled sample of mixed cells, which is what PALM already models. The question per niche:
does a pathway differ because the cell mix differs, or because a cell type changed its activity there?
`decompose` already gives this split.

**Data types:**
- **Microdissection / GeoMx DSP** (whole-transcriptome probe counts per ROI/AOI, ~50–1000 cells):
  - Each ROI is a bulk sample, so PALM runs unchanged. The replicate is the patient, with many ROIs per patient.
  - Marker-segmented AOIs (CD45+, CD68+, PanCK+) give strongly varying cell mixes. This is the best case for
    unmixing (compare Kang, where proportions barely varied and B/NK effects were not identifiable).
  - Competitor: SpatialDecon gives proportions only, with no per-cell-type pathway activity or uncertainty.
- **Spot-based** (Visium 55 µm): pool spots by niche × section, which turns them into bulk samples and
  removes shallow-depth and spot-correlation problems. Niches are defined by histology, or by clustering the
  composition of neighbourhoods.
- **Single-cell spatial** (Xenium, CosMx, Visium HD): niches are defined from cell neighbourhoods. A
  segmentation spillover correction (transcripts leaking between touching cells) is a possible module; the
  leakage resembles the spillover PALM corrects.

**Questions it would answer:**
1. Which cell types change which pathways between niches, beyond the composition difference.
2. Niche composition as a continuous covariate, e.g. whether macrophage interferon activity rises with local
   T-cell density.

**Known issues:**
- Platform effect: probe- or spatial-chemistry capture differs from scRNA-seq. Calibration/misfit may absorb
  part of it; this must be tested.
- GeoMx normalisation (Q3), background (negative probes) and low-nuclei AOIs.
- Inference must be at the patient level. ROIs and spots within a patient are correlated, and pathologist
  ROI choice is not random. Niche comparisons are observational, so report adjusted associations plus
  `robustness_value`, not causal effects.

**Answer-key benchmark:** sum segmented cells from single-cell spatial data (or matched scRNA-seq) into
pseudo-ROIs per niche. The per-cell-type truth comes from the other half of the cells, as in §3.9.

**Steps:**
- First: run region-pooled PALM on a public TB granuloma GeoMx or Visium dataset as a feasibility check.
- Then: a pseudo-ROI benchmark with C-SIDE (RCTD, Nat Methods 2022) and SpatialDecon as competitors.
- Scope: a paper section or a separate paper, decided after the feasibility check.
- Talk mention (2026-10-30): one line in the close, with no claims.

## 8. Open decisions

- [ ] Which public PBMC dataset(s) for E2 and the Paper 2 pilot
- [ ] Number of donors and samples in the paired tissue cohort (cross-fitting needs ~6–10 donors)
- [ ] Whether the paired cohort includes a treatment or condition (needed to validate the effect split)
- [ ] Format of the in-house raw data (FASTQ vs count matrices)
- [ ] Final journal choice for Paper 1 after E2–E4 results

## 9. Key references

- BayesPrism (Chu et al. 2022); InstaPrism (Hu & Chikina 2024); BLADE; BayesPrism-DWLS (PMC13457527); BLUE (PLOS Comp Biol); EPIC-unmix (Genome Biology 2025)
- Cell-type-specific pathway analysis of IPF bulk RNA-seq (bioRxiv 2024.07.30.605740)
- Meng et al. 2023 and Jaakkola & Elo 2022: csDE method benchmarks (Briefings in Bioinformatics)
- DestVI (spatial within-type states)
- BulkFormer: bioRxiv 2025.06.11.659222; Cell Systems 2026; weights on Hugging Face (zhfanrui/BulkFormer-*)
- Tripso (bioRxiv 2026.03.24.713961): gene-program-centric single-cell model; 94% of discovered programs lineage-specific
- Kang et al. 2018 (IFN-β PBMC); Monaco et al. 2019 (PBMC bulk + flow)
