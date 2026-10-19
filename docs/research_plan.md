# PALM-bulk research plan: two papers

*Status as of 2026-10-16. Owner: Duy Pham.*

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
- Tutorial notebook (new realistic demo, recovery checks, benchmark section), README, tests.

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
| Demo recovery | fraction RMSE 0.012; per-cell-type response coverage 95% | tutorial |

### 3.3 Defaults (decided)

`init="svi"`, `num_warmup=200`, `dispersion="gene"`, `precision="float64"` for real data (float32 only for small problems), covariates whenever known, `reparam=False`.
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
| 7 | Competitor comparison | BayesPrism → PROGENy (strongest), CIBERSORTx high-res → PROGENy, BLADE → PROGENy, TOAST/TCA csDE → enrichment, naive two-step. Harness in `palm_bulk.compare` + `scripts/run_compare.py` (2026-10-19); R/Docker/BLADE wrappers still to be run against the real tools |
| 8 | Modest application on the in-house unpaired cohort | optional; group-level validation |
| 9 | Write and submit | |

### 3.5 Ground-truth design (for E2–E4)

- **Pseudo-bulk from held-out donors:** exact truth, no platform difference. Build the reference from the other donors (cross-fitting; needs ~6–10 donors). Add realism step by step: platform factor estimated from the paired data, real-looking noise. Stress tests: remove a cell type, vary sample size and depth.
- **Paired real bulk:** realistic but noisy truth (scRNA-seq capture bias). Compare **across-sample correlations** per cell type, which cancel a fixed capture bias, plus lineage-level fractions and effect directions. Flag cell types known to be lost in scRNA-seq (neutrophils, plasma cells).
- Truth tables per sample: cell fractions; read fractions (cells × UMIs per cell); per-type pseudobulk → program scores (rank comparison); attribution shares; effects if conditions exist.
- Require ~50–100 cells per cell type per sample for per-type program truth.
- **Never** build the reference from the test donors (leakage).

### 3.6 Paper 1 outline

1. Introduction: bulk cohorts are large; one bulk program score confounds "which cells" and "how active"; two-step approaches lack uncertainty and effect decomposition.
2. Results: model overview → calibration and identifiability limits (E1) → PBMC pseudo-bulk vs competitors and level guidance (E2) → known stimulation (E3) → paired tissue real bulk (E4) → (application) → practical guidance and scaling.
3. Discussion: what bulk can and cannot identify; missing cell types; causal limits.
4. Methods; code and data (package, tutorial, benchmark scripts, Zenodo).

### 3.7 Novelty positioning (from quick literature search)

- Existing: fractions only (CIBERSORTx, MuSiC, Bisque); fractions + gene-level per-cell-type expression (BayesPrism, BLADE, BayesPrism-DWLS, BLUE, EPIC-unmix, TCA, bMIND, ENIGMA); csDE from bulk (TOAST, CellDMC, CARseq, TCA); two-step cell-type pathway analysis (IPF preprint 2024; GTM-decon); within-type states in spatial data (DestVI).
- Our difference: program-level latent structure (far fewer unknowns than gene-level); attribution with uncertainty; exact intrinsic/composition split with intervals; calibration as a design goal.
- Main reviewer question: "Isn't this BayesPrism + PROGENy?" Answer with a head-to-head comparison.
- To do before writing: a systematic literature search (PubMed, bioRxiv, Scholar, 2023–2026).

---

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
| 2 | 2–3 months | Add Teacher 2 (ARCHS4, GTEx blood); test on real bulk with truth: flow-cytometry PBMC (e.g. Monaco et al. 2019), IFN-stimulated vs control, our paired tissue cohort | Beats CIBERSORTx and BayesPrism + PROGENy on real bulk; honest intervals |
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
- BayesPrism extension module (PALM-BP): BayesPrism for fractions and per-type reads, then the program model per cell type. A fast mode and an ablation.
- Hybrid Gibbs sampler (Pólya–Gamma augmentation): only if speed remains the bottleneck.

## 8. Open decisions

- [ ] Which public PBMC dataset(s) for E2 and the Paper 2 pilot
- [ ] Number of donors and samples in the paired tissue cohort (cross-fitting needs ~6–10 donors)
- [ ] Whether the paired cohort includes a treatment or condition (needed to validate the effect split)
- [ ] Format of the in-house raw data (FASTQ vs count matrices)
- [ ] Final journal choice for Paper 1 after E2–E4 results

## 9. Key references

- BayesPrism; BLADE; BayesPrism-DWLS (PMC13457527); BLUE (PLOS Comp Biol); EPIC-unmix (Genome Biology 2025)
- Cell-type-specific pathway analysis of IPF bulk RNA-seq (bioRxiv 2024.07.30.605740)
- Meng et al. 2023 and Jaakkola & Elo 2022: csDE method benchmarks (Briefings in Bioinformatics)
- DestVI (spatial within-type states)
- BulkFormer: bioRxiv 2025.06.11.659222; Cell Systems 2026; weights on Hugging Face (zhfanrui/BulkFormer-*)
- Tripso (bioRxiv 2026.03.24.713961): gene-program-centric single-cell model; 94% of discovered programs lineage-specific
- Kang et al. 2018 (IFN-β PBMC); Monaco et al. 2019 (PBMC bulk + flow)
