# PALM-uni plan: one probabilistic model for fractions and effects

*Status 2026-11-13: plan only, nothing run (2026-11-16: §3.4 cell2location terms, §3.5 programs and unknown programs, steps U2b, U2c, P0 added; U0 tempering test running). Not part of Paper 1. Freeze v2 stays the Paper 1 method
(`method_spec.md`), v5 stays the candidate for splits 30–39 (`weak_points_plan_2026-11-12.md` §4b).
No held-out split is used for this work without the user's agreement.*

## 0. Why

program2cell is a pipeline: stage A (Bayesian model, fractions) → B (reallocation) → C (calibration) →
D (effects router: unmix vs calibrated reallocation, guard, design check) → E (readouts). Each step is a
standard technique, but the whole needs a diagram to describe and contains rules with no probabilistic
form (router, guard, v4 gate, v5 sizing). A reviewer can read that as researcher degrees of freedom.

The first version (October 2026) *was* one model: fractions, δ and β from one posterior. It fell short.
Head to head on 8 datasets × splits 0–2 (`methods_cut_model.md` §3.1): its β had coverage 0.50, r 0.50
and called 28% of null pairs, against 0.97, 0.86 and 3% for freeze v2. Earlier snapshots:
- its own δ ranked activity worse than reallocation on its fractions (PBMC ρ 0.52–0.60 vs 0.85,
  `benchmarking.md` §7);
- its β intervals undercovered (PBMC 0.79–0.84, OneK1K 0.73; `benchmarking.md` §7, cellGeometry comparison).
Reallocation on the model's fractions *without* calibration was worse still (coverage 0.17–0.35, 19 of 20
calls wrong, `benchmarking.md` §5): the cut alone is not enough; calibration is what makes it honest.

The likely cause is **misspecification**: a misspecified posterior still concentrates as N grows, on the
wrong value (Kleijn & van der Vaart 2012). The model's known gaps (`model_dag.md` §5):

| Assumption in the model | Reality | What the pipeline does instead |
|---|---|---|
| B fixed | donor-to-donor variation of each type's profile | calibration folds over reference donors |
| G the same in every type | footprints differ by lineage | design check with jittered weights (E22) |
| every change goes through a known program | off-program response | misfit term in the unmixed interval |
| RNA content fixed from the atlas | uncertain, maybe disease-dependent | not handled (limitation) |
| linear leakage (Jacobian at δ = 0) | nonlinear for strong responses | nonlinear correction, guard |

The pipeline handles these by *measuring* their error on data with a known answer. PALM-uni has to either
*model* them or *calibrate the posterior* against the same known answers. Otherwise it repeats October.

## 1. Two routes and the decision

| Route | What | Changes the method? | Use |
|---|---|---|---|
| **R1. Cut model** | Describe freeze v2 as a modular Bayesian model with a deliberate cut: the fraction module's posterior feeds the effect module, and nothing feeds back | No | **Paper 1 Methods**, first |
| **R2. PALM-uni** | Extend the generative model with the missing error sources, then calibrate the posterior (tempering) on atlas donors | Yes, a new estimator | Research, after Paper 1 |

(User decision 2026-11-13: keep R1 and R2 only, R1 first.)

## 2. R1: cut-model framing (writing task, no new runs)

*Drafted 2026-11-13: `methods_cut_model.md` (Methods text, supplementary table, reviewer questions).*

- Methods paragraph: PALM is a two-module model. Module 1 is the generative model of section 2 of
  `model_dag.md`, used for π. Module 2 is the effect model (B–D), conditioned on draws of π. The cut stops
  module 2's misspecification (leakage, footprints) from distorting π, and stops module 1's misspecification
  from narrowing β's intervals.
- Evidence already in hand: full posterior β (October) vs cut (freeze v2): calls, coverage, false calls.
  Collect the numbers from `benchmarking.md` §5 and the freeze v1/v2 tables into one table for the
  supplement.
- Cite: Plummer 2015 (cuts); Jacob, Murray, Holmes & Robert 2017 (modules, misspecification); Carmona &
  Nicholls 2020 (semi-modular inference) as the bridge to R2.
- Say plainly that the router and guard are not part of the probabilistic model; they are a pre-registered
  selection rule, checked on held-out splits 20–29.

## 3. R2: the PALM-uni model

### 3.1 Generative model

Indices as in `model_dag.md` (i sample, c type, p program, g gene).

```
Profile of type c in sample i:
  log Ẽ_icg = log B_cg + b_icg + Σ_p δ_icp (G_pg + Γ_cpg) + r_icg
  E_icg     = softmax_g(log Ẽ_icg)

  b_icg ~ N(0, s²_cg)    reference donor variation; s²_cg from the atlas (between-donor variance
                         of pseudo-bulk log shares), fixed, never learned from PALM output
  Γ_cpg ~ N(0, τ²_Γ)     type-specific footprint deviation, shrunk to the shared G; only on the
                         top-k genes of each program
  r_icg ~ N(0, σ²_r)     off-program response; heavy-tailed option (Student-t)

Mixing:      f_ig = Σ_c π_ic ρ_c E_icg / Σ_c π_ic ρ_c       (ρ_c: RNA content, log-normal prior
                                                              centred on atlas UMIs per cell)
Observation: Y_ig ~ NB(L_i m_g f_ig, θ_g)                    (m_g: platform scale, prior as use_m)

Composition: η_ic = a_c + γ_c T_i + s_η ε_ic + Σ_k X_ik A_kc,  π_i = softmax(η_i)   (as now)
Activity:    δ_icp = κ_cp + β_cp T_i + λ_p u_ip + σ_d z_icp + Σ_k X_ik D_kcp
             β_cp = μ_p + v_cp,   v_cp ~ N(0, ω²_p)            (shared response + type deviation,
                                                              replaces the router's REML shrinkage)
```

Every readout comes from the same posterior: π, δ, β, calls (90% interval excludes 0), contributions
(Shapley split per draw), shared response (μ_p), design limit (posterior width under the prior at β = 0).

### 3.2 Computation

b_icg and r_icg are N × C × G latents: too many for NUTS. Plan:
1. **Collapse** them: to first order they add variance to log f_ig. Replace them by an inflated,
   sample-and-gene-specific variance in the stage-1 Gaussian summary (`twostage.sample_summaries`), i.e. a
   per-gene extra dispersion φ_ig = Σ_c w_icg² (s²_cg + σ²_r) with w the reallocation weights. This keeps
   stage 1's latent size at C + C·P per sample.
2. Γ: only top-k genes per program (k ≈ 20–50) and only lineage-level (myeloid / lymphoid) to start.
3. Keep the two-stage fit: stage 1 Laplace per sample, stage 2 NUTS on population parameters
   (`run_palm_fast`). New code in `bayes/uni.py`; it must reproduce the current model when
   s² = σ²_r = τ²_Γ = 0 and ρ, m fixed (unit test).

### 3.3 Posterior calibration

Extra terms may still not make the model right. Calibrate the posterior on data with a known answer:
- **Tempered likelihood**: multiply the log-likelihood by η ∈ (0, 1]. Choose η so the 90% intervals of β
  cover the truth at 0.90 on synthetic cohorts built from reference donors (the thinning of stage C:
  half 1 makes the bulk, half 2 the truth). One η per study, chosen like the current design check, from
  this study's fractions and design. References: Syring & Martin 2019; Grünwald & van Ommen 2017.
- **Check, not tune:** simulation-based calibration (Talts et al. 2018) on the rarest types
  (`v3_plan.md` item 15).
- Rule kept from `v4_plan.md`: never learn a prior or η from earlier PALM outputs.

### 3.4 Terms borrowed from cell2location (added 2026-11-16)

cell2location (Kleshchevnikov et al. 2022) shares module 1's structure; the comparison is in
`methods_cut_model.md` §2b. Three of its terms fit PALM-uni; its inference (mean-field VI) does not, because
calibration is the point here.

| Term | In cell2location | PALM-uni form | Targets | Why it may work where the free extra profile did not (v4_plan §10B) |
|---|---|---|---|---|
| **Batch background** s_bg | additive s_eg per experiment | μ_ig = L_i (Σ_c π_ic E_icg + s_bg), s_bg ≥ 0, sparse prior, **one per batch**, shared by all samples of the batch | W9 (missing types), part of W2 | shared across a batch it cannot follow a per-sample treatment effect; the earlier extra profile was free per sample |
| **Priors in physical units** | expected cells per location | prior on a_c (mean composition) from blood counts / CyTOF, converted to read fractions by RNA content ρ_c | W6, W4 | informative where bulk is weak (near-identical types); never learned from PALM output |
| **Reference regression with technology covariates** | NB regression of signatures with batch/technology effects | build B_cg by NB regression on the atlas with chemistry / platform covariates, report its standard errors (feed s²_cg of U1) | W6, U1 | separates platform from type identity at the reference step, before any bulk is seen |

Not taken: y_s (equal to L_i once f is normalised); the co-location factor prior (our composition has a design
regression instead); mean-field VI.

**Caveat for confounded designs.** Where group = batch (PTB: disease = run, W8), a per-batch background can
absorb the group difference. There it must be shared across batches or switched off, and the choice reported.

### 3.5 Program inputs and unknown programs (added 2026-11-16)

**What G is.** The programs define the question: β_cp is "change of program p in type c", and p means what
G_pg says. So programs must come from outside the data, and a database is a legitimate source. But a raw
database is a poor input, for three measured reasons:
1. *Member sets are not footprints.* Member genes (Hallmark, Reactome) include pathway components that do not
   change in expression; footprint genes (PROGENy, CytoSig) are the genes that respond. Holland et al. 2020
   found footprints better; PTB with member sets gave 0 of 48 calls (W7).
2. *Same footprint in every type.* Responses differ by lineage (e.g. IFN-γ in monocytes vs T cells); the
   design check with jittered weights (E22) shows the cost.
3. *Programs that contain markers.* Program genes that are also type markers in B make program shifts and
   composition trade off (the reason for `delta_given_eta`). E9 showed weight columns are not collinear with
   each other (|r| ≤ 0.43); collinearity with B has not been audited.

**Why not build a database of all possible programs.** Bulk carries a fixed amount of information per sample.
Each added program adds C parameters per sample, widens every interval (the design limit) and opens leakage
between programs. A universal catalogue would also make "program p" mean whatever the factorisation found. The
target is a small, question-specific, near-orthogonal panel (PTB uses 5), built from a curated library.

**Program library (P-steps).** A versioned resource, database-anchored and atlas-refined:
- P0 *audit* (offline): for each program, the share of weight on B markers, collinearity with the Jacobian
  of composition, overlap with other programs, and the design-check width. Programs that fail are flagged,
  not silently dropped.
- P1 *atlas footprints*: re-estimate G per lineage from perturbation atlases (huCIRA / Parse cytokines, CytoSig,
  Immune Dictionary for mouse checks), with the database set as a prior (Spectra-style gene-set-guided
  factorisation, Kunes et al. 2023; PLIER, Mao et al. 2019). This is the data source for U3 (Γ_cpg).
  Leakage rule: a footprint built from Parse is never scored on Parse; evaluate on Kang, PTB, COMBAT, with
  leave-cytokine-out on Parse.
- P2 *de-markering*: down-weight program genes in the top reference markers of any type (VIPER-style shadow
  correction for program–program overlap, the same idea for program–type overlap).

**Unknown programs: yes, but only in forms the data can identify.**

| Form | What it is | Identified? | Use |
|---|---|---|---|
| (a) residual variance | σ²_r of r_icg, collapsed into the stage-1 dispersion (U2) | yes, as variance | calibration: absorbs off-program response so known-program intervals stay honest (W2) |
| (b) **sample-level residual factors** | K factors W_kg with scores per sample, constrained orthogonal to the span of log B and the program Jacobian; K by held-out likelihood | yes, at bulk level (as RUV / PEER factors) | nuisance in the fit, plus a **diagnostic**: does a residual factor associate with T? If yes, report "response outside the program panel"; its genes go to enrichment and a reference lookup to *suggest* a type |
| (c) type-resolved unknown program | free W_kg with per-type activity | **no**: gene weights and cell type both unknown from bulk; the earlier free extra profile absorbed platform signal | not built |

Form (b) and the W9 residual-NMF (E12, Ivich et al. 2025) are the same machinery: one residual
decomposition serves "unknown program" (factor associates with T) and "missing type" (factor tracks
composition). In confounded designs (PTB), a factor that associates with group is confounded with run and is
reported as such.

**Paper 1 note (user decision needed).** The diagnostic in (b) does not change any estimate, so it could be
reported in Paper 1 as a model check next to freeze v2 without breaking the freeze. The nuisance-factor fit
belongs to PALM-uni.

## 4. Experiments (development splits only)

Each step is a go/no-go against freeze v2 and v5 on the same splits, scored by `causal_loop.py`
(new estimator `uni_U*`). Scores as in the held-out protocol: coverage (S1 ≥ 0.90 per dataset),
null calls (S2), power, effect r.

| Step | Adds | Data | Go if | Cost |
|---|---|---|---|---|
| **U0** | Tempering only, on the current model (no new terms). Report β from the model's own posterior | Kang, McFarland, Zhang; splits 0–2 | coverage ≥ 0.90 on all three, null calls ≤ v2 + 0.02, power ≥ 0.5 × v2 | ~1 day code, overnight run |
| U1 | + reference donor variance (collapsed b) | as U0 | U0 rules, and η closer to 1 than in U0 | ~3 days |
| U2 | + off-program residual r | as U0 + Tahoe trametinib, RMC-6236 | McFarland idasanutlin null calls < v2's 0.05 | ~2 days |
| U2b | + sample-level residual factors (§3.5 b), orthogonal to B and the program Jacobian | as U2 + a planted off-panel response (simulated program outside G) | known-program null calls ≤ U2 + 0.01 and power ≥ U2 − 0.02; planted response found by the T-association diagnostic in ≥ 8/10 splits, false diagnostic ≤ 1/10 on Zhang | ~4 days |
| U2c | + batch background s_bg (§3.4) | COMBAT, McFarland, a planted missing type (drop one type from B) | missing-type share recovered (r ≥ 0.6 with the dropped type); null calls not worse than U2 | ~3 days |
| P0 | program library audit (offline, no fit) | all program sets in `data/` | — (a report; flags programs for P2) | ~1 day |
| U3 | + lineage footprints Γ, data from P1 (atlas footprints, leakage rule §3.5) | Kang, PTB programs; Parse leave-cytokine-out | PBMC power ≥ v5 | ~1 week |
| U4 | + RNA content ρ, platform m priors; composition prior from blood counts / CyTOF; B by NB regression with technology covariates (§3.4) | COMBAT real bulk | null calls ≤ 0.01 on real bulk, as v2; fraction error vs CyTOF below v2 | ~1 week |
| U5 | Full 13 datasets, splits 20–29, vs v5 | all | S1 on every dataset; null-call difference upper CI ≤ +0.02; power difference > 0 | 1–2 nights |

**Stop rules.**
- U0 is the decisive cheap test. If tempering alone cannot reach coverage 0.90 without η so small that
  power collapses (< half of v2), the problem is structural. Go to U1–U2 once; if they do not move η
  towards 1, stop R2 and write up R1 only.
- Every step gets an ablation row (with / without its term) in a results file
  `docs/palm_uni_results.md`; no step is kept for a gain inside split noise (0.03, as in the v4 protocol).
- No retuning on 20–29 after U5 starts.

## 5. Held-out confirmation (only with the user's agreement)

Blocks already spoken for: 20–29 (v2, run), 30–39 (reserved, v5 candidate), 40–49 (v4, run),
50–59 (association section). **Proposed for PALM-uni: 60–69**, never run (confirm before freezing).
- Freeze `bayes/uni.py` and the decision rule (git tag) before any 60–69 output.
- Primary rule: S1 on every dataset; null-call difference vs the then-current default, upper CI ≤ +0.02;
  power difference > 0 (two-stage cluster bootstrap over datasets and splits, as in the v4 protocol).
- If v5 has not been confirmed on 30–39 by then, decide the order with the user first: two candidates
  must not share one held-out block unless both are pre-registered together.

## 6. Risks

1. **Identifiability.** b, r and δ compete to explain one bulk sample per patient; the priors decide.
   Mitigation: s² fixed from the atlas, σ²_r and τ²_Γ with tight priors, report them as assumptions next to
   the design limit.
2. **Collapsing is an approximation.** The inflated variance ignores that b and r are shared across genes
   of one type. Check against full latents on a small subset (N ≈ 20, 200 genes).
3. **Tempering trades power for coverage.** If η must be small, PALM-uni will be honest but weaker than v5.
4. **Same lesson as v4:** a more efficient estimator can be more fragile under misspecification. The
   held-out safety rules (S1/S2) come first, not power.
5. **Competitors already unify.** BLADE and TCA are joint models with per-sample type-specific expression.
   PALM-uni's difference must be shown: program structure (G) and posterior calibration on atlas truth.

## 7. Code

| Piece | Where |
|---|---|
| Model and two-stage fit | `src/program2cell/bayes/uni.py` (reuses `twostage.sample_summaries`, `model._NegBin2`) |
| Atlas variance s²_cg | `src/program2cell/core/` next to the reference builder |
| η selection | reuse the design-check cohort builder in `core.unmix` |
| Estimator entry | `program_deconvolution(estimator="uni")`; `causal_loop.py` ESTIMATORS `uni_U0` … `uni_U5` |
| Tests | reduction to the current model at zero variance; η = 1 equals the untempered fit; parity on a toy cohort |

## 8. References

- Plummer M. Cuts in Bayesian graphical models. Stat Comput 2015.
- Jacob PE, Murray LM, Holmes CC, Robert CP. Better together? Statistical learning in models made of
  modules. arXiv 1708.08719, 2017.
- Carmona CU, Nicholls GK. Semi-modular inference: enhanced learning in multi-modular models by tempering
  the influence of components. AISTATS 2020.
- Kleijn BJK, van der Vaart AW. The Bernstein–von Mises theorem under misspecification. Electron J Stat 2012.
- Syring N, Martin R. Calibrating general posterior credible regions. Biometrika 2019.
- Grünwald P, van Ommen T. Inconsistency of Bayesian inference for misspecified linear models, and a
  proposal for repairing it. Bayesian Anal 2017.
- Kleshchevnikov V et al. Cell2location maps fine-grained cell types in spatial transcriptomics. Nat
  Biotechnol 2022.
- Holland CH et al. Robustness and applicability of transcription factor and pathway analysis tools on
  single-cell RNA-seq data. Genome Biol 2020.
- Kunes RZ et al. Supervised discovery of interpretable gene programs from single-cell data (Spectra). Nat
  Biotechnol 2023.
- Mao W et al. Pathway-level information extractor (PLIER). Nat Methods 2019.
- Ivich A et al. 2025 (residual NMF for missing cell types; see `weak_points_plan_2026-11-12.md` W9).
- Talts S, Betancourt M, Simpson D, Vehtari A, Gelman A. Validating Bayesian inference algorithms with
  simulation-based calibration. arXiv 1804.06788, 2018.
- Andrade Barbosa B et al. Bayesian log-normal deconvolution for enhanced in silico microdissection of bulk
  gene expression data (BLADE). Nat Commun 2021.
- Rahmani E et al. Cell-type-specific resolution epigenetics without the need for cell sorting or
  single-cell biology (TCA). Nat Commun 2019.

*Citation details written from memory on 2026-11-13; verify before they enter a manuscript.*
