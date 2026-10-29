# PALM method specification (frozen core, 2026-10-27)

*The method as it will be written up in the paper. It records what is fixed, what each step does, its
defaults, and the evidence behind each choice. Open items O1–O5 were settled by the 2026-10-27 overnight
run (section 7). The second round (section 7b, 2026-10-28) adds:
- the leakage-aware effect width;
- proxy calibration for PALM;
- Kang IFN-β validation;
- CIBERSORT;
- a fix to `estimate_delta`.

Its results are pending. For the plain-language explanation see `palm_method_explained.md`. For metrics and results
see `benchmarking.md`.*

---

## Method freeze v1 (2026-10-29; git tag `method-freeze-v1`)

**What is frozen** (the method reported in Paper 1):

| Step | Setting |
|---|---|
| Fractions | full PALM, `run_palm_fast(likelihood="nb", activity="common", use_m=False)`, 50 posterior draws (PALM-lite or external fractions in module mode) |
| Activity | posterior reallocation + `estimate_delta`; calibration on reference donors (`calibration_from_table`, 5 donor folds), `effect_interval="lk"`, level 0.9 |
| Effects | `program_deconvolution` defaults: `estimator="auto"`, `UNMIX_DEFAULTS = dict(pev=True, misfit=True)`, `sim_widen=True`, `n_sim=10`; `covariates` / `groups` when the design has them |
| Targets | coverage ≥ 0.85, r ≥ 0.9 (where max \|β\| ≥ 0.1), calls on null pairs ≤ 0.1, responder AUROC ≥ 0.95, truth-based TPR ≥ 0.9 |

**Rules after the freeze:**
- Defaults and the code paths they use do not change. A bug fix that changes results is recorded here, with
  old and new numbers.
- New engines or options are opt-in only and reported as post-freeze.
- **Held-out splits 10–12** of every dataset are run once with the frozen setting, and their numbers are the
  ones the paper reports. Splits 0–9 stay for development. The runner is
  `scripts/experiments/heldout_chain.sh`. Its rows are the s10–s12 files in the usual directories, and the
  summaries (`causal_loop.py summary heldout`, `practice_vs_palm.py summary heldout`) go to
  `benchmarks/dev/heldout_summary.txt`. Development summaries exclude splits ≥ 10.
- Competitor and current-practice comparisons (C1/C2, `scripts/experiments/practice_vs_palm.py`) use the
  same held-out splits.

**Known limits, reported and not fixed:**
- Minor cell types with little spread in composition (Kang B/NK) get no calls: their effects can't be
  identified from the bulk.
- Coverage when the response varies between patients is untested; this is for the disease arm.

---

## 0. Summary

PALM deconvolves gene-program activity per cell type from bulk RNA-seq in three steps:

1. **Fractions.** Estimate cell-type read fractions π, with uncertainty, from one of three engines:
   - full PALM (Bayesian), the accurate option;
   - PALM-lite (multinomial EM), the fast option;
   - any external method, used as a module.
2. **Activity.** Split each bulk sample's reads between cell types in proportion to π × B ("posterior
   reallocation"). Then estimate each type's program shift δ against its reference profile.
3. **Calibration and effects.**
   - Map δ̂ to the truth scale with a regression learned on reference donors.
   - Give conformal intervals for activity.
   - Estimate treatment effects β per cell type and program, with intervals that account for fraction,
     calibration and residual uncertainty.

What is *not* in the method: the joint model's own δ estimate is not the reported activity. Reallocation on
its fractions beats it on the δ truth (section 6).

---

## 1. Inputs

| Symbol | Shape | Meaning |
|---|---|---|
| Y | N × G | Bulk counts, after gene selection |
| B | C × G | Reference profiles: per cell type, the mean over reference donors of per-donor pseudobulk shares (rows sum to 1) |
| G | P × G | Program weights from the prior set |
| T | N | Treatment / contrast indicator (optional; covariates X optional) |
| Reference donors | — | Single-cell or per-donor pseudobulk counts. Used for B and for the calibration set (section 4). They must not overlap the bulk donors. |

**Gene selection:** Fisher (Bayesian D-optimal) selection, `select_genes(method="fisher")`, with
`max_genes=1000` in the benchmark.

**Program prior sets:**
- **Fitting:** PROGENy top-500, 7 pathways, used only to fit fractions.
- **Reallocation:** any prior set; the user chooses. Benchmarked: PROGENy-7, PROGENy-14 (top-200),
  CollecTRI immune TFs (26) and Hallmark immune (15).
- Fitting once and reallocating under another set matches a native fit at about 1/15 of the cost
  (section 7, O1).

---

## 2. Step 1: fractions

All engines return read fractions π (N × C) on the modelled genes. Where possible they also return S draws
or an approximate posterior.

### 2a. Full PALM (accurate engine, default)

- **Function:** `run_palm_fast(inputs, likelihood="nb", activity="common", use_m=False)`.
- **Sampler:** joint two-stage fit, NUTS, 2 chains × 500 warm-up + 500 draws, `target_accept=0.95`.
- **Model:**

      y_ng ~ NB(L_n · Σ_c π_nc softmax(log B_c + δ_nc G)_g, θ)

  δ is shared across types through the common activity prior. The treatment enters δ through β.
  π comes from a softmax over η.
- **Output used downstream:** posterior draws of π, thinned to 100.
- **Evidence:** best fractions on PBMC (RMSE 0.062 against 0.096 for cellGeometry). Ties on OneK1K
  (0.049 vs 0.048). As a host for reallocation it gives the highest activity ρ on both datasets.
- **Likelihood: NB (decided, O2).** quasi+sandwich is equivalent on every metric, so it remains an option.

### 2b. PALM-lite (fast engine)

- **Model:** multinomial maximum likelihood, y_n ~ Mult(n_n, π_n B), fitted by EM:

      π_nc ← π_nc · Σ_g (y_ng / (π_n B)_g) B_cg / n_n

  Run to `max |Δπ| < 1e-10` or 500 iterations.
- **Uncertainty:** none from the EM itself. Fraction error enters through `calib fr` (section 4.1, step 5).
  Bootstrap draws were tested and add nothing (O4).
- **Evidence, PBMC, 10 splits:** fraction RMSE 0.076 and activity ρ 0.830, in 5–15 s per split. That is
  cellGeometry's level, against 0.117 and 0.791 for NNLS.
- **To do:** vectorise or accelerate the EM (SQUAREM). Target: under 1 s.

### 2c. External fractions (module mode)

- **Input:** any method's fractions (cellGeometry, BayesPrism, CIBERSORTx, …).
- **Unit harmonisation:** convert to *read* fractions on the modelled genes.
  - Methods that report cell fractions are converted with each type's mean library size from the
    reference, then `to_modelled_read_fractions`.
  - This conversion is required. Without it cellGeometry's fraction RMSE is 0.229; with it, 0.093.
- **Uncertainty:** draws if the method provides them. Otherwise pass the method as a fraction function so
  `calib fr` can run it on the calibration bulks. If neither is possible, the intervals undercover
  (NNLS: 0.78 instead of 0.88).
- **Design:** `research_plan.md` §3.8. The planned API is `pb.program_deconvolution(...)`.

---

## 3. Step 2: activity by posterior reallocation

For each fraction draw π^(s):

    w_ncg = π_nc B_cg / Σ_c' π_nc' B_c'g                 (share of gene g's reads from type c)
    X_ncg = y_ng · w_ncg                                (reallocated counts)
    δ̂^(s)_nc = estimate_delta(X_n c ·, B_c, G)           (multinomial ML shift of X_nc against B_c,
                                                          ridge 1e-3, Newton steps)

**Point estimate:** the median over draws. The draws are kept for step 3.

**Fixed choices:**
- The profiles used for splitting are the reference B, not program-shifted profiles. Shifted profiles were
  worse in every variant tried (section 6).
- `estimate_delta` is the single estimator used for PALM, for every baseline and for the truth. Methods
  therefore differ only in how they split the reads.
- `estimate_delta` uses Newton steps with backtracking line search (fixed 2026-10-28, section 7b).

**Evidence:**
- Gain over naive (whole bulk against each B_c): +0.065 ρ on PBMC and +0.09 to +0.106 on OneK1K
  (δ truth).
- This is close to the oracle, i.e. reallocation with the true π: 0.869 on PBMC, 0.750 on OneK1K.

**Known limit:**
- Reallocation attenuates. Own-type gain is about 0.4 even with the oracle π.
- It leaks the other types' shifts into minor types.
- This is why step 3 recalibrates rather than reporting δ̂ raw.

---

## 4. Step 3: calibration, intervals, effects

### 4.1 Calibration set (built from the reference, no extra data)

1. **Donors.** Take the reference donors that are not bulk donors and have every cell type at ≥ 20 cells.
   Use one specimen per donor and at most 300 donors.
2. **Folds.** Split the donors into K = 5 folds. For fold k, compute B_k from the *other* reference
   donors only.
3. **Thinning.** Binomially thin each calibration specimen's per-type counts 50/50.
   - Half 1, summed over types, is a synthetic bulk. Its true read fractions are π.
   - Half 2 gives the truth: d = `estimate_delta(half 2, B_k)`.
4. **Estimate.** Compute d̂ = reallocation of half 1 with the true π and B_k.

5. **Fraction error.**
   - Engines with draws (PALM): d̂ uses the true π. Fraction error is added at test time from the draws
     (4.3).
   - Engines without draws (PALM-lite, external): d̂ uses the engine's *own* fractions, estimated on the
     synthetic bulk with B_k (`calib fr`). The calibration map and its residuals then include fraction
     error.

### 4.2 Calibration map

- **Model:** per (c, p), a ridge regression (λ = 1e-3, intercept unpenalised) d_c,p ~ 1 + d̂_c,p.
  - This "own type" form is the default: one slope and intercept.
  - The "all types" form (d_c,p ~ 1 + d̂_·,p) performs the same (coverage 0.87 vs 0.88, FSR 0.07 vs
    0.09). It is kept as an option.
- **Out-of-fold residuals:** computed with the same 5 donor folds.
- **Final map:** fitted on all calibration donors and applied to every draw δ̂^(s).

### 4.3 Activity: point and 90% interval

    δ̃ = f(median δ̂)
    half-width = sqrt( q_c,p² + h_n,c,p² )

- q_c,p is the split-conformal quantile of |out-of-fold residual| at level ⌈(n+1)·0.9⌉/n.
- h is the half-width of the 5–95% range of f(δ̂^(s)) over fraction draws. It is 0 when an engine has no
  draws.

### 4.4 Effects

- **Point:** β̃ = design(δ̃). This is the treatment contrast per (c, p), with covariates if given.
- **Variance, three parts:**
  1. Bootstrap over calibration donors, 200 refits of f: coefficient uncertainty.
  2. Residual variance σ²_c,p × (1/n₁ + 1/n₀).
  3. Variance of design(f(δ̂^(s))) over fraction draws.
- **Interval: stratified conformal (decided, O3).**
  - For each type c' and program p, split random two-thirds subsets of the calibration donors at the median
    of the true d_c',p.
  - The contrast of out-of-fold residuals between the halves gives one pseudo-effect error for every
    (c, p). This includes leakage from a real effect in c'.
  - q^β_c,p is the 90% conformal quantile of |pseudo-effect error|, rescaled to the test group sizes.
  - Half-width = sqrt( q^β² + z²·(bootstrap var + fraction-draw var) ).
- **Calls:** an effect is called when its interval excludes 0.

**Evidence (PALM NB, 4 priors × 10 splits):**

| | Effect cov90 | False sign rate (calls) |
|---|---|---|
| PBMC, normal interval | 0.70 | 0.07 (33) |
| PBMC, stratified | 0.88 | 0.05 (19) |
| OneK1K (near-null contrast), normal | 0.81 | 12.8 false calls |
| OneK1K, stratified | 1.00 | 1.8 false calls, intervals about 2.7× too wide |

**Refinement (implemented 2026-10-28; full results pending):** the leakage-aware width (`lk`, section 7b).
It is a candidate to replace the stratified interval. Point estimates are unchanged by any interval choice.

### 4.5 Unmixed effects (`estimator="unmix"` / `"auto"`; added 2026-10-28, `core/unmix.py`)

**Problem found on MIX-Seq (McFarland, idasanutlin):** p53 rises in 3 of 6 lines (+0.36 to +0.68), but the
reallocated effect is +0.33 to +0.46 in **every** line.
- **Cause:** reallocation splits each gene's reads with the unshifted profiles B. A shift in type k multiplies
  every type's reallocated profile by nearly the same mixture-averaged factor. On these mixtures a type keeps
  about 30% of its own shift; the rest spreads evenly.
- **Consequence:** no per-sample map can undo this (one bulk does not say which type moved). Calibration
  rescales it (AUROC 0.89, 1 of 3 responders called); planted shifts in the calibration set (section 4.1
  option `n_plant`) do not fix it either (tested: coverage up, localisation not).
- **What identifies it:** across samples, the leakage depends on the fractions, which vary.

**Estimator:**
1. **Leakage operator** A_n = ∂(reallocated activity)/∂(true shift), per sample, in closed form (linearisation
   of estimate_delta at the reference):
   A_nc·k· = π_nk S_c⁻¹ [G diag(v) Gᵀ − (Gv) m_kᵀ − m_c (Gv)ᵀ + (1ᵀv) m_c m_kᵀ],
   v = B_c ∘ B_k / (π_n B), m_k = G B_k, S_c = G diag(B_c) Gᵀ − m_c m_cᵀ.
   - All programs jointly (cross-program leakage: shared genes and the joint fit). Without it, simulated errors
     were 2× larger.
   - Checked against finite differences (max error 0.001); a lone type gives the identity.
2. **Regression across samples**, all programs stacked:
   d̂_ncq = a_cq + Σ_kp A_ncqkp (γ_kp + T_n β_kp [+ covariates, donor effects]) + e.
   - β_kp = m_p + u_kp: type effects shrunk toward a shared effect by a ridge on u (and on the baseline
     deviations), sized by REML. A shared response selects strong shrinkage.
3. **Nonlinear correction:** the linear operator under-predicts leakage by about 25% for drug-sized shifts.
   Chord iterations with the exact forward model (shifted profiles mixed, then reallocated):
   working response d̂ − (F(Tβ) − F(0)) + A Tβ. Only the treatment part enters (the baseline is confounded with
   the intercepts). Stops if a step grows (keeps the last contracting step).
4. **Intervals:** cluster bootstrap (donors, else samples within arms), a fraction draw per replicate.
   Widened to max(t, q) standard errors, where q is the 90% quantile of studentised errors in simulated cohorts
   (next point).
5. **Design check (`design_check`):** about 10 synthetic cohorts with **this study's** fractions, treatment and
   size, built from the reference specimens' real per-type counts, with planted null / shared / localised
   effects sized to the study (2 × the largest calibrated effect per program). It measures the RMSE of
   unmixing vs calibrated reallocation per program, and q. `auto` unmixes the programs where unmixing wins.

**Ruled out on the way:**
- GLS with a per-sample random-shift covariance (τ² A Aᵀ + σ² I): worse on MIX-Seq (r 0.72 vs 0.94).
- Sample-level shared random effects: worse.
- Cross-validated shrinkage: too noisy with few samples.

**First results (split 0, PROGENy-7, PALM posterior fractions):**

| | RMSE | r | Effect cov90 | p53 responder AUROC | p53 calls (3 true) |
|---|---|---|---|---|---|
| MIX-Seq idasanutlin, calibrated (`diff`) | 0.084 | 0.89 | 0.62 | 0.89 | 1 |
| MIX-Seq idasanutlin, `unmix` | 0.056 | 0.94 | 0.83 | 1.00 | 3 (+ BICR31, true +0.12) |
| MIX-Seq idasanutlin, `auto` (NFkB, p53 unmixed) | 0.051 | 0.95 | 0.67 | 1.00 | 3 |

- **When it does not help:** with few samples (Kang: 14) or tiny effects (PBMC), unmixing is noisier than
  calibrated reallocation (Kang JAK-STAT r 0.83 vs 0.98). This is why `auto` exists.
- **Cost:** about 5 minutes per study (bootstrap with nonlinear refits, plus the design check).

#### 4.5b Development loop on real perturbations (2026-10-28 evening, `scripts/experiments/causal_loop.py`)

Scoreboard: every estimator (none removed) on four real datasets with a known answer, splits 0–2.

**Targets:**
- coverage of 90% intervals ≥ 0.85;
- r ≥ 0.9, only where the truth has effects (max |β| ≥ 0.1; Zhang's truth is null, max |β| 0.05);
- calls on truly null pairs (|β| < 0.05) ≤ 0.1;
- known program: responder AUROC ≥ 0.95, and responders called with the right sign ≥ 0.9.

**Finding 1: the treatment response outside the programs is large, and it is the main bias.** The bulk log
fold change per gene, minus the one the true program shifts predict (forward model), has a spread well beyond
sampling noise (`offprogram_tau`):

| | τ per type and gene (log) |
|---|---|
| idasanutlin | 0.69 |
| trametinib | 0.58 |
| Kang | 1.0 |
| Zhang | 0 |

- It is the drug's response, the same in every treated sample, so it does not average out. Its projection
  leaks into reallocation and is amplified by unmixing. This is why the same pairs were missed in every split
  (LNCaP NFkB/TNFa, SNU1079 Hypoxia, BICR31 JAK-STAT).
- **Fix (`misfit=True`):** draw independent N(0, τ²) off-program log shifts per type and gene, push them through
  the exact forward model and the unmixing fit (fixed shrinkage, linear), and compare with what they do to the
  realised truth. The RMS error per (type, program) is added to the interval in quadrature (`misfit_sd`).
- The design check simulates cohorts with the same response, so `auto`'s choice and the widening of calibrated
  programs see it too. The unmixed intervals then use the t quantile alone (no simulated q).
- Split 0, unmix with PEV, without → with misfit:

  | | cov | null calls | known program |
  |---|---|---|---|
  | idasanutlin | 0.71 → 0.95 | 0.30 → 0.04 | 3 p53 responders still called |
  | trametinib | 0.81 → 1.00 | 0.28 → 0.06 | MAPK called in every line |
  | Kang | 0.89 → 0.93 | 0.33 → 0.00 | JAK-STAT called in Monocyte, T |

**Finding 2: shrinkage bias.** A type whose effect differs from the others (trametinib RCC10RGB) is pulled to the
shared mean, and a bootstrap at a fixed λ does not see that. **Fix (`pev=True`):** se = max(bootstrap sd,
prediction-error sd), where the prediction-error variance is σ²(Z'Z+Λ)⁻¹ at the REML λ.

**Finding 3: Kang's minor types are not identifiable from bulk.**
- B cells are 4% of reads and keep 8% of their own shift; NK cells keep 22%. Fractions barely vary across
  14 samples (sd 0.01–0.07), and the mean operator's smallest singular value is 0.01.
- The forward model fits Kang almost exactly (contrast residual ≤ 0.005). Reallocation still turns any per-type
  JAK-STAT effect into about 0.32 in every type.
- Calibrated reallocation is close only because the true effects happen to be nearly shared (0.26–0.35). Honest
  intervals for B and NK are wide, so TPR ≤ 0.5 there is the right answer, not a failure of validity.
- Tried and kept as options (not defaults), none of which made B or NK identifiable:
  - separate REML λ for the baseline and the effect deviations (`separate=True`), or per program
    (`"program"`, coordinate ascent);
  - simulation-calibrated intervals in place of the calibrated ones (`sim_widen="replace"`; wider on Kang).

**Direct engine, two options** (`direct_kw`):
- `weights="nb"`: quasi-likelihood. A weighted Poisson likelihood with each sample's depth profiled out, which is
  unbiased for any fixed weights, with NB weights 1/(1+φ_g μ_g). The multinomial lets a few deep genes steer
  every effect. Kang RMSE 0.26 → 0.11, r 0.57 → 0.86; idasanutlin 0.11 → 0.10.
- `gene_offset=True`: a per-gene log offset shared by the types, a batch correction. It recovers a planted
  gene-wise batch difference, but on real data it is worse (Kang RMSE 0.16): the mismatch is type-specific.
- Coverage is still 0.3–0.6. The off-program response biases the direct likelihood too, and it has no misfit term
  yet.

**Scoreboard, splits 0–2** (`benchmarks/dev/causal_loop_summary.txt`; PALM posterior fractions throughout,
trametinib's earlier PALM-lite rows kept in `causal_loop/lite/`):
- The TPR target is truth-based: known-program pairs with |β| ≥ 0.1, called with the right sign.
- The label "MAPK down in every line" is wrong for LNCaP: its true MAPK effect is −0.04, so calling it would be
  a null call. The label-based TPR stays in the table.

| | coverage | r | null calls | AUROC | TPR (truth) | pass |
|---|---|---|---|---|---|---|
| idasanutlin, `diff` | 0.63 | 0.89 | 0.01 | 0.89 | 0.25 | – |
| idasanutlin, `unmix` | 0.83 | 0.92 | 0.11 | 1.00 | 0.92 | – |
| idasanutlin, **`auto_m`** | 0.92 | 0.92 | 0.05 | 1.00 | 0.92 | **yes** |
| idasanutlin, `unmix_m` | 0.96 | 0.92 | 0.06 | 1.00 | 0.92 | yes |
| trametinib, `diff` | 0.67 | 0.86 | 0.00 | – | 0.07 | – |
| trametinib, `unmix` | 0.91 | 0.92 | 0.28 | – | 1.00 | – |
| trametinib, **`auto_m`** | 0.97 | 0.91 | 0.035 | – | 1.00 | **yes** |
| Zhang (null truth), **`auto_m`** | 0.99 | – | 0.01 | – | – | **yes** |
| Kang, `diff` | 0.99 | 0.96 | 0.00 | – | 0.25 | – |
| Kang, `auto_m` | 1.00 | 0.96 | 0.00 | – | 0.00 | – |
| Kang, `unmix_m` | 0.96 | 0.49 | 0.00 | – | 0.50 | – |
| Kang, `direct_nb` | 0.57 | 0.85 | 0.36 | – | 1.00 | – |

- **Validity:** `auto_m` meets every validity target on all four datasets (coverage ≥ 0.92, null calls
  ≤ 0.05).
- **Power:** it meets every target on three of them. Kang misses only on power, for B and NK (finding 3). The
  only engine that calls them, `direct`, does so with invalid intervals.
- Taking the narrower of the `unmix_m` and `auto_m` intervals per pair (emulated offline) does not change Kang
  (TPR 0.5, r 0.52).
- **Default since 2026-10-29 (`program_deconvolution`, `UNMIX_DEFAULTS`):** `estimator="auto", unmix_kw=dict(pev=True, misfit=True)`. Fast setting: `estimator="diff"`. Experiment scripts pin `pev=False, misfit=False` for the older scoreboard rows.

**Scoreboard estimator names** (`causal_loop.ESTIMATORS`):
- diff, ols_pair;
- unmix;
- auto (with `sim_widen`), auto_nowiden, auto_pair, auto_sim, auto_sim_pair;
- unmix_m, auto_m, auto_m_pair (pev + misfit);
- direct, direct_pair, direct_nb.

---

## 5. Outputs

| Output | Shape | Notes |
|---|---|---|
| Fractions π | N × C | Median and 90% interval (draws, if the engine has them) |
| Activity δ̃ | N × C × P | Calibrated, with a 90% conformal interval |
| Effects β̃ | C × P | Per contrast, with a 90% interval and a call flag |
| Attribution / intrinsic vs composition | — | As in `palm_method_explained.md` §3.3–3.4, computed from δ̃ and π |
| Diagnostics | — | Sampler checks (full PALM), calibration set size, conformal q per (c, p) |

---

## 6. Ruled out (with evidence; for the paper's "what did not work")

| Variant | Result | Where |
|---|---|---|
| The joint model's own δ as activity | Lower than reallocation on the δ truth. It wins on the ULM truth by +0.10. **Kept as a secondary check, not reported as the activity.** | benchmarking §7 |
| Two-step fit (δ given η) as default | Does not improve fractions on OneK1K | benchmarking §7 |
| Independent (per-type) activity prior | Worse than the common prior | night2 run |
| EM and iterated reallocation (program-shifted split profiles) | Worse fractions and activity | dev_bench rows |
| PALM-lite with program-adjusted profiles | RMSE 0.087 vs 0.076; ρ 0.826 vs 0.830 | lite test |
| De-mixing by inverting the leakage Jacobian | No gain under either ridge | benchmarking §7 |
| Inverse calibration of effects | Effect r falls to 0.03 | benchmarking §5c |

---

## 7. Open items: results of the 2026-10-27 overnight run

**Setup.** `scripts/experiments/openitems.py`, dev splits 0–9, PBMC (COVID vs healthy) and OneK1K (sex
contrast). There are four prior sets: PROGENy-7, PROGENy-14, Hallmark and CollecTRI. Fraction sources are
PALM NB, PALM quasi+sw, PALM-lite, NNLS, cellGeometry and the true fractions. PALM was fitted once with
PROGENy-7 and its fractions were reallocated under each prior set. Summaries are in
`benchmarks/dev/openitems/summary_<dataset>__<prior>.txt`.

| ID | Decision | Status |
|---|---|---|
| O1 | PALM is fitted **once** with PROGENy-7. The prior set is applied afterwards, at reallocation. No default prior; PROGENy-7 is the fitting prior. | Decided |
| O2 | **Joint NB.** It ties quasi+sw everywhere and is slightly better on type-specific ρ on OneK1K. | Decided |
| O3 | **Stratified conformal effect intervals.** They fix coverage when real effects exist. They are too wide when effects are null (OneK1K); see the refinement below. | Decided, refinement open |
| O4 | Engines without draws: **calibrate with the engine's own fractions (`calib fr`).** No bootstrap. | Decided |
| O5 | OneK1K competitors are complete; see the table below. | Done |
| O6 | Kang IFN-β validation. Design in section 7b. | Running (2026-10-28) |

### O1: fitting once, and what richer priors do

**Transfer is valid.** On PBMC Hallmark split 0:
- PALM's native Hallmark fit took 3.5 h and gave ρ 0.787, fraction RMSE 0.067.
- PALM fractions fitted with PROGENy-7 and reallocated under Hallmark gave ρ 0.789, RMSE 0.068.

**Paired difference, PALM NB minus the other engine.** Raw reallocation activity ρ across samples, 10
splits, 95% CI:

| Dataset | Prior | vs cellGeometry | vs PALM-lite | vs NNLS |
|---|---|---|---|---|
| PBMC | PROGENy-7 | +0.017 [+0.009, +0.026] | +0.004 [−0.003, +0.011] | +0.072 |
| PBMC | PROGENy-14 | +0.007 [+0.004, +0.009] | +0.005 [+0.002, +0.008] | +0.030 |
| PBMC | Hallmark | +0.002 [−0.003, +0.007] | +0.011 [+0.005, +0.017] | +0.165 |
| PBMC | CollecTRI | +0.012 [+0.008, +0.016] | +0.005 [+0.001, +0.009] | +0.066 |
| OneK1K | PROGENy-7 | +0.025 [+0.018, +0.033] | −0.008 [−0.019, +0.003] | +0.082 |
| OneK1K | PROGENy-14 | +0.020 [+0.012, +0.029] | +0.019 [+0.009, +0.029] | +0.159 |
| OneK1K | Hallmark | +0.001 [−0.007, +0.010] | +0.045 [+0.036, +0.055] | +0.197 |
| OneK1K | CollecTRI | +0.014 [+0.007, +0.020] | +0.006 [−0.001, +0.012] | +0.045 |

**Reading:**
- **Fraction error matters a lot for richer priors, but only until fractions are "good".**
  - Against NNLS (fraction RMSE 0.07–0.15), PALM gains up to +0.20 ρ on Hallmark.
  - Against cellGeometry and PALM-lite (RMSE 0.04–0.10), the gain is 0 to +0.025 on ρ.
  - On type-specific ρ and effect r the differences are mostly within CI. PBMC PROGENy-7 is the
    exception: +0.05 type-specific ρ vs cellGeometry.
- The split-0 impression that "PALM's lead grows with richer priors" held **against NNLS only**.
- PALM is never worse than any engine on ρ. It is significantly better than cellGeometry in 6 of 8
  settings, and better than or tied with PALM-lite in 7 of 8.
- **Consequence for the paper:**
  - Most of the activity gain comes from steps 2–3 (reallocation and calibration), which work with any
    good fractions.
  - PALM's own fit adds a small, consistent gain and the fraction uncertainty.
  - The module framing (section 2c) is the stronger story.

### O2: likelihood

PBMC and OneK1K, pooled over the 4 priors × 10 splits, `calib` row:

| | PBMC ρ | PBMC spec | PBMC act cov | PBMC eff cov (strat) | OneK1K ρ | OneK1K spec |
|---|---|---|---|---|---|---|
| joint NB | 0.809 | 0.519 | 0.855 | 0.877 | 0.684 | 0.491 |
| joint quasi+sw | 0.810 | 0.514 | 0.859 | 0.859 | 0.682 | 0.469 |

They are equivalent. NB is the default because it is simpler (no sandwich) and slightly better on
OneK1K type-specific ρ.

### O3: effect intervals

Pooled over the 4 priors × 10 splits, PALM NB, own-type calibration:

| Effect interval | PBMC cov90 | PBMC iscore | PBMC FSR (calls) | OneK1K cov90 | OneK1K iscore | OneK1K calls (all false*) |
|---|---|---|---|---|---|---|
| raw (draws only) | 0.15 | 0.67 | 0.17 (58) | 0.22 | 0.14 | 49 |
| normal (section 4.4) | 0.70 | 0.27 | 0.07 (33) | 0.81 | 0.054 | 12.8 |
| random-split conformal | 0.70 | 0.27 | 0.07 (33) | 0.81 | 0.053 | 12.6 |
| **stratified conformal** | **0.88** | **0.25** | **0.05 (19)** | 1.00 | 0.146 | 1.8 |

\*On OneK1K every true |β| ≤ 0.1, so every call is a false discovery (FDR = 1). The sex contrast is close to
null.

- **Random-split conformal equals the normal interval.** The shortfall therefore comes from error that
  moves with the effect, i.e. leakage and nonlinearity, not from heavy tails.
- **Stratified conformal fixes PBMC** and improves its interval score. It removes about 85% of OneK1K's
  false calls.
- **But it is about 2.7× too wide when effects are null.** The pseudo-contrasts split donors at the median
  of a type's true activity, which is a larger effect than most real contrasts.
- **Refinement (to do):** a leakage-aware width. On the calibration set, regress the pseudo-contrast
  residual of (c, p) on the stratifying contrast of each type c'. At test time, add
  Σ_c' L_cc'·|β̃_c'| instead of the worst-case quantile. Effects that are actually small then get narrow
  intervals.
- Own-type calibration is kept as the default. The "all types" variant lowers PBMC ρ by about 0.01 and
  does not change coverage.

### O4: fraction uncertainty for engines without draws

Activity cov90, pooled over priors:

| Engine | PBMC calib | PBMC calib fr | OneK1K calib | OneK1K calib fr |
|---|---|---|---|---|
| PALM-lite, bootstrap draws | 0.845 | 0.872 | 0.860 | 0.902 |
| PALM-lite, point only | 0.845 | 0.872 | 0.860 | 0.901 |
| NNLS | 0.780 | 0.879 | 0.807 | 0.901 |
| PALM NB (posterior draws) | 0.855 | — | 0.878 | — |

- **Bootstrap draws for PALM-lite change nothing.** Drop them.
- **`calib fr` gives nominal activity coverage for any engine.** It recomputes the calibration set's d̂ with
  the engine's own fractions on the synthetic calibration bulks. With `calib fr all` it even recovers
  point accuracy for a poor engine: NNLS on OneK1K goes from ρ 0.563 to 0.646.
- **Still to do:**
  - PALM's own activity coverage is 0.855 on PBMC.
  - Option: run `calib fr` with PALM-lite as a proxy for PALM's fraction error. It is conservative, because
    PALM-lite's error is larger.

### O5: OneK1K competitors (PROGENy-7, dev_bench rows)

Splits 0–2 for the BayesPrism rows; cellGeometry is over 10 splits.

| Method | Fraction RMSE | ρ | Type-specific ρ | Effect r | Runtime |
|---|---|---|---|---|---|
| PALM NB, posterior realloc | 0.046 | 0.751 | 0.531 | 0.771 | ~10 min |
| BayesPrism, own Z (pseudobulk ref) | 0.064 | 0.732 | 0.543 | 0.746 | 1.9 h |
| BayesPrism + realloc | 0.064 | 0.744 | 0.543 | 0.774 | |
| BayesPrism with cells + realloc | 0.064 | 0.747 | 0.539 | 0.776 | 50 min |
| cellGeometry with cells | 0.053 | 0.745 | 0.462 | 0.724 | 71 s |
| True fractions + realloc | 0 | 0.750 | 0.502 | 0.789 | |

- On OneK1K every good method sits at the true-fraction ceiling (ρ about 0.75). The dataset cannot
  separate them.
- PBMC (splits 0–2) does separate them:

  | PBMC, splits 0–2 | ρ | Fraction RMSE |
  |---|---|---|
  | PALM | 0.859 | 0.059 |
  | cellGeometry | 0.850 | 0.087 |
  | BayesPrism + realloc | 0.806 | 0.176 |
  | BayesPrism, own Z | 0.690 | |

- PBMC, where real effects exist, is the informative dataset. Kang (O6) would be a second one.

---

## 7b. Second round (2026-10-28): fixes, Kang, CIBERSORT, new interval variants

### Bug fix: `estimate_delta` did not converge for strong responses with large weights

**The bug.** Newton steps were capped at ±1 per coordinate.
- PROGENy weights are raw: up to about 17, for IFIT1 in JAK-STAT.
- So a capped step moved a gene by up to e¹⁷, and the iteration cycled between δ = 0 and δ = 1.
- After 100 iterations it returned whichever end it was on.

**The fix.** Newton steps with per-profile backtracking line search. The log-likelihood is concave, so this
always converges. A regression test, `test_estimate_delta_large_weights_converge`, fails on the old code
(returns 0.0) and passes on the new (0.300).

**Impact on split 0** (old vs new on the truth δ):

| Dataset / prior | Entries changed > 0.01 | Old–new correlation |
|---|---|---|
| Kang, PROGENy-7 | 50% | −0.07. The truth was wrong: the IFN-β effect showed up on p53 (+1.4) instead of JAK-STAT (≈0). It is now JAK-STAT +0.26 to +0.35 in all types, p53 ≈ 0. |
| PBMC, PROGENy-7 | 1.2% (max change 2.2) | 0.99. COVID samples with strong interferon responses. |
| PBMC and Kang, Hallmark; OneK1K, both priors | 0 | 1.00. Weights of ±1 never trigger it. |

**What has to be redone:**
- PBMC PROGENy results (truth, baselines, reallocation rows, competitor activity) are being recomputed in
  the v2 run.
- The old files are in `benchmarks/dev/pbmc{,__progeny14}/old_estimator/`.
- Section 7's PBMC PROGENy numbers are slightly contaminated. The conclusions are unlikely to change
  (old–new correlation 0.99), but the v2 numbers replace them.
- PALM's own model δ (sampled) was not affected.

### Kang IFN-β benchmark (O6)

**Data.** Kang et al. 2018 via `pertpy` (`pt.dt.kang_2018()`): 24,673 PBMCs from 8 lupus donors, control
vs IFN-β stimulated, raw counts. Built by `scripts/experiments/build_kang.py` into
`data/kang/kang_pseudobulk.h5ad`.
- **Cell types:** mapped to the dev_bench types. CD4/CD8 T → T cell; CD14+/FCGR3A+ Mono → Monocyte;
  B and NK kept. DC and megakaryocytes are dropped.
- **Pseudo-replicates:** each donor × condition is split into max(1, n_cells // 600) specimens. That gives
  35 specimens, each with ≥ 20 cells of every type.

**Design** (`DATASETS["kang"]` in `scripts/dev_bench.py`):
- 4 test donors per split, with **all** their specimens. Stim and ctrl are paired, giving 14–20 bulk
  samples.
- The reference and the calibration set come from the other 4 donors' **control** cells only
  (`make_benchmark(ref_filter=("label", "ctrl"))`). This mimics a resting reference atlas, so the IFN-β
  response is entirely "program activity".
- Calibration uses all of a calibration donor's control specimens, with folds by donor.
- **Competitors needing single cells** (cellGeometry, BayesPrism) use the reference donors' control cells,
  `data/kang/kang_ctrl_cells.h5ad`, through `--cells`.
  - With only 4 control donors, there are too few pseudo-bulk rows: cellGeometry's `cellMarkers` failed on
    split 1.
  - The split-0 cellGeometry row below used pseudo-bulk rows. That file was moved to `kang/rows_ref/`.

**Split 0, PROGENy-7:**

| | Fraction RMSE | ρ | Type-specific ρ | Effect r | Localisation |
|---|---|---|---|---|---|
| PALM NB + realloc | 0.034 | 0.762 | 0.469 | 0.976 | 0.71 |
| cellGeometry | 0.079 | 0.702 | 0.438 | 0.977 | 0.57 |
| CIBERSORT | 0.084 | 0.715 | 0.438 | 0.978 | 0.71 |
| NNLS + realloc | 0.095 | 0.705 | 0.322 | 0.971 | 0.43 |
| True fractions + realloc | 0 | 0.749 | 0.480 | 0.975 | 0.71 |

- Effects are large and shared by every type, so effect r is easy (about 0.97 for all methods).
- Activity ρ and type-specific ρ separate the methods, and PALM's fractions are 2–3× more accurate.

### CIBERSORT

- **Code:** `compare.methods.cibersort`. It runs the published R script v1.02 (`compare/R/CIBERSORT.R`,
  through `compare/R/run_cibersort.R`; needs e1071 and preprocessCore, both now in palm-r).
- **Algorithm:** nu-SVR with a linear kernel, nu ∈ {0.25, 0.5, 0.75}, best by RMSE.
- **Signature:** the R script cannot build one (that is CIBERSORTx), so `cibersort_signature` uses an
  LM22-style stand-in. For each type it takes the 100 genes with the largest log2 fold change over the
  highest other type, with ≥ 10 CPM, using reference CPM over all genes.
- **Fractions:** mixture is bulk CPM, QN off. The weights are read fractions over all genes and are moved to
  the modelled genes.
- **Activity:** by reallocation, as for cellGeometry.
- **Speed:** about 1–5 s per split.
- **PBMC split 0:** fraction RMSE 0.135, ρ 0.793 (NNLS 0.109 and 0.810).

### New pipeline variants (in `scripts/experiments/openitems.py`)

- **Leakage-aware effect width (`calib lk`, `calib lk+`).**
  - On the calibration set, per program p, regress each type's out-of-fold residual on all types' true
    activity; this gives L_cc'p.
  - Width = random-split conformal width (combined with bootstrap and fraction-draw variance) +
    Σ_c' |L_cc'p| · |β̃_c'p|.
  - `lk+` uses |β̃| + z·se instead of |β̃|.
  - Null effects get the plain width; real effects get the leakage they would cause.
- **Proxy fraction error (`calib px`).** For engines without a fraction function (PALM, cellGeometry,
  CIBERSORT, BayesPrism), the calibration set's d̂ is computed with PALM-lite fractions. This is
  conservative, because PALM-lite's error is at least as large.
- **Competitor sources:** `comp:<name>` reads any dev_bench competitor's saved fractions and reallocates.

**Two-split preview** (PROGENy-7, old estimator; PALM NB):

| | PBMC eff cov | PBMC iscore | OneK1K eff cov | OneK1K iscore |
|---|---|---|---|---|
| normal | 0.84 | 0.112 | 0.84 | 0.021 |
| stratified | 0.91 | 0.139 | 1.00 | 0.054 |
| leakage-aware (`lk`) | 0.96 | 0.118 | 1.00 | 0.023 |

- `lk` reaches coverage without the over-width on near-null effects.
- `calib px` raised PALM's activity coverage slightly: PBMC 0.89 → 0.90, OneK1K 0.86 → 0.88.

### v2 run (started 2026-10-28 09:55)

**Launchers** are in the session scratchpad. **Logs:** `benchmarks/dev/v2_bench.log` and
`benchmarks/dev/v2_oi.log`.
- **dev_bench:**
  - Kang splits 0–9: baselines, CIBERSORT, PALM joint NB, and cellGeometry on control cells.
  - PBMC baselines and cellGeometry redone with the fixed estimator, plus CIBERSORT and PROGENy-14
    baselines.
  - OneK1K CIBERSORT.
  - `rescore` of PBMC.
  - Then BayesPrism on Kang (splits 0–4) and PBMC (splits 0–2).
- **openitems_v2:** Kang, PBMC and OneK1K × PROGENy-7, Hallmark, PROGENy-14 and CollecTRI.
  - Sources: PALM NB, PALM-lite, NNLS, cellGeometry, CIBERSORT, true fractions; BayesPrism where it ran.
  - Summaries: `benchmarks/dev/openitems_v2/summary_<dataset>__<prior>.txt`.

---

## 8. Implementation status (2026-10-28)

| Part | Where | Status |
|---|---|---|
| Full PALM fit | `palm_bulk.bayes.twostage.run_palm_fast` | Done |
| Gene selection (Fisher) | `palm_bulk.core.data.select_genes` | Done |
| `estimate_delta` | `palm_bulk.core.scores` | Done; line-search fix 2026-10-28 |
| Posterior reallocation | `palm_bulk.reallocate`, `realloc_counts` (`core/realloc.py`) | Done; accepts draw axes; fraction floor 1e-4 |
| PALM-lite EM | `palm_bulk.em_fractions` (`core/realloc.py`) | Done; SQUAREM-accelerated, under 1 s per split |
| Calibration set | `palm_bulk.build_calibration_set`, `calibration_from_table` (`core/calibrate.py`) | Done; folds by donor, `ref_filter`, `all_specimens` |
| Calibration map, intervals | `palm_bulk.Calibrator` | Done; activity: conformal; effects: `normal`, `conf`, `strat`, `lk`, `lk+`, `lka` |
| Module API | `palm_bulk.program_deconvolution` → `DeconvolutionResult` (`deconvolution.py`) | Done |
| Tests | `tests/test_deconvolution.py` (7 tests, simulated studies) | Pass; quick suite 55/55 |

**Using the module:**

    import palm_bulk as pb
    inp = pb.prepare_inputs(bulk, reference, programs, treatment=T)
    cal = pb.calibration_from_table(table, bulk_donors, inp.cell_types, inp.genes, inp.Gmat)
    res = pb.program_deconvolution(inp, cal, engine="palm")             # or engine="lite"
    res = pb.program_deconvolution(inp, cal, fractions=other_method_pi)  # any method's read fractions
    res.table("effects")                                                 # effect, lo, hi, called

How fraction error enters the intervals is chosen automatically:
- PALM draws: calibration with the true fractions, plus the spread over draws.
- A fraction function (PALM-lite): the engine's own fractions on the calibration bulks.
- An external point estimate: PALM-lite as a proxy.

**Two changes made while packaging:**
- **Own-type slope shrinkage.**
  - The calibration slope of each (c, p) is shrunk toward 1 with the weight of 5 specimens and floored at
    0.05.
  - On Kang (11 calibration specimens) one slope was −1.2, which reversed that pair's ranking: calibrated
    ρ 0.724 vs raw 0.762. With shrinkage, calibrated ρ is 0.764 and effect r goes from 0.908 to 0.948.
  - PBMC (86 specimens) has no negative slopes and barely changes.
  - The `openitems_v2` script still uses the old unshrunk map, so its Kang calibration rows may carry flipped
    slopes; use `benchmarks/dev/intervals/` for Kang.
- **`lka`: asymmetric leakage-aware effect interval.**
  - The own-type leakage slope L_cc is attenuation: a known-direction shrink toward 0, estimated 0.87 vs true
    1.13 in a simulation.
  - Symmetric widening by L·|β̃| (`lk`) therefore costs real effects their power: the planted effect was
    missed in 2 of 3 simulated studies.
  - `lka` widens both sides only by the other types' leakage and extends the side away from zero to the
    de-attenuated value β̃/(1 − L_cc).
  - **Simulations, 3 studies:**

    | Interval | Coverage | Planted effect called |
    |---|---|---|
    | normal | 0.17–0.50 | 3/3 |
    | strat | 0.50–0.67 | 3/3 |
    | lk | 1.00 | 1/3 |
    | lka | 0.83–1.00 | 3/3 |

  - **Real data:** `scripts/experiments/intervals.py` runs Kang, PBMC and OneK1K × PROGENy-7 and Hallmark ×
    10 splits through the package, for PALM and PALM-lite. Results go to `benchmarks/dev/intervals/`. The
    run started 2026-10-28 11:25.

### Interval decision (2026-10-28, `benchmarks/dev/intervals/summary.txt`)

**Data:** packaged pipeline, PALM draws, 10 splits × Kang, PBMC and OneK1K × PROGENy-7 and Hallmark.

**Effect coverage (calls per split):**

| Interval | Kang P7 | Kang Hallmark | PBMC P7 | PBMC Hallmark | OneK1K P7 (null) | OneK1K Hallmark (null) |
|---|---|---|---|---|---|---|
| normal | 0.78 (19) | 0.55 (36) | 0.69 (14) | 0.75 (36) | 0.83 (7.4 false) | 0.76 (13 false) |
| strat | 0.80 (19) | 0.58 (34) | 0.90 (7.2) | 0.90 (24) | 1.00 (2.1), iscore 2.4× | 1.00 (2.4), iscore 2.4× |
| **lk** | **0.98 (5.3)** | **0.89 (9)** | **0.93 (2.9)** | **0.97 (8.2)** | **0.96 (2.3)** | **0.93 (5.6)** |
| lka | 0.98 (10) | 0.85 (15) | 0.90 (6.4) | 0.95 (19) | 0.93 (4.8) | 0.88 (9.3) |

**Decision.**
- **Default `effect_interval="lk"`.**
  - It is the only interval with coverage ≥ 0.89 in every setting, including the real stimulation (Kang).
  - On null contrasts its width equals the normal interval's (OneK1K interval score 0.024 vs 0.022). Stratified
    is 2.4× too wide there.
  - It makes the fewest false calls under the null.
- **`lka` is the higher-power option.** It calls 2–3× more effects, at coverage 0.85–0.98 and with more false
  calls under the null.
- **`strat` is retired.** It fails on Kang.

**Limits to report:**
- Kang activity coverage is 0.70–0.83. A resting calibration reference doesn't span stimulated activity.
- Kang Hallmark has a false sign rate of about 0.11 among calls, under every interval type. Leakage under
  strong responses shared by all cell types is not fully corrected.
