# Algorithm improvement plan (draft v2, 2026-11-11)

Scope: changes to the algorithm only. This draft leaves out data, references, programs as inputs and cohort design.
Each item names:
- the weakness, with its evidence;
- the stage it changes;
- the change;
- the test and the gate.

Nothing here changes freeze v2 (`estimator="auto"`).

**Protocol:**
- development on splits 0–2 and 20–29, plus COMBAT real bulk with CyTOF;
- splits 30–39 kept as the held-out set for any new freeze;
- splits 50–59 stay reserved for the association section.

**Current algorithm, for reference:**

| Stage | What it does |
|---|---|
| A | fractions (PALM or PALM-lite EM, unshifted reference profiles B) |
| B | reallocation: per-type activity `dhat` |
| C | atlas calibration: attenuation and error quantiles |
| D | unmixing, `dhat_ncp = a_cp + Σ_k A_nckp (γ_kp + T_n β_kp) + e`, with `β_k = m + u_k`, `u_k ~ N(0, τ_p²)` (one shared value, isotropic deviations, τ by REML per program); router: unmixed vs calibrated by width, with a guard |
| E | call when the 90% interval excludes 0; Shapley decomposition; robustness value |

---

## 1. Weak points the algorithm can address

| # | Weak point | Evidence | Stage |
|---|---|---|---|
| A1 | The prior over cell-type effects allows only "shared + isotropic deviations". It can't say "one type responds" or "one lineage responds" | Low power on weak or broad responses (Tahoe encorafenib 0.05, gemcitabine 0.00, Kang 0.22); gated v4's shared structure raised McFarland null calls to 0.12 / 0.11 | D |
| A2 | Shrinkage doesn't target the directions the design can't resolve | Near-identical types split one signal into opposite signs (PTB CD4 T +0.47 / CD8 T −0.41, which nearly cancel); merging 8 → 5 types narrowed intervals by a quarter | D |
| A3 | Fractions treat an activated cell as composition | The fraction step uses unshifted B. On COMBAT the NK estimate follows CyTOF neutrophils (r 0.32), not NK (r 0.09 anchored) | A |
| A4 | Platform offsets: every gene counts equally, and the error model has no bias term | Real-bulk fraction error is bias (RMS log bias 3.3–4.1), not scatter (v4_plan §10A); calibration simulates scatter | A, C |
| A5 | Calls have no multiplicity control | Each pair is called at 90% (`deconvolution.py:209`), with 30–102 pairs per run | E |
| A6 | The robustness value is computed at sample level, but some confounders act at group level | A run-level or site-level confounder is assigned by cluster | E |
| A7 | No missing-type detector | Method limit; the rejected learned extra profile absorbed platform signal (v4_plan §10B) | self-tests |
| A8 | Program weights are the same in every type | Forced by `G (P, G)` in `leakage_operator` and `estimate_delta` | B, D |

---

## 2. Changes, in order

### Phase 1: the effect prior (stage D). Targets A1 and A2. Expected to matter most.

**1a. Structured mixture prior on each program's effect vector** (MASH idea, adapted to few effects).
- Today: `β_p = m_p 1 + u`, `u ~ N(0, τ_p² I)`.
- Proposed: `β_p ~ Σ_j w_j N(0, U_j)` over a **fixed** set of covariance patterns:

| Pattern | Meaning |
|---|---|
| null | small ε I |
| shared | `σ² 11ᵀ + ε I` |
| single type | `σ² e_k e_kᵀ + ε I`, one per type |
| lineage | `σ² 1_L 1_Lᵀ + ε I`, one per lineage block from the reference's lineage tree |
| independent | `σ² I` |

How it is fitted:
- **Why a fixed set:** a study gives only C × P ≈ 30–100 effects. That is too few to learn covariance matrices from the data, as MASH does with thousands. So the patterns are fixed, and only the mixture weights w (pooled across programs) and a scale grid σ are estimated, by EB marginal likelihood. These are a handful of parameters.
- **Full-rank components** (the ε I term): rank-deficient prior covariances make the lfsr anti-conservative (too small), which inflates the false sign rate (Xie 2025, arXiv 2502.16118, §3, Fig. 4; remedy §4: each pattern plus a positive diagonal, preferably data-adaptive, §5). That is a candidate explanation for gated v4's McFarland excess, since v4's shared structure is effectively low rank. Caveats from the full text (checked 2026-11-12): Xie assumes no true effect is exactly 0 and warns that with exact nulls "the FSR is much harder to control" (§7); all simulations use 1000–2000 units. Yang et al. 2024 (§7) add that lfsr can stay anti-conservative even with full-rank components and advise a stricter threshold (0.01 rather than 0.05). Our 1a test (section 4b) has exact nulls and ~7 units per study, so its excess null calls are consistent with these caveats and not a contradiction of Xie.
- **Likelihood:** the unmixed estimate and its covariance, `β̂_p ~ N(β_p, V_p)`. V_p is the cluster-bootstrap covariance, which already exists in `unmix_effects`, plus the misfit term. The posterior is closed form for a Gaussian mixture.
- **Calls:** lfsr < 0.05, alongside the existing interval rule for comparison.
- **Safety by design:** the single-type and lineage patterns let the data say "only this type responded". The shared-only structure forces some response onto every type, which is how a real effect in one type becomes null calls in the others.

**1b. Shrink along the unresolved directions** (A2).
- Rotate β_p into the eigenbasis of the design information `Ã'Ã`, which already exists (the resolvable-contrast basis, v4 idea 8).
- Apply shrinkage in proportion to how poorly each direction is identified. This is a generalised ridge whose penalty is large only on near-null directions such as CD4 − CD8, and leaves well-identified directions (lineage means, T − myeloid) untouched.
- This is not the rejected `v4kt` tree shrinkage, which shrank every type toward its parent and lost power. Here the design's own information decides where to shrink.
- Combined with 1a, use the rotated basis for the likelihood and the original basis for the prior.

**1c. Hierarchical calls** (A2, readout).
- Call lineage-level effects first (fraction-weighted means within each lineage block).
- Call children within a parent only as contrasts and only when the parent is called, with error control over the tree (Yekutieli 2008 tree FDR).
- Flag opposite-sign siblings whose fraction-weighted sum is near zero as "split unresolved".

**Code:** a new `effect_prior.py` that takes `unmix_effects` output (`effects`, the bootstrap draws → V_p), the lineage map and the Ã eigenbasis. `estimator="v5mix"` would be opt-in.

**Test:** all 13 datasets, development splits.
- **Metrics:** coverage, null calls, power, r, within-program r.
- **Specific checks:**
  - full-rank vs low-rank patterns on McFarland: does the null-call excess appear only with low rank?
  - the weights w per dataset: expected "shared" on Kang and Parse IFN-β, "single type" on McFarland.

**Gate:** the S1/S2 criteria of the v4 protocol, unchanged:
- every dataset: coverage ≥ 0.85 and null calls ≤ 0.10;
- null-call difference vs v2: upper CI ≤ +0.02;
- primary: power, not inferior on r.

### Phase 2: the fraction step (stage A). Targets A3 and A4.

**2a. Fractions with state-shifted profiles** (A3).
- Today the fraction EM uses unshifted B, so a program switched on looks like more of the type whose profile already carries those genes.
- Proposed: alternate the steps.
  1. Fractions π given profiles `B_c(δ) ∝ B_c ∘ exp(G'δ_c)`, with δ_c the per-type activity at the sample's group level (treatment arm or condition), not per sample, so that composition stays identified.
  2. δ from reallocation given π.
  3. Iterate to convergence, which usually takes 2–3 rounds.
- This is the fraction-side counterpart of the attenuation fix in stage C.

**Test:** COMBAT real bulk against CyTOF (`ptb_anchor.py validate` harness).
- **Success:** NK r rises clearly above 0.09, and the correlation between NK and CyTOF neutrophils falls, without loss for T, myeloid or neutrophils.
- **Also:** coverage on the 13 datasets. Group-level δ must not absorb the treatment effect into composition: the effect estimate must stay unbiased on Kang and Parse, where the truth is known.

**2b. Gene weights from offset stability** (A4, DeMixSC idea; Guo et al. 2024, Genome Res).
- Weight each gene in the fraction fit by 1/(variance of its platform offset across specimens).
- The variance is estimated on the anchor specimens, or in a self-offset study from the spread of the per-sample log ratios. Unstable genes count less.
- **Test:** the COMBAT harness, measuring bias and r per group.

**2c. Bias in the calibration error model** (A4; v4 idea 2).
- In the stage C / design-check simulation, add a per-type log bias to the fractions, with sd taken from the measured real-bulk bias, alongside the current scatter. Intervals then carry fraction bias, not only fraction noise.
- **Test:**
  - coverage on COMBAT real bulk, where it is currently 0.97–1.00 with few calls; the interval width cost is measured on the 13 datasets;
  - the bias sd is a declared assumption, reported next to the design limit.

**Gate for Phase 2:** better COMBAT fraction accuracy (2a, 2b) or calibrated coverage (2c), with no S1/S2 failure on the 13 datasets.

### Phase 3: inference and readouts (stage E). Targets A5 and A6. Cheap; no change to estimates.

**3a. Multiplicity.**
- For each run, report alongside the existing calls:
  - lfsr, if Phase 1 is adopted;
  - otherwise, false-coverage-rate (FCR) adjusted intervals for the called pairs, using the directional-selection version (Frostig & Benjamini 2025, JRSS-B), which matches the rule "interval excludes 0 on one side".
- **Metric:** held-out false-call rate among *called* pairs (false discovery proportion), next to the per-pair null-call rate.

**3b. Cluster-level robustness.** When the treatment, or a named confounder, is assigned by cluster (run, site, donor), compute the robustness value from cluster-level regressions. Strezhnev 2026 (arXiv 2607.13334) shows the unit-level value can mislead in that case. Before implementing, read the full paper.

### Phase 4: diagnostics and generalisation. Targets A7 and A8. Lower priority.

**4a. Missing-type diagnostic** (report only; Ivich et al. 2025, Genome Biol).
- Run a non-negative factorisation of the stage A fit residuals across samples.
- Report a component that is consistent across samples, together with its top genes.
- It never enters the fit, which is the difference from the rejected learned profile.
- **Test:** drop one type from the reference on COMBAT and Parse. The diagnostic should flag it, and stay quiet with the full reference.

**4b. Type-specific program weights** (A8).
- Generalise `G (P, G)` to `G (C, P, G)` in:
  - `leakage_operator`: S_c, m_k and the cross terms;
  - `estimate_delta`;
  - the forward model.
- With identical slices it must reproduce v2 exactly. This is infrastructure only; whether to use non-identical weights is a separate decision.

---

## 3. What is not proposed, and why

| Idea | Reason |
|---|---|
| Learning prior covariances in-study (full MASH) | 30–100 effects per study are too few; fixed patterns with estimated weights instead (1a) |
| Priors from earlier PALM outputs | Standing rule: undetectable overconfidence |
| A third stacked estimator | v4_plan §10C: extra weights add variance |
| Tree shrinkage toward parents | Rejected as `v4kt`; 1b shrinks only unresolved directions |
| Learned extra reference profile | Rejected (v4_plan §10B); 4a only diagnoses |
| Loosening the call rule | Changes the error rate, not the evidence |

---

## 4. Order and stopping

1. **Phase 1 (1a + 1b, then 1c).** On development splits.
   - If 1a fails the gate, keep 1c and 3a as readouts on v2.
   - Stop the phase if neither 1a nor 1b gains power without breaking S1/S2.
2. **Phase 3.** Can run alongside Phase 1, since it changes readouts only.
3. **Phase 2 (2a first, then 2b, then 2c).** On COMBAT, then on the 13 datasets.
4. **Phase 4.** When Phases 1–3 are settled.
5. **Freeze and held-out.** A candidate that passes is frozen with a pre-registered rule and run once on splits 30–39, as for v4.

## 4b. First experiments (2026-11-11, development splits 20–29, 13 datasets)

**Offline, on the saved freeze v2 outputs** (`scripts/experiments/v5_prior_offline.py`, `v5_width_offline.py`):

| Method | Coverage | Null calls | Power | Called pairs that are null | S1 |
|---|---|---|---|---|---|
| freeze v2 | 0.982 | 0.017 | 0.269 | 0.108 | pass |
| BH q < 0.10 on v2 | 0.982 | 0.002 | 0.187 | 0.009 | pass |
| Mixture prior (1a), diagonal errors, lfsr < 0.05 | 0.648 | 0.016 | 0.276 | 0.077 | fail (credible intervals under-cover) |
| Interval width × k, k chosen leave-one-study-out | 0.960 | 0.036 | 0.309 | 0.158 | fail (McFarland null calls 0.13 / 0.17) |
| Same, separate k per route | 0.953 | 0.045 | 0.376 | 0.249 | fail (McFarland) |

**Readings:**
- **1a with diagonal errors does not help.** The calls hardly change, and the posterior intervals miss the small non-zero truths. The saved files have no cross-type error covariance, so this is the weak form of 1a.
- **Freeze v2's intervals are too wide.** Coverage is 0.98 on average and 1.00 on PBMC. Shrinking them raises power: with a separate k per route, power goes 0.27 → 0.38 and Parse power rises clearly.
- **The excess width is study-specific.** McFarland already sits at nominal width, so no k chosen on other studies transfers. Width needs an in-study rule.
- **On gemcitabine, 69% of v2's calls are null pairs.** BH removes almost all of them, at a cost of 30% of power overall.

**Overnight (launched 17:31):**
- freeze v2 rerun with its intermediate parts saved (`estimator="auto"`, `unmix_kw=dict(keep_parts=True)`; `causal_loop.py ... auto_v2_parts`): both routes' estimates and half-widths, the simulated error, the atlas interval and the bootstrap draws;
- the rerun reproduces freeze v2 to 1e-10 (checked on Kang split 20).

**Morning check:** `python scripts/experiments/v5_parts_offline.py 20-29` scores these variants:
- `simonly`: the calibrated route drops the max with the atlas interval and uses the simulated error alone;
- `calonly`: atlas interval alone (a reference, expected unsafe);
- `modt` / `modt_prog`: moderated unmixed variance, limma-style (Ignatiadis & Roquain 2026 give the EB basis);
- `lfsr`: the mixture prior with the full bootstrap error covariance (the strong form of 1a);
- `bh`: BH multiplicity on v2.

**Overnight results** (`v5_parts_offline.py 20-29`, 130 runs; the v2 rebuild from the parts equals the saved json exactly). Mean of the 13 dataset means; the difference vs v2 is given with a 95% bootstrap CI over datasets:

| Variant | Coverage | Null calls | Power | FDP | S1 | Δ null calls | Δ power |
|---|---|---|---|---|---|---|---|
| v2 | 0.982 | 0.017 | 0.269 | 0.108 | pass | – | – |
| simonly | 0.979 | 0.021 | 0.285 | 0.151 | pass | +0.004 [+0.001, +0.006] | +0.017 [−0.007, +0.051] |
| calonly | 0.842 | 0.035 | 0.319 | 0.265 | fail (6: McFarland, Tahoe) | +0.018 | +0.050 |
| modt | 0.982 | 0.016 | 0.269 | 0.107 | pass | −0.001 | +0.000 |
| modt_prog | 0.982 | 0.017 | 0.271 | 0.109 | pass | +0.000 | +0.002 |
| simonly_modt | 0.980 | 0.019 | 0.285 | 0.145 | pass | +0.002 [−0.001, +0.005] | +0.017 [−0.008, +0.052] |
| lfsr (full covariance) | 0.982 | 0.043 | 0.280 | 0.125 | fail (McFarland 0.12 / 0.18, Tahoe gem) | +0.026 [+0.005, +0.050] | +0.011 |
| bh | 0.982 | 0.002 | 0.187 | 0.009 | pass | −0.015 | −0.082 [−0.129, −0.039] |

**Readings:**
- **No variant is a clear gain.** The safe ones (simonly, modt, simonly_modt) change power by at most +0.017, with a CI that includes 0.
- **The max with the atlas interval is not what makes v2 wide.** Dropping it (simonly) narrows width only 0.304 → 0.284. The simulated error itself is the wide part, and the atlas interval alone (calonly) under-covers badly on McFarland and Tahoe (0.61–0.79). So the excess width sits in the simulated error quantile, which is the target of a simulation-based bias correction in the form of PPI (see the next list for why only the form).
- **Moderated variance does nothing.** The unmixed se are already stable across pairs.
- **Full-covariance lfsr makes too many null calls** (McFarland and gemcitabine). The fixed-pattern mixture prior is over-confident even with correlated errors, so 1a in this form is dropped.
- **BH stays the safety option.** It cuts null calls to 0.002 and FDP to 0.009, at 30% of power.

**Next techniques from the search, not yet testable from the parts:**
- **Bias correction in the form of prediction-powered inference** (Angelopoulos et al. 2023, Science; PPI++, Ann Appl Stat 20(3), 2026): subtract the simulated bias per pair and add its variance over the simulated cohorts, in place of widening to error quantiles. This targets the conservative widths directly. Full texts checked 2026-11-12: PPI's coverage guarantee needs labelled pairs i.i.d. from the study's own population (PPI §1.2; PPI++ §2), or a known covariate or label shift (PPI §4.2); synthetic labelled data are not covered. Our 10 simulated cohorts are neither, so we borrow the form only, and validity rests on the empirical check against known truths on the 13 datasets (S1/S2), not on PPI's theorem. With 10 cohorts the variance term is itself noisy. Possible framing for the sim-to-real gap, not read: Tan & Zrnic 2026, "Valid inference with synthetic data via task exchangeability" (arXiv 2606.13629).
- **Data fission** (Leiner et al. 2023, JASA): choose the route on one part of the data and infer on the other, which removes the router's selection effect.
- ~~Multivariate heteroscedastic NPMLE empirical Bayes (Soloff, Guntuboyina & Sen 2024, JRSS-B)~~: dropped after reading the full text (2026-11-12). The model matches ours in form (X_i ~ N(θ_i, Σ_i) with observation-specific Σ_i, nonparametric G), but Σ_i is treated as known, every guarantee is an asymptotic rate in n with (log n)^(d+1) factors (Theorem 7), the smallest application has n = 100 at d = 2, and the paper gives posterior means only (no credible sets, no lfsr; posterior quality is listed as open in §6). With ~7 vectors of dimension 4–9 per study it would overfit, and it does not address calibrated intervals, which is our problem. Cite only as a nonparametric point-estimation idea.
- **Inference with estimated proportions** (Cai et al. 2024, JASA): for Phase 2c, fraction error in the calibration.

## 5. References (OpenAlex metadata; abstracts read; full texts read 2026-11-12 for the four marked ‡)

- Xie D. Local false sign rate and the role of prior covariance rank in multivariate empirical Bayes multiple testing. arXiv 2025 (v2, Apr 2025). 10.48550/arxiv.2502.16118 ‡
- Angelopoulos AN, Bates S, Fannjiang C, Jordan MI, Zrnic T. Prediction-powered inference. Science 2023;382:669–674 (arXiv 2301.09633). ‡
- Angelopoulos AN, Duchi JC, Zrnic T. PPI++: efficient prediction-powered inference. Ann Appl Stat 2026;20(3). 10.1214/26-AOAS2215 (arXiv 2311.01453). ‡
- Soloff JA, Guntuboyina A, Sen B. Multivariate, heteroscedastic empirical Bayes via nonparametric maximum likelihood. JRSS-B 2024 (arXiv 2109.03466). ‡
- Singhania A, et al. A modular transcriptional signature identifies phenotypic heterogeneity of human tuberculosis infection. Nat Commun 2018;9:2308. 10.1038/s41467-018-04579-w ‡ (module gene lists: data/singhania2018_ifn_modules.csv; lightgreen = type I IFN, 86 genes; yellow = IFN/complement/myeloid, 360; tan = Th1/NK, 157)
- Yang Y, Carbonetto P, Gerard D, et al. Improved methods for empirical Bayes multivariate multiple testing and effect size estimation. arXiv 2024. 10.48550/arxiv.2406.08784
- Urbut SM, et al. Flexible statistical methods for estimating and testing effects in genomic studies with multiple conditions (MASH). Nat Genet 2019. 10.1038/s41588-018-0268-8 †
- Chen L, Li Z, Wu H. CeDAR: cell type hierarchy in cell type-specific differential analyses. Genome Biol 2023. 10.1186/s13059-023-02857-5
- Yekutieli D. Hierarchical false discovery rate-controlling methodology. JASA 2008. †
- Guo S, et al. DeMixSC: single-cell sequencing plus a small benchmark data set. Genome Res 2024. 10.1101/gr.278822.123
- Ghaffari S, et al. BEDwARS: deconvolution with noisy reference signatures. Genome Biol 2023. 10.1186/s13059-023-03007-7
- Frostig T, Benjamini Y. Improving the false coverage rate adjusted confidence intervals. JRSS-B 2025. 10.1093/jrsssb/qkaf024
- Benjamini Y, Yekutieli D. FDR-adjusted multiple confidence intervals for selected parameters. JASA 2005. 10.1198/016214504000001907 †
- Strezhnev A. Omitted variable bias sensitivity analysis with clustered treatment assignment. arXiv 2026. 10.48550/arxiv.2607.13334
- Ivich A, et al. Missing cell types in single-cell references impact deconvolution of bulk data but are detectable. Genome Biol 2025. 10.1186/s13059-025-03506-9

† From my knowledge, not retrieved in the search; verify before citing.
