# Prior knowledge from shared databases for program2cell: plan (2026-11-18)

*Plan only, no method change. Paper 1 (freeze v5) is untouched. Companion to `dl_plan_2026-11-16.md` (amortized
inference, FM footprints D5, atlas teacher T1) and `palm_uni_plan.md` §3.5 (program library). Written after the
PTB GO-program analysis and the independent-fit work on `feature/independent-programs`. No held-out split is used
for this work without the user's agreement.*

## 1. Question and short answer

**Question.** Can knowledge shared across studies (gene-set databases such as GO and Reactome, expression and
perturbation atlases, pretrained models) be used as a prior to improve cell-type program effects, especially the
independent fit, which screens many overlapping programs and is the least precise?

**Short answer.** Yes, in four ways that differ in what they need and what they risk (§3). But a prior only helps
once the reported uncertainty is honest. Today the independent intervals are 1.5–3× wider than the actual errors
(§2), and any prior placed on top of them shrinks too hard. So the order is: first make the independent
uncertainty honest (P0, already running as the exclusive-gene test), then add the cheapest prior that needs no
outside data (M1, term-graph pooling), and leave the atlas-trained priors (M3, M4) for Paper 2, where learning
from a large database is the main idea anyway.

## 2. What we already know (development evidence, 2026-11-18)

| Finding | Source | Consequence for priors |
|---|---|---|
| Independent v5 (calibrated route only) has coverage 0.972 at nominal 0.90; mean \|error\| / SE is 0.27–0.64 per dataset (≈ 0.8 if SEs were right) | `benchmarks/dev/indep_shrink_offline.txt`, splits 20–24 | the SEs are inflated, mainly by the simulated leakage widening of reallocation |
| Per-study adaptive shrinkage (ash-type normal mixture, EM) on those SEs puts 92–100% of pairs at 0: coverage 0.02, power 0.01 | same | shrinkage trusts the SEs; with inflated SEs it destroys the signal. **Priors need honest SEs** |
| Per-program unmixing with `forward_independent` (only program r shifts) was biased because other programs also move | `method_spec.md`, "Tried and not adopted (2026-11-18)" | any per-program model must let the other programs absorb their own genes |
| Exclusive-gene joint v5 (other programs' weights set to 0 on r's genes): truth matches the independent truth (r ≈ 0.99); first 5 Kang splits coverage 0.96–1.00, narrower than independent v5 on split 20 | `indep_exclusive_bench.py`, run in progress | the candidate for P0 |
| PTB: joint and independent estimates of the same 9 GO terms correlate r = 0.14; joint calls not reproduced independently | PTB report, `results/ptb/report/` | overlapping terms make joint (partial) effects fragile; pooling over the term structure (M1) targets exactly this setting |
| Fractions are estimated once per study on all modelled genes; the independent mode only changes reallocation | `deconvolution.py` | nothing to gain from a shared fraction fit |

## 3. What "prior knowledge" can mean

Gene-set databases list members of terms; they contain no cell-type information. The useful shared knowledge
comes from four sources:

| Kind | Source | What it informs | Needs outside data | Main risk |
|---|---|---|---|---|
| **K1 term structure** | GO DAG, Reactome hierarchy, gene overlap between terms | which program effects should be similar | no (the database only) | over-smoothing distinct child terms |
| **K2 expression support** | reference profiles B; expression atlases (e.g. CELLxGENE Census) | which cell types *can* carry a program's shift (its genes are expressed there) | atlas optional | small; B already carries most of it |
| **K3 response atlas** | perturbation atlases: Immune Dictionary (Cui et al. 2024), Tahoe-100M, Parse 10M PBMC | which cell types *usually respond* to a pathway family | yes | leakage into benchmarks; missing unexpected responders |
| **K4 effect-size and error distribution** | many studies with known truth (atlas pseudobulk, benchmarks of other perturbations) | how large true effects usually are, and how the estimator's error relates to its SE | yes | transport to new tissues and platforms |

Pretrained foundation-model features (scGPT, Geneformer, State) are covered by `dl_plan` D5 (FM footprints,
expected kill) and are not repeated here. Amortized inference (`dl_plan` §3) is the full form of K4: the network
learns the prior and the error model together.

## 4. Methods

Notation as `method_spec.md`: e_{c,t} the estimated effect of program (term) t in cell type c, s_{c,t} its
standard error, β_{c,t} the true effect. All four methods act on (e, s) after the fit, so the fit itself and its
estimand stay unchanged; they differ in the prior on β.

### M1. Term-graph pooling (K1): for independent screens of many overlapping terms

Within one study and one cell type c, the effects of related terms share information:

  e_c ~ N(β_c, S_c),  β_c ~ N(0, τ_c² K + σ_c² I)

- **K**, the term similarity: the gene-overlap kernel K_tu = |S_t ∩ S_u| / √(|S_t| |S_u|) (Ochiai), optionally
  plus a GO-DAG kernel (shared ancestors). Fixed from the database, not fitted.
- **S_c, the error covariance, is essential.** Independent effects of overlapping terms are estimated from shared
  genes, so their errors are correlated too. Ignoring that double-counts the shared genes and makes pooling
  overconfident. S_c is estimated with the same group bootstrap of the contrast that the calibrated route already
  uses (`core.unmix.contrast_se`), keeping the off-diagonal terms, then scaled to the calibrated diagonal.
- τ_c², σ_c² by marginal likelihood per study and cell type (2 numbers from tens of terms: stable for screens of
  ≥ 20 terms; for fewer, M1 is not run).
- Output: posterior mean and interval of β_c; calls by interval or lfsr.

A data-driven variant replaces the fixed kernel by covariance patterns learned from all pairs in the screen, as in
multivariate adaptive shrinkage (mashr; Urbut et al. 2019), over cell types instead of terms or over both.

M1 suits the independent screen (the PTB 59-term case) and does not change the estimand: each β_{c,t} is still
the total shift of term t's genes.

### M2. Expression-scaled prior variance (K2)

Prior variance of β_{c,t} proportional to how much of term t's expression comes from type c:
v_{c,t} = τ² · max(a_{c,t}, a_min), with a_{c,t} the share of the term genes' reads in type c (from B, or an atlas
for the same tissue). A floor a_min keeps every pair estimable. Combined with M1 as a diagonal scaling of K.
Expected gain is small (B already enters the leakage model); worth testing only because it is cheap.

### M3. Response-atlas prior (K3)

A spike-and-slab prior whose inclusion probability depends on the cell type and pathway family:
β_{c,t} ~ (1 − p_{c,f(t)}) δ_0 + p_{c,f(t)} N(0, τ²), with p learned from a perturbation atlas (fraction of
perturbations of family f with a clear response in type c). This is the prior most likely to raise power, and the
most dangerous one: a type that responds against the atlas would be shrunk away, and that is often the finding a
study is looking for.

Conditions before any use:
- the leakage exclusions of `dl_plan` §3 (no benchmark drugs, cytokines, IFN family, cell lines, OneK1K donors;
  COMBAT, GSE60424 and in-house cohorts never in training), checked in code;
- a **prior-violation test set**: benchmark pairs where the atlas says "no response" but the truth has one;
  coverage and power there are reported, not only on average;
- every call is labelled with whether it holds without the prior.

### M4. Learned effect-size prior and error calibration (K4)

From a corpus of studies with known truth (T1 atlas pseudobulk under the leakage rules), learn:
- g, the distribution of true effects (spike-and-slab or normal mixture, per cell-type class);
- the error model: how the actual error relates to the reported SE (a calibration curve, as in conformal
  recalibration).

In a new study, apply the learned SE correction and then empirical-Bayes shrinkage with prior g. This is the
cheap, interpretable version of the amortized posterior in `dl_plan`. It fixes the failure of §2 directly: the
SEs are corrected before shrinkage, and g comes from many studies instead of the 28–56 pairs of one study.

## 5. Guardrails (for every method)

1. **Honest uncertainty first.** No prior is evaluated on SEs whose |error| / SE ratio is outside 0.7–0.9 on the
   development splits.
2. **Prior off = current method.** Each method has a switch; results with the prior off must reproduce the current
   fit exactly.
3. **Prior-driven calls are visible.** Report, for each call, whether it is also a call without the prior; report
   the posterior-to-prior width ratio per pair (as `dl_plan` §6).
4. **No leakage.** Data that trains a prior (M3, M4) is disjoint from every benchmark and evaluation cohort.
5. **Misspecification test.** Each method is also scored with a deliberately wrong prior (M1: shuffled term
   labels; M3: permuted families; M4: corpus from another tissue). Coverage must hold (S1 ≥ 0.90 per dataset).

## 6. Experiments (development splits 20–29 only; each a go/no-go)

Scored as the U- and D-series: S1 coverage ≥ 0.90 per dataset; null calls ≤ current + 0.02; power compared with
independent v5 and joint v5; no step kept for a gain inside split noise (0.03).

| Step | What | Data | Go if | Cost |
|---|---|---|---|---|
| **P0** | Honest independent SEs: exclusive-gene joint v5 (running), and if still conservative, an SE calibration from reference-donor simulations (not from benchmark truth) | 5 datasets × splits 20–24 (P0a, running); then 20–29 | \|error\| / SE in 0.7–0.9 per dataset, S1 holds, power above independent v5 by > 0.03 | P0a ~10 h CPU (in progress) |
| **P1** | M1 term-graph pooling with bootstrap error covariance | needs a **new benchmark arm with many overlapping programs**: GO BP / Hallmark terms on Kang, Parse IFN-β/γ, Tahoe (truth from `b.truth(program_fit="independent")` for any set) | power gain > 0.03 over P0 alone; S1 and null calls hold; holds with shuffled-label kernel (guardrail 5) | ~1 week code + benchmark arm |
| **P2** | M2 expression-scaled variance on top of P1 | as P1 | gain > 0.03 over P1; otherwise drop (expected) | ~2 days |
| **P3** | M3 response-atlas prior | as P1 + prior-violation set | power gain > 0.05 **and** coverage ≥ 0.90 on the prior-violation set | ~2 weeks (atlas pipeline shared with `dl_plan` D3) |
| **P4** | M4 learned prior + SE calibration, leave-dataset-out | T1 corpus; evaluated on the 13 dev datasets | beats P1 by > 0.03 with S1 on every dataset, including leave-tissue-out | ~2 weeks; merges into `dl_plan` D3 if both run |

**Order.** P0 first: it decides whether the independent SEs are usable at all. P1 next, because it needs no
outside data and has no leakage risk; its benchmark arm (many overlapping terms) is also what the PTB GO screen
needs. P2 only alongside P1. P3 and P4 wait for the atlas pipeline of `dl_plan` D3 and belong to Paper 2.

**Stop rules.** Stop P1 if the error covariance cannot be estimated stably (bootstrap S_c not positive definite
after shrinkage on ≥ 2 datasets) or if pooling fails S1 with the shuffled kernel. Stop P3 if coverage on the
prior-violation set is below 0.90 after one revision of the prior.

## 6b. Results (development)

**P0 (exclusive-gene fit), 2026-11-20: not adopted** after its held-out check (splits 90-99; worktree
`docs/heldout_excl_results.md`): power +0.276, but null calls 0.222 on McFarland trametinib (S1, S2 fail).

**P1 (M1 term-graph pooling), 2026-11-23: stopped at guardrail 1.** Benchmark arm: all 50 Hallmark sets, independent
v5, Kang, Parse IFN-beta, Parse IFN-gamma, Tahoe trametinib, splits 20-24 (20 studies; worktree
`scripts/experiments/term_pool_bench.py`, `term_pool_score.py`; output `benchmarks/dev/term_pool_score.txt`).
- No available SE is honest. Mean |error| / SE (0.80 if right): reported v5 0.34-0.47; bootstrap noise only
  3.1-27.5; simulated leakage RMSE + noise 0.31-0.36. The simulated leakage term is 95-100% of the reported
  variance and overstates the actual error about 2.3-fold.
- Pooling on these SEs shrinks to zero (fitted prior variances ~0): coverage 0.11 (reported SEs) or 0.19 (RMSE SEs)
  vs 0.991 unpooled. The shuffled kernel does as well as the real one, so the term structure adds nothing.
- The truth follows the kernel only weakly (Spearman of overlap with |beta_t - beta_u| -0.05 to -0.14).
- Independent v5 on 50 Hallmark terms: coverage 0.991, power 0.012, mean width 1.31.

**Consequence.** Every prior (M1-M4) needs an honest error model of the calibrated route first, and the
bottleneck is the simulated leakage error, not the sampling noise. P2 (on top of P1) is not run. Next candidate:
P0b, a leakage-error model that matches the actual error (e.g. planted effects scaled to the study's observed
effects, or the leakage term recalibrated across development studies as in M4's SE calibration), checked by the
|error| / SE ratio before any prior is tried again.

**P0b (honest SEs for the independent fit), 2026-11-23: both variants fail.**
- (a) `unmix_kw=dict(plant="observed")` (simulation plants the study's own effect pattern), Hallmark arm: no change
  (|err|/SE 0.31-0.48 vs 0.34-0.47; `benchmarks/dev/p0b_observed.txt`). The calibrated interval itself is about as
  wide as the leakage term (q_leak / h_cal 1.0-1.8), so both components are too wide in PBMC.
- (b) leakage term times a factor k learned leave-one-dataset-out (`p0b_recal.py`). Hallmark, 4 datasets: looked
  good (floor dropped: coverage 0.901, |err|/SE 0.76, power 0.019 -> 0.273, null calls 0.025). On the 13-dataset
  progeny7 benchmark (`TP_PROGRAMS=progeny7`, splits 20-24, `benchmarks/dev/p0b_recal_progeny7.txt`): pooled
  coverage 0.898, power 0.065 -> 0.400, but null calls 0.075 pooled and McFarland trametinib 0.198, Tahoe
  gemcitabine 0.122, Zhang 0.113 (fails S1). With the h_cal floor kept: null calls 0.015, cell-line coverage
  0.74-0.77, power +0.04.
- Reason: the current SEs are near honest in cell lines (|err|/SE 0.56-0.72) and 2-3x too wide in PBMC
  (0.27-0.43). One factor cannot serve both; the same cell-line vs PBMC split as the exclusive-gene and gated-v4
  failures. A recalibration would need a study-level predictor of the regime that does not use the truth.

**Decision (2026-11-23): the prior-knowledge line is closed for Paper 1.** The independent fit keeps its current
(conservative) intervals, stated as a limitation. Learning the error model and the priors (M3, M4, and a
study-adaptive version of P0b) moves to Paper 2, where the amortized model learns them from many training studies
(`docs/research_plan.md` §4.0, step D1b). Splits 80-89 stay unused.

## 7. Held-out confirmation

Nothing above touches held-out splits. Before any held-out run, a protocol in the style of `method_spec.md` is
written and frozen: the method and its switch, the prior and its data manifest, the decision rule. Blocks in use
or proposed: 30–39 (v5, used), 40–49 (v4, used), 50–59 (association section), 60–69 (PALM-uni, proposed), 70–79
(`dl_plan`, proposed). Proposed for this plan: **80–89**, to be confirmed unused. Real-bulk check: PTB (with its
run confounding stated) and COMBAT, which no step trains on.

## 8. Where it sits in the papers

- **Paper 1:** unchanged. The independent fit and P0 can appear as an option of the software with its development
  evidence; M1 only after its own held-out test.
- **Paper 2:** M3 and M4 are natural parts of "learn from the database": the response-atlas prior and the learned
  error model are interpretable components that the amortized posterior can be compared with (ablation: prior
  only, amortized only, both).

## 9. Risks

| Risk | Sign | Response |
|---|---|---|
| Prior hides real, unexpected responses | calls lost on the prior-violation set | report prior-off calls alongside; M3 stays opt-in |
| Error covariance underestimated (M1) | coverage drops as term overlap rises | stratify coverage by overlap; inflate S_c off-diagonals to the bootstrap upper bound |
| Atlas transport (M3, M4) | gains on immune benchmarks, none on tissue | leave-tissue-out in P4; report by tissue |
| Benchmark too simple for M1 | PROGENy 7 has little overlap, so no gain is possible | the GO / Hallmark benchmark arm in P1 is a prerequisite, not an extra |
| Calibration tuned on benchmark truth | good dev numbers, poor held-out | SE calibration from reference-donor simulations only; benchmark truth only scores |

## 10. Key references (from memory; verify with nature-citation before citing)

- Stephens M. False discovery rates: a new deal. Biostatistics 2017 (ashr, adaptive shrinkage, lfsr).
- Urbut SM, Wang G, Carbonetto P, Stephens M. Flexible statistical methods for estimating and testing effects in
  genomic studies with multiple conditions. Nature Genetics 2019 (mashr).
- Efron B. Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction. Cambridge 2010.
- Alexa A, Rahnenführer J, Lengauer T. Improved scoring of functional groups from gene expression data by
  decorrelating GO graph structure. Bioinformatics 2006 (topGO).
- Cui A, et al. Dictionary of immune responses to cytokines at single-cell resolution. Nature 2024.
- Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty
  quantification. 2021 (SE recalibration).
