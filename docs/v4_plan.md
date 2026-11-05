# Method v3 (narrowed) and v4: plan after the Quantitative Core review (draft, 2026-11-05)

Inputs: `palm_v3_methods_review.md` and `palm_v3_idea_register.md` (Quantitative Core, 2026-11-05), written
against the "PALM Open Problems" status report of the same date. Each claim was checked against the code
before being adopted; corrections are noted. Supersedes the order of work in `docs/v3_plan.md` (its results
and candidate list stay valid).

## 0. Decision proposed

Split the remaining work into two freezes with different jobs:

- **v3 = correctness and scoring.** No new effect estimator. The v2 estimator with:
  - the routing made valid (no choice on realised interval widths);
  - a residual-variance gate;
  - the in-study self-tests;
  - the new scoring and pre-registered decision rule.

  Short (about 2–3 weeks). This is the version Paper 1's claims rest on. Held-out: splits 30–39, with **v2
  and v3 both run there** so that the comparison is out of sample for both.
- **v4 = the estimator redesign.** A shared-plus-deviation parameterisation: GLS for the deviations, a
  hierarchical between-type spread, an explicit off-program component and data-anchored variance. It
  replaces the router and the pooling variants. Development on splits 0–2 and 20–39 after the v3 report;
  held-out on new splits 40–49.

Alternative (PI decision): skip the v3 freeze and make one freeze with everything. This means fewer freezes
to correct for multiplicity, but Paper 1 waits for the v4 estimator (about a quarter). Recommended: two
freezes. The v3 changes are needed whatever the estimator, and they fix a validity issue in the current
claims.

## 1. Review claims checked against the code

| # | Claim | Check | Verdict | Goes to |
|---|---|---|---|---|
| R1 | Splits 20–29 are not held-out for v3 | Protocol already labels them v3 development and reserves 30–39. Gap: plan item 5 would have shown a v2-vs-v3 table on 20–29 | **Agree in part.** The 20–29 comparison is go/no-go only, labelled as biased towards v3. Run v2 and v3 on 30–39 together. Pre-declare the variant before 30–39 | v3 |
| R2 | The narrower-interval guard voids coverage | `choose="width_guard"` (deconvolution.py:288) picks unmix vs calibrated per program by realised width. Held-out coverage 0.93–1.00 is measured end to end, so the *average* claim stands empirically. Coverage per route and per type is not known | **Agree, overstated.** Not "void", but selection on realised widths has no conditional guarantee, and the shortfall would sit on the contrasts that matter. Route by quantities fixed before the outcome (design limit, conditioning), or report the union. Report coverage per route | v3 |
| R3 | Fix the scoring before choosing estimators | Pooling and GLS were judged on overall r plus an untargeted within-program r | **Agree** | v3, step 1 |
| R4 | P1, P2, P3 and P7 are one ill-conditioned design: μ is well determined, δ is set by the centred design Ã | Matches the design-limit results (Kang 0.19 vs 0.04–0.05). `pool="partial"` already writes β = common + w·d but *shrinks d*; `gls_unmix` estimates d without shrinkage | **Agree.** Make (μ, δ) the estimator's parameterisation, not only a scoring split. Drop the router | v4 |
| R5 | Hierarchical τ_p across programs, studies and the atlas | Only 7 programs (PROGENy), and 4–9 types | **Agree, smaller gain than implied.** The atlas prior carries most of the weight | v4 |
| R6 | P7 is confounding, not calibration | `single_group.py` permutes z, which breaks the z–composition link, so a near-nominal permutation null is *expected* under confounding | **Agree.** Run the signed-error vs corr(z, π_c) diagnostic before building item 14. If confounding: residual permutation (composition regressed on z), and a partial design limit | v3 diagnostic |
| R7 | Residual-variance gate as a freeze criterion | v2 already measures off-program τ for its own intervals. The GLS calibration lacked it (the McFarland failure, found by inspection) | **Agree.** Generalise to every estimator: observed vs simulated residual variance, blocking | v3 |
| R8 | Data-anchored variance instead of the simulator as the only scale | v2 unmix already uses a cluster bootstrap plus simulation widening; GLS has none | **Agree** for the new estimator (sandwich or cluster bootstrap); the simulator kept for validation | v4 |
| R9 | Split the design limit into collinearity and noise before building gene-level unmixing | `identifiability.py` has the pieces | **Agree.** Cheap, and it decides v3-plan item 5 | v3 diagnostic |
| R10 | COMBAT paired samples: cross-fitted bulk↔pseudobulk mapping | COMBAT is also our only real-bulk test | **Agree with cost.** Calibrating on it uses up part of its evaluation role; strict cross-fitting, and transfer to TB stated as an assumption | v4 |
| R11 | RNA content is a structural non-identifiability of activity; CBC breaks it | π is the **read** fraction. RNA content only converts read fractions to cell fractions (and "more cells" claims), so β does not depend on it | **Correction.** CBC anchors the composition (cells) and the cell-number interpretation, not β. Still recommend the EGA request, with an expected-gain number attached | PI |
| R12 | Bound the harm of a missing type by simulation | — | **Agree.** Cheap; reported as the sensitivity statement for P6 | v3 diagnostic |
| R13 | The truth is noisy (attenuated r); errors-in-variables | Truth split-to-split SD median 0.006–0.017, larger for rare types | **Agree.** Bootstrap SE over cells for each truth value; disattenuated r beside raw r | v3 scoring |
| R14 | Splits share donors, so effective df ≈ donors | Kang 8 donors, Parse 12 | **Agree.** Donor-aware intervals on every mean and on coverage | v3 scoring |
| R15 | False-call target per pair or per study? | Currently per pair | **Agree.** State it per pair; add a study-level rate (any false call per study) as a secondary readout | v3 scoring |
| R16 | Pre-register the 30–39 decision rule | — | **Agree.** Written before the freeze, with multiplicity across freezes | v3 |
| R17 | Footprints shared across types: test first | v3-plan item 11 (not started) | **Agree** | v4 gate |
| R18 | Ambient RNA inflates "shared" response in the Kang truth | Kang is droplet data; truth is per-type pseudobulk of the other half of the reads | **Agree, cheap and important.** Recompute the truth after ambient correction (decontX, cluster-based; Kang's empty droplets are probably unavailable). Same for Parse | v3 diagnostic |
| R19 | Trametinib loss may be saturation or non-linearity | GLS attenuates Tahoe MAPK (−0.13 to −0.21 vs −0.26); planting larger effects did not help | **Agree.** Residual-vs-fitted on the strong-response datasets | v3 diagnostic |
| R20 | Two-step vs full fit for rare types: compare coverage, not r | — | **Agree** | v4 |

## 2. Idea register: adopted, deferred, adjusted

| # | Idea | Decision | Note |
|---|---|---|---|
| 3 | Marker-program leakage assay | **Adopt, v3** | Truth-free leakage on every study, including COMBAT and TB. Exclusivity checked in the atlas; size and expression matched |
| 4 | Matched null programs | **Adopt, v3** | 20 per study; in-study false-call rate. Extends the random gene sets already planned for the in-house PTB data |
| 8 | Report in the resolvable-contrast basis (SVD of Ã) | **Adopt, v3 reporting** | Extends the design-limit readout from a number to its structure |
| 16 | P(\|δ\| > t) instead of binary calls | Adopt, v4 output | Calls kept for comparability with v2 |
| 7 | Explicit off-program component | **v4 core** | Follows directly from the McFarland finding; also a missing-type detector. Needs an orthogonality constraint |
| 2 | Composition measured with error | v4, diagnostic first | Partly in already: the calibration reruns fraction estimation on simulated data, so sampling error of π is inside coverage. Missing: the platform error of real bulk. First, simulate with fractions perturbed to the COMBAT accuracy and check the induced leakage |
| 1 | Low-rank type × program deviations | v4 candidate, lower priority | With 7 programs the gain is at most about √7 on the type profile; it cannot touch Ã's null space. Test on Tahoe and McFarland (prediction: no gain on Kang) |
| 6 | Conformal intervals on COMBAT pairs | v4, shares COMBAT with R10 | One cross-fitting scheme for both |
| 9 | Coarser functionals (shared vs restricted; number of responders) | v4 | Check against the design limit first |
| 10 | Type-specific footprints | Gated on R17 | — |
| 11 | Gene-level count likelihood | Gated on R9 | Only if Kang's limit is noise-dominated |
| 14 | Reference-ensemble sensitivity | v4 reporting | — |
| 5 | Orthogonalised estimating equations | Long term | Already in v3-plan framing |
| 12 | Dose or severity curves | Later | Where several doses exist |
| 13 | Cross-study responsiveness prior | Later (deployment) | Kept out of the benchmark loop |
| 15 | c-optimal sample selection | When the TB contrast is named | Also gives the number for the EGA request |

## 3. v3 (narrowed): steps and gates

Development data only (splits 0–2, 20–29). Stop before the freeze for the PI's go-ahead.

1. **Scoring and relabelling** (R3, R13–R15, P8, P10):
   - truth-based and label-based measures, with their disagreement rate;
   - per-program thresholds from target-gene fold change;
   - shared and deviation scores;
   - interval score;
   - truth standard errors and disattenuated r;
   - donor-aware confidence intervals;
   - per-route coverage;
   - encorafenib and gemcitabine relabelled from target genes and TP53/BRAF status.

   Re-score v2 under it (as v2's held-out result on 20–29, and as development).
2. **Diagnostics, in parallel**, each deciding a build:
   - R6: confounding plot for the single-group path;
   - R9: collinearity vs noise;
   - R18: ambient-corrected Kang truth;
   - R19: saturation on trametinib;
   - R12: missing-type harm.
3. **Routing without realised widths** (R2): choose per program by the design limit and conditioning, fixed
   before the outcome; compare with the union. Pick the rule on development data, and report coverage per
   route.
4. **Residual-variance gate** (R7) and the **self-tests** (ideas 3 and 4) in the package, reported per study.
5. **Pre-register** the decision rule for 30–39 (R16), then **freeze v3** after the PI's go-ahead, and run v2
   and v3 once on 30–39.

Exit gate for v3: at no cost in coverage, any routing change must keep held-out power within the
development noise of v2's.

## 4. v4: the estimator redesign

After the v3 report. Development on 0–2 and 20–39; held-out on 40–49.

1. **(μ_p, δ_cp) parameterisation:**
   - μ from the overall signal;
   - δ by GLS on the centred design Ã;
   - a hierarchical τ_p (across programs, with an atlas prior) integrated into δ's interval;
   - no router.

   Start from `gls_unmix` and `gls_calibrated` with `offprog_sd=τ`. Development results so far, splits
   20–22: within-program r better than v2 on 4 of 6 drugs, coverage 0.86–0.99, and the loss on trametinib.
2. **Off-program component** (idea 7) in the model, replacing the simulated τ.
3. **Data-anchored variance** (R8): sandwich or cluster bootstrap. The simulator becomes a check.
4. Gated additions:
   - type-specific footprints (if R17 shows heterogeneity);
   - gene-level likelihood (if R9 says noise);
   - measurement-error fractions (if idea 2's diagnostic reproduces the cross-type errors);
   - low-rank deviations.
5. COMBAT cross-fitted calibration and conformal intervals (R10, idea 6), with transfer to TB stated as an
   assumption.

Success for v4 (development): within-program r and true effects called above v3's on cell lines with
coverage ≥ 0.85, false calls ≤ 0.1, no loss on strong shared responses (trametinib), and the self-tests
passing. Kang's power is not expected to change (design limit).

## 5. Decisions for the PI

1. Two freezes (v3 correctness and scoring now, v4 estimator later) or one combined freeze.
2. EGA request for COMBAT blood counts, with the expected gain computed first (design-limit calculator,
   about a day). Framed as anchoring composition, not β (R11 correction).
3. Download COMBAT whole-blood CyTOF (public, 0.9 GB): yes recommended.
4. A whole-blood single-cell dataset with granulocytes: decide after the missing-type harm simulation (R12).

## 6. Idea register, first results (2026-11-05, splits 20–22; `scripts/experiments/v4dev_summary.py`)

- **Self-tests (ideas 3 and 4; `program2cell.selftest`, `selftest_bench.py`; v2, splits 20–21):**
  - Null programs are 3 × 7 per split, matched gene by gene on expression, from genes outside every program and
    not specific to one type, with gene selection rerun.
  - Raw null call rate: Kang, Zhang and Parse TNF-α 0.006–0.007; Tahoe RMC-6236 0.07; McFarland idasanutlin
    **0.27**. The McFarland calls are correct calls of tiny real off-program shifts (called truth ≈ 0.04,
    coverage 0.99; τ ≈ 0.6 there). The null rate therefore needs an **effect floor**: calls with
    |estimate| ≥ 0.1 give 0.000–0.037 on all five datasets.
  - Marker leakage (calls in non-owner types): 0.000 on Kang, Zhang, Parse and McFarland; 0.02 on Tahoe.
  - Truth for marker programs in non-owner types is noise (genes barely expressed there), so leakage is
    judged in-study only.
  - Caveat: in assembled benchmarks the treatment barely moves composition, so the leakage assay is weak
    there. It has to run on COMBAT real bulk, where composition shifts with disease.
- **Fractions measured with error (idea 2; `fraction_error.py`, split 20):**
  - Per-sample fraction noise at real-bulk level (per-type r ≈ 0.4–0.5) breaks the GLS estimator on McFarland:
    coverage 0.50, false calls 0.43, within-program r −0.13. v2 degrades less (0.81 / 0.14). Tahoe GLS:
    coverage 0.84.
  - Shared per-type bias (platform-like level error, r unchanged) is nearly harmless for both.
  - PALM-lite on the benchmarks is accurate (r 0.96–0.98 on cell lines, 0.71 on Kang), so this is a
    real-bulk risk. The v4 estimator must propagate fraction error; the simulation alone re-estimates
    fractions only from noise-free-platform data.
- **Low-rank deviations (idea 1; `lowrank_dev.py`):** rejected. Rank 1 lowers within-program r on cell lines
  (Tahoe gemcitabine 0.79 → 0.04, encorafenib 0.37 → 0.17, RMC-6236 0.61 → 0.47); rank 2 ≈ full; no gain
  anywhere. The review predicted a gain on cell lines. With 7 programs there is too little shared structure
  to borrow.

## 7. Fraction-error propagation (2026-11-05, splits 20–22; `frac_prop.py`, `combat_fracnoise.py`, `frac_prop_summary.py`)

Change: `gls_calibrated(..., frac_sd=s)` perturbs the simulated cohorts' re-estimated fractions per sample,
softmax(log π + N(0, s²)), so the calibration (slope k and error quantile) sees the study's fraction error.
The level s comes from `estimate_frac_sd`: fraction misfit of the bulk (depth-corrected squared relative
error beyond Poisson), matched between study and simulations.

**Real-bulk fraction noise (COMBAT sepsis, splits 20–24).** PALM-lite fractions vs the same specimens'
single-cell read fractions:

| Bulk | Per-type r | MAE | Matching s |
|---|---|---|---|
| Single-cell pseudobulk | 0.90 | 0.037 | 0.27 (0.25–0.30) |
| Real whole blood | 0.47 | 0.137 | 1.03 (0.85–1.20) |
| Real, self-offset | 0.57 | 0.156 | 0.81 (0.65–0.95) |

Real bulk carries per-sample fraction noise s ≈ 1, more than the `rand` condition of §6 on Kang (0.40) and
similar to McFarland and Tahoe (0.87–0.90).

**Propagation with the true level (oracle), noisy fractions:**

| Dataset | Coverage none → oracle | False calls | Power | Within-program r |
|---|---|---|---|---|
| Tahoe RMC-6236 | 0.85 → 0.97 | 0.05 → 0.00 | 0.63 → 0.38 | 0.24 → 0.23 |
| McFarland idasanutlin | 0.59 → 0.76 | 0.31 → 0.13 | 0.47 → 0.47 | −0.03 → −0.03 |
| Kang | 1.00 → 0.99 | 0.06 → 0.00 | 0.28 → 0.34 | 0.69 → 0.65 |

Propagation repairs the intervals on Tahoe and Kang. On McFarland it falls short of nominal even at the
exact level (0.74–0.79 per split), and the within-program ranking is lost (r ≈ 0): at s ≈ 1 the cell-line
design no longer separates the types, and no interval can recover a ranking the estimate does not contain.

**Estimating the level from the bulk.**
- `anchor="absolute"` (raw misfit) fails both ways: it reads baseline misfit (biology, platform) as fraction
  error (Kang true 0 → 1.20, sys 0 → 1.13) and misses real error on Tahoe (0.90 → 0.00). Rejected.
- `anchor="excess"` (misfit beyond the study's own PALM-lite fit, in study and simulations alike): mean
  estimates against the injected level: Kang 0.43 vs 0.40, McFarland 0.94 vs 0.87, Tahoe 0.63 vs 0.90.
  False readings without per-sample noise are 0.07–0.36, and they cost power on McFarland (true fractions:
  0.73 → 0.59), not coverage. Single splits vary (Kang 0.00 / 0.43 / 0.84 at true 0.45 / 0.35 / ~0.4).
- Limit: with PALM-lite fractions the excess is 0 by construction. The bulk cannot show an error its own
  best fit shares, which is the real-bulk case (reference mismatch).

**Decision for v4.**
- Fraction noise s is an input of the v4 estimator with two sources:
  - external fractions (CyTOF, cell counts, another tool): `excess` estimate;
  - PALM-lite fractions on real bulk: a fixed prior level from COMBAT, s = 1.0 (raw) / 0.8 (self-offset),
    reported with the result; assembled bulk: s = 0.
- On real bulk at s ≈ 1, expect cell-line-like designs to lose type resolution (McFarland result). The
  report should then say so: shared response only, with type-specific calls withheld when the design limit
  at s exceeds the effect floor. This is a new design-limit input (the limit at the fraction noise level),
  to be built next.
- Sensitivity: on McFarland, an overestimate of 0.2–0.3 costs 0.1–0.15 power and raises false calls to
  0.10–0.17 on two of three splits (the slope k inflates the effects). An underestimate costs coverage.
  Neither direction is free.

## 8. v4 estimator in the package, first comparison, and more idea-register tests (2026-11-05)

**v4 in program2cell** (`core/gls.py`, `program_deconvolution(estimator="v4", v4_kw=...)`; opt-in, not frozen):
GLS unmixing, off-program τ and stated fraction noise in the calibration, shared response μ and deviations in
`info["v4"]`. About 1 minute per study, against several minutes for v2.

**v4 vs v2, 13 datasets, splits 20–29** (`v4_compare.py`; Tahoe gemcitabine 7 splits):

| | Coverage | r | Within-program r | False calls | Power | Width |
|---|---|---|---|---|---|---|
| v2 | 0.98 | 0.78 | 0.62 | 0.02 | 0.27 | 0.30 |
| v4 | 0.95 | 0.47 | 0.30 | 0.04 | 0.26 | 0.48 |

- Cell lines: v4 better or equal (McFarland idasanutlin within-program r 0.36 → 0.72, power 0.49 → 0.67;
  Tahoe encorafenib power 0.22 → 0.50; Tahoe gemcitabine and RMC-6236 within-program r up); trametinib worse
  (Tahoe power 0.79 → 0.45).
- PBMC: v4 fails. Parse r 0.85–0.93 → 0.07–0.30 with intervals 2–3 times wider; Zhang (true effects ≤ 0.05)
  coverage 0.75, false calls 0.23.

**Two fixes, quick test (6 datasets, splits 20–22; estimator `v4s`):**
- Study-vs-simulation noise ratio (intervals scaled up when the study's GLS standard errors exceed the
  simulated cohorts'): Zhang coverage 0.80 → 0.93, false calls 0.18 → 0.06. **Keep.**
- Shrinkage of deviations towards μ (method of moments, calibrated as shrunk): Parse unchanged (r 0.23 / 0.09).
  Not the cause.

**Idea 9, coarser functionals** (`coarse_functionals.py`, saved outputs; programs with max |β| ≥ 0.1):
- v2 recovers the shared response μ almost everywhere (r ≥ 0.94 except Parse TNF-α 0.81 and Tahoe
  gemcitabine), including Kang, where the per-type table is weaker (0.82).
- v4's Parse failure is in μ itself (r 0.34–0.85 vs v2 0.81–0.99), so it is not a deviation problem.
- Shared vs restricted (h = sd / (|μ| + sd)): v4 better on cell lines (Tahoe encorafenib r 0.54 → 0.92,
  RMC-6236 0.89 → 0.96, trametinib 0.85 → 0.95).
- Number of responding types: poor for both (mean error 1–3 types; conservative calls).
- Decision: report μ and shared-vs-restricted alongside the per-type table (v3 reporting).

**Idea 10 gate, footprint heterogeneity (R17)** (`footprint_heterogeneity.py`; single-cell truth side):
mean pairwise r of per-type gene log fold changes over a program's genes, responding programs.
- PBMC cytokines share one footprint: Parse IFN-β 0.51–0.81, TNF-α 0.66–0.76, Kang 0.38–0.85 (JAK-STAT
  0.81–0.85).
- Cell lines do not: McFarland 0.09–0.37, Tahoe 0.38–0.64.
- So type-specific footprints do not explain v4's Parse failure. They are a cell-line issue (where v4 already
  does well). Idea 10 stays gated; its target is cell lines.
- Open: the Parse cause. A broad shared cytokine response moves the reallocated activity of every program
  (NF-κB and TNF-α respond with weak weight agreement, 0.13–0.41), which GLS attributes through the leakage
  operator while v2's calibrated reallocation absorbs it per type. To test next: v4 μ with and without the
  off-program τ, and GLS on Parse with v2's calibrated slopes.

**F6, per-sample fit score as a missing-type detector** (`missing_type_score.py`, COMBAT, split 20 reference):
- Drop-one-type on single-cell pseudobulk: the score rises with the dropped type's true read fraction
  (Spearman 0.45–0.88 for 6 of 7 types; B cells 0.22).
- COMBAT real whole-blood bulk, 131 specimens with CyTOF: Spearman of the score with the granulocyte
  frequency (neutrophil + eosinophil + basophil; absent from the PBMC reference) **0.72**. Control, the same
  specimens' single-cell pseudobulk (no granulocytes): 0.18.
- **Adopt** as a per-sample report next to the self-tests (cheap; validated on real data with an
  independent measurement).

**Self-tests on COMBAT real bulk** (`combat_selftest.py`, v2, sepsis, splits 20–21):

| Bulk | Null-program calls | Marker leakage calls | Leaked \|effect\| | Real-program calls |
|---|---|---|---|---|
| Single-cell pseudobulk | 0–0.02 | 0 | 0.06–0.08 | 0 |
| Real whole blood | 0–0.02 | 0 | 0.15 | 0 |
| Real, self-offset | 0–0.04 | 0 | 0.17–0.21 | 1 |

v2 makes no false calls on real bulk, but it also makes essentially no calls: its intervals absorb a leaked
effect size twice that of the pseudobulk. Safe, no power on this contrast.

**Not tested, with reasons:** F1/F2 (fraction uncertainty) is covered by section 7's stated fraction-noise
input; F3 EPIC-unmix and F5 robust neural posterior estimation are large builds; F4 (tensor decomposition
through the mixing operator) is the principled form of idea 1, which was rejected at 7 programs; F7 and idea 6
(conformal on COMBAT) need calls on COMBAT, which v2 does not make; ideas 5, 11, 12, 13 and 15 are long-term
or need data not in hand; 8, 14 and 16 are reporting.
