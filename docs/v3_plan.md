# Method v3: candidate changes (draft, 2026-11-04)

**Superseded order (2026-11-05):** after the Quantitative Core review, the order of work and the split into a
narrowed v3 (correctness and scoring) and a v4 (estimator redesign) are in `docs/v4_plan.md`. Results below stand.

Status: candidates only. Nothing here changes the frozen method (v2, tag `method-freeze-v2`). The final plan
is written after the held-out v2 report, matching each candidate to where held-out fell short.

**Protocol.**
- v3 is developed on development splits 0–2 and on 20–29, which become development data for v3 once their
  v2 results are reported.
- New behaviour is opt-in until frozen; v2 stays the package default.
- Before the v3 freeze and its held-out run (new splits 30–39, run once), the development results are shown
  to the PI for a go-ahead. The freeze is a pre-registration decision.
- A candidate that needs data we do not have is reported as blocked; missing data is not simulated in its
  place.

Sources: our own list (2026-11-03), an external critique (`PALM_biological_critique.md`, 2026-11-04; points
checked against the code) and a statistics literature scan (`v3_cs_stats_scan.md`, 2026-11-04; arXiv IDs
given below are from that scan and **not yet verified**: check each before citing). What was not adopted, and
why, is listed at the end.

**Expected gain.** Most items make the method more defensible, not more accurate. The ones that can change
results are 5, 6 (with blood counts), 8, 9 and 10; of the method changes, **test 9 (MASH-style sharing)
first**: it targets the weakest held-out regime (broad responses).

## Matched to held-out v2 (2026-11-04; docs/heldout_v2_results.md)

| Where held-out fell short | Evidence | Candidates |
|---|---|---|
| **Unmixing a broad response costs accuracy.** For calibrated-route programs PALM's point estimate equals `diff`'s, so its r loss is from unmixed programs | Kang r 0.82 vs `diff` 0.94; unmixed blood pairs RMSE 0.27 vs 0.03 calibrated; Kang's unmixed program is JAK-STAT (interferon, all types respond) | **9** (sharing as the estimand: a shared response plus per-type deviations, instead of unmixing each type alone) |
| **Power in blood** | Kang: 40% of true effects called, mean interval width 0.75 (current practice 0.08) | 9, 5 |
| **Weak drug responses** | Tahoe encorafenib / gemcitabine r 0.42 / 0.54 (best comparator 0.61 / 0.75), almost no calls | 9 (if unmixed programs drive the loss, as on Kang: to check), 10 |
| **Real bulk: specific but no power** | COMBAT ≤ 1 call per split; fraction r 0.09–0.70 on real bulk | 6, 7, 12 (globin and platelets: testable now on COMBAT development splits) |
| **Reporting gaps** | guard firings not recorded; no conditioning readout | 4, 4b, 4d |
| Not a method problem | McFarland p53 called in TP53-mutant lines (57% of 3 pairs), which show a small real induction | none (report with the measured induction) |

**Order of work for step 6 (development data only; stop before the freeze):**
1. Record the guard and the conditioning per study (4, 4b): cheap, and needed to read every later comparison.
2. Candidate 9, sharing across cell types: first the simple form (shared s_p + deviations d_cp with REML
   shrinkage, routed by the same width rule), then the MASH-style learned sharing if the simple form helps.
   Success: Kang r back towards `diff`'s 0.94 with at least `auto_v2`'s power and coverage ≥ 0.9, no loss on
   McFarland and Tahoe trametinib.
   - **First result (2026-11-04, splits 20–29, all-or-nothing pooling, `pool=True`):** Kang r 0.82 → 0.95,
     Tahoe RMC-6236 r 0.72 → 0.84 and true effects called 0.78 → 0.94; but false calls rose on McFarland
     idasanutlin (0.05 → 0.11, above target) and Zhang (0.00 → 0.09, r 0.85 → 0.80). Cause: the new false
     calls are tiny true effects (|β| 0.01–0.05) given the average over types, with an interval that took
     the between-type spread as zero (4–6 types cannot estimate it); not leakage of a large response.
   - **Next (`pool="partial"`):** empirical-Bayes partial pooling (no test; weight τ²/(τ²+s²)), interval
     from the upper confidence bound of τ² (Q-profile). Screen on Kang, Zhang, McFarland idasanutlin,
     Tahoe gemcitabine and RMC-6236, splits 20–24.
   - **Partial pooling, splits 20–24 (v2 / all-or-nothing / partial):** false calls back to v2 level or below
     (Zhang 0.00 / 0.11 / 0.00; McFarland 0.04 / 0.09 / 0.03; gemcitabine 0.05 / 0.11 / 0.01), and the r gains
     kept (Kang 0.90 / 0.95 / 0.96; gemcitabine 0.54 / 0.64 / 0.65; RMC-6236 0.73 / 0.84 / 0.83), coverage ≥ 0.95.
     But power on truth |β| ≥ 0.1 is a little lower (Kang 0.22 → 0.17, McFarland 0.50 → 0.43), and
     within-program r across types still falls (McFarland 0.36 → 0.24, RMC-6236 0.21 → −0.02). Not yet a v3
     candidate: the shrinkage weight still pulls responding lines towards the mean.
   - **Headroom check (2026-11-04, splits 20–24; `scripts/experiments/gls_headroom.py`):** design limit
     (`identifiability.py` mde_full) median 0.04–0.05 on McFarland and Tahoe (97–100% of |β| ≥ 0.1 and 64–89% of
     type deviations ≥ 0.05 resolvable), 0.19 on Kang (49%). A plain GLS unmixing with every sample's own leakage
     operator ranks types within a program well (r ≈ 0.7) but overstates sizes 1.3–5× (type-specific). With a
     per-(type, program) scale learned on simulated cohorts (`gls_calibrated`):
     Tahoe RMC-6236 within-program r 0.21 → 0.61, true effects called 0.69 → 0.73, r 0.73 → 0.78; gemcitabine
     r 0.55 → 0.62; encorafenib r 0.39 → 0.47, called 0.27 → 0.40; false calls ≤ 0.07, coverage 0.87–0.91. But
     Tahoe trametinib worse (called 0.81 → 0.59, within-program r 0.63 → 0.33) and on McFarland its intervals
     are far too narrow (coverage 0.59–0.63, false calls 0.41): the simulation misses an error source there.
     Next: find that source (fraction error of PALM-lite on paired mixtures?) and route by the design limit.
   - **Source found (2026-11-05, splits 20–22; `scripts/experiments/gls_diag.py`):** the simulated cohorts
     lacked the off-program response (`offprogram_tau`, τ ≈ 0.57–0.69 on McFarland), which v2's design check
     already plants. With `offprog_sd=τ`: McFarland idasanutlin coverage 0.64 → 0.92, false calls 0.35 → 0.06
     (v2 0.98 / 0.03), true effects called 0.67 (v2 0.55), within-program r 0.66 (v2 0.44); McFarland
     trametinib coverage 0.86, false calls 0.04, called 0.87 (v2 0.83), within-program r 0.66 (v2 0.85).
     Fraction estimator not the cause (PALM draws same or worse than PALM-lite). Tahoe: coverage 0.96–0.99,
     false calls ≤ 0.05; within-program r vs v2: RMC-6236 0.61 / 0.23 (called 0.53 / 0.69), gemcitabine
     0.79 / 0.64, encorafenib 0.37 / 0.16 (called 0.50 / 0.33), trametinib 0.36 / 0.58 (called 0.56 / 0.88).
   - Trametinib (both datasets) is the remaining loss. A shared/spread rule does not separate it from
     RMC-6236 (ratio 1.4–3.2 vs 2.1–3.4). Estimates are attenuated (Tahoe MAPK −0.13 to −0.21 vs true −0.26),
     but planting larger effects in the calibration (`plant_mult=2`) is worse everywhere (Tahoe trametinib
     called 0.56 → 0.20, RMC-6236 0.53 → 0.15): rejected. Next: route by comparing the two estimators'
     intervals per program (e.g. the narrower honest one), checked on all development datasets.
   - **Metric check against the biology (2026-11-04, splits 20–29):**
     - Encorafenib known answer (BRAF-V600 lines LOX-IMVI, HT-29, RKO) is not supported by the measured truth:
       MAPK −0.02 / −0.14 / −0.06, while KRAS-G12C NCI-H2030 is −0.10. Unlike dabrafenib, it was never checked
       against target genes. Its true-effect share and AUROC are not evidence either way until relabelled.
     - Gemcitabine p53: truth +0.08 to +0.10 in the labelled TP53-wild-type lines and +0.08 in LOX-IMVI
       (labelled non-responder; TP53 status to verify). Every response is in the 0.05–0.1 grey zone, so the
       target is out of reach for any interval method at this effect size.
     - Zhang: max |mean truth| 0.02; only 2 of 10 splits have any pair ≥ 0.1, so its r (a held-out "miss") is
       noise. Zhang is a null / specificity dataset; its v3 false calls (0.09) are the real signal there.
     - Parse IFN-β "pass" is on coverage and r only: JAK-STAT truth is +0.07 to +0.13 in every type, and PALM
       calls 2% of the pairs ≥ 0.1. Parse TNF-α: truth ≤ 0.08 (3 splits with any pair ≥ 0.1).
     - The same pair can count as a true effect and a false call: label-based responders with null truth
       (McFarland trametinib LNCaP MAPK −0.04; encorafenib LOX-IMVI).
     - Overall r over all type × program pairs is dominated by the largest program and hides cell-type
       specificity: within-program, across-type r falls with all-or-nothing pooling (Tahoe trametinib
       0.55 → 0.32, RMC-6236 0.28 → −0.20) while overall r rises.
     - To add as readouts: true effects called defined by the measured truth (|β| ≥ 0.1, right sign);
       wrong-sign calls; within-program across-type r. Fix the encorafenib and gemcitabine labels from
       target genes and TP53/BRAF status before using them.
3. Candidate 12 on COMBAT (globin / platelet masking): fraction r and effect r on real bulk.
4. Candidate 5 (gene-level unmixing) if 9 does not recover Kang's power.
5. Blocked until data: 1 (granulocyte truth), 6 and 7 (blood counts), 13 (ordered TB axis needs the cohort).

**Data check (2026-11-04):**
- COMBAT blood counts: the data dictionary (`CBD-KEY-CLINVAR`, Zenodo 6120249) lists `neutrophil_count`,
  `lymphocyte_count`, `monocyte_count`, `white_cell_count` and `platelet_count` at sampling (±24 h), but only in
  the controlled clinical data (EGA, EGAS00001005493, data access agreement). The public file has none.
  Candidates 6–7 on COMBAT need that agreement: a PI decision.
- COMBAT whole-blood CyTOF (`CBD-KEY-CYTOF-WB`, 0.9 GB, public) gives granulocyte *composition* truth for the
  same samples: the fraction half of candidate 1, available now. No activity truth.
- COMBAT FACS is on PBMC (no granulocytes).
- Candidate 12 on COMBAT: globin genes are 0.1% of bulk reads (median; max 0.6%), platelet genes 0.3% (max
  1.3%): the library was evidently globin-depleted, so masking would change little here. It stays relevant for
  TB cohorts sequenced without globin depletion.

## A. Validation before any whole-blood TB claim

1. **Granulocyte-containing benchmark.** Every current benchmark is cell lines or PBMC; COMBAT's single-cell
   arm is PBMC. Nothing could have shown a neutrophil failure.
   - composition truth: COMBAT whole-blood CyTOF (check availability; not in our download);
   - activity truth: a held-out split of a whole-blood single-cell dataset that captures granulocytes
     (RBC-lysed whole blood on Seq-Well, Parse or BD Rhapsody); to be found.
2. **Separate cell-line and primary-immune results** in every headline table (no averaging across them).
   Kang's recall (0.40–0.48) is the expectation for blood, not the cell-line numbers. *Reporting only; also
   applied to the held-out v2 report.*
3. **"All cell types respond" as a named regime** with its own row (interferon in TB is broad). *Reporting
   only; held-out v2 report where the truth allows it.*
4. **Coverage by route** (unmixed / calibrated; guard fired / silent). Choosing the narrower interval can hide
   a branch with poor coverage. *Held-out v2: program-level route is recorded, the per-type guard is not;
   record both in v3.* Formal counterpart: group-conditional (Mondrian) conformal coverage.
4b. **Coverage against the conditioning of the unmixing.** With little variation in the cell mix, intervals
   could be narrow *and* wrong, not only wide (ill-conditioned orthogonal scores, arXiv:2512.07083). Report
   coverage against the condition number of each study's unmixing design, including the McFarland "tight"
   variant. *Reporting only.*
4c. **False coverage rate among called boxes.** End-to-end coverage already includes the routing choice on
   average; for the paper, control or report the false coverage rate of the *called* cell type × program
   boxes (selective inference, arXiv:2203.12572).
4d. **Design limit as a pre-registered readout.** `identifiability.py` computes it analytically from π, B, G
   and n before any effect is estimated; report it up front for every study.

## B. Model and estimator changes

5. **Power when composition barely varies** (Kang recall 0.40–0.48): gene-level unmixing pooled across a
   program's genes.
6. **Blood counts as a second likelihood on π** (CBC differential counts; available in most TB cohorts).
   Anchors the cell mix, identifies RNA content, and turns "more cells" into cells per µL.
   - Model for the reference scale: one scale per cell type plus one per sample (two-step linear mixing
     under variability, arXiv:2502.17212). The per-sample scale is not identifiable from bulk alone: only
     with blood counts.
   - **RNA content:** currently mean UMIs per cell in the atlas (capture efficiency mixed in, worst for
     neutrophils), a point value with no uncertainty, and fixed across disease status: a state-driven change
     in RNA output can only appear as "more cells". Give it a prior and propagate it.
7. **Real-bulk platform gap** (COMBAT r 0.2–0.6 on real bulk): anchor the average composition (blood counts,
   or a subset with paired bulk + single-cell).
8. **Missing-reference detector:** a non-negative "unassigned" component with a sparsity penalty, reported per
   sample, and/or a held-out-gene check of the fit. If it grows with disease, suppress the neighbouring types'
   effects. Whole-blood reference with neutrophils and low-density granulocytes.
9. **Shared response as the estimand:** report each type's deviation from the shared response
   (β_cp = s_p + d_cp, d shrunk to 0) instead of falling back to a program-level change. The unmixing already
   shrinks towards a shared value (REML); this makes the deviation the reported quantity.
   `plant="shared"` (v2 development) failed and is not the route.
   - **Preferred form: multi-condition empirical Bayes (MASH-style; arXiv:2406.08784)** with cell types as
     the conditions: the pattern of sharing across types is learned instead of imposed, which generalises
     s_p + d_cp. **First method change to test.**
10. **Attenuation as a function of effect size:** calibrate on stimulated donors (Kang, Parse) or on spiked
    effects at TB-like size, instead of one slope from between-donor variation near zero. The design check
    already plants study-sized effects and held-out coverage held on large effects; this tests it directly.
    - Tool: SIMEX: estimate attenuation as a curve over injected effect / error size and extrapolate
      (nonparametric and count variants: arXiv:2111.02863, arXiv:2409.14256). Expected gain: small, at large
      effects.
11. **Cell-type-specific program footprints** (G_pcg = G_pg + u_pcg, u shrunk hard). Cheap first step: report,
    per program, how consistent its footprint is across cell types in the atlas, and down-weight
    inconsistent programs.
12. **Whole-blood contaminants:** mask globin (HBB, HBA1/2, HBD) and platelet genes (PPBP, PF4, …), or model
    an erythroid/platelet component; report the masked read share per sample.
13. **Ordered disease axis:** uninfected / latent / incipient / active / on treatment, through the continuous
    path, instead of a binary D. Most "healthy" controls are IGRA-positive.
14. **Single-group path** (unmixing along z, 10 development splits: false calls 0.16–0.17 McFarland gradient,
    0.12–0.24 COMBAT COVID-only, against ≤ 0.1; permutation null near nominal, so the excess is leakage between
    types; per-patient activity intervals too narrow without T): calibrate along z (benchmarking.md §0.8).

## C. Smaller checks

15. **Stage A two-step fit for rare types:** compare against full NUTS on a few samples across the abundance
    range; report the interval-width ratio for the rarest types.
    - Proper version: simulation-based calibration of PALM-lite (posterior SBC, conditioned on data like the
      study's; arXiv:2502.03279), with rare-type π and the reported β̂ as test quantities, not global
      parameters (arXiv:2211.02383). If intervals are too narrow, recalibrate (arXiv:2603.20068). Too slow
      with full NUTS.
16. **Differentiation vs activation states:** write the partition policy (naive / memory / effector) and
    report at two granularities (lineage and fine type; supported via lineage groups).

## Framing for the paper (no method change)

- PALM is a **modular (cut) Bayesian pipeline**: the Bayesian model for the cell mix, a calibrated estimator
  downstream (JRSS-B 2025, arXiv:2211.03274). Semi-modular inference (AISTATS 2020, arXiv:2003.06804) is a
  family between full Bayes and full cut; an optional research direction, not a v3 item.
- Long-term option: a Neyman-orthogonal estimator for β with (π, B) as nuisance, which would make β
  first-order insensitive to fraction error. Research-level; not v3.

## Not adopted from the statistics scan

- **Checking the "sufficiently scattered condition"** (ranked first in the scan): it governs recovery of
  *unknown* simplex factors in matrix factorisation. Our unmixing is a regression with a *known* leakage
  operator, so identifiability is the rank and conditioning of that design, already computed analytically
  (item 4d).
- **"Carrying draws does not remove first-stage bias"** as a v3 fix: the calibration and the design check run
  on estimated fractions (the fraction step is rerun), so fraction error is inside the measured coverage. The
  orthogonal-moment route is kept as a long-term option above.
- **Covariate-shift conformal methods** (weighted, sketched): they formalise the calibration-donor vs study
  shift (critique 3.1); low priority, and the scan itself notes they degrade when overlap is poor.

## Not adopted from the critique (checked against the code)

- "No covariate adjustment": covariates enter the model and the unmixing; paired designs and IPW/AIPW are
  supported. The explainer did not say so (fixed).
- "Program weights are ±1": PROGENy's continuous weights are used, rescaled so the median |weight| is 1
  (explainer fixed).
- "π → δ arrow": no such arrow in the model or the slides.
