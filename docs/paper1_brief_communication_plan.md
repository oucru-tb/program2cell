# Paper 1 writing plan: Nature Methods Brief Communication

*Written 2026-11-23. Method: freeze v5 (joint fit, `method_spec.md` "Method freeze v5"), held-out splits 30-39
passed. Nothing here changes the method; the paper reports what is frozen.*

## 1. Format (verify first)

The Nature Methods guidelines page needs a login, so these limits are from public snippets and memory. **Task 0
is to check them against the current author guide** (content types, formatting, reporting summary, software
policy):

| Item | Planned limit (to verify) |
|---|---|
| Scope | "concise report of ... practical tweaks to an existing method or tool, software platforms, resources of broad interest, and technical critiques of widely used methodologies" |
| Main text | short; plan for ~1,500 words and treat 3,000 as the ceiling; no section headings |
| Abstract | ~70 words, unreferenced (to verify) |
| Display items | 2 (up to 3 with editorial discretion) |
| References | to verify (plan for ~20-30 in main text; Methods references extra) |
| Methods | Online Methods with subheadings; where the detail goes |
| Extended Data | figures and tables for benchmark breadth (count to verify) |
| Also required | Code Availability (versioned DOI, licence; "on request" not accepted), Data Availability, Reporting Summary, software summary; reviewers may install the package |

The Brief Communication fits two of the listed scopes at once: a **technical critique of a widely used
methodology** (deconvolve, score a pathway per cell type, test) and a **practical tool** that fixes it.

## 2. The one message

> Per-cell-type pathway effects from bulk RNA-seq, as usually computed, call changes in roughly half of cell
> types that did not change. program2cell (PALM) splits a bulk program change into cell types with calibrated
> intervals, and on held-out known-answer data its calls are right at the nominal rate.

Everything that does not serve this message goes to Extended Data, Supplementary or Paper 2.

Title candidates:
- "Calibrated cell-type program effects from bulk RNA-seq"
- "Bulk RNA-seq pathway analysis per cell type needs calibration"
- "Which cell types change a pathway? Calibrated answers from bulk RNA-seq"

## 3. Main-text story (~1,500 words, 5 paragraphs)

1. **Problem.** Bulk cohorts ask which cells change which programs. The common answer (deconvolution, per-type
   score, t-test) has never been checked against a truth. Bulk change = composition change + cells changed, and
   deconvolution leaks one type's change into the others.
2. **Method in one paragraph.** Fractions (any source) -> reallocation of program activity to cell types ->
   calibration on reference donors -> effects split into composition and intrinsic parts, each with an interval
   widened for leakage by a design-matched simulation. Model as a cut (modular) Bayesian model
   (`methods_cut_model.md`).
3. **Known-answer benchmark and the critique.** 13 datasets with exact per-type truth (McFarland MIX-Seq,
   Tahoe-100M, Kang IFN-beta, Parse cytokines, Zhang paclitaxel), split by donor; development and held-out splits
   kept apart; pre-registered decision rules. Current practice: null-call rates near 0.5; PALM at the nominal
   rate with higher responder AUROC.
4. **Real bulk.** COMBAT: 125 whole-blood bulk samples with paired single-cell data from the same people. Raw bulk
   loses attribution (platform gap); the platform offset restores it; no wrong-sign calls. The idasanutlin / TP53
   case as the worked example (only TP53 wild-type lines respond).
5. **Limits and use.** What bulk cannot identify (attenuation ceiling, missing types, resting reference),
   conservative intervals in PBMC, run time, when to use it.

## 4. Display items

| Item | Panels | Source |
|---|---|---|
| **Fig. 1** Problem and method | (a) bulk change = sum over types of (more cells + cells changed) schematic; (b) leakage: a change in one type appears in others under current practice; (c) PALM pipeline in one row | `docs/figures/talk/` schematic (simplified slide, 2026-11-20), `model_dag.md` |
| **Fig. 2** Held-out evidence | (a) null-call rate and power, PALM vs 6 practice pipelines, 13 datasets; (b) coverage per dataset at nominal 0.90; (c) idasanutlin p53 per cell line, PALM vs InstaPrism + score; (d) COMBAT real bulk: raw vs offset, per-type r and calls | held-out v5 (30-39) and v4 (40-49) outputs; `heldout_figures.py` |
| (Fig. 3, only if allowed) Real application | a disease cohort finding with orthogonal support | none clean yet; see §6 |

Extended Data (breadth): per-dataset tables; calibration curves; transport set (9 perturbations); fraction
sources (PALM, PALM-lite, InstaPrism, CIBERSORT); program sets (PROGENy, Hallmark, CollecTRI, Reactome);
composition vs intrinsic split; runtime; the independent fit and its conservative intervals (limitation).

## 5. Evidence map: what exists, what is missing

| Claim | Evidence in hand | Gap |
|---|---|---|
| Current practice over-calls | held-out 40-49 (v2): PALM - InstaPrism null-call rate -0.529 [-0.583, -0.476], coverage +0.579; CIBERSORT, NNLS, bulk score similar; TOAST, TCA (4 datasets) | **rerun the practice comparison with v5 on 30-39** (v5 estimates = v2 by C0, intervals differ); one table in the final numbers |
| PALM calibrated, held out | v5 held-out 30-39: S1-S3 pass, power +0.047 over v2 | done: `docs/heldout_v5_results.md` (2026-11-23), with practice (30-39: PALM null calls 0.023 vs 0.53 InstaPrism + score) and COMBAT v5 |
| Responder localisation | AUROC +0.06 to +0.34 over practice (40-49) | recompute on 30-39 with v5 |
| Real bulk | COMBAT 40-49 (v2): raw r 0.03, paired-offset 0.48, wrong sign 0; `benchmarks/dev/combat_v5` exists | summarise combat_v5; decide offset framing (paired vs self) |
| Worked example | idasanutlin / TP53 slide (talk) | final figure panel |
| Fair comparison | competitors at defaults; PALM frozen before held-out | state it; list versions; InstaPrism (not BayesPrism) everywhere |
| Novelty vs prior work | `research_plan.md` §3.6 framing | **systematic literature check 2023-2026** (BLADE, Statescope, IPF preprint, CODEFACS, EPIC-unmix, TOAST/TCA); verify with nature-citation |
| Code | package, tests, tutorial | Zenodo DOI + licence; install test on a clean machine; software summary |

## 6. Decisions for the user

1. **Decided 2026-11-23: COMBAT** is the real-bulk validation in Fig. 2; a further real dataset is being sought
   (user consulting). **PTB: MDR vs DS PTB** (148 vs 148, all sequencing run 1, so no run confounding; library
   batches R1_b1 and R1_b3 in both groups, used as a covariate) is the PTB application. PTB vs healthy is not used
   in the paper. The PTB run waits until no other job runs (PTB jobs run alone). Original options:
   (a) COMBAT as the real-bulk validation only (safe, no new biology); (b) PTB, not recommended for the main text:
   all PTB samples are in run 1 and all healthy in run 2, so every effect is confounded with run, and the bridge
   adjustment leaves no calls; (c) a new or in-house cohort without the confounding. Recommendation: (a) in Fig. 2,
   and PTB only in Supplementary with the confounding stated, or left out.
2. **Association section** (false associations of per-type scores with a sample variable; held-out 50-59 not run).
   It strengthens the critique but costs a display item. Recommendation: run it and put it in Extended Data.
3. **Independent fit.** Recommendation: Extended Data, with its conservative intervals as a stated limitation;
   the full fix is Paper 2.
4. **Presubmission enquiry** to Nature Methods with the abstract and Fig. 2: cheap and tells whether the editors
   see it as a Brief Communication.

## 7. Work plan

| Step | Work | Output |
|---|---|---|
| 0 | Verify format limits, reporting summary, software policy | §1 updated |
| 1 | `heldout_v5_results.md`; practice comparison and AUROC with v5 on 30-39; COMBAT v5 summary | final numbers, one table |
| 2 | Literature check, claim-to-source map | references, novelty paragraph |
| 3 | Fig. 1 and Fig. 2 (dataviz rules; vector; colour-blind safe) | `docs/figures/paper1/` |
| 4 | Main text + abstract (nature-writing), Online Methods from `method_spec.md` and `methods_cut_model.md` | manuscript draft |
| 5 | Extended Data and Supplementary | ED figures and tables |
| 6 | Code release: version tag, Zenodo DOI, licence, clean-install test, tutorial run in place | Code Availability |
| 7 | Internal review (nature-reviewer mock review), statistics check (nature-statistics), data availability (nature-data) | revision |
| 8 | Presubmission enquiry or submission; cover letter | |

Steps 1 and 2 can run in parallel; figures wait for the final numbers. Paper 2 work (D1, D1b) does not compete for
held-out splits with Paper 1.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Editors see a full method, not a brief one | presubmission enquiry; lead with the critique |
| "Benchmarks only, no biology" | COMBAT real bulk in the main figure; worked example; association section in ED |
| Fairness (tuned method vs default competitors) | method frozen before held-out; competitors at documented defaults; same fractions where possible |
| Novelty challenged (BLADE, Statescope) | claim calibration and the known-answer critique, not "first per-type pathway activity" (`research_plan.md` §3.6 claims to avoid) |
| PBMC intervals conservative | state it; it errs on the safe side; Paper 2 targets it |
