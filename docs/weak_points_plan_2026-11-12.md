# Weak points of freeze v2 and the experiment plan (2026-11-12)

Scope: every weakness the benchmarks and the PTB application have shown, with its evidence, the candidate fixes from
the literature, and the cheapest experiment that can confirm or kill each fix. Freeze v2 (`estimator="auto"`)
stays the Paper 1 method; everything here is development on splits 0–2 and 20–29, held-out splits 30–39 reserved
for any new freeze, 50–59 for the association section. The earlier algorithm-only plan
(`docs/improvement_research_2026-11.md`, items A1–A8) is folded in and superseded by this list.

Evidence sources: `benchmarks/dev/v5_prior_offline/parts_20-29.csv` (freeze v2, 13 datasets × splits 20–29),
`docs/heldout_v4_results.md` (splits 40–49), `benchmarks/dev/heldout_v4_out/combat_summary.txt` (COMBAT real
bulk), `results/ptb/*` (PTB application), the talk figures `docs/figures/talk/2_3*_idasanutlin_*.png`.

## 1. Freeze v2 per dataset (development splits 20–29, means over 10 splits)

| Dataset | Coverage | Null calls | Power | FDP | r | Width | Unmixed share |
|---|---|---|---|---|---|---|---|
| Kang IFN-β (8 donors, paired) | 0.996 | 0.000 | 0.200 | 0.000 | 0.82 | 0.750 | 0.21 |
| McFarland idasanutlin (6 lines) | 0.962 | 0.049 | 0.488 | 0.163 | 0.92 | 0.247 | 0.45 |
| McFarland trametinib (6 lines) | 0.962 | 0.076 | 0.831 | 0.117 | 0.92 | 0.188 | 0.82 |
| Parse GM-CSF | 1.000 | 0.000 | 0.161 | 0.000 | 0.85 | 0.299 | 0.00 |
| Parse IFN-β | 1.000 | 0.003 | 0.023 | 0.100 | 0.90 | 0.453 | 0.00 |
| Parse IFN-γ | 1.000 | 0.000 | 0.020 | 0.000 | 0.91 | 0.361 | 0.00 |
| Parse IL-4 | 1.000 | 0.000 | 0.050 | 0.000 | 0.69 | 0.406 | 0.00 |
| Parse TNF | 0.998 | 0.000 | 0.000 | 0.000 | 0.86 | 0.176 | 0.02 |
| Tahoe encorafenib (9 lines) | 0.949 | 0.024 | 0.217 | 0.232 | 0.42 | 0.201 | 0.40 |
| Tahoe gemcitabine (9 lines) | 0.930 | 0.049 | 0.000 | 0.692 | 0.54 | 0.143 | 0.37 |
| Tahoe RMC-6236 | 0.978 | 0.011 | 0.712 | 0.049 | 0.72 | 0.299 | 0.56 |
| Tahoe trametinib | 0.986 | 0.006 | 0.794 | 0.017 | 0.81 | 0.317 | 0.59 |
| Zhang (negative control) | 1.000 | 0.004 | – | 0.033 | 0.84 | 0.106 | 0.01 |
| **Mean of datasets** | **0.982** | **0.017** | **0.269** | **0.108** | **0.79** | **0.304** | **0.26** |

Power = called with the right sign among |β| ≥ 0.1; null calls = called among |β| < 0.05; FDP = share of calls
that are on null pairs; width = mean 90% interval width on the effect scale (true effects are mostly 0.1–0.5).

## 2. Weak points, with evidence

Grouped by where they sit in the pipeline (A fractions, B reallocation, C atlas calibration, D unmixing + router,
E calls, I inputs, S study design). "Costs" is what the weakness costs in the numbers above.

| # | Weak point | Evidence | Costs | Stage |
|---|---|---|---|---|
| W1 | **Intervals far too wide where the calibrated route is used.** Coverage 1.00 on all 7 PBMC datasets; the unmixed route is never chosen on Parse (0%). The excess sits in the simulated-error quantile, not in the atlas interval: dropping the max with the atlas interval (`simonly`) narrows width only 0.304 → 0.284, while the atlas interval alone (`calonly`) under-covers McFarland and Tahoe (0.61–0.79). The simulation plants effects up to 2× the study's largest and adds the study's noise on top of cohorts that already carry sampling noise. | Power 0.02–0.16 on Parse, 0.20 on Kang: effects of 0.1–0.5 are not called | C, D |
| W2 | **Leakage residue on non-responding types in pooled cell lines.** McFarland idasanutlin: a TP53-mutant line with true change 0.05 is estimated at 0.23 and flagged in 8/10 held-out splits; null calls 0.049 (ida), 0.076 (tram), FDP 0.16 / 0.12. The standard workflow flags all 6 lines (10/10). PALM on InstaPrism's fractions flags that line 2/10 but misses the weakest true responder 4/10, so the fraction step carries part of this. | FDP 0.12–0.16 on McFarland; the "only to the responders" claim does not fully hold | A, D |
| W3 | **Small true effects with many types: calls are dominated by nulls.** Tahoe gemcitabine: FDP 0.69, power 0; encorafenib: r 0.42, FDP 0.23. Every pair is called at 90% with 30–102 pairs per run and no multiplicity control. BH (q < 0.10) cuts FDP 0.108 → 0.009 and null calls 0.017 → 0.002 at a cost of 30% of power (0.269 → 0.187); every dataset stays safe. | Gemcitabine and encorafenib results are not usable as calls | E |
| W4 | **Unmixing is unstable where types are near-identical or units are few.** Condition numbers 8–13 (6 lines); Kang NK unmixed −1.34 vs truth +0.15 (caught by the guard); PTB CD4 T +0.47 / CD8 T −0.41 nearly cancel; merging 8 → 5 types narrowed PTB intervals by a quarter. The prior allows only "shared + isotropic deviations" (A1), shrinkage does not target the unresolved directions (A2). | Sign flips and wide unmixed intervals; the router then falls back to the wide calibrated route (W1) | D |
| W5 | **The structured effect prior, as tested, is unsafe.** The MASH-type mixture (fixed full-rank patterns + ε I, EB weights, lfsr < 0.05) gave null calls 0.12 / 0.18 on McFarland and failed Tahoe gemcitabine, even with the full bootstrap error covariance. Consistent with Xie 2025 §7 (exact nulls are harder) and Yang 2024 §7 (lfsr anti-conservative; use 0.01). | Phase 1a of the earlier plan is not a route as designed | D |
| W6 | **Fractions: activation is read as composition, and real-bulk platform error is bias, not scatter.** Fractions use unshifted reference profiles (A3); on COMBAT the NK estimate tracks CyTOF neutrophils (r 0.32), not NK. Real bulk without offsets: r 0.02 with the truth; anchored offsets 0.69; self offsets 0.40; retention 0.83. Calibration simulates scatter while the measured error is bias (RMS log bias 3.3–4.1, v4_plan §10A). | Real-bulk results depend on an external composition anchor | A, C |
| W7 | **Program inputs overlap and are the same in every type.** Hallmark IFN-α / IFN-γ share 73 of 97 genes; Reactome TLR / NLR 31 of 61; BCR / TCR 56 of 113. Programs are modelled jointly, so overlap is collinearity. Member sets gave 0 of 48 calls in PTB (self offsets) and 1 of 48 (anchored). Program weights are forced equal across types (A8). | PTB: IFN-α vs IFN-γ cannot be separated; near-zero calls | I, B, D |
| W8 | **PTB design: group = sequencing run.** PTB n = 296 all Run1, healthy n = 30 all Run2; no run correction (user decision); offsets anchored on healthy CyTOF composition; 5 resequenced pairs give a sensitivity bound only. The robustness value is sample-level while this confounder is cluster-level (A6). | Any PTB call is a joint statement about disease and run | S, E |
| W9 | **No missing-type detector.** The 8-type reference has no neutrophil states or plasmablasts; the rejected learned extra profile absorbed platform signal (v4_plan §10B). | Unknown share of the PTB signal is unassigned or misassigned | self-tests |
| W10 | **The router chooses on the same data it reports.** Route = narrower mean half-width per program, guard on the same estimates. Coverage of the chosen route is only checked empirically (it over-covers, so the selection effect is hidden today); it would surface once W1 is fixed. | Risk to any narrower-interval fix | D |

Not weaknesses, for the record: McFarland is at nominal width already (a global width factor fails it); moderated
variance changes nothing (the unmixed se are stable); Zhang (negative control) makes 0.5 calls per run.

## 3. Candidate fixes from the literature

Three searches on 2026-11-12 (OpenAlex + web; the nature-academic-search MCP tools were unavailable, the PubMed
connector is not authorised). Abstracts only unless marked ‡ (full text read). Each candidate names the weak point
it targets and the tier of the cheapest test (section 4).

**W1, intervals too wide (calibrated route).** Candidates: (a) *bias correction in the form of PPI* ‡ (subtract the
simulated per-pair bias, add its variance over cohorts; PPI's own guarantee does not cover simulated cohorts, so
validity rests on S1/S2); (b) *drop the double-counted study noise* (the synthetic cohorts already carry sampling
noise, and `sim_widen="abs"` adds `(t·se_r)²` on top); (c) *match the planted effect size to the study* (the
simulation plants up to 2× the study's largest effect; the saved `sim_beta` lets the error quantile be taken from
cohorts whose planted sizes match the study's calibrated estimates). (d) Interval-calibration literature: see 3b
below. Not a cause: composition spread (Parse has more spread than McFarland/Tahoe, power ≈ 0.03 vs 0.5–0.8; only
Kang, 8 donors, is design-limited; the cypress 2024 power model would not explain it).

**W2, leakage residue on non-responders (unmixed route).** Diagnosis from the parts: all 10 COLO680N false flags
(dev splits) come from the unmixed route, with eff_u 0.08–0.26 vs truth 0.04–0.06 in every split, i.e. a bias
from the Gaussian deviation prior around the shared mean (responders pulled down, non-responders pulled up).
Candidates: (a) *heavy-tailed deviation prior* (horseshoe on per-type deviations from the shared effect; Alt et
al. 2025 hierarchical grouped horseshoe; Thomas et al. 2021: horseshoe beats spike-and-slab for heterogeneity
with few units); (b) *two-group (responder / non-responder) prior on deviations*, a 2-component version of what
the full mixture failed to do; (c) *spillover-spreading diagnostic* (Nguyen et al. 2013, flow-cytometry
compensation): from the leakage operator A, the variance each type inherits from the others scales with the
spillover coefficient × source signal; a per-pair spread index to flag pairs whose call is driven by inherited
signal; (d) *tree prior sharing DE status across related types* (CeDAR, Chen et al. 2023): a rerun-level
alternative. Not useful: an abundance gate (DALE-Eval, Hu et al. 2025, recommends fraction > 0.1); here the
residue sits in the *abundant* types (null calls 0.11 at fractions 0.2–0.4 vs 0.01 below 0.1 on McFarland/Tahoe).
Already done by freeze v2: fraction-draw propagation through A (the DECALS point, Cai et al. 2024).

**W3, calls dominated by nulls with many types and small effects.** BH q < 0.10 over the study's pairs
(Benjamini–Hochberg; FCR variant Benjamini & Yekutieli 2005, Frostig & Benjamini 2025 for the intervals): tested,
FDP 0.108 → 0.009, every dataset safe, power −30%. BH q < 0.05: power 0.166, no further safety gain. Decision
candidate: BH as an opt-in "strict" call mode next to the interval rule, not the default; must be in the
false-association protocol before splits 50–59 are run.

**W4, unstable unmixing (near-identical types, few units).** (a) *generalised ridge along the design's own weak
directions* (A2 / 1b of the earlier plan): penalise only the eigen-directions of the design information with
small information (CD4 − CD8), leave lineage contrasts free; (b) the W2 prior change also acts here, because the
isotropic Gaussian deviation is what forces every type toward the mean.

**W5, structured mixture prior.** Closed as designed: lfsr thresholds 0.05 / 0.02 / 0.01 / 0.005 give null calls
0.043 / 0.027 / 0.020 / 0.015 with worst-dataset 0.18 / 0.14 / 0.12 / 0.10 and power 0.280 / 0.262 / 0.253 /
0.246 (v2: 0.017, 0.076, 0.269). Consistent with Xie 2025 ‡ (exact nulls; 1000+ units) and Yang et al. 2024 ‡.
MASH on the saved estimates with canonical covariances (the leakage search's top suggestion) is this test.

**W6, fractions.** (a) *PALM on InstaPrism's fractions* on all 13 datasets (McFarland idasanutlin: mutant-line
false flags 8/10 → 2/10 but weakest responder 10/10 → 6/10): decides whether a state-aware fraction step (A3)
is worth building; (b) *EPIC-style "other" fraction* (Racle et al. 2017) as a template for unknown content;
(c) platform bias (A4): calibration should simulate bias, not scatter.

**W7, overlapping programs; member vs response sets.** (a) *collinearity audit* of the program matrix per cell
type (decoupleR's `check_corr`; its MLM refuses collinear sources; VIF); (b) *shadow correction* (VIPER, Alvarez
et al. 2016): down-weight genes shared by two enriched programs instead of dropping a program; (c) *disjoint IFN
sets*: Singhania et al. 2019 (Nat Commun, 10.1038/s41467-019-10601-6) resolves separate type I and type II IFN
networks in blood; Singhania 2018 ‡ lightgreen (86 genes) is a clean type I module, its "IFNG" module is Th1/NK
content, the IFN-γ-inducible genes sit in a 360-gene mixed module (data/singhania2018_ifn_modules.csv);
(d) *per-cytokine response signatures*: CytoSig (Jiang et al. 2021; IFN1 and IFNG separate), huCIRA (human
PBMC cytokine dictionary, 2025 preprint, Parse barcoding, per-type consensus signatures), Immune Dictionary
(Cui et al. 2024, mouse); (e) PROGENy's own benchmark (Holland et al. 2020): footprint genes beat member genes,
so 0/48 with member sets is expected; (f) PLIER (Mao et al. 2019) lets related sets share one latent program.

**W8, run = group.** (a) Nygaard, Rødland & Hovig 2016 (Biostatistics): correcting batch in an unbalanced design
inflates confidence, a published justification for "no run correction"; (b) *bridge-sample ratio anchoring*
(Quartet, Nat Biotechnol 2023): express PTB and healthy program activity as ratios to the 5 resequenced samples
measured in both runs; pwrBRIDGE (2022) gives what n = 5 bridges can support; (c) RUV-III with pseudo-replicates
(Molania et al. 2022) using the bridges and the CyTOF-anchored healthy composition as the strata; (d)
*negative-control-outcome bias bounds* (Du, Roeder & Wasserman 2024) and *sensitivity intervals* for unmeasured
confounding (Biometrics 2024): report each PTB effect with a run-bias interval, not a point call. No TB blood
study was found that reports and bounds run = group confounding.

**W9, missing types.** Ivich et al. 2025: subtract the reconstructed bulk, NMF the residual; factors track the
missing type's true proportion. Needs saved residuals from `run_ptb.py` (one line) and a rerun.

**W10, router selection.** Data fission / thinning (Leiner et al. 2023, JASA): choose the route on one split of
the data, infer on the other. Only matters once W1 is fixed; a T2 rerun.

### 3b. Interval-calibration literature (W1, W10)

Third search (interval calibration with non-exchangeable simulations; selective inference for the router;
small-n bootstrap intervals; coverage reporting in deconvolution CTS-DE tools). Findings that change the plan:

- **Synthetic-powered predictive inference** (Bashari, Lotan, Lee, Dobriban, Romano; NeurIPS 2025, arXiv
  2505.13432): pools a small trusted *real* calibration set with many *synthetic* scores through an empirical
  quantile "score transporter"; finite-sample coverage holds with no assumption that the real and synthetic
  distributions match. Maps directly: real set = studentised errors |est − truth| / se from the other 12
  known-answer datasets (leave-one-dataset-out), synthetic set = the study's own 10-cohort errors; replaces
  `max(calibrated, simulated)`. The guarantee is then for "a study exchangeable with the benchmark studies",
  which is the assumption the whole known-answer benchmark already makes (and the one the PPI check said we
  lacked). **E16, T0 after the sims run.**
- **Weighted conformal under non-exchangeability** (Barber, Candès, Ramdas, Tibshirani, Ann Stat 2023): coverage
  gap bounded by the weighted total-variation distance between calibration and test; needs per-cohort weights.
  Background to E16.
- **Simulation-calibrated critical value** (Beran prepivoting; Palmer et al. 2022 recalibrate bootstrap intervals
  from the empirical z-score distribution as a function of predicted width, not a global factor): per cell type,
  the 90% quantile of the studentised cohort errors pooled over programs, times se, dropping the max. Freeze v1
  used the studentised quantile per program (`sim_widen=True`); v2 moved to absolute units (`"abs"`) and the max.
  **E17, T0 after the sims run:** studentised quantile per type / per program, with and without the max.
- **Tiny calibration sets** (Zwart 2025, arXiv 2509.15349): with 10 cohorts the 90% empirical quantile is itself
  noisy and biased; a Beta-distribution correction of the level. Fold into E16/E17 as a sensitivity.
- **Conformal prediction after data-dependent model selection** (Liang, Zhu, Barber, JASA 2026, arXiv
  2408.07066): valid sets after choosing a model by a hold-out criterion such as narrowest width, without extra
  splitting. Our router is this selection. Note from the search: routing on width (a function of se, not of the
  estimate) is nearly ignorable under normality. **E18, T0:** correlation between route choice and the estimate's
  sign/size on the saved pairs; if null, document ignorability, else apply the selection-adjusted quantile.
  Data fission / generalised data thinning (Leiner 2025; Dharamshi 2025) stay as the T2 fallback; at n = 8 the
  power cost is large.
- **Few-cluster bootstrap-t is conservative** (MacKinnon & Webb 2017; MacKinnon, Nielsen & Webb 2023; G ≲ 12
  clusters): documented over-coverage with few clusters, which supports the Kang diagnosis. Leverage-based
  effective df (Pustejovsky & Tipton 2018 CR2; Hansen 2026 jackknife) usually widen, so not a fix.
- **Local coverage as a function of (effect size, abundance)** (LF2I, Dalmasso et al. 2024; FreB, Carzon et al.
  2026): learn coverage from labelled simulations instead of one quantile; needs many more than 10 cohorts, T2.
- **Gap we can claim:** no 2023–2026 bulk CTS-DE tool (bMIND, swCAM, DeMixSC, TCA, CellDMC, TOAST, CeDAR,
  CSeQTL) reports per-cell-type effect-interval coverage; DECALS / MEAD / DeepDeconUQ cover proportions only;
  C-SIDE (spatial) is the one calibrated CTS-DE test.

## 4. Experiments, cheapest first

Cost tiers:
- **T0, seconds to minutes:** offline from the saved parts of splits 20–29 (`<ds>__s<k>__auto_v2_parts.npz`,
  `..._auto_v2_sims.npz`: both routes' estimates, se, bootstrap draws, simulated-cohort errors and planted
  effects, fractions, truth). Scorer: `scripts/experiments/v5_parts_offline.py`. Safety rules as in the held-out
  protocol (S1: every dataset coverage ≥ 0.85 and null calls ≤ 0.10; S2: differences vs v2 with a dataset
  bootstrap CI).
- **T1, minutes to 1 h:** one dataset, a short rerun, or the PTB cohort (`run_ptb.py` ≈ 45 min).
- **T2, overnight:** 13 datasets × splits 20–29 with a changed stage (6 lanes, 2–5 h).

| # | Targets | Experiment | Tier | Status / result | Gate |
|---|---|---|---|---|---|
| E1 | W3 | BH q < 0.10 and 0.05 on freeze v2 calls | T0 | **done:** q 0.10 FDP 0.108 → 0.009, power −30%, all datasets safe; q 0.05 no safer, power 0.166 | adopt as opt-in strict mode |
| E2 | W5 | lfsr thresholds 0.05 → 0.005 | T0 | **done:** still fails one dataset at 0.005, power < v2 | closed |
| E3 | W1 | composition-spread index vs power | T0 | **done:** no relation; Parse spread > Tahoe | closed |
| E4 | W2 | abundance gate (fraction > 0.1) | T0 | **done:** residue sits in abundant types | closed |
| E5 | W2 | route and parts behind the COLO680N flags | T0 | **done:** unmixed route, eff_u 0.08–0.26 vs 0.05, consistent bias | → E7 |
| E6 | W1 | bias correction from the simulated cohorts: (a) subtract the mean simulated bias per pair, carry its variance (PPI form); (b) no added study noise; (c) error quantile from the cohorts whose planted sizes match the study's estimates | T0 (`scripts/experiments/v5_sims_offline.py`, output `benchmarks/dev/v5_prior_offline/sims_20-29.txt`) | **done, 130 runs, 13 datasets:** (a) fails all 13 (coverage 0.72); (b) no change; (c) with the route frozen as v2 (`sized_fx`): **power 0.269 → 0.316, +0.047 [+0.017, +0.081]**, null calls +0.002 [+0.001, +0.005], coverage 0.982 → 0.970 (worst 0.91), width −16%, every dataset passes S1. Gains on PBMC: Kang 0.20 → 0.28, Parse GM-CSF 0.16 → 0.30, IFN-γ 0.02 → 0.20, IFN-β 0.02 → 0.09; cell lines unchanged. With the route free (`sized`) the narrower calibrated interval wins the width contest and costs cell-line power. **Implemented opt-in** as `sim_widen="sized"` (routing and guard keep the v2 width, the report uses the size-matched quantile; `causal_loop.py` estimator `auto_v5s`); reproduces the offline result to 1e-10 on Kang and McFarland split 20; tests 90 passed. | **candidate for the next freeze** (held-out splits 30–39) |
| E7 | W2, W4 | heavy-tailed deviation prior: Student-t on the per-type deviations u from the shared effect, by iteratively reweighted ridge (EM for the t scale mixture; weights (ν+1)/(ν+u²/mean u²), 4 refits at the REML λ; bootstrap refits the same way). Opt-in `unmix_kw=dict(robust_nu=ν)`, estimators `auto_v5t1`, `auto_v5t4` | T1 | **pilot, McFarland idasanutlin split 20:** p53 estimates move by ≤ 0.02 (COLO680N 0.13 → 0.11 vs truth 0.06; BICR31 0.13 → 0.12, now not called); coverage and null calls unchanged. The deviation prior is not the main source of the residue. | parked; E20 decides |
| E8 | W2 | spillover-spread index from A (needs B, G via `cl._setup`, ≈ 1 min per dataset): does it predict the false flags on McFarland/Tahoe? | T1 | next | AUROC of the index for null-pair calls |
| E9b | W7 | PTB with the whole Reactome IFN sets (α/β 78, γ 99; 24 shared in the model) | T1 | **done:** 1 call of 56 (DC TCR); no innate call; type I vs IFN-γ correlate −0.33 across types (NK +0.15 vs −0.43). Figure results/ptb/figures_response/anchored_decomposition_innate_ifn_reactome.png. Interferome was offline (maintenance) | — |
| E9 | W7 | collinearity audit of the program weight columns; PTB rerun with the IFN split (immune_response8) | T0 + T1 | **done:** weight columns correlate ≤ 0.43 (VIF ≤ 1.05) in every set, so overlap is not collinearity of the weights; the PTB split run (5439 s) gives 2 calls of 64, both CD4 T (type I only +0.57 [0.01, 1.14]; shared ISGs +0.49 [0.02, 0.96]). Across types, type I only vs IFN-γ only correlate 0.18 (separated), type I only vs shared ISGs 0.74 (the 18 type-I-only genes behave like the shared set). CD4 T / CD8 T opposite signs again (W4). Figure: results/ptb/figures_response/anchored_decomposition_innate_ifn_split.png | IFN-γ separates; type I needs a better set (E10) |
| E10 | W7 | PTB with disjoint IFN programs from Singhania 2019 / 2018 lightgreen + a curated IFN-γ-inducible set; then CytoSig IFN1/IFNG | T1 × 2 | after E9 | same |
| E11 | W8 | 5 bridge pairs as ratio anchors → per-program run-bias interval added to every PTB interval; pwrBRIDGE statement for n = 5 | T0/T1 | next | a defensible bound on each PTB call |
| E12 | W9 | save residuals in `run_ptb.py`, Ivich residual-NMF | T1 | after E9 | a factor that tracks a plausible missing type, or none |
| E13 | W6 | PALM on InstaPrism fractions, 13 datasets × splits 20–29 | T2 (≈ 5 h) | tonight if E6/E7 leave the machine free | decides whether a state-aware fraction step is built |
| E14 | W4 | generalised ridge along the design's weak eigen-directions (1b) | T2 | after E7 | S1/S2, PTB CD4/CD8 cancellation gone |
| E15 | W10 | data-fission routing | T2 | only with a narrower-interval variant from E6 | coverage of the chosen route at nominal |
| E20 | W2, W6 | **fraction oracle:** freeze v2 with the TRUE read fractions vs InstaPrism's vs PALM's own, McFarland idasanutlin and trametinib, dev splits 20–24 (`SOURCE=truth|instaprism python insta_palm_v2.py`) | T1 | **done, negative:** true fractions do not remove the residue; idasanutlin null calls 0.036 (own) / 0.018 (InstaPrism) / **0.062 (true)**, mutant lines flagged 5 / 1 / **9** of 15 line-splits; non-responders still +0.08 high with true fractions (TEN 0.12 vs 0.04). InstaPrism looks safer only through attenuation (SNU1079 0.24 vs truth 0.34, loses calls). Trametinib: no difference between sources. **The fraction step is not the cause; a state-aware fraction step (A3) would not fix W2.** | A3 deprioritised |
| E21 | W2 | is the residue an estimator bias? Non-responder error in the 10 simulated cohorts per run (model correct by construction), programs with mixed responders, 13 datasets × splits 20–29 | T0 | **done:** no bias in simulation (−0.003 overall; McFarland idasanutlin 0.000; Kang −0.03, Zhang +0.05 on few cohorts). **W2 is a model-to-data gap:** the simulated response is exactly δ·G (+ independent off-program noise), the real response is not; the unexplained structured part of the responders' change leaks onto non-responders and the calibration never sees it (sim-to-real gap, as for PPI) | → E22 |
| E22 | W2 | (a) **realistic simulated responses + widened unmixed route**: in the design-check cohorts each type responds with its own jittered copy of the program weights (`weight_sd`, per-gene log-normal, mean-preserving; truth = realised shift as in the real benchmarks), and the unmixed interval is widened to the simulated unmixed-error quantile (`unmix_widen`), on top of `sized`; the route contest keeps freeze v2's widths (a widened unmixed interval in the contest misroutes: pilot coverage 0.98 → 0.86). Estimators `auto_v5u0` (widening, exact responses; control) and `auto_v5u10` (widening, jitter 1.0). (b) robust gene weights; (c) report as a limitation | T2 (13 datasets × splits 20–24, `scripts/experiments/v5_e22_score.py`) | **pilot, McFarland idasanutlin split 20:** u0 null calls 0.048 → 0.000, p53 calls unchanged; u10 null 0.000, mutant lines no longer called, all 3 responders still called, coverage 0.93; u05 too wide (10 cohorts make the quantile noisy). **Full run (stopped at 24 of 65 pairs, 6 datasets; answer clear):** vs sized_fx, u0 null calls −0.014 [−0.022, −0.007] but power −0.205 [−0.367, −0.071]; u10 power −0.306 (Tahoe trametinib 0.88 → 0.33). Offline on all 130 runs (v5_sims_offline.py `uw_all` = u0, `uw_sized`): widening to the size-matched quantile recovers most power (0.286 vs 0.316) but buys only −0.006 null calls; McFarland idasanutlin 0.05 → 0.04. **Uniform widening is too blunt: the residue is a consistent bias on particular pairs, not general noise.** | McFarland null calls down without power loss; S1/S2 |
| E23 | W2 | **exposure-targeted widening** (spillover spreading, Nguyen et al. 2013): a pair's expected leaked error grows with its exposure x_cp = Σ_{k≠c} |leak_ckp|·|effect_kp| (the study's within-program leakage operator, now saved in the parts as `leak`). From simulated cohorts with realistic responses (`auto_v5j05_sims`, `auto_v5j10_sims`: weight jitter 0.5 / 1.0, **20 cohorts**), fit |unmixed error| ≈ b0 + b1·x per program (and pooled), take the 90% ratio quantile, and widen the unmixed half-width to r_q·(b0 + b1·x_study), with x from the study's own unmixed estimates. Estimates, route and calibrated widths stay sized_fx; only the simulated errors come from the jittered runs (`scripts/experiments/v5_exposure_offline.py`) | T2 overnight (260 runs, 2026-11-12/13) + T0 scoring | **done, negative** (130 runs per jitter level, 13 datasets): every widening cuts null calls only 0.010–0.016 (sized_fx 0.019, already safe: worst dataset 0.076) and costs 0.04–0.20 power. Best trade uw05_sized: null −0.011 [−0.019, −0.003], power −0.042 [−0.074, −0.013]; expd05 (difference exposure): null −0.013, power −0.055; per-program / pooled exposure (exp, exp_pool) −0.11 to −0.20 power (they widen Tahoe trametinib / ruxolitinib responders: 0.79 → 0.08–0.42). The widening is not targeted: Kang loses the sized gain (0.28 → 0.15) though it has no residue. Exposure from the study's own unmixed estimates is large exactly where responses are strong, so it widens true calls more than leaked ones. **No unmixed widening in v5; the McFarland residue is reported as a measured limitation (W2).** Scorer output `benchmarks/dev/v5_prior_offline/exposure_20-29.csv` | closed |
| E19 | W10 | routing rule when the calibrated interval is narrowed: (a) route frozen by v2's widths (`sized_fx`); (b) route by simulated RMSE (freeze v1's `choose="rmse"`, needs `rmse_unmix/realloc` saved); (c) route by width but with the v2 width for the contest and the sized width for the report | T0 | (a) done in E6; (b), (c) next | S1/S2; cell-line power kept |
| E16 | W1 | synthetic-powered calibration: leave-one-dataset-out real studentised errors (12 datasets) + the study's 10-cohort synthetic errors through the score transporter; replaces the max() widening | T0, after the sims run | next after E6 | S1 all datasets, S2, power gain; coverage closest to 0.90 |
| E17 | W1 | simulation-calibrated critical value: studentised cohort-error quantile per type (pooled over programs) and per program, with and without the max; Beta correction for 10 cohorts | T0 | **done (preview):** studentised quantile per program (freeze v1 form) fails Tahoe gemcitabine (null calls 0.12, coverage 0.85); per type fails McFarland trametinib and gemcitabine; the leave-one-dataset-out real quantile fails 4 datasets (coverage 0.77–0.84). Absolute-unit quantiles (v2) stay. | closed |
| E18 | W10 | router ignorability: correlation of route choice with estimate sign/size on the saved pairs; selection-adjusted quantile if non-null | T0 | **done:** route choice correlates with the true effect size (r = 0.33, through the width) and negatively with the unmixed estimate's size (r = −0.16): the router avoids wild unmixed estimates, it does not pick large ones; no winner's-curse selection on the reported estimate. But see E6: with a narrower calibrated interval the width contest misroutes. | documented; routing rule to revisit with E6 |

Order for the next 24 h: E6, E16, E17 together as soon as the sims run finishes (one scorer, same safety
rules) → E18 → E7 → E8 and E9 in parallel → E11 → E13 overnight. Each is a decision, not a tuning: a variant that fails S1 on any dataset is dropped; one that passes
goes to splits 30–39 only as part of a frozen candidate.

## 4b. v5 candidate (2026-11-13, development evidence only)

After E1–E23 the candidate is small:
- **Default change: `estimator="v5"` = freeze v2 with `sim_widen="sized"`** (E6 sized_fx; package option
  added 2026-11-13, `program_deconvolution(estimator="v5")`). Same estimates and routes as freeze v2; the
  reported calibrated interval uses the error quantile of the design-check cohorts whose planted effects are no
  larger than the study's. Dev splits 20–29, 130 runs: power 0.269 → 0.316 (+0.047 [+0.017, +0.081]), null calls
  +0.002, coverage 0.970, every dataset passes S1. Largest gains in PBMC (Kang 0.20 → 0.28, Parse IFN-γ 0.02 → 0.20).
- **Opt-in strict calls: `calls="bh"`** (E1; BH over the study's pairs at `fdr=0.10`, q-values in
  `info["qvalues"]`). FDP 0.108 → 0.009, power −30%. For many-type, small-effect studies (Tahoe gemcitabine,
  encorafenib) and screening use; not the default.
- **Not adopted:** PPI, studentised quantiles, LODO, mixture prior / lfsr, abundance gate, t-robust joint fit
  (E7), uniform or exposure-targeted unmixed widening (E22, E23).
- **Known limitation kept:** pooled cell lines with strong responders leave +0.08 on non-responders through
  the unmixed route (W2; McFarland idasanutlin null calls 0.05 at dev); not present in PBMC; true fractions do
  not remove it (E20).

Confirmation needs held-out splits 30–39 (reserved for a new freeze) under a protocol written before any
result: primary = sized vs freeze v2 power difference > 0 with null-call difference upper CI ≤ +0.02 and S1 on
every dataset. Run only with the user's agreement.

## 5. References

Full texts read 2026-11-12 are marked ‡; the rest are abstracts (OpenAlex / publisher pages).

Intervals and selection
- Bashari M, Lotan I, Lee Y, Dobriban E, Romano Y. Synthetic-powered predictive inference. NeurIPS 2025. arXiv 2505.13432
- Barber RF, Candès EJ, Ramdas A, Tibshirani RJ. Conformal prediction beyond exchangeability. Ann Stat 2023. 10.1214/23-AOS2276
- Liang Z, Zhu Y, Barber RF. Conformal prediction after data-dependent model selection. JASA 2026. 10.1080/01621459.2026.2663588
- Palmer G, et al. Calibration after bootstrap for accurate uncertainty quantification. npj Comput Mater 2022. 10.1038/s41524-022-00794-8
- Zwart. Conformal level correction for tiny calibration sets. arXiv 2025. 2509.15349
- MacKinnon JG, Webb MD. Wild bootstrap inference for wildly different cluster sizes. J Appl Econ 2017. 10.1002/jae.2508
- Dalmasso N, et al. Likelihood-free frequentist inference (LF2I). EJS 2024. 10.1214/24-EJS2307
- Cable DM, et al. C-SIDE. Nat Methods 2022. 10.1038/s41592-022-01575-3
- Angelopoulos AN, Bates S, Fannjiang C, Jordan MI, Zrnic T. Prediction-powered inference. Science 2023;382:669–674. ‡
- Angelopoulos AN, Duchi JC, Zrnic T. PPI++: efficient prediction-powered inference. Ann Appl Stat 2026;20(3). 10.1214/26-AOAS2215 ‡
- Leiner J, Duan B, Wasserman L, Ramdas A. Data fission. JASA 2023. 10.1080/01621459.2023.2270748
- Benjamini Y, Yekutieli D. FDR-adjusted multiple confidence intervals for selected parameters. JASA 2005. 10.1198/016214504000001907
- Frostig T, Benjamini Y. Improving the false coverage rate adjusted confidence intervals. JRSS-B 2025. 10.1093/jrsssb/qkaf024

Priors and multivariate EB
- Xie D. Local false sign rate and the role of prior covariance rank in multivariate EB multiple testing. arXiv 2025. 10.48550/arxiv.2502.16118 ‡
- Yang Y, Carbonetto P, Gerard D, et al. Improved methods for EB multivariate multiple testing. arXiv 2024. 10.48550/arxiv.2406.08784 ‡
- Urbut SM, et al. MASH. Nat Genet 2019. 10.1038/s41588-018-0268-8
- Alt EM, et al. Hierarchical grouped horseshoe for subgroup effects. Stat Med 2025. 10.1002/sim.70246
- Thomas S, et al. Horseshoe vs spike-and-slab for treatment-effect heterogeneity. Pharm Stat 2021. 10.1002/pst.2150
- Chen L, Li Z, Wu H. CeDAR. Genome Biol 2023. 10.1186/s13059-023-02857-5
- Soloff JA, Guntuboyina A, Sen B. Multivariate heteroscedastic EB via NPMLE. JRSS-B 2024. ‡ (dropped: point estimates only, asymptotic n)

Leakage, fractions, CSDE
- Nguyen R, et al. Spillover spreading matrix. Cytometry A 2013. 10.1002/cyto.a.22251
- Roca CP, et al. AutoSpill / AutoSpread. Nat Commun 2021. 10.1038/s41467-021-23126-8
- Hu Y, Zhang Y, Chikina M. DALE-Eval: benchmark of cell-type-specific expression methods. bioRxiv 2025. 10.1101/2025.07.31.667984
- Meng Y, et al. Benchmark of cell-type-specific DE methods. Brief Bioinform 2023. 10.1093/bib/bbac516
- Cai Z, et al. DECALS: inference with estimated proportions. JASA 2024. 10.1080/01621459.2024.2382435
- Racle J, et al. EPIC. eLife 2017. 10.7554/eLife.26476
- Ivich A, et al. Missing cell types are detectable. Genome Biol 2025. 10.1186/s13059-025-03506-9
- Zhang W, et al. cypress: power for cell-type-specific DE. Bioinformatics 2024. 10.1093/bioinformatics/btae511

Programs
- Badia-i-Mompel P, et al. decoupleR. Bioinform Adv 2022. 10.1093/bioadv/vbac016
- Alvarez MJ, et al. VIPER (shadow correction). Nat Genet 2016. 10.1038/ng.3593
- Holland CH, et al. Footprint vs member gene sets. Genome Biol 2020. 10.1186/s13059-020-1949-z
- Mao W, et al. PLIER. Nat Methods 2019. 10.1038/s41592-019-0456-1
- Jiang P, et al. CytoSig. Nat Methods 2021. 10.1038/s41592-021-01274-5
- Cui A, et al. Dictionary of immune responses to cytokines. Nature 2024. 10.1038/s41586-023-06816-9
- Singhania A, et al. Modular transcriptional signature of TB. Nat Commun 2018. 10.1038/s41467-018-04579-w ‡
- Singhania A, et al. Type I and type II IFN networks across diseases. Nat Commun 2019. 10.1038/s41467-019-10601-6
- Altman MC, et al. Fixed blood module repertoire. Nat Commun 2021. 10.1038/s41467-021-24584-w

Confounded design
- Nygaard V, Rødland EA, Hovig E. Batch correction in unbalanced designs. Biostatistics 2016. 10.1093/biostatistics/kxv027
- Molania R, et al. RUV-III with pseudo-replicates. Nat Biotechnol 2022. 10.1038/s41587-022-01440-w
- Quartet project. Ratio-based profiling with reference materials. Nat Biotechnol 2023. 10.1038/s41587-023-01867-9
- pwrBRIDGE. Stat Appl Genet Mol Biol 2022. 10.1515/sagmb-2022-0003
- Du J, Roeder K, Wasserman L. Negative-control outcomes for latent batch heterogeneity. arXiv 2024. 10.48550/arXiv.2410.04996

DOIs are from the search agents' reports; verify before citing in a manuscript.
