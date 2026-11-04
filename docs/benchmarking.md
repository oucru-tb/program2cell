# How program2cell is benchmarked

A living document: what we compare, against what truth, with which metrics, and why. Update the
**Changelog** and **Current results** sections whenever the protocol or the numbers change.
Last updated: 2026-11-02.

**Two benchmarks live here:**
- **Section 0 (current, from 2026-10-28):** the known-answer benchmark for **attribution**, i.e. which cell type
  changed which gene program after a treatment. It is the one the paper rests on. A short version for external
  review is in `docs/benchmark_review_brief.md`.
- **Sections 1–8:** the earlier activity benchmark (PBMC COVID, OneK1K sex), which compares per-sample activity
  estimators. It is kept for the record.

---

## 0. Known-answer benchmark for attribution (current)

### 0.1 The question and the claims under test
The question has two layers, together called attribution:
1. **Contribution:** when a gene program changes in bulk, which cell types does the change come from, and how
   much is more cells (more reads) versus cells that changed? Readout: `program2cell.contributions`; check:
   `contributions_eval.py`.
2. **Per-cell change:** for each cell type c and program p, did p change *inside* c, by how much, and can the
   data tell?

The claims and metrics below mostly cover layer 2. Layer-2 results are in §0.6; the layer-1 results table is
in `method_spec.md` §4.7. It is distinct from deconvolution (how many cells) and from bulk
differential expression (what changed in the sample).

Claims tested:
1. **Calls:** unchanged (cell type, program) boxes are not flagged, and real changes are found with the right
   sign.
2. **Honest ranges:** the 90% range holds the true change about 90% of the time.
3. **Accuracy:** estimated changes track the true ones (r, error).
4. **Design limit:** the identifiability calculator predicts which boxes any method can resolve
   (`scripts/experiments/identifiability.py`, `method_spec.md` §4.6).

### 0.2 How the answer key is built (three rules)
1. **Hide the answer in a real sample.**
   - Real single-cell perturbation data: real drugs, real cytokines.
   - Each test specimen's cells are split by binomial thinning. One half, summed over cell types, is the
     bulk the method sees. The other half, kept per cell type, gives the true change β_cp. The method never
     sees it.
   - The truth is scored on 7 PROGENy footprint programs with `estimate_delta`.
2. **Controls fixed by biology before any run.**
   - Responders are set from genetics and confirmed on target genes (below).
   - Negative controls:
     - every box with |β| < 0.05;
     - the non-target programs in responding types;
     - the non-responder types;
     - DMSO vs DMSO (Tahoe).
3. **Honest test.**
   - Reference profiles and PALM's calibration come only from other donors, or from a separate control-cell
     pool.
   - Success targets are set in advance: unchanged flagged ≤ 10%, coverage ≥ 85%, r ≥ 0.9, responders
     found ≥ 90%.
   - Development splits 0–9.
   - The v1 freeze was checked once on held-out splits 10–12. The freeze was lifted 2026-10-30, so a new
     freeze plus fresh held-out splits is required before paper numbers.

### 0.3 Datasets (ladder of evidence)

| Rung | Dataset | Cell types | Design | Known answer |
|---|---|---|---|---|
| 2 | McFarland 2020 (MIX-Seq): `mcf_ida`, `mcf_tram` | 6 cancer lines | 20 paired mixtures per split, Dirichlet(2) proportions, same proportions in both arms | Idasanutlin: p53 up in the 3 TP53-wt lines only. Trametinib: MAPK down in the MEK-dependent lines (LNCaP flat) |
| 2 | Kang 2018 IFN-β: `kang` | B, Mono, NK, T | 4 test donors, paired stim/ctrl pseudo-replicates | JAK-STAT up in every type |
| 2 | Zhang 2021 TNBC blood: `zhang` | immune types | 10 test patients, before vs after paclitaxel ± anti-PD-L1 | No strong response (negative control for a whole dataset) |
| 2 | **Tahoe-100M plate 3** (5 µM): `tahoe_*` | 9 cancer lines pooled per well (LOX-IMVI, HT-29, RKO, NCI-H2030, HOP62, PANC-1, A549, A-172, J82) | 40 mixtures per arm, 12 arms, controls from the same plate's DMSO well | See below |
| 3 | GSE60424 (Linsley 2014): whole blood + sorted neutrophils, monocytes, B, CD4, CD8 of the same 20 donors | 5–6 | Real sequenced bulk; truth from the sorted cells | MS before/after IFN-β; sepsis, T1D, ALS vs healthy |
| 4 | In-house PTB vs healthy PBMC | PBMC + neutrophils | Application | Berry 2010 IFN signature (positive), random gene sets and shuffled labels (negative) |

**Tahoe known answers** (`build_tahoe.py`; the mean log2 FC of 10 target genes per line vs DMSO well 1 was
checked before scoring):

| Drug | Measured response |
|---|---|
| Trametinib | MAPK −2.5 to −3.3 in all lines |
| Dabrafenib | MAPK −0.39 to −0.72 in the BRAF-V600 lines; +0.23 to +0.28 in RAS-mutant HOP62 and PANC-1 (paradoxical activation) |
| Adagrasib | MAPK −0.72 to −0.78 in the G12C lines; RKO −0.45 off-target |
| Gemcitabine | p53 +0.52 to +0.63 in TP53-wt A-172, A549 and RKO |
| Oxaliplatin | No p53 response, so it has no known answer |
| DMSO well 2 vs well 1 | Negative control |

### 0.4 Comparators and calling rules
- **Comparators:** CIBERSORT, InstaPrism (fractions + per-type expression), NNLS, a bulk score, each followed by a
  paired t-test. Also our lab workflow (InstaPrism → GSVA → correlation) and TCA and TOAST (cell-type-specific
  DE, scored by a ULM of their per-type log fold changes on the program weights).
- **Calling rules, reported side by side** (`scripts/experiments/fair_rules.py`):
  1. BH q < 0.1;
  2. q < 0.1 and |change| ≥ 0.1 (a relevant change);
  3. threshold-free ranking (AUROC of |estimate|).
- **Why rules 2 and 3 are needed:** the assembled bulk is nearly noise-free, so a q-only t-test turns small
  biases into "significant" calls. On idasanutlin its false flags rise with the number of paired mixtures:
  24, 58, 79 and 96% at 3, 5, 10 and 20.

### 0.5 Metrics
- **unchanged flagged:** calls on |β| < 0.05; also a curve over 0.02–0.1.
- **real found:** calls on |β| ≥ 0.1 with the right sign; also a curve over 0.05–0.2.
- **responders found:** in the known program.
- **non-responders flagged:** in the known program.
- **coverage** of the 90% range; **r**; **MAE**; **AUROC**.
- **Second answer key:** ULM on the truth half, a different activity model. Only scale-free metrics are used
  against it (Spearman, sign agreement on the top 20% |ULM| boxes, AUROC top 20% vs bottom 50%).

### 0.6 Results so far (development splits)

**InstaPrism + paired t-test vs PALM (default `auto_m`), splits 0–2:**

| | p53 drug | MEK drug | Interferon |
|---|---|---|---|
| Unchanged flagged, q only / q + cut-off / PALM | 91 / 7 / 5% | 81 / 5 / 4% | 83 / 5 / 0% |
| 90% range holds the truth, t-test / PALM | 7 / 92% | 2 / 97% | 55 / 100% |
| Real found at 0.1, t-test with cut-off / PALM | 49 / 57% | 69 / 80% | 94 / 2% |
| Mean abs error, t-test / PALM | 0.066 / 0.044 | 0.038 / 0.032 | 0.047 / 0.039 |
| AUROC vs δ truth, t-test / PALM | 0.80 / 0.79 | 0.93 / 0.95 | 0.99 / 0.98 |
| Spearman vs the second (ULM) truth, t-test / PALM | 0.56 / 0.58 | 0.73 / 0.71 | 0.59 / 0.61 |
| p53-mutant lines flagged on p53 (t-test with cut-off) | all | n/a | n/a |

PALM flags only BICR31 among the label-based non-responders; its weak p53 response (0.12) is real. COLO680N
and TEN stay unflagged.

**Reading:**
- PALM's advantage is honest ranges, and attributions that survive a fair cut-off. It is not ranking: under
  either truth, both methods rank boxes about equally.
- Against the fair comparator, PALM finds more cell-type-specific changes. It finds far fewer changes shared
  by every type (interferon), where it reports the program-level change instead (`shared_response`
  `net_change`, 10 of 10 correct calls on Kang).
- **v1 held-out check** (splits 10–12, run once): PALM flagged 0–1% of unchanged boxes, CIBERSORT + t-test
  69–78%.

**Design limit** (420 boxes, splits 0–2):
- Real changes below the limit: 28; PALM called 0% of them.
- Above the limit: 73; PALM called 58%.

**Tahoe** (fast fraction engine; one full NB fit on `tahoe_dab` split 0, kept as `s0__joint_nb_full`):
- Dabrafenib splits 0–2: coverage 0.89–0.98, no false flags, AUROC 0.89–0.94, no responders found.
- The true MAPK changes are only −0.07 to +0.08: the response is near the noise floor. Even oracle
  reallocation recovers about 0.54 of the effect size.
- Split 0 against InstaPrism + t-test: r 0.55 vs 0.66; coverage 0.94 vs 0.22; unchanged flagged 0 vs 53%
  (q only) or 0% (with a 0.05 cut-off).
- Other arms: running.

**GSE60424** (real whole blood, sorted cells as truth):

| Comparison | PALM | InstaPrism + t-test |
|---|---|---|
| MS IFN-β (3 v 3) | r 0.85, 0% unchanged flagged, 29% found, coverage 0.94 | r 0.88, 0%, 82%, 0.57 |
| Sepsis (3 v 4) | 0% flagged | 24% flagged |
| T1D, ALS | no calls | no calls |

- Caveat: sorting shifts stress and NF-κB genes in the sorted "truth".
- Only 0–2 of 35 boxes are above the design limit at n = 3–4.

**COMBAT** (real whole-blood bulk, truth from the same samples' PBMC single-cell data; `build_combat.py`,
`platform_gap.py`, `combat_real.py`; 125 paired samples, 7 broad PBMC types, splits 0–2, PALM-lite fractions):
- Platform gap: real bulk vs the same sample's single-cell pseudobulk correlate at median r 0.35 (log CPM). A
  per-gene offset explains 89% of the gap, and offsets from two donor halves agree at r 0.998; after removing it
  the per-sample r is 0.96.

| Bulk given to PALM | Sepsis vs healthy (20 test donors): r / RMSE | Severe vs mild COVID (40): r / RMSE | Unchanged flagged | Coverage |
|---|---|---|---|---|
| Single-cell pseudobulk (the usual assembled bulk) | 0.58 / 0.066 | 0.60 / 0.055 | 0% | 97–99% |
| Real bulk, raw | 0.12 / 0.101 | 0.21 / 0.148 | 0–1% | 97% |
| Real bulk ÷ offsets from paired samples of other donors | 0.36 / 0.156 | 0.21 / 0.108 | 1–2% | 96–99% |
| **Real bulk ÷ offsets estimated from the study bulk itself** | **0.52 / 0.102** | **0.64 / 0.061** | 0–1% | 96–99% |

- Raw real bulk loses most of the signal: the main sepsis change (CD14 monocytes TNF-α +0.27, NF-κB −0.26) is
  estimated at +0.03 to +0.09 and −0.03 to −0.06. Methods validated only on assembled bulk are over-optimistic.
- Study-estimated offsets restore single-cell-level accuracy. They are now in the package as
  `program2cell.correct_platform` (PALM-lite fractions and a per-gene median log ratio, alternated four times; no
  paired data). They cost proportion accuracy (read fractions vs truth r 0.25–0.40, against 0.58–0.92). An
  offset on a type's markers and a different mean fraction fit the bulk equally well, so the study's mean
  composition is not identifiable from the bulk alone. Read proportions from the uncorrected fit, or anchor
  them with paired samples or measured counts.

Comparator on the same bulks (InstaPrism + Welch t-test, BH q < 0.1; splits 0–2):

| Bulk | Sepsis: unchanged flagged, InstaPrism / PALM | Sepsis: r, InstaPrism / PALM | Severe: unchanged flagged | Severe: r | Coverage, InstaPrism / PALM |
|---|---|---|---|---|---|
| Single-cell pseudobulk | 25% / 0% | 0.53 / 0.58 | 34% / 0% | 0.57 / 0.60 | 57–63% / 97–99% |
| Real bulk, raw | 59% / 0% | −0.03 / 0.12 | 67% / 1% | 0.00 / 0.21 | 29–33% / 97% |
| Real bulk, study offsets | 56% / 0% | 0.35 / 0.52 | 63% / 1% | 0.42 / 0.64 | 35% / 96–99% |

- On real bulk, the usual workflow flags most unchanged cell-type × program boxes, and its estimates do not
  track the truth (r ≈ 0).
- The platform correction helps InstaPrism too (r 0.35–0.42), but its intervals stay overconfident.
- Built-in caveat: the bulk is whole blood and the reference PBMC, so neutrophils and red-cell RNA are only in the
  bulk. Paired offsets learned on other patients probably miss each study's average whole-blood component, which
  the study's own offsets absorb.

**Parse 10M PBMC cytokines** (real cytokine responses in blood; `fetch_parse.py`, `build_parse.py`; 12 donors,
6 test donors per split, 7 broad PBMC types, reference from the other donors' PBS cells; splits 0–2):

| Cytokine vs PBS | Unchanged flagged: InstaPrism + t-test / PALM | 90% coverage: InstaPrism / PALM | r with truth: InstaPrism / PALM |
|---|---|---|---|
| IFN-γ | 32% / 0% | 65% / 100% | 0.86 / 0.93 |
| IFN-β | 13% / 0% | 69% / 100% | 0.87 / 0.93 |
| GM-CSF | 22% / 0% | 63% / 100% | 0.81 / 0.87 |
| IL-4 | 70% / 0% | 47% / 100% | 0.81 / 0.74 |
| TNF-α (near-null at this timepoint) | 0% / 2% | 50% / 97% | −0.30 / 0.60 |

- True per-cell program changes are small on this scale (≤ 0.16; e.g. IFN-γ: monocyte NF-κB +0.14, JAK-STAT
  +0.04–0.07 in every type), although the raw genes respond strongly (monocyte CXCL9 +3.7 log2). PALM's estimates
  track the truth but it makes almost no calls; this is the shared / small-effect regime.
- With real donor-to-donor variation the t-test is less overconfident than on cell lines (CIBERSORT 5–71%,
  reallocation 4–40% unchanged flagged), so this is the fairest comparison so far; the ordering holds.
- IL-4 (cell-type-specific, small) is PALM's weakest cytokine: CD4 T TGF-β (+0.12) is partly spread onto CD8 T,
  DC and NK when the design check keeps calibrated reallocation.

**Tahoe plate 3** (9 lines, splits 0–2, PALM-lite):
- **Adagrasib on plate 3 is at 0.05 µM** (every other plate-3 drug is at 5 µM), so its weak, misplaced MAPK
  responses (true −0.08 to −0.11 in the G12C lines; PALM and InstaPrism both spread the drop over all lines) are
  not a 5 µM result. A strong-dose set replaces it (`TAHOE_SET=x`: RMC-6236, RAS-mutant responders; encorafenib,
  BRAF-V600 responders).
- Dabrafenib and gemcitabine (5 µM): true per-line changes ≤ 0.1; no method places them; PALM makes no false
  calls (gemcitabine AUROC 0.83).
- Trametinib: MAPK down in every line (−0.17 to −0.36). PALM finds all 9 lines in split 1 and none in splits 0
  and 2: the design check chooses calibrated reallocation over unmixing, and more simulated cohorts (40) do not
  change the choice (the simulation plants mostly type-specific effects; this response is shared). InstaPrism
  copies the bulk change to every line and is right here. **Fixed in freeze v2** (method_spec 4.8): the route
  is now the narrower simulation-calibrated interval with a per-type guard, which finds 0.89 of the lines over
  splits 0–2 (Kang 0.23 → 0.48). Planting the shared-response common effect in the design check was tried
  first and rejected: the flip moved, and Zhang coverage fell to 0.82–0.89.
- Vehicle vs vehicle (`tahoe_dmso2`): PALM 0 calls, coverage 0.90–0.98; CIBERSORT or reallocation + t-test flag
  66–70% of boxes.
- Oxaliplatin at 5 µM barely acts in these lines (max true change 0.087): a second near-null arm.

| Arm (splits 0–2, BH q < 0.1) | Unchanged flagged: deconvolution + t-test (4 tools) | PALM | PALM module on the tools' fractions | 90% coverage: tools / PALM | AUROC: InstaPrism / PALM |
|---|---|---|---|---|---|
| Vehicle vs vehicle | 72–76% | 0% | 0% | 22–33% / 94% | n/a |
| Oxaliplatin | 68–90% | 1% | 0–2% | 22–32% / 93% | n/a |
| Trametinib | 63–87% | 1% | 0–1% | 9–20% / 91% | n/a (all lines respond) |
| Dabrafenib | 64–78% | 1% | 2–6% | 10–17% / 92% | 0.83 / 0.91 |
| Gemcitabine | 73–83% | 1% | 0–2% | 19–31% / 90% | 0.89 / 0.81 |
| Adagrasib (0.05 µM) | 69–95% | 1% | 0–3% | 22–30% / 94% | 0.26 / 0.29 |
| RMC-6236 (strong-dose set) | 59–71% | 1% | 0–1% | 12–14% / 94% | real changes found: tools 74–100%, PALM 93% |
| Encorafenib (strong-dose set) | 73–100% | 1% | 0–1% | 11–16% / 92% | real changes found: tools 100%, PALM 67% |

- Strong-dose set, against the genetics: RMC-6236 lowers MAPK in all 9 lines (−0.13 to −0.32), not only the
  RAS-mutant ones, so its label was changed to "shared" after split 0 (post hoc; the measured truth is
  unaffected). PALM unmixed MAPK in all 3 splits (r 0.79–0.82), unlike trametinib. Encorafenib moves MAPK clearly
  only in HT-29 (−0.135; LOX-IMVI and RKO ≤ 0.05); PALM finds HT-29 in 2 of 3 splits and overestimates NCI-H2030
  (true −0.087, estimated −0.19) in one.

**Composition shift** (`mcf_ida_shift`, responder lines ×1.6 under treatment): responders found, unchanged
flagged 0–9%, coverage 0.86–0.98, but part of the responders' p53 signal moves to COLO680N (true +0.04–0.05,
estimated +0.16 to +0.48) in all three splits: a false flag in two, "can't tell" everywhere in the third.
With trametinib (`mcf_tram_shift`, the non-responder LNCaP ×1.6) nothing moves: all real MAPK drops found,
LNCaP left alone, coverage 0.95–1.00, r 0.92–0.94. The risk is specific to responders that expand.

**Less composition spread** (`mcf_ida_tight`, Dirichlet 10): split 0 "can't tell" everywhere (intervals ±0.8),
split 1 all responders plus one non-responder flagged, split 2 clean (AUROC 1.00, r 0.93): fewer calls, as the
design limit predicts, and the unmix/calibrate choice varies between splits.

### 0.7 Fairness review and stress tests (2026-11-02)

| # | Concern | Status |
|---|---|---|
| 1 | Composition is identical in both arms (paired mixtures); real disease shifts composition | **Idasanutlin done:** `mcf_ida_shift` moves part of the responders' signal onto a non-responder (above). Trametinib shift running |
| 2 | "Home field": the truth is scored with PALM's activity model | **Done:** the second (ULM) truth gives the same ranking for both methods (table above). No home-field advantage in ranking |
| 3 | Reference and calibration from the same experiment (no batch or platform shift) | **COMBAT done:** real bulk vs single-cell reference loses most signal; study-estimated per-gene offsets restore it (above). Still planned: reference from another study |
| 4 | Development and test overlap; freeze lifted | **Freeze v2 done (2026-11-03, tag `method-freeze-v2`):** protocol written first (method_spec, freeze v2); held-out splits 20–29 of 14 datasets + COMBAT running |
| 5 | Strongest competitors and their native tests missing (CIBERSORTx high-resolution, bMIND, ENIGMA; TCA/TOAST native tests) | Planned. TCA/TOAST rescored after the bug fix (BH q < 0.1, unchanged flagged / real found): TOAST 0–12% / 100% on ida, tram, Kang; 21% on Zhang. TCA 11–34% / 58–100%; 49% on Zhang. TOAST is the strongest comparator on per-cell calls (no intervals) |
| 6 | Assembled bulk is nearly noise-free | Calling rules 2 and 3; real bulk with same-sample truth: GSE60424, COMBAT (125 paired samples) |
| 7 | Arbitrary thresholds | **Done:** threshold curves; conclusions hold over 0.02–0.1 (unchanged) and 0.05–0.2 (real) |
| 8 | Splits reuse the same cells, so they aren't independent | Planned: report per experiment; add more Tahoe arms and Parse cytokines |
| 9 | Composition spread more generous than in blood cohorts | **Running:** `mcf_*_tight` (Dirichlet 10 instead of 2) |

### 0.8 Planned: single-group cohorts (e.g. TB patients only, no comparison group)

**Status:** development run done 2026-11-04 (splits 0–9; `scripts/experiments/single_group.py`, summary
`benchmarks/dev/single_group_summary.txt`). Freeze v2 covers two-group effects only. **No path passes yet.**

**Development results (10 splits; "false calls" = pairs with no true association called at α = 0.1):**

| | McFarland gradient idasanutlin / trametinib | COMBAT COVID-only, severity (raw real bulk; other bulk arms in range) |
|---|---|---|
| S3 false calls: activity correlated with Z (`associate`) | 0.62 / 0.73 | 0.39 (0.38–0.81) |
| S3 false calls: unmixing along Z (`associate_unmix`) | **0.16 / 0.17** | **0.24** (0.12–0.24) |
| S3 false calls: InstaPrism + correlation | 0.68 / 0.64 | 0.68 (0.26–0.90) |
| S3 Z permuted (pure null), unmixing | 0.06 / 0.13 | 0.07 (0.07–0.13) |
| Power, unmixing | 0.66 / 0.94 | 0.06 (0.06–0.25) |
| Coverage of the change interval, unmixing (target 0.85) | 0.80 / 0.82 | 0.85 (0.78–0.85) |
| S1 per-patient activity coverage (target 0.85) | 0.52 / 0.64 | 0.38 (0.67–0.72 with platform offsets; 0.93 single-cell bulk) |
| S2 patient ranking (Spearman), PALM vs InstaPrism | 0.89 vs 0.90 / 0.93 vs 0.93 | 0.22 vs 0.21 (0.69 vs 0.65 single-cell bulk) |

Reading:
- Unmixing along Z is the best path (false calls 4× lower than correlation) but misses the ≤ 0.1 target
  (0.12–0.24), and its intervals cover slightly under target.
- The permutation null is near nominal, so the excess false calls come from real associations in other
  cell types leaking into unchanged ones, not from the test itself. On COMBAT the truth is itself estimated
  (single-cell per-type activity), so part of its excess may be truth noise.
- Per-patient activity intervals (S1) are too narrow without T; ranking (S2) equals InstaPrism's.
- Next: v3 item 14 (calibrate along Z); the single-group path stays "not validated" in the explainer and
  the slides.

Everything here is new evidence: develop on splits 0–9, then report on fresh held-out splits, as for v2.

**The use case.** Bulk from one disease group only. PALM is run without T. It gives each patient's fractions and
per-cell-type activity δ_icp with 90% ranges, and the user compares patients against a clinical measure Z
(bacterial load, cavitation, time on treatment, severity). Graphical model:
`docs/figures/talk/2_2d_graphical_model_tb_only.png`.

**Claims to test (each with a pass target fixed now):**

| # | Claim | Metric | Target |
|---|---|---|---|
| S1 | Per-patient activity ranges are honest without T | coverage of the 90% activity interval vs the per-sample truth | ≥ 0.85 |
| S2 | Patients are ranked correctly within a cell type | Spearman across patients, estimated vs true δ_icp, per (type, program) with true between-patient SD ≥ 0.05; median over pairs | report; ≥ comparator |
| S3 | An association with Z is found when it exists and not otherwise | (a) planted gradient: power at α = 0.1; (b) Z permuted: false-positive rate per (type, program) | (b) ≤ 0.1, with (a) reported |
| S4 | The association lands in the right cell type | AUROC of \|association\| for the responding type vs the others | ≥ 0.9 |
| S5 | Absolute level (δ relative to the reference) is or isn't trustworthy | mean (estimated − true) δ per (type, program): bias | report; expected biased on real bulk |
| S6 | The study-estimated platform correction absorbs signal shared by all patients | S1, S2 and S5 on real bulk with self-offsets from (i) the single group alone, (ii) group + controls, (iii) paired offsets | expected: (i) loses the shared level (S5) but keeps the ranking (S2) |

**Datasets:**
1. **Semi-synthetic gradient (McFarland; Kang).** New mixture variant (`build_mcfarland.py MIX_VARIANT=gradient`,
   to write). In each mixture, the cells of each responder line are a share z_i ~ U(0, 1) treated and 1 − z_i
   control, so its true δ rises smoothly with z_i. Z = z_i, plus noise for an imperfect clinical measure.
   - Truth: the held-out half, as usual.
   - Non-responder lines give the null cell types; a permuted Z gives the null association.
   - Kang the same: share of stimulated cells per mixture.
2. **COMBAT, COVID-19 only (real bulk, same-sample single-cell truth).**
   - 91 paired samples with severity HCW-mild 13, mild 18, severe 41, critical 19, used as an ordinal Z.
   - Truth: each sample's own single-cell per-type activity.
   - No healthy samples enter the run.
   - Also sepsis-only (24) as a second single group.
3. **Parse, one cytokine arm only:** 12 donors, real donor-to-donor variation; S1, S2 and S5 only (no Z).

**Methods and comparators (each on the same samples and truth):**
- **PALM:** `program_deconvolution` without treatment, with calibrated activity and intervals; then per
  (type, program) a regression of δ_icp on Z:
  - the patient-level intervals propagated by resampling the fraction draws;
  - p-values by permutation of Z.
  - This association step is new code (`program2cell.associate`, to write), opt-in, chosen on development
    splits.
- **Lab workflow:** InstaPrism → per-type GSVA → Spearman with Z → BH (`lab_workflow.py`, extended to a
  continuous Z).
- **InstaPrism or CIBERSORT + reallocation activity → Spearman with Z.**
- **Bulk-only baseline:** PROGENy score of the bulk vs Z (no cell type; for the "is it cell-type-specific"
  contrast).

**Platform arms on COMBAT (S6):** raw real bulk; `correct_platform` on COVID-only bulk; `correct_platform` on
COVID + healthy bulk (offsets then applied to the COVID samples); offsets from paired samples of other donors;
single-cell pseudobulk (ceiling).

**Splits:** development 0–9 (patients split into reference/calibration donors and test patients, as now), then
held-out on fresh splits after the association step is frozen.

**Scripts to write:** `single_group.py run|summary` (rows in `benchmarks/dev/single_group/`), the gradient
mixture variant, `program2cell.associate`.

**Compute:**
- McFarland gradient with PALM-lite fractions: about 5 min per split.
- COMBAT: about 3 min per split and arm.
- In total, about 3–4 h on 4 lanes for splits 0–9.

### 0.9 How to run

```
bash scripts/experiments/tahoe_chain.sh       # Tahoe plate 3 (build_tahoe.py scan|cells|mix first)
bash scripts/experiments/stress_chain.sh      # composition shift / tight spread
python scripts/experiments/causal_loop.py run DATASETS SPLITS auto_m,auto_abs
python scripts/experiments/practice_vs_palm.py run DATASETS SPLITS        # competitors + t-test, TCA/TOAST
python scripts/experiments/fair_rules.py      # three calling rules + sample-size curve
python scripts/experiments/robustness.py DATASETS SPLITS                  # second truth + threshold curves
python scripts/experiments/identifiability.py run|summary|power ...       # design limit
python scripts/experiments/real_truth_gse60424.py                         # real blood, sorted-cell truth
```

---

## 1. What the benchmark has to show

program2cell claims three things from bulk RNA-seq plus a single-cell reference:

1. **Composition**: read fractions π per sample and cell type, with honest intervals.
2. **Gene program deconvolution**: activity δ of known programs (PROGENy pathways) *per sample and per
   cell type*, not one bulk score per sample.
3. **Cell-type-specific effects**: the treatment shift β of each program in each cell type, with
   calibrated uncertainty.

Claim 2 is the one no standard deconvolution method makes, and the one a "bulk score copied to every
cell type" baseline can fake on some metrics. The metrics below are chosen so that faking it is
visible.

---

## 2. Data, splits and truth

### Datasets (development)

| Name | File | Contrast (β truth) | Test samples |
|---|---|---|---|
| `pbmc` | `pbmc_broad_clean.h5ad` | disease: COVID-19 vs healthy | all eligible test donors (about 38) |
| `o1k` | OneK1K, `onek1k_full_broad.h5ad` | sex: male vs female | 150 |

- **Cell types:** B cell, Monocyte, NK cell, T cell.
- **Programs:** 7 PROGENy programs: JAK-STAT, NFkB, TNFa, TGFb, Hypoxia, p53, MAPK.
- **Genes:** 1000 genes chosen by Fisher selection (`prepare_inputs(max_genes=1000)`).

### Splits

`make_benchmark(seed=split)` splits *donors*:

- **Test donors** give one specimen each, and that specimen becomes a pseudo-bulk sample (the sum of
  its cell types).
- **Reference donors** provide the reference profiles B. For competitors, they also provide the
  single-cell reference.

A donor is never in both. Each test specimen needs at least 20 cells of every type.

- **Splits 0–9: development.** These are used freely while we change the method.
- **Splits 10–19: held-out.** The script refuses them without `--heldout`. Run them **once**, at the
  method freeze, so the final numbers are not tuned to.

### Truth split (binomial thinning)

Each test specimen's per-cell-type counts are split into two independent halves:

- **Bulk half:** summed over cell types, this is what methods see.
- **Truth half:** used only to compute the truth.

Without this, a method is scored against the same counts it was given, and the truth's sampling noise
correlates with the method's input. In practice the effect was negligible, but the split is now
always on.

### Truth definitions

| Quantity | Definition |
|---|---|
| π (N×C) | Share of each cell type's reads on the modelled genes (read fractions, not cell fractions). |
| δ truth (N×C×P) | `estimate_delta` applied to each cell type's truth-half counts: the maximum-likelihood program shift of that cell type's own profile relative to the reference profile B_c. This is the model's estimand. |
| ULM truth (N×C×P) | decoupler ULM scores on the same truth-half counts. This is a model-free check: a method cannot win just by sharing the δ truth's estimator. It is a different scale, so only rank metrics are used against it. |
| β truth (C×P) | Mean δ truth of treated minus mean of controls (a design-based difference). |

**Why two truths.** The δ truth uses the same multinomial estimator as reallocation and as the model's
own definition of activity. A method built on that estimator has a home advantage. ULM weights genes
differently. When the two truths disagree (section 7), the disagreement tells us which gene weighting a
method favours, and the paper must state which estimand it targets.

---

## 3. Methods and rows

### Baselines

| Name | Fractions | Activity |
|---|---|---|
| `naive` | NNLS | One bulk program shift per sample, copied to every cell type. |
| `reallocation` | NNLS | Bulk counts split by π·B (the BayesPrism / InstaPrism expectation step with a fixed reference), then `estimate_delta` per type. |
| `oracle realloc` | true π | Reallocation with perfect fractions: the ceiling for any reallocation estimator. |

### Published competitors (`--competitors`)

Each runs on the same split, genes and truth as the rows above.

| Name | What it gives | Activity used |
|---|---|---|
| `cellgeometry` | Cell amounts, converted to read fractions with each type's mean library size in the reference. | Reallocation with its fractions. |
| `instaprism` | θ fractions, and per-type expression Z (BayesPrism's model, fixed-point solver instead of Gibbs sampling). | `model` row: δ from its own Z. `realloc` row: its fractions with plain reallocation. |

- **Reference:** by default, the reference donors' pseudobulk rows (the same information PALM's B uses).
  With `--cells`, the reference donors' real cells are used instead (OneK1K only). The row name then
  gets a " cells" suffix.
- **Planned after the method freeze:** BLADE, CIBERSORTx (needs Docker and a token), TOAST and TCA
  (effects only).

### PALM rows (per configuration)

| Row | Activity / effects |
|---|---|
| `model` | Posterior median δ; the model's β with its intervals. |
| `design beta` | Posterior δ; β = treated − control mean of δ per draw. |
| `stage-1 mode` | Each sample's own fit, with no pooling (a diagnostic). |
| `posterior realloc` | Bulk split by posterior π × B, then `estimate_delta` per type. Intervals come from π draws. |
| `realloc + alloc` | The same, with a multinomial split per draw (adds allocation noise). |
| `EM realloc`, `iter realloc` | The split uses program-shifted profiles. Both were worse; kept for the record. |

The main configurations are:

- `2step quasi`: NB fractions, then quasi δ given those fractions.
- `joint quasi+sw`: quasi likelihood with a sandwich covariance.
- `joint nb`
- `2step quasi+sw`
- `2step quasi indep`

---

## 4. Metrics, one by one

All metrics are averaged over splits. Comparisons are **paired by split**: the mean difference, with a
bootstrap 95% CI over splits (`summary --ref`). Within a split, cell types and programs share the same
donors, so the split is the unit of replication.

### 4.1 Fractions

| Metric | Definition | Reading |
|---|---|---|
| `frac_rmse` | √mean over (i,c) of (π̂ − π)² | Overall accuracy, including systematic bias between types. |
| `frac_pearson` | Pearson over all (i,c) pooled | Dominated by between-type differences; easy to make high. |
| `frac_r_per_type` | Pearson across samples within each type, averaged | Whether a method tracks variation between samples. This is what downstream association studies need. A method can have a poor RMSE (biased scale) but a good r. |
| `frac_cov90` | Share of the true π inside the 90% interval | Calibration. The target is 0.90; below that means overconfident. |

### 4.2 Program activity (the core of gene program deconvolution)

Let δ̂ be N×C×P estimates and δ the truth.

| Metric | Definition | What it shows |
|---|---|---|
| `act_spearman_across_samples` (**ρ**) | Spearman across samples for each (c,p), averaged over (c,p) with non-trivial truth variance. | "Within cell type c, which samples have high program p?" This is the paired-cohort question. It is the headline metric, but it **does not require cell-type resolution**: a bulk score copied to every type does well whenever samples differ mostly in a shared way. |
| `act_spearman_shared` | The same, on the mean over cell types (per sample and program). | The part any bulk method can get. |
| `act_spearman_specific` | The same, on deviations from that mean (δ_ic − mean over c of δ_ic). | **The part only true deconvolution can get.** The naive baseline is constant here, so it scores nan. This is the metric that shows program deconvolution. |
| `act_specific_share` | The truth's variance share that is type-specific. | How much the specific metric matters. It is about 0.49 on PBMC: half the signal is type-specific. |
| `act_spearman_across_types` | Spearman across cell types within (sample, program), averaged. | "In this sample, which cell type carries program p?" The naive baseline cannot answer this (nan). |
| `act_mae`, `act_pearson` | Absolute error and Pearson on the δ scale. | Scale accuracy. Only for methods on the δ scale. |
| `act_cov90` | Coverage of 90% intervals. | Calibration of the activity intervals. |
| `ulm_*` | The same rank metrics against the ULM truth. | Robustness to the estimand. |

### 4.3 Effects (cell-type-specific differential program activity)

| Metric | Definition | Reading |
|---|---|---|
| `eff_pearson` | Pearson of β̂ with β over all (c,p). | Whether the effect pattern is recovered. |
| `eff_auroc` | AUROC of \|β̂\| for separating \|β\| > 0.1 from the rest. | "Which (cell type, program) pairs respond?" It is nan when no true effect exceeds 0.1, which is the case for OneK1K's sex contrast. |
| `eff_sign_agree` | Share of true effects (\|β\| > 0.1) with the correct sign. | |
| `eff_mae` | Absolute error on the δ scale. | |
| `eff_cov90` | Coverage of 90% intervals for β. | The inferential claim: intervals we can test with. |

### 4.4 Cost and fit quality

- `time`: wall-clock seconds.
- `rhat`, `div`: convergence checks for NUTS. The target is R-hat below 1.01 and 0 divergences.

---

## 5. Leakage and inference metrics for gene program deconvolution

These are implemented in `compare/scoring.py` as of 2026-10-27, and `dev_bench rescore` adds them to
saved fits.

| Column | Metric |
|---|---|
| `act_gain` | Own-type gain |
| `act_leakage` | Leakage (also `ulm_act_leakage`) |
| `eff_localise` | Localisation accuracy |
| `act_coupling` | Spurious fraction–activity coupling |
| `act_iscore`, `eff_iscore` | Interval score |
| `eff_fsr`, `eff_n_called` | False sign rate, and the number of effects called |
| `eff_fdr` | Share of called effects that are not meaningful (wrong sign, or true \|β\| ≤ 0.1) |
| `act_prog_leakage` | Leakage across programs within a cell type (see 5b) |

`eff_fsr` used to combine wrong signs and too-small effects. It now means the wrong sign only (the
Stephens definition), and `eff_fdr` is the combined rate.

## 5b. Program prior sets, and what Tripso adds (2026-10-27)

**Source.** Moullet, Isobe, Vahidi, Leonardi et al., *Self-supervised learning for a gene
program-centric view of cell states* (Tripso), bioRxiv 10.64898/2026.03.24.713961.

- **Setting:** single-cell, not bulk. Tripso learns one transformer embedding per gene program,
  starting from known programs.
- **Benchmarks:**
  - Perturb-seq of 6 cancer cell lines stimulated with cytokines (Jiang et al., *Nat Cell Biol* 2025),
    scored on TGFβ and TNFα.
  - The human endometrium atlas (WNT, TGFβ).
- **Compared against:** Spectra, expiMap, Scanpy `score_genes`, and log-expression baselines.

### What we take from it

**1. Program priors beyond PROGENy-7.** These are now available through `dev_bench --prior`:

| `--prior` | File | Programs | Construction |
|---|---|---|---|
| `progeny7` (default) | `progeny_human_top500.csv` | 7 | The current set |
| `progeny14` | `progeny_human_top200.csv` | 14 | All PROGENy pathways, top 200 genes |
| `collectri` | `collectri_immune.csv` | 26 | CollecTRI regulons of immune TFs, 25–250 targets, signed |
| `hallmark` | `hallmark_immune.csv` | 15 | MSigDB Hallmark immune and state sets, unsigned (weight 1) |

Details of the construction:

- **PROGENy-14:** Tripso takes the 200 genes with the smallest p-values. Our file has only weights, so
  we take the top 200 by \|weight\|.
- **CollecTRI:** Tripso's rule keeps TFs with 25–250 targets and prunes overlapping regulons. Ours
  prunes at Jaccard > 0.5, which removed none.
  - Of 40 candidate TFs, 26 pass the size rule.
  - The rule drops the hubs STAT1, STAT3, NFKB1, RELA, SPI1, CEBPB, MYC, JUN, FOS, HIF1A and TP53
    (over 250 targets). A variant keeping them is worth testing, since they are central in blood.
- **Hallmark:** IFN-α and IFN-γ overlap with Jaccard 0.33, which `act_prog_leakage` should expose.
- **Downloads** (decoupler 2.2 `op.collectri`, `op.hallmark`): raw files `data/collectri_human.csv`
  and `data/hallmark_human.csv`.

Each prior writes to `benchmarks/dev/<dataset>__<prior>/`, because the truth δ is defined on that
program matrix.

The queue `priors.log` runs baselines, cellGeometry and PALM `joint quasi+sw` on 10 splits per prior,
PBMC first.

**First look, PBMC split 0 (baselines only):**

| Prior | naive ρ | NNLS realloc ρ | oracle realloc ρ | Program leakage (NNLS / oracle) |
|---|---|---|---|---|
| PROGENy-14 | 0.75 | 0.81 | 0.84 | 0.23 / 0.18 |
| CollecTRI | 0.59 | 0.73 | 0.81 | 0.32 / 0.23 |
| Hallmark | 0.66 | 0.67 | 0.84 | 0.35 / 0.15 |

- With TF regulons and Hallmark sets, the naive baseline falls far behind.
- The value of accurate fractions also grows: the gap from NNLS to the oracle is 0.08 to 0.16, against
  0.06 for PROGENy-7.
- These richer priors are therefore the more discriminating test of program deconvolution.

**2. Cross-program leakage** (`act_prog_leakage`). This mirrors cross-type leakage.

- **Definition:** regress δ̂_c,p on the true δ_c,· of all programs (same type, across samples). The
  metric is R²(all) − R²(own).
- **Requirement:** at least P + 5 samples; for CollecTRI (P = 26) that means OneK1K-sized cohorts,
  otherwise nan.

**3. Label-based truth: a stimulation benchmark** (proposed; not built).

Tripso scores programs against *known stimulation labels* instead of an estimated truth. That removes
the estimator-dependence problem we see between the δ and ULM truths. The bulk version:

- **Data:** PBMC with an in-vitro stimulus, for example Kang et al. 2018 (8 donors, IFN-β vs control;
  available through pertpy, not decoupler).
  - Or the Jiang 2025 Perturb-seq, using cell lines as "cell types" (cytokines: IFNβ, IFNγ, TNFα,
    TGFβ, insulin).
- **Mixtures:** per pseudo-sample, each cell type is drawn independently from stimulated or control
  cells of a test donor. Each (sample, cell type) then has a known 0/1 label.
  - For a graded truth, mix a share α ∈ {0, 0.1, …, 1} of stimulated cells within a type.
  - The reference comes from control cells of other donors.
- **Metrics:**
  - `stim_auroc`: AUROC of δ̂_{i,c,p*} for stimulated vs control (i,c), where p* is the matched program
    (PROGENy JAK-STAT; Hallmark IFN-α; CollecTRI STAT2, IRF7 and IRF1). Computed within each type and
    pooled.
  - `stim_specificity`: AUROC of p* minus the largest AUROC among unmatched programs, plus the full
    programs × stimuli AUROC matrix (Tripso's "TGFβ from the TGFβ embedding only").
  - `stim_offtype`: AUROC of δ̂_{i,c′,p*} in *unstimulated* types c′ when another type in the same sample
    is stimulated. The target is 0.5; above that is cross-type false activation, the failure
    naive-style methods cannot avoid.
  - `dose_spearman`: Spearman of δ̂ with α. A monotone, estimator-free check of graded activity.
  - Difficulty curve: the metrics against effect strength, like Tripso's knockout panels ranked by
    Mixscale. Here, against α and against the stimulated type's read fraction.
  - Linear probe (Tripso's protocol): a cross-validated logistic F1 or AUROC predicting the label from
    all programs of a type.

**4. Not adopted:**
- GP embeddings: we estimate scalar shifts by design.
- scIB batch metrics.
- The condition dominance and specificity summaries: descriptive, with no truth.
- The Jaccard novelty and reconstruction of discovered programs: we do not discover programs.

## 5c. Calibrated reallocation (in progress, 2026-10-27)

**Method.** Reference donors, never test donors, form a calibration set.
- 5-fold cross-fitting: the profiles B come from the other reference donors.
- The same binomial thinning as the test set.
- True π for the split; fraction uncertainty is added at test time from the posterior draws.
- Per (c,p), a ridge regression maps the reallocated δ̂ of all types to the truth.
- Out-of-fold residuals give split-conformal half-widths for activity.
- Effects get bootstrap (over calibration donors) plus residual variance.
- Script: scratchpad `calib.py`; results in `benchmarks/dev/calib_<dataset>_<config>.json`.

**PBMC joint NB, splits 0–2 (86 calibration donors):**

| Row | Activity cov90 | Activity iscore | Effect cov90 | Effect iscore | Effect FSR (calls) |
|---|---|---|---|---|---|
| posterior realloc | 0.17 | 0.61 | 0.21 | 0.33 | 0.25 (24) |
| calibrated, own type | 0.88 | 0.28 | 0.77 | 0.13 | 0.09 (10) |
| calibrated, all types | 0.87 | 0.29 | 0.77 | 0.14 | 0.07 (8) |

- Activity intervals are fixed, and point accuracy holds: ρ 0.83–0.86 against 0.86.
- Effect intervals are much better but still under 0.90.
- Inverse calibration of effects (invert the learned gain and leakage matrix): effect r falls to 0.03.
  Ruled out.

The metrics above say *how well* activity is ranked. They do not say *why* a method fails. The failure
that matters for program deconvolution is **leakage**: one cell type's program shift showing up in
another type's estimate.

The de-mixing diagnostic (2026-10-27) found that reallocation recovers only about 15% of a B or NK
cell's own shift, while the shifts leaking in from the other types add up to about 1 (PBMC split 0).
The proposed metrics make that visible for every method.

| Metric | Definition | Ideal | What it exposes |
|---|---|---|---|
| **Own-type gain** | For each (c,p): slope of δ̂_c on δ_c in the regression of δ̂_c on all types' true δ_c' (across samples). Averaged. | 1 | Attenuation. Reallocation shrinks the activity of minor types towards 0. |
| **Leakage** | Same regression: the share of δ̂_c variance explained by the other types' truth δ_c' (c'≠c), beyond δ_c itself. | 0 | Contamination. The naive baseline has maximal leakage by construction. |
| **Localisation accuracy** | For programs with a true effect: the share of (program, split) where the cell type with the largest \|β̂\| is the one with the largest \|β\|. | 1 | "Did the method put the effect in the right cell type?" This is the question users actually ask. |
| **Spurious fraction–activity coupling** | Corr(δ̂_c, π_c) − Corr(δ_c, π_c) across samples, averaged in absolute value. | 0 | Activity estimates that move with composition. This matters for the BayesPrism-style analysis in section 6. |
| **Interval score** (Gneiting–Raftery) | Width + (2/α) × miss distance, for 90% intervals. | Low | Calibration and sharpness in one number. Coverage alone rewards infinitely wide intervals. |
| **False sign rate** | Among effects whose 90% interval excludes 0: the share with the wrong sign (true \|β\| > 0.1), plus the share where the true \|β\| ≤ 0.1. | ≤ 0.1 | The error rate a user acting on "significant" effects would experience. |

Own-type gain and leakage come from the same per-(c,p) regression, so they are cheap to compute.
Localisation accuracy needs data with real effects, so it uses PBMC and the planned simulation
(experiment E3), not the OneK1K sex contrast.

**Implementation details:**

- **Gain and leakage:** for each (c,p), δ̂_c,p is regressed across samples on the true δ_·,p of all
  C types plus an intercept, over finite samples only.
  - `act_gain` is the coefficient on its own type. It is computed only for methods on the δ scale.
  - `act_leakage` is R²(all types) − R²(own type only), clipped at 0.
  - A (c,p) needs at least C + 3 samples and non-trivial truth variance.
- **Coupling:** Spearman correlation against the *true* π, so the metric measures the estimate's
  dependence on composition, not the fraction method's errors.
- **Localisation:** when the estimates tie across types (as with naive), each tied type gets 1/C of
  the credit. That gives naive exactly chance level.
- **False sign rate:** a call is "wrong" when the true \|β\| ≤ 0.1, or when its sign differs.

**Sanity values, PBMC split 0 (smoke-test fit):**

| | Gain | Leakage | Localisation |
|---|---|---|---|
| Naive | 0.22 | 0.29 | 0.25 (chance with 4 types) |
| Posterior realloc | 0.45 | 0.11 | 1.0 |

For posterior realloc, 19 of its 20 called effects were wrong. Its intervals are far too narrow, which
is what the false sign rate is meant to show.

---

## 6. BayesPrism and gene programs: what it does, and how ours differs

Related work. Checked against the BayesPrism package (2.2.3: `learn.embedding`, `learn.embedding.nmf`) and
the paper (Chu et al., *Nature Cancer* 2022). The competitor we run is InstaPrism (Hu & Chikina 2024), the
same deconvolution model with a fast fixed-point solver; this section describes the published BayesPrism
program analysis that the plug-in workflow follows.

**What BayesPrism does:**

1. **Deconvolution.** Fractions θ and per-type expression Z, by Gibbs sampling with the scRNA-seq
   reference as a prior. A second pass pools non-malignant expression across samples.
2. **Embedding learning, for the malignant type only.** This needs `key` = the tumor cell type.
   - NMF on the deconvolved malignant expression gives K programs ("malignant bases" η). K is chosen
     by cophenetic score: 5 in GBM, 4 in HNSCC, 7 in SKCM.
   - An EM then refines η, with the non-malignant parts fixed, and gives each bulk sample program
     weights ω.
   - The weights are normalised to sum to 1 for analysis.
   - A user can pass their own programs as `eta_prior` instead of NMF.
3. **Downstream analysis.** Spearman correlation of
   - program weights ω with non-malignant cell fractions (paper Fig. 4e–g), and
   - deconvolved malignant gene expression with macrophage and T-cell fractions (Fig. 3, then GSEA on
     the correlations).

   The authors warn that missing reference cell states can create spurious correlations, and add
   filters to reduce false positives.

So the user's recollection is right: BayesPrism's program analysis is NMF factors (or user programs) in
the malignant compartment, then correlated with fractions estimated from the same bulk.

**How program2cell differs, and how to show it:**

| | BayesPrism | program2cell | How the benchmark shows it |
|---|---|---|---|
| Which cell types have programs | Malignant only | Every cell type | `act_spearman_specific`, across-types ρ, localisation accuracy. |
| What a program is | NMF factors, data-specific and unlabelled (or a user prior) | Fixed, named programs (PROGENy), the same across studies | Interpretability and comparability; not a metric. |
| Scale of activity | Weights summing to 1 within a sample (relative; one program up forces another down) | An absolute log-scale shift of each program relative to the reference | Effects are interpretable per program; ω cannot express "all programs up". |
| Association with a design | Post-hoc Spearman of estimated ω with estimated θ | β estimated jointly with π, with covariates and intervals | `eff_cov90`, interval score, false sign rate. |
| Uncertainty in the association | Not propagated: two plug-in estimates are correlated | Propagated from counts to β | Coverage, false sign rate. |
| Confounding | ω and θ come from the same bulk, so their estimation errors can correlate and create associations | δ and π are estimated jointly, and the coupling is part of the posterior | **Spurious fraction–activity coupling metric, plus a null simulation** (below). |

**The decisive experiment: a null for fraction–program coupling.**

- Simulate bulk where programs are independent of composition. Then do the same with a known true
  coupling.
- Run each method's full pipeline:
  - InstaPrism plug-in (the BayesPrism-style analysis): deconvolution, per-type PROGENy activity from
    its Z, then Spearman activity ~ θ.
  - Reallocation: activity ~ fractions.
  - PALM: posterior of the activity–fraction association.
- **Report** the false-positive rate at nominal 5% under the null, and the power under the alternative.
- **Expectation:** the plug-in correlations inflate false positives because of shared estimation
  error, and PALM's joint posterior does not.
- **Caution:** this is an expectation, not a result. Run it before claiming it.

**Honest status:**

- Today, PALM's own model δ ranks activity *worse* than naive. Our activity gains come from posterior
  reallocation (PALM's fractions plus the BayesPrism / InstaPrism expectation step).
- The cell-type-specific claim therefore currently rests on reallocation plus PALM's fractions. The
  inference claim rests on β intervals, which are calibrated for the model but not for reallocation.
- De-mixing (section 7) and conformal intervals are the planned fixes.

---

## 7. Current results (dev splits 0–9)

Snapshot of 2026-10-27, morning. PBMC is complete; OneK1K has 6–10 splits per row.
Commands: `python scripts/dev_bench.py summary --dataset pbmc --ref "naive|model"`.

### Activity ρ across samples, δ truth (paired difference vs naive, 95% CI)

| | PBMC | OneK1K |
|---|---|---|
| naive | 0.783 | 0.639 |
| NNLS reallocation | 0.791 (+0.01) | 0.665 (+0.03) |
| posterior realloc (joint quasi+sw) | 0.847 (+0.064 [0.06, 0.07]) | 0.743 (+0.104 [0.10, 0.11]) |
| oracle realloc | 0.869 | 0.750 |
| PALM model δ | 0.52–0.60 | 0.43–0.49 |

**Type-specific ρ:**
- Posterior reallocation: 0.45–0.48 on PBMC, 0.50–0.52 on OneK1K.
- NNLS reallocation: 0.23 and 0.36.
- Naive: nan, since it is constant here.

**Effects r:**
- Posterior reallocation: 0.63–0.65 on PBMC, 0.63–0.71 on OneK1K.
- Naive: 0.43 on both.

**Against the ULM truth:**
- Posterior reallocation ties naive on PBMC and is slightly worse on OneK1K (−0.01 to −0.035). The
  oracle reallocation is also −0.03 there, so this is a limit of reallocation.
- The joint NB model's δ is +0.10 against naive on both datasets.

**Coverage:**
- Reallocation intervals: 0.17–0.26 for activity, 0.18–0.35 for effects. They are not usable as they
  are.
- Model intervals: 0.72–0.80 for activity, 0.82–0.84 for effects on PBMC.

### Competitors

Pending; the queue is logged in `benchmarks/dev/competitors.log`.

First check, PBMC split 0, cellGeometry after the unit fix:
- fraction RMSE 0.093
- activity ρ 0.85
- type-specific ρ 0.44
- effects r 0.63

That is at the level of posterior reallocation on activity, and worse on fractions (PALM is about
0.058).

**PALM posterior realloc vs cellGeometry + realloc** (paired by split, 10 splits each, 2026-10-27).
The reference is donor pseudobulk rows. Cells give the difference [95% CI]:

| Metric | PBMC, joint NB | PBMC, joint quasi+sw | OneK1K, joint NB | OneK1K, joint quasi+sw |
|---|---|---|---|---|
| Fraction RMSE (lower is better) | **−0.034** [−0.042, −0.027] | **−0.033** [−0.040, −0.026] | +0.001 [−0.003, +0.007] | +0.001 [−0.001, +0.003] |
| Activity ρ | **+0.017** [+0.011, +0.024] | **+0.015** [+0.006, +0.024] | **+0.027** [+0.019, +0.032] | **+0.030** [+0.024, +0.035] |
| Specific ρ | **+0.052** [+0.032, +0.074] | +0.028 [−0.008, +0.061] | +0.005 [−0.009, +0.018] | +0.003 [−0.007, +0.014] |
| Effect r | **+0.065** [+0.007, +0.116] | +0.067 [−0.001, +0.124] | −0.026 [−0.070, +0.016] | −0.010 [−0.043, +0.025] |
| ULM ρ | +0.003 [−0.002, +0.008] | +0.003 [−0.002, +0.008] | **+0.018** [+0.014, +0.021] | **+0.036** [+0.030, +0.041] |

What the table says:

- **PALM wins activity ρ everywhere.** The margin is small: +0.015 to +0.030.
- **Fractions:** PALM wins clearly on PBMC (0.062 vs 0.096). On OneK1K they are level: cellGeometry
  scores 0.048 and PALM's joint fits 0.049.
- **Specific ρ and effects:** PALM wins on PBMC. They are level on OneK1K, where the sex contrast has
  few real effects.
- **Localisation** (PBMC; OneK1K has no true effects): PALM 0.83–0.87, cellGeometry 0.76.
- **Speed:** cellGeometry takes about 1 minute per split; PALM takes about 2–10 minutes.
- **Intervals:** cellGeometry gives none for activity or effects. PALM's reallocation intervals are
  overconfident:
  - PBMC: 94% of called effects are wrong.
  - OneK1K: every called effect is wrong, because no true effect exceeds 0.1.

  The model's β intervals are conservative instead: PBMC coverage 0.79–0.84; OneK1K
  `joint quasi+sw` coverage 0.73, with 0 calls.

### De-mixing: ruled out (2026-10-27)

**Idea.** For each sample, compute the matrix A_i that maps the true δ to the expected reallocation
estimate under the model (A_i holds the attenuation and the leakage between types), then invert it
with a ridge. Script: scratchpad `demix.py` / `demix_r.py`; results in
`benchmarks/dev/demix*_pbmc_joint_nb.json`.

**What A_i looks like** (PBMC split 0):
- Own-type gain: about 0.15 for B and NK cells, about 0.5 for Monocytes and T cells.
- Summed leakage from the other types: about 0.6–1.1.

**Ridge towards zero** (10 splits). Every strength is worse than reallocation, also with the true π:

| Ridge | ρ | Specific ρ |
|---|---|---|
| none | 0.31 (blows up; MAE 0.95) | 0.31 |
| 0.01 | 0.52 | 0.40 |
| 1 | 0.77 | 0.10 |
| reallocation | 0.849 | 0.476 |

**Ridge towards the reallocation estimate** (5 splits). Results improve steadily as the ridge grows,
and the best only ties reallocation: ρ −0.006, specific +0.001.

**Conclusion.** The model-based leakage matrix does not describe reallocation's real errors: profiles
also differ from B outside the program genes, and inverting amplifies that. Reallocation's attenuation
acts as useful shrinkage for ranking. The remaining activity headroom is the oracle gap
(0.02 ρ, 0.05 specific ρ). Reallocation stays the activity estimator. Work moves to its intervals.

---

## 8. How to run

```
python scripts/dev_bench.py run --dataset pbmc --splits 0-9 --configs night        # PALM presets
python scripts/dev_bench.py run --dataset pbmc --splits 0-9 --configs {} --competitors cellgeometry,instaprism
python scripts/dev_bench.py run --dataset o1k  --splits 0-2 --configs {} --competitors instaprism \
       --cells C:/Users/duy.pham/Downloads/onek1k_full_cells.h5ad
python scripts/dev_bench.py summary --dataset pbmc --ref "joint quasi+sw|posterior realloc"
python scripts/dev_bench.py rescore --dataset pbmc      # after changing metrics: recompute from saved arrays
```

Every row file has a matching `.npz`:

- **PALM:** thinned draws, 100 of each.
- **Baselines and competitors:** `<row>__fractions`, `<row>__activity` and `<row>__effects`.

From the morning of 2026-10-27, metrics come from the 100 thinned draws, in both `run` and `rescore`.
Rows from before then have no `.npz` and keep their old metrics: the PBMC configurations
`2step quasi` and `joint quasi+sw` from phase 1.

- **Outputs:** one JSON per (split, configuration) in `benchmarks/dev/<dataset>/`. Fits also write
  `.npz` arrays, so new estimators can be scored without refitting.
- **Restarting:** runs are restart-safe; existing files are skipped.
- **R methods:** need `PALM_RSCRIPT`.

---

## Changelog

- **2026-11-04**
  - **Held-out v2 finished** (11:49; all four lanes, one pass). Results: docs/heldout_v2_results.md.
    Coverage and false calls met the targets on all 13 datasets and COMBAT; all five targets on 5 of 13; r and
    power missed on Kang, Zhang, Parse GM-CSF / IL-4 and the Tahoe drugs. cellGeometry failed on Kang in 8 of 10
    splits (R error, as in development).
  - `heldout_v2_report.py` sections 5a–5c: results by family, route and response regime (reporting only).
  - Raw held-out outputs zipped with a checksum (`benchmarks/heldout_v2_raw_2026-11-04.zip`, copy in OneDrive).
  - v3 candidate list: docs/v3_plan.md.

- **2026-11-03**
  - **Method freeze v2** (tag `method-freeze-v2`; method_spec "Method freeze v2" and 4.8):
    - `auto` now routes by the narrower simulation-calibrated interval, with a per-type guard;
    - `sim_widen="abs"` is the default;
    - older `causal_loop` estimators are pinned to their v1 settings;
    - the development comparison is in `benchmarks/dev/route_rule_2026-11-03.txt`.
  - `program2cell.correct_platform` / `platform_offsets`: study-estimated per-gene platform offsets for real bulk.
    Tested; the limit on mean composition is documented.
  - COMBAT: InstaPrism + t-test on the same real bulks (table in 0.6).
  - Held-out v2 (`heldout_v2_chain.sh`, splits 20–29) started 12:18.
  - Scratch caches: InstaPrism and cellGeometry work folders are deleted after use (disk).
  - McFarland TP53 status checked in Cellosaurus: genetics agree with the measured responders.

- **2026-11-02 (evening)**
  - COMBAT real bulk with same-sample single-cell truth (`build_combat.py`, `platform_gap.py`,
    `combat_real.py`): platform gap measured; study-estimated per-gene offsets restore single-cell-level accuracy.
  - Parse 10M PBMC cytokines (`fetch_parse.py`, `build_parse.py`, `parse_chain.sh`): pseudobulk subset (PBS + 9
    cytokines) by HTTP range reads; datasets `parse_*`.
  - Tahoe: adagrasib found to be at 0.05 µM on plate 3; strong-dose set (`TAHOE_SET=x`); trametinib design-check
    instability; vehicle-vs-vehicle control.
  - Parse results (5 cytokines × 3 splits). Disk filled during the run (`benchmarks/dev/work` scratch); truncated
    outputs were found by loading every recent file, deleted and recomputed.
- **2026-11-02**
  - Section 0 added: the known-answer attribution benchmark (design, datasets, calling rules, results,
    fairness review).
  - Fair calling rules (`fair_rules.py`): q only, q + |change| ≥ 0.1, and AUROC; sample-size curve.
  - Second-truth and threshold-curve analysis (`robustness.py`).
  - Stress-test datasets `mcf_{ida,tram}_{shift,tight}` (`build_mcfarland.py MIX_VARIANT`).
  - Tahoe-100M plate 3 (`build_tahoe.py`, `tahoe_chain.sh`): 9 lines, 12 arms, answers confirmed on target
    genes.
  - Fixed the TCA/TOAST scoring bug: the R wrappers name columns ct0.., which were looked up by cell-type name,
    so every value was NaN and no calls were made.
  - Design limit: power curves, paired designs and fraction uncertainty (`identifiability.py power`).
- **2026-10-31 to 11-01**
  - GSE60424 real-truth validation (`real_truth_gse60424.py`).
  - Identifiability calculator; scorecard and talk figures.
- **2026-10-30**
  - The BayesPrism-model competitor is now InstaPrism (Hu & Chikina 2024: same model, fixed-point solver,
    seconds instead of minutes); the BayesPrism wrapper and its outputs were removed.
- **2026-10-27**
  - Added program prior sets (PROGENy-14, CollecTRI immune, Hallmark immune), the `--prior` option and
    `act_prog_leakage`, following Tripso (section 5b). Designed a stimulation benchmark with label
    truth (not built).
  - `eff_fsr` is now the wrong-sign rate only; added `eff_fdr`.
  - Calibrated reallocation: first results (section 5c).
  - De-mixed reallocation tested and ruled out (section 7).
  - Implemented the leakage, gain, coupling, localisation, interval-score and false-sign-rate metrics,
    with tests.
  - Added the `rescore` command, and `.npz` arrays for every row.
  - Added the competitors (cellGeometry, InstaPrism) and the `--cells` reference to `dev_bench`.
  - Fixed cellGeometry's conversion from cells to reads: it now uses each type's mean library size,
    not signature column sums. Fraction RMSE went from 0.229 to 0.093.
  - Proposed the leakage, localisation, coupling and interval-score metrics.
  - Started the de-mixing diagnostic.
- **2026-10-26**
  - Added the oracle, EM and iterated reallocation rows, and `.npz` saving.
  - Ran the night2 preset.
  - Introduced the shared/specific activity metrics and the ULM truth.
  - Introduced 10 dev splits with a reserved held-out set, and the truth split on by default.
- **2026-10-28**
  - Open-items run (`scripts/experiments/openitems.py`): fraction sources × 4 prior sets × 10 splits through the calibrated pipeline.
  - Fitting PALM once and transferring its fractions to other prior sets is valid.
  - Stratified conformal effect intervals reach PBMC coverage 0.88.
  - `calib fr` gives nominal activity coverage for engines without draws.
  - OneK1K competitors are complete.
  - Results and decisions are in `method_spec.md` section 7.
  - **Bug fix** in `estimate_delta`: the capped Newton step cycled for strong responses with large program weights (raw PROGENy).
    - Now uses backtracking line search.
    - Kang PROGENy truth was wrong; PBMC PROGENy-7 had 1.2% of entries wrong; Hallmark and OneK1K were unaffected.
    - PBMC PROGENy results are being redone (old files in `old_estimator/`).
  - New dataset **Kang IFN-β** (`--dataset kang`):
    - 8 donors, paired stim/ctrl pseudo-replicates.
    - Reference and calibration from control cells of the non-test donors.
    - `make_benchmark(ref_filter=..., all_test_specimens=...)`.
  - New competitor **CIBERSORT** (`--competitors cibersort`): R script v1.02 with an LM22-style marker signature; activity by reallocation.
  - New pipeline variants: leakage-aware effect width (`lk`), proxy calibration (`calib px`), `comp:<name>` sources.
  - Details are in `method_spec.md` section 7b.
