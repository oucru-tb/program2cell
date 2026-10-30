# Talk outline: lab meeting, 2026-10-30 (15 minutes)

**Title:** *Which cells changed? Decoding cell-type attribution to pathway activity*

*Speaker notes in plain language for an audience with a biology background; statistical terms stay in the
backup slides. Benchmark numbers are PALM v1 (frozen 29 Oct) on development splits, unless marked otherwise.
The v1 held-out check (run once) is in the backup.*

| Part | Content | Slides | Time |
|---|---|---|---|
| 1 | TB cohort: proportions, pathway correlations | deck slides 1–6 | 6.5 min |
| Bridge | Correlation ≠ causality: which cells changed? | deck slide 7 | 0.5 min |
| 2.1 | The problem, and why it matters | 1 | 1 min |
| 2.2 | How PALM solves it | 1 | 1 min |
| 2.3 | General benchmark | 1 | 1.5 min |
| 2.4 | Case studies: PALM vs InstaPrism | 2 | 2 min |
| 2.5 | In-house PTB application (next week) | 1 | 1 min |
| 2.6 | Outlook: spatial transcriptomics | 1 | 0.5 min |
| 2.7 | Conclusion | 1 | 1 min |
| | **Total** | **15** | **15 min** |

**The story in one line:** in TB, myeloid cells expand and their inflammatory pathways rise together. But a
correlation can't tell whether that is *more* cells or cells that *changed*. PALM tells us which cells
changed, and how sure we can be. We test it where the answer is known, then ask it of our TB data.

**Tone for part 1 → part 2:** this is not "part 1 was wrong". The associations stand. What needs checking is
which cell type they belong to, and every lab using this standard workflow faces the same issue.

**Every evaluation slide carries a small "Experimental design" box** (bottom-left, 4–5 lines): data, how
the truth is known, comparison, metrics, what counts as success. The full design for each one is below.

---

## Part 1: TB results (6.5 min, deck slides 1–6)

| Deck slide | Content | Time |
|---|---|---|
| 1 | Title | 15 s |
| 2 | PTB vs healthy, PBMC bulk RNA-seq: host response? | 45 s |
| 3 | Decoding bulk with a single-cell reference | 45 s |
| 4 | PBMC reference (includes neutrophils); predicted vs measured lymphocyte / myeloid % (r = 0.91) | 1 min |
| 5 | Cell fractions × pathway scores (Spearman) | 1.5 min |
| 6 | PTB: more myeloid cells, coordinated innate pathways | 1.5 min |

**Fill in on slide 2:** PTB (n = ), Healthy (n = ).

**Experimental design box for slide 4 (the one validation in part 1):**
- Data: PTB and healthy PBMC bulk RNA-seq; reference: PBMC single-cell atlas including neutrophils.
- Truth: measured lymphocyte and myeloid % (fill in: blood count / flow cytometry, same samples).
- Comparison: predicted vs measured, per sample. Metric: Pearson r (0.91).

### Conclusion box for slide 6
> **In PTB, myeloid cells expand, and their innate pathways (TLR, NLR, IFN-γ) rise *together*, while T-cell
> receptor signalling moves the opposite way.**

**What to say (three points):**
1. **More cells:** neutrophils, CD14 and CD16 monocytes are all higher in PTB.
2. **Tighter coordination:** in PTB (lower triangle) TLR, NLR and IFN-γ correlate more strongly than in
   healthy people (upper triangle): neutrophils TLR–NLR 0.80 vs 0.59, NLR–IFN-γ 0.96 vs 0.85; CD14 monocytes
   0.79 vs 0.66, 0.93 vs 0.90.
3. **Opposite direction for T-cell signalling:** TCR correlates negatively with the innate pathways in PTB
   (−0.4 to −0.6). In CD16 monocytes, TNF also turns against TLR, NLR and IFN-γ (−0.66 to −0.78).

**Careful wording:** say "associated" and "coordinated", not "drives" or "activates". Don't reveal on slide 6
that the standard workflow can make pathways *look* coordinated across cell types; part 2 shows it.

---

## Bridge (30 s, deck slide 7: correlation ≠ causality)

> "So in PTB we see more myeloid cells and stronger inflammatory pathways. Is that because there are *more*
> of these cells, or because each cell *switched on* its pathways? And is the signal really in the neutrophils,
> or did it come from the monocytes? A correlation can't tell us. That is what part 2 is about."

---

## Part 2: PALM

### 2.1 The problem, and why it matters (1 min)
- **Visual, left:** `2_1_three_causes.png`. **Right:** `2_5_heatmap_lab_workflow.png` (answer key | our
  workflow | PALM; or show only the first two panels and reveal PALM in 2.3).
- **Say (left):** "The neutrophil inflammatory score is higher in PTB. Three reasons give the *same* bulk
  signal: more neutrophils; monocytes switched on, and part of their signal is assigned to neutrophils; or
  neutrophils themselves switched on. Only the last one is what we want."
- **Say (right):** "Is this a real problem? We ran our own part-1 workflow on an experiment where we know the
  answer: a drug that switches on p53 in only 3 of 6 cell lines. Our workflow flags 6 of 7 pathways in all 6
  lines: 19 of the 23 boxes that did not change. It can't tell the responding lines from the others."
- **Why it matters:** "A wrong cell type means a wrong biological story, and wrong follow-up experiments. It
  can also make pathways look coordinated across cell types, as on slide 6, because they share one bulk
  signal."
- **Design box:** McFarland idasanutlin, 6 cell lines, 20 mixtures per arm; truth from held-out cells;
  workflow = InstaPrism → GSVA → correlation with treatment → BH q < 0.1 (our part-1 pipeline).

### 2.2 How PALM solves it (1 min)
- **Visual:** draw the slide-3 pipeline again, adding three boxes after it:
  1. **Proportions with uncertainty** (a Bayesian model of the bulk counts);
  2. **Correct the spillover** between cell types, learned on reference donors where the answer is known,
     and using how the bulk changes as proportions vary across samples;
  3. **A change per cell type and pathway, with an honest 90% range**, and "can't tell" when the data can't
     separate the cell types.
- **Say:** "PALM keeps the same first steps: proportions, then each cell type's share of the reads. It adds
  what was missing: it removes the spillover and the shrinkage, and it says how sure it is."
- Works with any gene set, including our TLR, NLR, IFN-γ, TNF, BCR and TCR pathways.

### 2.3 General benchmark (1.5 min)
- **Visual:** `2_5_controls.png` (main). Backup: `bench_calling.png` (all tools, every metric).
- **Say first (10 s), how to read it:** "Left of the line, the negative control: boxes that did not change
  but were flagged; shorter is better. Right, the positive control: real changes found; longer is better."
- **Say:**
  - "In the two drug experiments PALM finds 92–100% of the real changes, like every other tool. But it flags
    under 5% of the unchanged boxes, where the others flag 72–100%."
  - "Chemotherapy is a negative control for a whole dataset: nothing responds strongly, and nobody
    over-flags. The problem appears when one cell type responds strongly."
  - "Interferon is PALM's limit: no false flags, but it finds nothing either. I'll come back to it."
- **Experimental design (box + full version):**

  | Item | Design |
  |---|---|
  | Datasets | 4 single-cell experiments with a treatment, turned into bulk: McFarland 2020 idasanutlin and trametinib (6 cancer cell lines, 20 paired mixtures per split); Kang 2018 interferon-β (PBMC, 4 cell types, 14 samples from 4 test donors); Zhang 2021 TNBC blood before vs after paclitaxel ± anti-PD-L1 (10 test patients, paired) |
  | How the truth is known | Each sample's cells are split in two (binomial thinning): one half is pooled into a bulk sample, the other half gives the true change per cell type and pathway (PROGENy, 7 pathways) |
  | No leakage between test and training | Reference profiles and PALM's calibration come only from other donors (Kang, Zhang) or a separate pool of control cells (McFarland) |
  | Comparators | CIBERSORT, InstaPrism, NNLS, bulk score without deconvolution, each followed by the usual t-test (BH q < 0.1); our part-1 workflow (InstaPrism → GSVA → correlation) |
  | Metrics | Negative control: unchanged pairs (\|true\| < 0.05) flagged. Positive control: pairs in the drug's pathway with \|true\| ≥ 0.1 found with the right sign. Also r with truth and coverage of the 90% range |
  | Success targets (set in advance) | Unchanged flagged ≤ 10%; coverage ≥ 85%; r ≥ 0.9; responders found ≥ 90% |
  | Replication | 3 random splits (development); v1 held-out splits run once (backup) |

### 2.4 Case studies: PALM vs InstaPrism (2 min, 2 slides)
InstaPrism is the tool we use, so this is the head-to-head that matters for the lab.

**Slide 2.4a: every aspect, every dataset (1 min).** Visual: `case_scorecard_palm_vs_instaprism.png`.
Read it row by row, one sentence each:

| Aspect (row) | Result | Say |
|---|---|---|
| Proportions | PALM better on interferon (2.7 vs 6.0 pp error) and slightly on idasanutlin and chemotherapy; **InstaPrism better on trametinib (1.0 vs 2.8)** | "Proportions are close; neither wins everywhere." |
| Negative control | PALM 0–5% vs InstaPrism 81–91% unchanged pairs flagged (chemotherapy: 1% vs 0%) | "This is the big difference." |
| Positive control | Both find nearly all real changes in the drug experiments (92–100%); interferon: InstaPrism 100%, PALM 0% | "InstaPrism finds everything because it flags everything." |
| Effect accuracy | r 0.92 vs 0.78 (idasanutlin), 0.91 vs 0.87 (trametinib), 0.96 both (interferon) | "PALM's sizes are closer to the truth when only some cells respond." |
| Honest ranges | Truth inside the 90% range: PALM 92–100%, InstaPrism + t-test 2–55% in the three response datasets | "InstaPrism's ranges are confidently wrong." |

**Slide 2.4b: the biology, cell by cell (1 min).** Visual: `case_studies_palm_vs_instaprism.png` (top two
panels for the talk; the interferon panels are backup). ◆ = truth; filled = flagged; line = 90% range.
- **Idasanutlin → p53.** Idasanutlin blocks MDM2, so p53 accumulates, but only in lines with intact p53.
  - InstaPrism gives *every* line a similar rise (0.30–0.43), so p53-mutant COLO680N and TEN (true 0.04–0.05)
    look like responders, and the strongest responder LNCaP is underestimated (0.43 vs 0.68).
  - PALM recovers the order LNCaP 0.66 > RCC10RGB 0.55 > SNU1079 0.41, leaves COLO680N and TEN unflagged,
    and picks up BICR31's weak real response (0.12).
- **Trametinib → MAPK.** Trametinib blocks MEK. Five lines depend on it (true −0.14 to −0.27); LNCaP, a
  prostate line, doesn't (−0.03).
  - InstaPrism flags LNCaP at −0.18, the same as the responders. PALM leaves it unflagged (−0.06).
- **Key sentence:** "When a drug acts on some cells only, the standard approach paints every cell with the
  same brush. PALM tells responders from non-responders."
- **Experimental design (box):** same splits and truth as 2.3; InstaPrism with the same reference; calls at
  BH q < 0.1 (InstaPrism + t-test) and a 90% range excluding 0 (PALM); split 0 shown, splits 0–2 in the
  scorecard. The cell-line biology (p53 status, MEK dependence) is the external positive/negative control:
  it was not used to fit anything.

**Interferon, if asked (backup panels):**
- JAK-STAT rises in every cell type (0.26–0.36), so copying the bulk signal to every cell type happens to
  be right, and InstaPrism finds it.
- But monocytes also have their own changes (TGF-β −0.18, hypoxia −0.17, MAPK −0.16, ≈0 elsewhere).
  InstaPrism copies those into B, NK and T cells too.
- PALM says "can't tell" for B and NK: 14 samples with almost identical proportions (B cells 5% ± 1%) can't
  separate them. A fix in testing narrows the ranges and recovers the monocyte response with no new false
  flags.

### 2.5 In-house PTB application: next week (1 min)
- **Visual:** the slide-6 cell types (neutrophils, CD14 and CD16 monocytes) × pathways (TLR, NLR, IFN-γ, TNF,
  BCR, TCR), as an empty grid with two columns per box: InstaPrism | PALM. "Results next week."
- **Questions:**
  1. Are TLR, NLR and IFN-γ higher in each neutrophil and monocyte in PTB, or only because there are more
     of these cells?
  2. Which of the slide-5 associations remain once spillover is removed?
  3. Where do InstaPrism and PALM disagree, and which cell types does PALM call "can't tell"?
- **Experimental design (box + full):**

  | Item | Design |
  |---|---|
  | Data | PTB vs healthy PBMC bulk RNA-seq (n = / n =); same PBMC reference with neutrophils as in part 1 |
  | Pathways | The six part-1 pathways (TLR, NLR, IFN-γ, TNF, BCR, TCR), same gene sets |
  | Methods | InstaPrism → GSVA → test (part-1 workflow) vs PALM; same reference, genes and samples |
  | Contrast | PTB vs healthy per cell type × pathway, adjusted for age and sex |
  | Proportion check | Predicted vs measured lymphocyte / myeloid % for both methods (part 1: r = 0.91) |
  | Positive control | Known TB biology: the interferon-inducible signature in neutrophils and monocytes of active TB (Berry et al. 2010) should come out up in PTB |
  | Negative controls | (a) Size-matched random gene sets: should not be flagged; (b) PTB/healthy labels shuffled: should give no calls |
  | Robustness | Leave-one-reference-donor-out; and a reference without neutrophils, to see how much the calls depend on it |
  | Readout | Per box: change with 90% range (PALM) vs score difference and q-value (InstaPrism); agreement table; split of each change into "more cells" vs "cells changed" |

- **Can claim:** which cell types and pathways differ between PTB and healthy people, by how much, adjusted
  for age and sex. **Can't claim:** that TB caused it (patients vs controls). Before vs after treatment in the
  same patients would come closer.

### 2.6 Outlook: spatial transcriptomics (30 s)
- **Idea:** a microdissected region (e.g. GeoMx / NanoString ROI) is itself a small bulk sample. The same
  question applies: in a TB granuloma, is the inflammatory signal in the core vs the rim from *more*
  macrophages, or from macrophages that switched on?
- **Why PALM fits:** regions differ a lot in composition (exactly what PALM needs to separate cell types),
  and many regions per slide give the sample size.
- **Experimental design (proposed):** build "regions" from single-cell-resolution spatial data (Xenium, CosMx
  or Visium HD) by pooling the cells inside each region, so the true per-cell-type change is known; then
  compare PALM with spot-deconvolution tools (e.g. cell2location, RCTD) on the same regions. No results yet.

### 2.7 Conclusion (1 min)
1. **Bulk RNA-seq can tell us which cell type changed a pathway,** but only after correcting its built-in
   spillover. Without it, a response in some cells is painted onto all of them.
2. **PALM gives that answer with an honest range:** on known-answer experiments it finds the real changes
   (92–100% in the drug experiments) and flags under 5% of the unchanged boxes, where the standard tools
   flag 70–90%.
3. **It says "can't tell" when the data can't separate cell types** (interferon, B and NK cells), instead
   of guessing.
4. **Next week:** PALM vs InstaPrism on our PTB cohort. Which cells changed in TB?
- **Last line:** "We can now ask not only *what* changed in the blood of TB patients, but *which cells*
  changed, and how sure we are."

---

## Figures

Generated figures are in `docs/figures/talk/`: 300 dpi PNG plus SVG (editable text in PowerPoint), sized for a
16:9 slide, no titles (put the message in the slide title). Colours keep one meaning: teal = PALM, red =
standard tools, dark = truth. Rebuild with `python scripts/experiments/talk_figures.py [SPLIT] [figure,...]`
and `python scripts/experiments/benchmark_heatmaps.py`.

| Slide | Visual | File |
|---|---|---|
| 2.1 | Three causes of the same bulk signal; our workflow on idasanutlin | `2_1_three_causes.png`, `2_5_heatmap_lab_workflow.png` |
| 2.2 | Pipeline + three new boxes | draw |
| 2.3 | Negative / positive controls, all tools, 4 datasets | `2_5_controls.png` (backup: `bench_calling.png`, `2_5_false_flags_all.png`) |
| 2.4a | PALM vs InstaPrism scorecard: 5 aspects × 4 datasets | `case_scorecard_palm_vs_instaprism.png` |
| 2.4b | Per cell line: idasanutlin p53, trametinib MAPK | `case_studies_palm_vs_instaprism.png` (top panels) |
| 2.5 | Empty grid: 3 cell types × 6 pathways, InstaPrism \| PALM | draw |
| 2.6 | Granuloma core / rim cartoon with ROIs | draw |
| 2.7 | Four short lines | text |
| Backup | Proportions and activity, all tools; interferon ranges; estimated vs true; CIBERSORT heatmap | `bench_deconv.png`, `2_6_interferon_ranges.png`, `2_3_estimated_vs_true.png`, `2_5_heatmap_idasanutlin.png`, `2_6_fraction_error.png` |

**Figure reading notes:**
- `2_5_heatmap_lab_workflow`: the middle panel is coloured by correlation (ρ, its own scale); the other two
  by change in activity (shared scale).
- `case_scorecard`: larger dot and bold number = the better of the two; same splits for both methods.
- `case_studies`: the interferon panels use the post-freeze interval fix (marked in the footnote); show only
  the top two panels in the talk, which are identical to v1.
- `bench_calling` / `bench_deconv`: bars scaled per column (longest = best); 1–3 = rank. Rates use a fixed
  0–100% scale; other metrics are scaled from the worst to the best method, which magnifies small gaps.

---

## Backup slides (only if asked)

- **"Is it fair to our workflow?"**
  - The emulation is InstaPrism per-type expression → log CPM → GSVA → Spearman with treatment → BH q < 0.1.
  - Gene sets are the up-weighted genes of each PROGENy pathway, so they match the answer key.
  - Slide 5 correlated pathways with fractions, not with a treatment, so the leakage applies but the exact
    numbers differ.
  - Code: `scripts/experiments/lab_workflow.py`.
- **"Is it tuned to the test?"** The v1 result was checked once, on data held back until the method was
  frozen: PALM flagged 0–1% of unchanged pairs, CIBERSORT + t-test 69–78%.
- **Why the correction matters:** `2_3_estimated_vs_true.png`. Without correction, changes come out at about
  75% of their size and some non-responding lines appear to change; with correction, about 95%.
- **Method terms:** posterior reallocation of bulk reads by fractions × reference profiles; calibration
  learned on reference donors; effects estimated by unmixing across samples; intervals include fraction
  uncertainty and the treatment response outside the pathways.
- **Causality:** it comes from the study design (randomised or paired); PALM provides the measurement. For
  observational data it reports robustness values (how strong a hidden confounder would have to be).
- **Held-out scoreboard (PALM v1, run once):**

  | Dataset | Coverage | r | False calls | Known answer | CIBERSORT + t-test false calls |
  |---|---|---|---|---|---|
  | McFarland idasanutlin | 0.92 | 0.92 | 0.01 | AUROC 1.0, 75% of responders found | 0.78 |
  | McFarland trametinib | 0.93 | 0.87 | 0.00 | all responders found | 0.77 |
  | Zhang | 0.98 | — | 0.00 | — | 0.00 |
  | Kang | 1.00 | 0.94 | 0.00 | power-limited | 0.69 |

- **InstaPrism, same proportions, three ways (false calls on unchanged pairs):**

  | Dataset | InstaPrism + GSVA + correlation | InstaPrism + t-test | InstaPrism proportions + PALM |
  |---|---|---|---|
  | idasanutlin | 0.82 | 0.91 | 0.03 (responders found 0.67) |
  | trametinib | 1.00 | 0.81 | 0.08 (responders found 1.0) |
  | interferon | 0.41 | 0.83 | 0.00 (misses the response) |
  | chemotherapy | 0.00 | 0.00 | 0.00 |

  Most of the gain is PALM's correction step, not only its proportions.
- **In development (not in v1):** a per-cell-type leakage allowance that narrows the interferon ranges and
  recovers the monocyte response with no new false flags (all 10 development splits); a "shared vs
  cell-type-specific" readout (e.g. interferon: everything moved together, JAK-STAT 0.31 [0.27, 0.35]).
- **Run time:** seconds (fast screen), about 5 min (default).
