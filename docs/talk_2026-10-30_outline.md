# Talk outline: consortium meeting (lab + collaborators), 15 minutes, revised 2026-11-02

**Title:** *Which cells changed? Attributing bulk gene-program changes to cell types, and how to test it*

*Audience: clinicians, wet-lab scientists and analysts. The talk's purpose is three things: the **problem**,
the **approach**, and the **evaluation design**. Results appear only as proof that the design works; details
go to backup. Plain words on the slides, technical terms in backup. Benchmark numbers are development splits
unless marked "held-out".*

| Part | Content | Slides | Time |
|---|---|---|---|
| 1 | TB cohort, condensed: proportions and pathway correlations | deck 1–6 | 3.5 min |
| Bridge | Correlation ≠ causality: which cells changed? | deck 7 | 0.25 min |
| **A. Problem** | One bulk signal, three explanations; the question; it happens in our own workflow | 3 | **3 min** |
| **B. Approach** | How PALM answers it | 1 | **1.5 min** |
| **C. Evaluation design** | Building an answer key; ladder of evidence; can a study answer it at all? | 3 | **4 min** |
| D. Proof it works | Negative and positive controls, all tools | 1 | 1.5 min |
| E. Next + conclusion | PTB next week; three take-home messages | 1 | 1.25 min |
| | **Total** | **≈ 15** | **15 min** |

Weight: problem + approach + evaluation design = 8.5 of the 11.5 minutes in part 2.

**The story in one line:** bulk RNA-seq tells us *that* a gene program changed; we ask *which cells* changed
it. We define the question, show it bites in our own workflow, propose a method, and, most importantly, show
how to test any method on it fairly, including whether a study can answer it at all.

**Three take-home messages (they open part 2's last slide):**
1. **The problem:** attribution: where a bulk gene-program change comes from (which cell types; more cells or cells changed) and which cell types themselves changed. Standard tools answer "how
   many cells" or "what changed", not "which cells changed", and they spread one cell type's change onto all.
2. **The approach:** PALM undoes that spillover, at the gene-program level, and says how sure it is.
3. **The evaluation design:** an answer key from real experiments, controls fixed by biology in advance, and a
   design limit that says before sequencing which cell types a study can resolve.

**Updates of 2026-11-03, for the 4-section version** (PTB results / attribution and approach / evaluation
design / PTB application). Development numbers unless marked held-out.

*Section 2, attribution and approach: two principle slides in the talk, two graphical models in backup.*

All four use one colour code: 🔵 blue = how many cells (fraction), 🟠 orange = program switched on in a cell
(activity), ⚪ grey = reference profile (atlas), 🟢 green = program genes. Keep the same colours on the
results slides.

| Order | Slide | File (`docs/figures/talk/`) | Time | Message |
|---|---|---|---|---|
| 2.1 | **"One bulk sample = many cell types, each with programs switched on or off"** | `2_2_principle_formula.svg/.png` | 45 s | Bulk = Σ cell types 🔵 fraction × ⚪ reference profile × exp(🟠 activity × 🟢 program genes). Blue changed → more cells; orange changed → cells switched on |
| 2.2 | **"Where does a bulk change come from?"** | `2_2b_principle_contributions.svg/.png` | 30 s | The same equation read the other way: the bulk change = Σ cell types (more-cells part + cells-changed part), adding up exactly; contribution and per-cell change can disagree |
| 2.3 (optional) | **"PALM as a graphical model"**, simple | `2_2c_graphical_model_simple.svg/.png` | 30 s | Plate diagram (T → γ → π, T → β → δ, π + δ + B + G → Y) and the four steps from model to answers. For a mixed audience |
| Backup | **"PALM as a graphical model"**, detailed | `2_2c_graphical_model.svg/.png` (`scripts/experiments/talk_graphical_models.py`) | — | Faithful to the frozen method. Left: the model PALM fits (it gives π); B and G in the gene plate, the cell-profile node E = B·exp(δ×G). Right: the reported β comes from split reads → calibrate on atlas donors → unmix across patients; the atlas feeds both B and the calibration; contributions from π and β |
| Backup | **"PALM with TB patients only"**, simple | `2_2d_graphical_model_tb_only_simple.svg/.png` | — | No treatment node; what you can and can't claim; the comparison along Z is marked "being validated" |
| Backup | **"PALM with TB patients only"**, detailed | `2_2d_graphical_model_tb_only.svg/.png` | — | Same layout as the detailed two-group slide; unmixing along the clinical measure z gives the change over the z range; "not yet validated" box with the first-split numbers (`benchmarking.md` §0.8) |
| Backup | **"PALM as a Bayesian network"** (for statisticians) | `2_2e_graphical_model_bayes.svg/.png` (`scripts/experiments/talk_graphical_model_bayes.py`) | — | Standard plate notation of the stage-A generative model: latent / observed / deterministic nodes, priors, plates N, C, P, G, equations alongside; footnote that the reported β̂ comes from the calibrated unmixing (modular / cut inference) |

- **One sentence (slide 2.1, or a title):** "Bulk = a mix of cell types, each one its atlas profile with
  gene programs turned up or down; we estimate both the mix and the switches."
- **Script for slide 2.1 (≈ 30–45 s):**
  > "We treat each blood sample as a mix of cell types. For each cell type we know from a single-cell atlas
  > what it normally looks like, and from public databases which genes make up each program, such as the
  > interferon response. A cell type that switches a program on raises that program's genes above its normal
  > level. So the sample we sequence is the sum, over cell types, of how many of those cells there are times
  > their atlas profile, shifted by whichever programs they have switched on. PALM works backwards from
  > hundreds of genes and many patients to how much of each cell type there is and how strongly each one
  > switched each program on, each with a 90% range. When two cell types look too alike to separate, the
  > range widens: 'can't tell' instead of a guess."
- **Script for slide 2.2 (≈ 30 s):**
  > "The same equation, read the other way. The change we see in blood is a sum over cell types, and each
  > part splits into *more of them* and *each of them changed*. So instead of 'IFN-γ is up' we can say
  > 'three quarters of it comes from monocytes, mostly because they switched on'."
- **If asked "is the atlas profile a resting cell?":** no. It is the atlas's average cell of that type, so
  activity is relative to that average. We use healthy donors and merge activation-state clusters into their
  parent type (`model_dag.md`, review point 2).
- **If asked "can it run on TB patients only?":** yes, for each patient's mix and activity and for
  comparisons within TB. There is no TB-vs-healthy effect without controls, and the platform correction then
  needs outside offsets (backup slide `2_2d`).
- The numbers in the slide examples (75% monocytes, and so on) are illustrations, not results. Replace them
  with PTB results when available.

*Section 3, evaluation design: one new line and one new row on the ladder.*
- **Real bulk rung (COMBAT):** 125 people with whole-blood bulk *and* single-cell data from the same tube.
  - On real bulk, the usual workflow (InstaPrism + t-test) flags 59–67% of unchanged cell-type × program
    boxes, and its estimates do not track the truth (r ≈ 0).
  - PALM flags 0–1%. After the platform correction its r is 0.52–0.64, about the level reached on
    single-cell bulk (0.58–0.60).
  - Say: "Methods tested only on simulated or assembled bulk look better than they are."
- **Shared changes (the interferon weakness):** fixed in freeze v2.
  - Responses shared by every cell type are now found: Tahoe trametinib 0.33 → 0.89 of the lines; Kang
    interferon none (v1) → 0.48. Coverage stays ≥ 0.93.
  - Say: "PALM now also finds changes that every cell type shares, not only cell-type-specific ones."
- **Pre-registration:** the method was frozen (tag `method-freeze-v2`) and the held-out protocol written down
  *before* the held-out run. That run uses splits 20–29 of 14 datasets plus COMBAT, with the results to come.
- **Controls fixed by genetics:** the three idasanutlin responders are TP53 wild-type, and the three
  non-responders carry homozygous TP53 mutations (Cellosaurus), so the "3 TP53 WT and 3 TP53 mutant" wording
  is correct.

*Section 4, PTB application (what changes in the plan):*
- The PTB bulk is real PBMC bulk with an atlas reference, the COMBAT situation. Run
  `bulk = pb.correct_platform(bulk, ref)` first: one offset per gene, estimated from the study itself, which
  cannot absorb the PTB vs healthy difference.
- Read cell proportions from the uncorrected bulk, or check them against the measured blood counts. The
  correction fixes gene shapes, not the average composition.

**Words used on the slides (keep them the same throughout):**

| Word on slide | Meaning | Technical term (backup only) |
|---|---|---|
| gene program | a set of genes that switch on together for one job, e.g. the interferon response | program, gene-weight vector |
| a "box" | one cell type × one gene program | (c, p) pair |
| change | how much a program moved in that cell type | effect, β |
| flagged | the method says "this box changed" | a call (q < 0.1 or 90% interval excluding 0) |
| spillover | signal from one cell type (or program) counted in another | leakage |
| 90% range | where the true change lies, 9 times in 10 | 90% interval, coverage |
| answer key | the true change, measured separately | ground truth |
| can't tell | the data can't separate this cell type | not identifiable |

---

## Part 1: TB results, condensed (3.5 min, deck slides 1–6)

| Deck slide | Content | Time | Keep to one message |
|---|---|---|---|
| 1 | Title | 10 s | |
| 2 | PTB vs healthy, PBMC bulk RNA-seq | 30 s | "Host response in blood: PTB (n = ) vs healthy (n = )." |
| 3 | Decoding bulk with a single-cell reference | 30 s | "We estimate cell-type proportions and per-cell-type signals." |
| 4 | Predicted vs measured lymphocyte / myeloid % | 30 s | "Proportions agree with measurement (r = 0.91)." |
| 5 | Cell fractions × pathway scores | 45 s | "Pathway scores track myeloid cells." |
| 6 | PTB: more myeloid cells, coordinated innate pathways | 1 min | Conclusion box below |

> **Slide 6 conclusion box:** In PTB, myeloid cells expand, and their innate pathways (TLR, NLR, IFN-γ) rise
> *together*, while T-cell receptor signalling moves the opposite way.

- Say the three points in one breath each:
  - more neutrophils and CD14/CD16 monocytes;
  - TLR, NLR and IFN-γ correlate more tightly in PTB (e.g. neutrophil NLR–IFN-γ 0.96 vs 0.85);
  - TCR runs opposite (−0.4 to −0.6).
- Slide-4 design box (one line): bulk RNA-seq vs measured % in the same samples (fill in: blood count / flow).
- Wording: "associated", "coordinated", not "drives".

## Bridge (15 s, deck slide 7)
> "More myeloid cells and stronger inflammatory pathways. But is that *more* cells, or cells that *switched
> on*? And is it in the neutrophils, or borrowed from the monocytes? A correlation can't tell us."

---

## A. The problem (3 min, 3 slides)

### A1. One bulk signal, three explanations (1 min)
- **Slide title:** "The same rise in blood can mean three different things"
- **Visual:** `2_1_three_causes.png`.
- **Say:** "In part 1, the neutrophil inflammatory score was higher in PTB. Three explanations give exactly
  the same bulk signal:
  1. **more neutrophils**, each one unchanged;
  2. **monocytes switched on**, and part of their signal spills into the neutrophil estimate;
  3. **neutrophils switched on**: the neutrophils themselves changed.

  For biology and for follow-up experiments, only the third is a change *in* neutrophils."
- **Point at:** the dashed line (same bulk signal), then the highlighted panel ("what we want to know").

### A2. The question, and why no standard tool answers it (1 min)
- **Slide title:** "Attribution: where does a bulk program change come from?"
- **Top, large (the question, two layers):**
  1. *When a gene program changes in bulk, which cell types does the change come from, and how much is more
     cells versus cells that changed?* (**contribution**; e.g. "IFN-γ up: 75% from monocytes, 25% from
     neutrophils")
  2. *Inside each cell type, did the program itself change, by how much, and can the data tell?* (**per-cell
     change**; e.g. "monocyte IFN-γ +0.30 [0.18, 0.42]; neutrophils: can't tell")
- **Say (20 s):**
  - "We start where we always start: a program that changed in bulk. Then we ask where it comes from: which
    cells, and whether it's more cells or cells that changed. That is the three-explanations picture."
  - "The second layer is what you'd validate at the bench: did the program change *inside* the cell type?"
  - "The two differ: an abundant cell type can contribute most of the bulk signal with only a small change per
    cell."
- **One-line link (small, bottom of slide):** contribution = abundance × how much that cell type expresses the
  program × its per-cell change.
- **Gene programs (20 s):**
  - "A gene program is a set of genes that switch on together for one job, like the interferon response. Our
    part-1 pathways are programs."
  - "We ask at the program level on purpose. One gene in one cell type is too noisy to pin down from bulk; a
    program pools hundreds of genes, so there is enough signal."
- **The gap (bottom half of the slide, 40 s):**

  | What we use today | It answers | It doesn't answer |
  |---|---|---|
  | Deconvolution (CIBERSORT, InstaPrism) | How many cells of each type? | Did those cells change? |
  | Bulk differential expression | What changed in the sample? | In which cells? |
  | Per-cell-type expression (CIBERSORTx high-resolution, TCA, BayesPrism) | Each gene in each cell type | Did a program change, and in which cell type? |
  | **Attribution (this talk)** | **Where a program change comes from (contribution), and which cell types changed (per-cell change), how sure** | n/a |

  - **Say:** "These tools are rarely checked against a known answer on the question that matters to us: did
    the change land in the right cell type?"

### A3. It happens in our own workflow (1 min)
- **Slide title:** "Our workflow paints one cell's change onto every cell"
- **Visual:** `2_5_heatmap_lab_workflow.png`, first two panels only (answer key | our workflow). PALM's panel
  is revealed in D.
- **Say:**
  - "We ran our part-1 workflow on an experiment with a known answer: a drug that switches on p53 in only 3
    of 6 cell lines."
  - "Look at the p53 column: our workflow gives the p53-mutant lines the same rise as the real responders. It
    can't tell responding lines from the others."
  - Don't lead with the count (19 of 23 unchanged boxes flagged): part of it is small biases made
    'significant' by a nearly noise-free test. The robust point is the mutant lines looking like responders
    (see D).
- **Why, in one picture (spoken, 20 s):**
  - "The workflow shares out each sample's reads by cell proportions, then scores each share. One cell type's
    extra reads land in every cell type."
  - "And programs share genes, so one program's change leaks into others: the drug acts on p53, yet 6 of 7
    programs get flagged. Spillover runs across cell types *and* across programs."
  - "It can also make programs look *coordinated* across cell types, as on slide 6, because they share one
    signal."
- **Why it matters (one line):** "A wrong cell type means a wrong story, and we'd sort neutrophils to validate
  a change that happened in monocytes."
- **Design box:** McFarland 2020 idasanutlin; 6 cell lines, 20 mixtures per arm; answer key from held-out
  cells; workflow = InstaPrism → GSVA → correlation → BH q < 0.1.

---

## B. The approach: how PALM answers it (1.5 min, 1 slide)

- **Slide title:** "PALM: undo the spillover, and say how sure"
- **Visual:** the slide-3 pipeline again, with three new boxes after it (draw).
  1. **Proportions with uncertainty.** How sure we are about the cell counts feeds into everything after.
  2. **Undo the spillover, across cell types and programs.**
     - The spillover map is known from the reference and the proportions.
     - It is corrected using samples that differ in composition: "a change that rises and falls with the
       monocyte proportion belongs to monocytes."
     - The correction is checked on reference donors, where the answer is known.
  3. **A change per box with a 90% range**, and "can't tell" when the data can't separate cell types.
- **Say:**
  - "PALM keeps our familiar first steps. It adds the two things that were missing: it removes the spillover,
    and it says how sure it is, including 'can't tell'."
  - "It works directly at the program level. You bring the programs: our TLR, NLR, IFN-γ sets, hallmark sets,
    or programs from single-cell data. A program is just a list of genes with weights."
- **Hand-off line:** "Any method can claim this. The real question is how to *test* it. That is the core of
  this talk."

---

## C. The evaluation design (4 min, 3 slides)

### C1. Building an answer key: three rules (1.5 min)
- **Slide title:** "How to test attribution: hide the answer inside a real sample"
- **Visual:** `2_3_answer_key.png`.
- **Rule 1, hide the answer in a real sample (30 s):**
  - "Take a sample's cells and split them in two. Half A is pooled into bulk: that's what the method sees.
    Half B stays separated by cell type: that's the answer key, never shown to the method."
  - "Like splitting a blood tube: one for bulk sequencing, one sorted."
  - "The responses are real (real drugs, real interferon), not simulated."
- **Rule 2, decide the controls from biology, before running anything (30 s):**
  - "Idasanutlin switches on p53, but only in cell lines with an intact p53 gene. So those lines are positive
    controls, and the p53-mutant lines are negative controls."
  - "The method is never told which lines respond. Genetics is the external check, like a knockout control."
  - "Inside each line, the programs the drug doesn't target are negative controls too."
- **Rule 3, keep the test honest (30 s):**
  - "The reference and the correction are learned from *other* donors."
  - "Success targets are written down before the run."
  - "The method is frozen, then checked once on held-back data."
- **Design box:** answer key scored per cell type × program with PROGENy footprint programs (genes that
  *respond* when a pathway fires, rather than its parts list).

### C2. A ladder of evidence (1 min)
- **Slide title:** "No single test is enough: each rung covers the last one's weakness"
- **Visual:** ladder graphic (draw), four rungs, bottom to top:

  | Rung | Data | Answer key from | Proves | Leaves open |
  |---|---|---|---|---|
  | 1. Simulation | Planted changes | We put them in | The maths works | Real responses are messier |
  | 2. Real responses, built bulk | 2 drugs on 6 cell lines; interferon on blood cells; chemo ± anti-PD-L1 in patients' blood; next: 93 drugs on 50 lines (Tahoe) | Split cells + genetics | Real biology, known answer | Bulk is assembled, not sequenced |
  | 3. Real bulk, sorted cells | Whole blood + sorted neutrophils, monocytes, B, CD4, CD8 from the same 20 donors (GSE60424) | Sorted cells, same people | Works on real sequenced blood | Sorting shifts stress genes; small groups |
  | 4. Our cohort | PTB vs healthy | Known TB biology (Berry 2010) | Useful for our question | No full answer key |

- **Say (one sentence per rung):** the "Proves" and "Leaves open" columns.
- **One line on rung 3 (lesson for the field):** "Sorting cells changes their stress genes, so a sorted 'answer
  key' differs from the cells in the tube. Sorted-cell validations need care."

### C3. Can a study answer it at all? The design limit (1.5 min)
- **Slide title:** "Know before you sequence: which cell types can this study resolve?"
- **Visual:** `2_4_design_power.png`.
- **Say:**
  - "Before a clinical trial we do a power calculation. The same question exists here, per cell type and
    program, and it hadn't been asked."
  - "From the design alone (cell proportions, how much they vary, number of samples, noise) we compute the
    smallest change that *any* method could detect. No answer key needed, so it runs before the study."
- **Left panel, interferon, 7 paired donors:**
  - "Monocytes, 64% of the reads: a change of 0.06 is detectable."
  - "B cells, 5%: only 0.8 or larger, far above any real change. You would need more than 200 donors per arm.
    That's why PALM says 'can't tell' for B cells, and it's right."
- **Right panel:** "Samples that differ in composition help. Halve the spread and you need about 2.5× the
  donors."
- **It works:** "Across 420 benchmark boxes, PALM never flagged a real change below this limit, and found 58%
  of those above it."
- **Programs matter too (one line):** "The limit differs 2–3× between programs in the same study, so choose
  programs before the study."
- **Design box:** limit = smallest change detectable with 90% confidence and 80% power by any unbiased method
  given the design; checked on 420 boxes, 4 datasets, 3 splits.

---

## D. Proof the design works: negative and positive controls (1.5 min, 1 slide)

- **Slide title:** "A fair test: the standard approach is confidently wrong, and more data makes it worse"
- **Visual:** `2_5_fair_rules.png` (three panels). Backup: `2_5_controls.png` (all tools).
- **Say, panel by panel (the evaluation design in action):**
  - **Left, "we gave the standard approach a fair chance":**
    - "With the usual t-test, InstaPrism flags 81–91% of unchanged boxes."
    - "That's partly unfair to it: our test mixtures have very little noise, so tiny biases become
      'significant'. So we also required a change of at least 0.1."
    - "Then it drops to 5–7%, close to PALM's 0–5%. Both find the real changes in the drug experiments."
  - **Middle, "but its confidence is wrong":**
    - "Its 90% ranges hold the true value only 2–55% of the time. PALM's hold it 92–100%."
    - "And the error that matters survives the cut-off: in the p53 drug, the p53-mutant lines still look like
      responders. PALM leaves them alone."
  - **Right, "more samples make it worse":**
    - "Give the t-test more samples and it flags *more* unchanged boxes: 24% with 3 samples, 96% with 20."
    - "That's the signature of bias, not noise: more data makes a biased method more confidently wrong."
- **Interferon, both layers (20 s):**
  - "Per-cell change: PALM finds little for B and NK cells, exactly as the design limit predicted."
  - "Contribution: it still gets right *where* the interferon signal comes from. Its share error is 0.07–0.10
    vs 0.10–0.15 for InstaPrism, and it correctly says almost none of it is more cells: 4–7%, against InstaPrism's
    21–26%."
  - "So when cells can't be told apart one by one, you still learn where the bulk change comes from."
- **Design box:**
  - Data: 3 known-answer experiments (p53 drug and MEK drug on 6 cell lines, interferon on blood cells),
    answer key from split cells, same reference for all tools, 3 splits.
  - Calling rules: q < 0.1; q < 0.1 + change ≥ 0.1; and a threshold-free ranking (AUROC, similar for both:
    0.79–0.99).
  - Targets set in advance.

---

## E. Next, and conclusion (1.25 min, 1 slide)

- **Next week, PTB (30 s; small panel on the slide):**
  - "We run PALM and our InstaPrism workflow side by side on the PTB cohort."
  - "We compute the design limit first, so we know which cell types our cohort can resolve before looking."
  - "We check how much the TLR, NLR and IFN-γ gene sets share genes: shared genes could build part of slide 6's
    'coordination' in."
  - "We add a positive control (Berry 2010 interferon signature in neutrophils and monocytes) and negative
    controls (random gene sets, shuffled labels)."
- **Conclusion (45 s), the three take-home messages:**
  1. **Problem:** where a bulk program change comes from, and which cell types changed. Standard tools spread one cell's change onto all, with confidence that grows with sample size, not accuracy.
  2. **Approach:** PALM undoes the spillover at the program level and says how sure it is; where cell types can't be separated, it reports the program-level change instead of guessing.
  3. **Evaluation design:** answer keys from real experiments, controls set by biology, and a design limit that
     tells before sequencing what a study can resolve. Any method can be tested this way.
- **Last line:** "We can now ask not only *what* changed in the blood of TB patients, but *which cells*
  changed, how sure we are, and whether our study is big enough to tell."

---

## Figures

Generated figures are in `docs/figures/talk/`: 300 dpi PNG plus SVG (editable text in PowerPoint), sized for a
16:9 slide, no titles (put the message in the slide title). Colours keep one meaning: teal = PALM, red =
standard tools, dark = truth. Rebuild with `python scripts/experiments/talk_figures.py [SPLIT] [figure,...]`.

| Slide | Visual | File |
|---|---|---|
| A1 | Three explanations of the same bulk signal | `2_1_three_causes.png` |
| A2 | The question; table of tools vs questions | text + table |
| A3 | Answer key vs our workflow (first two panels) | `2_5_heatmap_lab_workflow.png` |
| B | Pipeline + three new boxes | draw |
| 2.1 (new) | Colour-coded model equation + smoothie + two answers | `2_2_principle_formula.png` |
| 2.2 (new) | Contribution split: more cells / cells changed, stacked bar | `2_2b_principle_contributions.png` |
| 2.3 (optional) | Graphical model, simple | `2_2c_graphical_model_simple.png` |
| Backup | Graphical model, detailed (model + calibrated estimator) | `2_2c_graphical_model.png` |
| Backup | TB patients only, simple / detailed | `2_2d_graphical_model_tb_only_simple.png` / `2_2d_graphical_model_tb_only.png` |
| Backup | PALM as a Bayesian network (plate notation) | `2_2e_graphical_model_bayes.png` |
| C1 | Three rules for an answer key | `2_3_answer_key.png` |
| C2 | Ladder of evidence | draw (four rungs) |
| C3 | Design limit vs donors; composition spread | `2_4_design_power.png` |
| D | Fair test: three calling rules, 90% ranges, false flags vs sample size | `2_5_fair_rules.png` (backup: `2_5_controls.png`) |
| E | Small PTB panel + three lines | text |

**Reading notes:**
- `2_5_heatmap_lab_workflow`: the middle panel is coloured by correlation (ρ, its own scale); the others by
  change in activity.
- `2_4_design_power`: log scales; dashed line = a typical real change (0.1); dotted line = the study's size.
  A cell type is resolvable where its line is below the dashed line.

---

## Backup slides (only if asked)

**Results in more depth**
- **PALM vs InstaPrism, cell by cell:** `case_studies_palm_vs_instaprism.png`.
  - Idasanutlin: InstaPrism gives every line 0.30–0.43; PALM ranks LNCaP 0.66 > RCC10RGB 0.55 > SNU1079 0.41
    and leaves COLO680N and TEN (true 0.04–0.05) unflagged.
  - Trametinib: InstaPrism flags LNCaP (true −0.03) at −0.18; PALM leaves it unflagged.
- **Scorecard:** `case_scorecard_palm_vs_instaprism.png`.
  - Proportions are similar.
  - Unchanged boxes flagged: PALM 0–5% vs InstaPrism 81–91%.
  - 90% range holds the truth: PALM 92–100% vs InstaPrism 2–55%.
- **All tools, every metric:** `bench_calling.png`, `bench_deconv.png`.
- **Real blood (rung 3, GSE60424):**

  | Comparison (donors per arm) | PALM | InstaPrism + t-test |
  |---|---|---|
  | MS before vs after interferon-β (3 v 3) | 0% unchanged flagged; 29% of real changes found | 0% flagged; 82% found |
  | Sepsis vs healthy (3 v 4) | 0% flagged | 24% flagged |
  | Type 1 diabetes, ALS vs healthy | no flags | no flags |

  With 3 donors per group, the design limit says almost nothing is resolvable, and PALM reports exactly that.
- **Held-out check (PALM v1, run once on the frozen method):** PALM flagged 0–1% of unchanged boxes,
  CIBERSORT + t-test 69–78%.

**Method questions**
- **"What exactly is the model?"** The model fits the cell mix; the per-cell change comes from a calibrated
  estimator on top of it (split reads, calibrate on atlas donors with a known answer, unmix across patients).
  Detailed graphical model: `2_2c_graphical_model.png`; full pipeline graph: `docs/model_dag.md`.
- **"Is it a Bayesian method?"** For the cell mix, yes (posterior draws). The per-cell change is not the
  model's posterior: it is a calibrated, spillover-aware estimate, because it was more accurate and its ranges
  more honest on the benchmarks.
- **"Does it need a healthy group?"** No, but then there is no TB-vs-healthy effect: `2_2d_graphical_model_tb_only.png`.
- **"Is it fair to our workflow?"**
  - The emulation is InstaPrism per-type expression → log CPM → GSVA → Spearman with treatment → BH q < 0.1.
  - Gene sets are the up-weighted genes of each PROGENy program.
  - Code: `scripts/experiments/lab_workflow.py`.
- **"How is the design limit computed?"** Each cell type's share of the bulk change is a known mix of all
  cell types' true changes (the spillover map). The limit is the best precision any unbiased method can reach
  when unmixing that map across samples (Gauss–Markov), × 2.49 for 90% confidence and 80% power. Paired donors
  and uncertainty in the proportions are included. See `docs/method_spec.md` §4.6.
- **"Why footprint programs?"** A pathway's parts list (receptor, kinases) barely changes at the RNA level when
  the pathway fires; its downstream responsive genes do. PROGENy footprints (~500 weighted genes each) are
  learned from hundreds of perturbation experiments.
- **"Programs overlap?"** Yes. In PROGENy, NF-κB and TNF-α share 291 of ~500 genes, so they can't be fully
  separated. The same check applies to our gene sets.
- **Interferon in detail:**
  - JAK-STAT rises in every cell type (0.26–0.36), so copying the bulk signal to every cell type happens to be
    right.
  - Monocytes have their own changes (TGF-β −0.18, hypoxia −0.17, MAPK −0.16) that InstaPrism copies into B,
    NK and T cells.
- **Causality:** comes from the study design (randomised or paired); PALM provides the measurement.
- **Run time:** seconds (fast screen), about 5 min (default).

**Outlook: spatial transcriptomics**
- A microdissected region (GeoMx ROI) is a small bulk sample. In a TB granuloma: more macrophages, or
  macrophages that switched on?
- Regions differ a lot in composition, which the design limit says helps.
- Proposed test: pool cells inside regions of single-cell-resolution spatial data (Xenium, CosMx) to build
  regions with a known answer.

---

## Appendix: full experimental designs (for the speaker; not slides)

**Benchmark (C1–C2 rung 2, D):**

| Item | Design |
|---|---|
| Datasets | McFarland 2020 idasanutlin and trametinib (6 cancer cell lines, 20 paired mixtures per split); Kang 2018 interferon-β (PBMC, 4 cell types, 14 samples from 4 test donors); Zhang 2021 TNBC blood before vs after paclitaxel ± anti-PD-L1 (10 test patients, paired) |
| Answer key | Each sample's cells split in two (binomial thinning): one half pooled into bulk, the other gives the true change per cell type × program (PROGENy footprints, 7 programs, ~500 weighted genes each) |
| No leakage | Reference and calibration only from other donors (Kang, Zhang) or a separate control-cell pool (McFarland) |
| Comparators | CIBERSORT, InstaPrism, NNLS, bulk score, each + t-test (BH q < 0.1); our workflow (InstaPrism → GSVA → correlation) |
| Metrics | Negative control: unchanged boxes (\|true\| < 0.05) flagged. Positive control: boxes in the drug's program with \|true\| ≥ 0.1 found with the right sign. Also r with truth and 90%-range coverage |
| Success targets (set in advance) | Unchanged flagged ≤ 10%; coverage ≥ 85%; r ≥ 0.9; responders found ≥ 90% |
| Replication | 3 development splits; held-out splits run once on the frozen method |

**PTB application (E):**

| Item | Design |
|---|---|
| Data | PTB vs healthy PBMC bulk RNA-seq (n = / n =); same reference as part 1 |
| Programs | The six part-1 gene sets; their overlap table; versions without shared genes (sensitivity); PROGENy footprints as a cross-check. BCR and TCR read only in the cell types that express them |
| Methods | Part-1 workflow (InstaPrism → GSVA → test) vs PALM; same reference, genes and samples |
| Contrast | PTB vs healthy per box, adjusted for age and sex |
| Before results | Design limit per box from our cohort's proportions and size |
| Positive control | Interferon-inducible signature up in neutrophils and monocytes of active TB (Berry et al. 2010) |
| Negative controls | (a) Size-matched random gene sets: not flagged; (b) PTB/healthy labels shuffled: no calls |
| Robustness | Leave-one-reference-donor-out; reference without neutrophils |
| Readout | Per box: change with 90% range (PALM) vs score difference and q-value (InstaPrism); agreement table; "more cells" vs "cells changed" |
| Can claim / can't | Which cell types and programs differ, adjusted for age and sex / not that TB caused it |

**Tahoe-100M (rung 2, next):** plate 3 (5 µM, 93 drugs, ~50 lines pooled per well).
- Positive controls by genetics: BRAF inhibitor → MAPK down in BRAF-V600 lines only; KRAS-G12C inhibitor →
  only G12C lines; DNA-damaging drugs → p53 up in TP53-intact lines only.
- Negative control: DMSO vs DMSO, from the same plate.
