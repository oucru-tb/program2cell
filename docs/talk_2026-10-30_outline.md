# Talk outline: consortium meeting (lab + collaborators), revised 2026-11-02

**Title:** *Which cells changed? Attributing bulk gene-program changes to cell types, and how to test it*

*Audience: clinicians, wet-lab scientists and analysts. Method-focused, but every idea gets an everyday picture
first and the technical term second. Statistical terms stay in the backup. Benchmark numbers are development
splits unless marked "held-out".*

| Part | Content | Slides | Time |
|---|---|---|---|
| 1 | TB cohort: proportions, pathway correlations | deck slides 1–6 | 6.5 min |
| Bridge | Correlation ≠ causality: which cells changed? | deck slide 7 | 0.5 min |
| 2.1 | The question: attribution (three explanations; which cell type changed which gene program) | 2 | 2 min |
| 2.2 | Why it is hard: spillover, shown on our own workflow | 1 | 1 min |
| 2.3 | **How to test attribution: building an answer key** | 2 | 2.5 min |
| 2.4 | How PALM answers it | 1 | 1 min |
| 2.5 | Results: negative and positive controls, PALM vs InstaPrism | 2 | 2.5 min |
| 2.6 | **Can a study answer it at all? The design limit** | 1 | 1.5 min |
| 2.7 | Real blood check (whole blood vs sorted cells) | 1 | 1 min |
| 2.8 | Next: our PTB cohort, and a 50-cell-line drug screen | 1 | 1 min |
| 2.9 | Outlook: spatial transcriptomics | 1 | 0.5 min |
| 2.10 | Conclusion | 1 | 1 min |
| | **Total** | **20** | **≈ 21.5 min** (incl. ~1 min of program lines) |

If time is short, drop 2.7 and 2.9 to backup (−1.5 min); the program lines in 2.3 (footprints) and 2.6 can also go to backup (−0.5 min).

**The story in one line:** bulk RNA-seq tells us *that* something changed; we ask *which cells* changed it.
We defined that question, built a fair way to test any method on it, built PALM to answer it, and worked out
when a study can answer it at all.

**Three contributions to name explicitly (they return in the conclusion):**
1. **The question:** attribution, i.e. which cell type changed a gene program. This is different from "how many
   cells" (deconvolution) and "what changed in the bulk" (differential expression).

**The gene-program thread (easy to under-sell; it runs through every section):**

| Where | Program point |
|---|---|
| 2.1 | Programs, not single genes, make the question answerable from bulk |
| 2.2 | Spillover also runs *between programs* that share genes |
| 2.3 | The answer key is per cell type × program; footprint programs read the *response*, not the pathway's parts list |
| 2.4 | PALM takes any program as a gene-weight list (bring your own) |
| 2.5 | The non-target programs are the negative control inside each cell |
| 2.6 | The design limit differs by program, by 2–3× |
| 2.8 | For PTB, check how much the six gene sets overlap before reading "coordination" |
2. **The test:** an answer key built from real experiments, with negative and positive controls decided by
   biology before any method is run.
3. **The tool and the limit:** PALM, and a design calculator that says in advance which cell types a study
   can resolve.

**Tone for part 1 → part 2:** not "part 1 was wrong". The associations stand; which cell type they belong to
is the open question, and every lab using this workflow faces it.

**Every evaluation slide carries a small "Experimental design" box** (bottom-left, 4–5 lines): data, how the
truth is known, comparison, metric, success target.

**Words used on the slides (keep them the same throughout):**

| Word on slide | Meaning | Technical term (backup only) |
|---|---|---|
| gene program (pathway) | a set of genes that switch on together for one job, e.g. the interferon response | program, gene-weight vector |
| a "box" | one cell type × one gene program | (c, p) pair |
| change | how much a program moved in that cell type | effect, β |
| flagged | the method says "this box changed" | a call (q < 0.1 or 90% interval excluding 0) |
| spillover | signal from one cell type counted in another | leakage |
| 90% range | where the true change lies, 9 times in 10 | 90% interval, coverage |
| answer key | the true change, measured separately | ground truth |
| can't tell | the data can't separate this cell type | not identifiable |

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

**Careful wording:** "associated" and "coordinated", not "drives" or "activates". Don't reveal on slide 6 that
the standard workflow can make pathways *look* coordinated across cell types; part 2 shows it.

---

## Bridge (30 s, deck slide 7: correlation ≠ causality)

> "So in PTB we see more myeloid cells and stronger inflammatory pathways. Is that because there are *more*
> of these cells, or because each cell *switched on* its pathways? And is the signal really in the neutrophils,
> or did it come from the monocytes? A correlation can't tell us. That is what part 2 is about."

---

## Part 2: Which cells changed?

### 2.1 The question: attribution (2 min, 2 slides)
Goal: by the end, everyone in the room can say the question in one sentence, and sees that no standard tool
answers it.

**Slide 2.1a: one bulk signal, three explanations (1 min).**
- **Slide title:** "The same rise in blood can mean three different things"
- **Visual:** `2_1_three_causes.png`.
- **Then our data (45 s):** "In part 1, the neutrophil inflammatory score was higher in PTB. Three explanations give exactly the same bulk signal:
  1. **more neutrophils**, each one unchanged;
  2. **monocytes switched on**, and part of their signal spills into the neutrophil estimate;
  3. **neutrophils switched on**: the neutrophils themselves changed.

  For biology and for follow-up experiments, only the third is a change *in* neutrophils. Today's tools can't  tell the three apart."
- **Point at the figure:** the dashed line, where all three reach the same bulk signal; then the highlighted
  third panel, "what we want to know".

**Slide 2.1b: the question, and the gap (1 min).**
- **Slide title:** "Attribution: which cell type changed which gene program?"
- **Top of slide, the question in one line (large font):**
  > *For each cell type and each gene program: did it change, by how much, and can the data tell?*
- **Define "gene program" in plain words (10 s):** "A gene program is a set of genes that switch on together
  for one job, like the interferon response or TLR signalling. Our part-1 pathways are gene programs."
- **Why programs and not single genes (20 s; one line on the slide: "programs pool many genes → enough
  signal")**:
  - "One gene in one cell type is too noisy to pin down from bulk. A program pools tens to hundreds of genes
    that move together (the PROGENy programs we test have about 500 each), so there is enough signal to tell
    cell types apart."
  - "It is also how we think: we say 'interferon response in monocytes', not 'IFI44L in monocytes'."
- **Why this is a new question (30 s; the table, bottom half of the slide):**

  | What we use today | It answers | It doesn't answer |
  |---|---|---|
  | Deconvolution (CIBERSORT, InstaPrism) | How many cells of each type? | Did those cells change? |
  | Bulk differential expression | What changed in the sample? | In which cells? |
  | Per-cell-type expression (CIBERSORTx high-resolution, TCA, BayesPrism) | Each gene's level in each cell type | Did a *program* change, and in which cell type? Programs are scored afterwards, and the borrowed signal adds up gene by gene |
  | **Attribution (this talk)** | **Which cell type changed a gene program, by how much, how sure** | n/a |

  - **Say:** "Several tools give per-cell-type expression, but they are rarely checked against a known answer
    on the question that matters to us: did the program change land in the right cell type? Answering that
    needs a method, and a fair way to test it. That is the rest of this talk."
- **Transition to 2.2 (one line):** "First, let me show that this is not a theoretical worry: it happens with
  our own workflow."

### 2.2 Why it is hard: spillover, on our own workflow (1 min)
- **Visual:** `2_5_heatmap_lab_workflow.png` (answer key | our workflow; reveal PALM's panel in 2.5).
- **Say:** "We ran our part-1 workflow on an experiment where we know the answer: a drug that switches on p53
  in only 3 of 6 cell lines. Our workflow flags 6 of 7 pathways in all 6 lines: 19 of the 23 boxes that did
  not change. It can't tell the responding lines from the others."
- **Why:** the workflow first splits each sample's reads by cell-type proportions, then scores each part. If
  one cell type changes, its extra reads are shared out to every cell type, so every cell type looks changed.
  It also makes pathways *look* coordinated across cell types, as on slide 6, because they share one signal.
- **Spillover has a second direction, between programs (say it, 15 s):**
  - "The drug acts on one program, p53, yet our workflow flags 6 of 7 programs. Programs share genes, and a
    strong change in one program leaks into the others in the same cell."
  - "In the PROGENy set we test, the NF-κB and TNF-α programs share 291 of their ~500 genes; no method can
    fully separate those two."
  - "So the question is two-dimensional: which *cell type*, and which *program*."
- **Why it matters:** "A wrong cell type means a wrong biological story and wrong follow-up experiments: we
  would sort neutrophils to validate a change that happened in monocytes."
- **Design box:** McFarland idasanutlin, 6 cell lines, 20 mixtures per arm; truth from held-out cells;
  workflow = InstaPrism → GSVA → correlation with treatment → BH q < 0.1.

### 2.3 How to test attribution: building an answer key (2.5 min, 2 slides)
This is the part to slow down on: it is new, and it is what lets anyone, with any method, check a claim of
the form "program X changed in cell type Y".

**The answer key is per cell type × program, using footprint programs (one spoken line, 15 s):**
- "We score the answer key with PROGENy *footprint* programs: genes that respond downstream when a pathway is
  active, learned from hundreds of perturbation experiments."
- "A pathway's parts list (the receptor, the kinases) barely changes at the RNA level when the pathway fires.
  Its footprint does. So footprints measure the response, which is what we want to attribute."
- Backup if asked: "Any program works in PALM; footprints give the cleanest answer key."

**Slide 2.3a: three rules for a fair test.** Visual: draw a three-panel schematic (left to right). Proposed
file: `2_3_answer_key.png`, to be drawn.

1. **Hide the answer inside a real sample (split each sample's cells in two).**
   - Picture: one tube of cells, split in half. Half A is pooled into a "bulk" sample, which is what the method
     sees. Half B stays sorted by cell type: that is the answer key, which the method never sees.
   - Responses are real (real drugs, real interferon), not simulated.
   - Wet-lab analogy: "like splitting a blood sample, sending one tube for bulk sequencing and sorting the
     other."
2. **Decide the controls from biology, before running anything.**
   - Negative control: boxes that did not change. A good method must not flag them.
   - Positive control: changes we expect from known biology.
     - Idasanutlin turns on p53 only in lines with an intact p53 gene.
     - Trametinib turns off MAPK in lines that depend on it.
     - Interferon turns on JAK-STAT in every immune cell.
   - The method is never told which lines respond: genetics is the external check, like a knockout control.
3. **Keep the test honest.**
   - The reference profiles and PALM's correction are learned from *other* donors or a separate control pool,
     never from the test samples.
   - Success targets are written down before the run.
   - The method is frozen, then checked once on data held back (held-out splits).

**Slide 2.3b: a ladder of evidence.** Each rung closes a gap the one below leaves open. Visual: a table or
ladder graphic.

| Rung | Data | How the truth is known | Proves | Leaves open |
|---|---|---|---|---|
| 1. Simulation | Planted changes | We put them in | The maths works | Real responses are messier |
| 2. Real responses, built bulk | McFarland (6 cell lines, 2 drugs), Kang (interferon, PBMC), Zhang (TNBC blood, chemo ± anti-PD-L1); next: Tahoe (50 lines, 93 drugs) | Split cells (rule 1) + genetics (rule 2) | Real biology, known answer | Bulk is assembled, not sequenced |
| 3. Real bulk, sorted cells | GSE60424: whole blood + sorted neutrophils, monocytes, B, CD4, CD8 from the same 20 donors | Sorted cells of the same people | Works on real sequenced blood | Sorting itself shifts some genes; small groups |
| 4. Our cohort | PTB vs healthy PBMC | Known TB biology (Berry 2010) as positive control | Useful for our question | No full answer key |

**Say:** "No single test is enough. Each rung has a weakness, and the next rung covers it."

**Full experimental design (rung 2, used in 2.5):**

| Item | Design |
|---|---|
| Datasets | McFarland 2020 idasanutlin and trametinib (6 cancer cell lines, 20 paired mixtures per split); Kang 2018 interferon-β (PBMC, 4 cell types, 14 samples from 4 test donors); Zhang 2021 TNBC blood before vs after paclitaxel ± anti-PD-L1 (10 test patients, paired) |
| How the truth is known | Each sample's cells split in two (binomial thinning): one half pooled into bulk, the other gives the true change per cell type and program (PROGENy footprints, 7 programs: JAK-STAT, NF-κB, TNF-α, TGF-β, hypoxia, p53, MAPK; ~500 weighted genes each) |
| No leakage between test and training | Reference and calibration only from other donors (Kang, Zhang) or a separate pool of control cells (McFarland) |
| Comparators | CIBERSORT, InstaPrism, NNLS, bulk score, each + t-test (BH q < 0.1); our part-1 workflow (InstaPrism → GSVA → correlation) |
| Metrics | Negative control: unchanged boxes (\|true\| < 0.05) flagged. Positive control: boxes in the drug's pathway with \|true\| ≥ 0.1 found with the right sign. Also r with truth and 90%-range coverage |
| Success targets (set in advance) | Unchanged flagged ≤ 10%; coverage ≥ 85%; r ≥ 0.9; responders found ≥ 90% |
| Replication | 3 development splits; held-out splits run once on the frozen method (backup) |

### 2.4 How PALM answers it (1 min)
- **Visual:** the slide-3 pipeline again, with three new boxes after it:
  1. **Proportions with uncertainty**: how sure we are about the cell counts feeds into everything after.
  2. **Undo the spillover**:
     - learned on reference donors where the answer is known;
     - helped by samples that differ in composition, because a change that follows the monocyte proportion
       belongs to monocytes.
  3. **A change per box with a 90% range**, and "can't tell" when the data can't separate the cell types.
- **Say:** "PALM keeps the familiar first steps. It adds what was missing: it removes the spillover, and it
  says how sure it is, including when it can't tell."
- **Programs are an input you choose (say it, 15 s):**
  - "PALM estimates each program's change per cell type directly, using all its genes and their weights,
    instead of estimating 20,000 genes per cell type and scoring programs afterwards."
  - "You bring the programs: our TLR, NLR, IFN-γ, TNF, BCR and TCR sets, MSigDB hallmarks, PROGENy footprints,
    or programs found in single-cell data. A program is just a list of genes with weights."
  - "It corrects spillover between cell types *and* between programs at the same time."

### 2.5 Results: controls, and PALM vs InstaPrism (2.5 min, 2 slides)

**Slide 2.5a: negative and positive controls, all tools.** Visual: `2_5_controls.png`. Backup:
`bench_calling.png`.
- **How to read it (10 s):** "Left of the line: unchanged boxes that were flagged; shorter is better. Right:
  real changes found; longer is better."
- **Say:**
  - "In the two drug experiments PALM finds 92–100% of the real changes, like every tool. But it flags under
    5% of the unchanged boxes, where the others flag 72–100%."
  - "Chemotherapy is a negative control for a whole dataset: nothing responds strongly, and nobody
    over-flags. The problem appears when one cell type responds strongly."
  - "Interferon is PALM's limit: no false flags, but it finds little. Slide 2.6 explains why."

**Slide 2.5b: PALM vs InstaPrism, cell by cell.** Visual: `case_studies_palm_vs_instaprism.png` (top two
panels); scorecard `case_scorecard_palm_vs_instaprism.png` as backup or a quick flash.
- **Idasanutlin → p53:**
  - InstaPrism gives *every* line a similar rise (0.30–0.43), so p53-mutant COLO680N and TEN (true 0.04–0.05)
    look like responders.
  - PALM recovers the order LNCaP 0.66 > RCC10RGB 0.55 > SNU1079 0.41 and leaves COLO680N and TEN unflagged.
- **Trametinib → MAPK:** five lines depend on MEK (true −0.14 to −0.27); LNCaP doesn't (−0.03).
  - InstaPrism flags LNCaP at −0.18, like a responder.
  - PALM leaves it unflagged (−0.06).
- **Key sentence:** "When a drug acts on some cells only, the standard approach paints every cell with the
  same brush. PALM tells responders from non-responders."
- **The program dimension (one line):** "Inside each responding line, the other six programs are a negative
  control too: the drug targets one program, and PALM leaves the rest unflagged, where our workflow flagged
  6 of 7."
- **Scorecard in one line each:**
  - proportions: similar, neither wins everywhere;
  - unchanged flagged: PALM 0–5% vs InstaPrism 81–91%;
  - real changes found: both 92–100% in the drug experiments, and on interferon InstaPrism 100% vs PALM 0%;
  - 90% range holds the truth: PALM 92–100% vs InstaPrism 2–55%.

### 2.6 Can a study answer it at all? The design limit (1.5 min)
- **Visual:** `2_4_design_power.png`.
- **Everyday picture:** "Before a clinical trial we do a power calculation: how many patients do we need to
  see an effect of this size? The same question exists here, per cell type, and nobody asked it before."
- **What we did:**
  - We worked out the smallest change that *any* method could detect in each cell type, from the study design
    alone: the cell proportions, how much they vary between samples, the number of samples and the noise.
  - It needs no answer key, so it can be run before a study.
- **Say (left panel, interferon, 7 paired donors):**
  - "Monocytes are 64% of the reads: a change of 0.06 is detectable with 7 donors."
  - "B cells are 5%: the smallest detectable change is 0.8, much larger than any real change. To reach 0.1
    you need more than 200 donors per arm. That is why PALM says 'can't tell' for B cells, and it is right
    to."
- **Say (right panel, composition spread):**
  - "Samples that differ in composition help. Halve the spread between samples and you need about 2.5 times
    as many donors; double it and you need less than half."
  - "Spillover can only be undone if cell proportions vary."
- **It depends on the program too (one line, 10 s):**
  - "The limit differs 2–3× between programs in the same study." In the drug experiment, MAPK can be
    resolved down to 0.04, p53 only to 0.11.
  - "Programs whose genes are expressed in many cell types, or overlap other programs, are harder to
    attribute. So choose and check your programs before the study, not after."
- **Check against the answer key:**
  - Across all benchmark boxes, PALM never flagged a real change below this limit (0 of 28).
  - It found 58% of those above it (73 boxes).
  - "So the limit predicts where any method can and can't attribute."
- **Honest note (one line):** PALM is still more cautious than the limit in some cases, e.g. interferon in
  monocytes. Closing that gap is our next method step.
- **Design box:** limit = the smallest change detectable with 90% confidence and 80% power by any unbiased
  method, given the design; checked on 420 boxes across 4 datasets (3 splits each).

### 2.7 Real blood: whole blood vs sorted cells from the same people (1 min)
- **Visual:** a small table (draw).
- **Data:** GSE60424: 20 donors; whole blood plus sorted neutrophils, monocytes, B, CD4 and CD8 cells, all
  sequenced. The whole blood is the bulk; the sorted cells are the answer key.

  | Comparison (donors per arm) | PALM | InstaPrism + t-test |
  |---|---|---|
  | MS before vs after interferon-β (3 v 3) | 0% unchanged flagged; 29% of real changes found | 0% flagged; 82% found |
  | Sepsis vs healthy (3 v 4) | 0% flagged | 24% flagged |
  | Type 1 diabetes, ALS vs healthy | no flags | no flags |

- **Say:**
  - "On real blood, PALM made no false calls in any comparison."
  - "With 3 donors per group the design limit says almost nothing is resolvable, which is exactly what PALM
    reports. The design calculator predicted it."
  - "One lesson for the field: sorting cells changes their stress genes. The sorted 'answer key' differs from
    the cells in the tube, so sorted-cell validations need care."

### 2.8 Next: our PTB cohort, and a 50-cell-line drug screen (1 min)
- **Visual:** the slide-6 cell types (neutrophils, CD14 and CD16 monocytes) × pathways (TLR, NLR, IFN-γ, TNF,
  BCR, TCR) as an empty grid with two columns per box, InstaPrism | PALM, plus a column "design limit". Label
  it "results next week".
- **PTB questions:**
  1. Are TLR, NLR and IFN-γ higher in each neutrophil and monocyte in PTB, or only because there are more of
     these cells?
  2. Which slide-5 associations remain once spillover is removed?
  3. Which cell types does the design limit say our cohort *can* resolve? Run before looking at results.
  4. How much do the six gene sets share genes? If TLR, NLR and IFN-γ share many genes, part of the
     "coordination" on slide 6 is built in. Check the overlap first, and test each program on its unique
     genes too.
- **PTB experimental design:**

  | Item | Design |
  |---|---|
  | Data | PTB vs healthy PBMC bulk RNA-seq (n = / n =); same reference as part 1 |
  | Programs | The six part-1 gene sets, as in part 1; plus their overlap table, versions without shared genes (sensitivity), and PROGENy footprints of the matching pathways (JAK-STAT for IFN-γ, NF-κB / TNF-α) as a cross-check. Cell-type-restricted programs (BCR, TCR) are only read in the cell types that express them |
  | Methods | Part-1 workflow (InstaPrism → GSVA → test) vs PALM; same reference, genes and samples |
  | Contrast | PTB vs healthy per box, adjusted for age and sex |
  | Before results | Design limit per box from our cohort's proportions and size |
  | Positive control | Interferon-inducible signature up in neutrophils and monocytes of active TB (Berry et al. 2010) |
  | Negative controls | (a) Size-matched random gene sets: not flagged; (b) PTB/healthy labels shuffled: no calls |
  | Robustness | Leave-one-reference-donor-out; reference without neutrophils |
  | Readout | Per box: change with 90% range (PALM) vs score difference and q-value (InstaPrism); agreement table; "more cells" vs "cells changed" |

- **Can claim:** which cell types and pathways differ between PTB and healthy people, adjusted for age and
  sex. **Can't claim:** that TB caused it. Before vs after treatment in the same patients would come closer.
- **Also coming (one sentence):** "We are adding Tahoe-100M, a public screen of 93 drugs on about 50 cancer
  cell lines pooled in one well. It gives many more known answers, e.g. a BRAF inhibitor should act only in
  BRAF-mutant lines and a KRAS-G12C inhibitor only in G12C lines."

### 2.9 Outlook: spatial transcriptomics (30 s)
- **Idea:** a microdissected region (GeoMx ROI) is itself a small bulk sample. In a TB granuloma, is the
  inflammatory signal in the core vs the rim from *more* macrophages, or macrophages that switched on?
- **Why PALM fits:** regions differ a lot in composition, which is exactly what the design limit says helps,
  and one slide gives many regions.
- **Proposed design:** build "regions" from single-cell-resolution spatial data (Xenium, CosMx, Visium HD) by
  pooling the cells in each region, so the per-cell-type truth is known. Compare with cell2location and
  RCTD. No results yet.

### 2.10 Conclusion (1 min)
1. **A new question:** which cell type changed a gene program (pathway). Working at the program level, not
   gene by gene, is what makes it answerable from bulk. It still needs spillover undone; without that, a response in some cells is painted onto all of them.
2. **A fair test for it:** an answer key from real experiments (split cells, sorted cells, drug screens),
   with negative and positive controls set by biology in advance. Any method can be checked this way.
3. **PALM:**
   - finds 92–100% of real changes in the drug experiments;
   - flags under 5% of unchanged boxes, where standard tools flag 70–90%;
   - made no false calls on real blood.
4. **Know before you sequence:** the design limit tells, per cell type and program, whether a study can
   resolve it. Rare cell types need many donors, variation in composition helps, and the choice of programs
   matters as much as the sample size.
- **Last line:** "We can now ask not only *what* changed in the blood of TB patients, but *which cells*
  changed, how sure we are, and whether our study is big enough to tell."

---

## Figures

Generated figures are in `docs/figures/talk/`: 300 dpi PNG plus SVG (editable text in PowerPoint), sized for a
16:9 slide, no titles (put the message in the slide title). Colours keep one meaning: teal = PALM, red =
standard tools, dark = truth. Rebuild with `python scripts/experiments/talk_figures.py [SPLIT] [figure,...]`
and `python scripts/experiments/benchmark_heatmaps.py`.

| Slide | Visual | File |
|---|---|---|
| 2.1a | Three explanations of the same bulk signal | `2_1_three_causes.png` |
| 2.1b | The question in one line; table of tools vs questions | text + table |
| 2.2 | Our workflow on idasanutlin | `2_5_heatmap_lab_workflow.png` |
| 2.3a | Answer key: split cells, controls from biology, honest test | draw (`2_3_answer_key.png`, to make) |
| 2.3b | Ladder of evidence | table or ladder graphic |
| 2.4 | Pipeline + three new boxes | draw |
| 2.5a | Negative / positive controls, all tools, 4 datasets | `2_5_controls.png` (backup: `bench_calling.png`) |
| 2.5b | Per cell line: idasanutlin p53, trametinib MAPK; scorecard | `case_studies_palm_vs_instaprism.png`, `case_scorecard_palm_vs_instaprism.png` |
| 2.6 | Design limit vs donors per arm; composition spread | `2_4_design_power.png` |
| 2.7 | Whole blood vs sorted cells table | draw |
| 2.8 | Empty grid: 3 cell types × 6 pathways, InstaPrism \| PALM \| limit | draw |
| 2.9 | Granuloma core / rim cartoon with ROIs | draw |
| 2.10 | Four short lines | text |
| Backup | Proportions and activity, all tools; interferon ranges; estimated vs true; CIBERSORT heatmap | `bench_deconv.png`, `2_6_interferon_ranges.png`, `2_3_estimated_vs_true.png`, `2_5_heatmap_idasanutlin.png`, `2_6_fraction_error.png` |

**Figure reading notes:**
- `2_5_heatmap_lab_workflow`: the middle panel is coloured by correlation (ρ, its own scale); the other two
  by change in activity (shared scale).
- `2_4_design_power`: log scales; dashed line = a typical real change (0.1); dotted line = the study's actual
  size. A cell type is resolvable where its line is below the dashed line.
- `case_scorecard`: larger dot and bold number = the better of the two; same splits for both methods.
- `bench_calling` / `bench_deconv`: bars scaled per column (longest = best); 1–3 = rank. Rates use a fixed
  0–100% scale.

---

## Backup slides (only if asked)

- **"Is it fair to our workflow?"**
  - The emulation is InstaPrism per-type expression → log CPM → GSVA → Spearman with treatment → BH q < 0.1.
  - Gene sets are the up-weighted genes of each PROGENy pathway, so they match the answer key.
  - Code: `scripts/experiments/lab_workflow.py`.
- **"Is it tuned to the test?"** The v1 result was checked once, on data held back until the method was
  frozen: PALM flagged 0–1% of unchanged boxes, CIBERSORT + t-test 69–78%. Post-freeze changes will get a
  new freeze and new held-out splits before any paper numbers.
- **"How is the design limit computed?"** The bulk change in each cell type's share of reads is a known mix
  of the true changes of all cell types (the spillover map, from proportions and reference profiles). The
  limit is the best precision any unbiased method can reach when unmixing that map across samples
  (Gauss–Markov), × 2.49 for 90% confidence and 80% power. Paired donors and uncertainty in the proportions
  are included. Details: `docs/method_spec.md` §4.6.
- **Why the correction matters:** `2_3_estimated_vs_true.png`. Without correction, changes come out at about
  75% of their size and some non-responding lines appear to change; with correction, about 95%.
- **Interferon in detail:**
  - JAK-STAT rises in every cell type (0.26–0.36), so copying the bulk signal to every cell type happens to
    be right, and InstaPrism finds it.
  - Monocytes also have their own changes (TGF-β −0.18, hypoxia −0.17, MAPK −0.16). InstaPrism copies those
    into B, NK and T cells too.
  - A fix in testing narrows PALM's ranges and recovers the monocyte response with no new false flags.
- **Causality:** it comes from the study design (randomised or paired); PALM provides the measurement. For
  observational data it reports how strong a hidden confounder would have to be.
- **Held-out scoreboard (PALM v1, run once):**

  | Dataset | Coverage | r | False calls | Known answer | CIBERSORT + t-test false calls |
  |---|---|---|---|---|---|
  | McFarland idasanutlin | 0.92 | 0.92 | 0.01 | AUROC 1.0, 75% of responders found | 0.78 |
  | McFarland trametinib | 0.93 | 0.87 | 0.00 | all responders found | 0.77 |
  | Zhang | 0.98 | — | 0.00 | — | 0.00 |
  | Kang | 1.00 | 0.94 | 0.00 | power-limited | 0.69 |

- **InstaPrism, same proportions, three ways (unchanged boxes flagged):**

  | Dataset | InstaPrism + GSVA + correlation | InstaPrism + t-test | InstaPrism proportions + PALM |
  |---|---|---|---|
  | idasanutlin | 0.82 | 0.91 | 0.03 (responders found 0.67) |
  | trametinib | 1.00 | 0.81 | 0.08 (responders found 1.0) |
  | interferon | 0.41 | 0.83 | 0.00 (misses the response) |
  | chemotherapy | 0.00 | 0.00 | 0.00 |

  Most of the gain is PALM's correction step, not only its proportions.
- **Shared vs cell-type-specific readout (in development):** e.g. interferon: everything moved together,
  JAK-STAT 0.31 [0.27, 0.35].
- **Run time:** seconds (fast screen), about 5 min (default).
