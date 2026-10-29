# Flagship experimental design (target: Nature Methods / Nature Biotechnology)

Companion to `experiment_cellgeometry.md`, which covers the head-to-head with fraction-only
deconvolution. This document covers what a top methods venue needs beyond that: a new capability
validated against ground truth that does not come from our own model, generality across tissues and
platforms, scale, and a biological finding that only this method could make.

Planned for **after the method freeze** (see `experiment_cellgeometry.md` §7), except where marked
*early*.

## 1. The claims the paper must support

| # | Claim | Why it is new |
|---|---|---|
| C1 | From bulk RNA-seq alone, PALM-bulk estimates **per-sample, per-cell-type program activity** with calibrated uncertainty | Fraction methods stop at proportions; cell-type-specific DE methods (TOAST, TCA, CellDMC, bMIND, Unico) give gene-level effects without program structure or calibrated per-sample activity |
| C2 | It **decomposes a bulk treatment effect into composition and intrinsic parts**, with uncertainty | The question "is this signature immune-cell shift or real activation?" cannot be answered by current tools |
| C3 | Its intervals **cover** under realistic mismatch between reference and bulk | Competitors give no intervals, or SEs validated only by correlation |
| C4 | It scales to atlas references and cohorts of 10⁴ samples | Bayesian deconvolution (BayesPrism) is hours to days |
| C5 | Applied to large public cohorts, it **changes a biological conclusion** | Shows the method matters in practice |

Each claim needs evidence from at least two tiers below, and at least one tier must not rely on the
model's own definitions.

## 2. Evidence tiers

| Tier | Ground truth | Strength | What it can prove |
|---|---|---|---|
| T1 Simulation | Exact, from our generative model | Weak (circular) | Calibration (SBC), identifiability limits, power |
| T2 Pseudobulk from scRNA | Exact fractions; δ from held-out counts (`truth_split`) | Medium | Accuracy under real biology and real donor variation |
| T3 Pseudobulk with **known stimulation** | Which cell type responded, by experimental design | Strong for C1/C2 | Cell-type attribution of activity is causal, not inferred |
| T4 Real bulk + orthogonal measurement | Flow/Coulter fractions, sorted-cell bulk, histology | Strong for fractions and C3 | Behaviour under real platform shift |
| T5 **Designed physical mixtures** (wet lab) | Fractions and stimulated compartment set by design | Strongest, unique | C1–C3 on real bulk sequencing with an exactly known answer |

## 3. Experiments

### F1. Calibration by construction (T1, *early*, already partly done)

- Simulation-based calibration (SBC) of full PALM and palm_fast.
- Rank histograms for π, δ, β, and the composition and intrinsic shares.
- Grid: N ∈ {24, 96, 384, 1,536}, C ∈ {4, 8, 16}, effect sizes, overdispersion, reference noise.
- Output: coverage vs nominal at every level, which separates "the method is calibrated" from "the
  benchmark is noisy".
- Also the **identifiability map**: posterior contraction of δ_c as a function of π_c and of
  cell-type specificity of program genes. It shows where per-type activity is learnable at all, and
  justifies the resolution chosen by the data.

### F2. Breadth of pseudobulk ground truth (T2)

- At least 8 scRNA cohorts across tissues and platforms, each with ≥ 30 donors:
  - blood: COVID PBMC, OneK1K, a CellTypist-type atlas;
  - lung and kidney (e.g. HLCA subsets, KPMP);
  - tumour (a pan-cancer atlas subset);
  - brain (snRNA);
  - a 5′ or full-length chemistry for cross-chemistry tests.
- Always cross-donor (reference and test donors disjoint), with `truth_split=True`.
- Metrics: fraction RMSE/CCC and coverage; activity ρ across samples and types, plus coverage; effect
  r, AUROC, coverage; composition/intrinsic share error.
- Comparators:
  - fractions: cellGeometry, BayesPrism/InstaPrism, CIBERSORTx, MuSiC, DWLS;
  - cell-type-specific effects: TOAST, TCA, CellDMC, bMIND, Unico, ENIGMA;
  - activity: deconvolution then ULM/decoupler, naive bulk activity.

### F3. Stimulation ground truth: who responded? (T3, the key experiment for C1)

- Use scRNA data where a stimulus acted on **known** cell types, then build pseudobulk:
  - IFN-β PBMC (Kang et al. 2018, GSE96583): all types respond via JAK-STAT; monocytes most.
  - A cytokine perturbation atlas (e.g. Parse 10M PBMC, 90 cytokines × 12 donors): for each
    cytokine, the receptor-bearing cell types respond. That gives a cytokine × cell-type response map
    as ground truth.
  - LPS (monocytes) and anti-CD3/CD28 (T cells), where available.
- Design: per donor, a stimulated and a control pseudobulk. Vary which cell types come from the
  stimulated arm (for example only monocytes stimulated, the rest control) to create
  **cell-type-restricted** activation.
- Metrics: AUROC for "which cell type responded", sign accuracy, calibration of
  P(δ_c,stim > δ_c,ctrl).
- **Success:** correct attribution of the responding cell type with AUROC ≥ 0.85, where
  deconvolution-then-ULM pipelines fail.

### F4. Composition vs intrinsic, with known truth (T3 + T5, the key experiment for C2)

- *In silico*, from F3 data: construct contrasts where
  - (i) only composition changes (more monocytes, no stimulation);
  - (ii) only intrinsic activation changes (same composition, monocytes stimulated);
  - (iii) both, in the same or opposite directions (masking).
- Show that the bulk DE / bulk-activity signal is identical across (i)–(iii), while PALM-bulk recovers
  the true split with intervals.
- **Wet-lab version (T5, optional but decisive):** with a collaborating lab, sort PBMC compartments,
  stimulate one compartment (e.g. LPS on monocytes, IFN on all), and physically mix at designed
  fractions (a 3 × 3 design of monocyte fraction × stimulation) before bulk RNA-seq.
  - About 30–50 libraries.
  - Ground truth for fractions *and* for which compartment is activated, on real bulk chemistry
    against a 10x reference, so it includes the real platform shift.
  - This is the experiment reviewers cannot argue with.

### F5. Real bulk with orthogonal measurements (T4, fractions and calibration)

- Sorted-cell and flow-cytometry datasets: Monaco et al. (GSE107011), FlowSorted blood sets.
- PEAC whole blood with Coulter counts and R4RA/STRAP synovium with histology, as in the cellGeometry
  head-to-head.
- Metrics: fraction correlation and **coverage of the orthogonal measurement**.
- Effect of m_g (covariate model) and the unknown component.

### F6. Robustness to reference mismatch (T2/T4)

- Reference from another cohort, chemistry (3′ vs 5′ vs snRNA), or tissue state (healthy reference,
  diseased bulk).
- Fewer reference donors (1, 3, 10, 30), to show how propagating reference uncertainty keeps coverage
  as the reference shrinks.
- Missing cell types (in the bulk only, and in the reference only).
- Programs: PROGENy vs cell-type-specific G_c vs a mis-specified program set (shuffled genes, as a
  negative control).

### F7. Scale (C4)

- Reference: atlas-scale signatures (CELLxGENE Census subsets, 10⁶–10⁷ cells), with streaming means.
- Bulk: N = 10², 10³, 10⁴ (GTEx whole blood; a large TCGA cohort).
- Report wall time and memory for stage 1, palm_fast with subsampled stage 2, and full PALM where
  feasible, on CPU and GPU.
- **Success:** N = 10⁴ with posterior output in under 1 h on one GPU; stage-1 point estimates in
  minutes.

### F8. Biological application that changes a conclusion (C5)

Pick 2–3 large public cohorts with clinical outcomes where composition confounding is plausible and
the question matters. Candidates (accessions to verify):

- **Sepsis endotypes / severity** (blood bulk): are endotype signatures intrinsic myeloid
  reprogramming or neutrophilia and lymphopenia?
- **TB progression signatures** (blood, longitudinal): how much of the progression signal is
  composition vs intrinsic IFN activity in monocytes?
- **Dengue / severe febrile illness** (OUCRU cohorts, if available): host response per cell type,
  and whether intrinsic activity predicts severity beyond cell counts.
- **COVID severity** (several bulk cohorts with matched scRNA in some): there the scRNA arm validates
  the bulk-only conclusions in the same patients.

Analysis for each cohort:

1. The composition vs intrinsic split of each published signature.
2. Per-cell-type program activity vs outcome, adjusted for composition.
3. Where matched scRNA exists for a subset, confirm the cell-type attribution directly. That is T4
   evidence for C1 on real disease.

**Deliverable:** at least one published signature reinterpreted as mostly composition, and at least
one intrinsic cell-type program that predicts outcome independently of cell counts.

### F9. Genetic validation (stretch)

- OneK1K has genotypes and scRNA. Map QTLs for cell-type program *activity* from PALM-bulk estimates
  on pseudobulk, and compare with QTLs mapped directly on the scRNA per cell type.
- Concordance shows the per-type activity is biologically real, not a model artefact.

### F10. Usability

- Design / power calculator: samples needed to detect an intrinsic effect of size β in a cell type at
  fraction π. It comes straight from the Fisher information and is validated in F1.
- Package: pip + conda, a Snakemake workflow for every figure, containers for the competitors, and a
  web or Colab demo.

## 4. Figure plan (main text)

1. Concept: bulk = composition × per-type activity; the decomposition; model and two-stage inference.
2. Calibration and identifiability (F1) plus the multi-tissue accuracy summary (F2).
3. **Who responded?** Stimulation ground truth (F3) vs competitor pipelines.
4. **Composition vs intrinsic** (F4): in silico masking plus designed physical mixtures.
5. Real bulk: orthogonal measurements and reference mismatch (F5, F6); scale (F7).
6. Biology (F8): a reinterpreted signature and an outcome-predictive intrinsic program.

The head-to-head with cellGeometry (fractions, speed, noise) goes to Extended Data, cross-referenced
to `experiment_cellgeometry.md`.

## 5. Order, dependencies, budget

| Phase | Experiments | Depends on | Effort |
|---|---|---|---|
| Now (development) | F1 SBC grid, F3 on Kang 2018 (small, *early* check of C1) | Truth split, ideas 1–4 in §7.1 of the cellGeometry doc | ~1 week compute |
| After freeze | F2, F5, F6, the cellGeometry head-to-head | Comparator runners, dataset downloads | 2–3 weeks |
| In parallel | F4 wet-lab mixtures (if a lab partner agrees) | Protocol + ~40 libraries | 1–2 months lab |
| Then | F7 scale (GPU), F8 biology | Frozen method | 3–4 weeks |
| Stretch | F9 genetics, F10 calculator and website | — | 2 weeks |

## 6. Risks

- **Weak cross-type identifiability** (F1 shows it): narrow C1 to lineage resolution where needed. The
  identifiability map then becomes a result, not a hole.
- **No wet-lab partner:** F4 relies on in-silico T3 only, so the composition/intrinsic claim is
  weaker. Mitigate with F8 cohorts that have matched scRNA.
- **Program definitions (PROGENy) limit accuracy:** this is why cell-type-specific G_c (development
  idea 5) and the shuffled-program negative control are in F6.
