# Spatial extension: cell-type program activity in spots, cells and niches (plan, 2026-11-03)

Status: plan only, nothing run. It extends research_plan §7.1. It is outside Paper 1 and outside freeze v2:
the frozen method and its held-out results are not changed by anything here. A spatial method gets its own
freeze (v3 or a separate tag) and its own held-out sections before any result is reported.

## 1. Why spatial data has the bulk problem

PALM's question in bulk is: a program changed; which cell types changed it, and how sure are we? The same
question arises in spatial data whenever a measured unit holds more than one cell, or holds transcripts from
cells it should not:

| Setting | Unit that acts as a "bulk sample" | Source of spillover | Fit with the current method |
|---|---|---|---|
| **Visium / Visium v2** (55 µm spots, ~1–10 cells) | spot, or spots pooled by niche | real mixing of cells in the spot | direct: fractions per unit, reallocation, unmixing |
| **GeoMx / microdissection** (ROI, ~50–1000 cells) | ROI / AOI | real mixing | unchanged; best case (marker-segmented AOIs give strongly varying mixes) |
| **CosMx / Xenium / Visium HD** (single cell or 2–8 µm bins) | segmented cell, or niche | segmentation error: transcripts of touching cells assigned to the wrong cell | same idea, but the leakage operator comes from neighbours, not from fractions |
| **Niche level** (granuloma core / rim / lymphoid aggregate / normal lung) | niche × section, pooled | mixing inside the niche | closest to bulk; z = niche label or a continuous position (distance to lesion) |

The niche is the natural unit. Pooling spots or cells by niche × section gives bulk-like depth, removes most
spot-to-spot correlation, and asks a biological question directly: does a pathway differ between niches
because the cell mix differs, or because a cell type changed its activity there? `decompose` already gives
that split.

## 2. Data model

Indices: patient i, section s (nested in i), unit u (spot, cell or niche pool, nested in s), cell type c,
program p, gene g.

- Mixed units (spots, ROIs, niche pools): Y_ug ≈ Σ_c π_uc · B_cg · exp(Σ_p δ_ucp G_pg) · offset_g, exactly the
  bulk model with u in place of a patient. offset_g is the platform gap (probe chemistry vs scRNA-seq).
- Segmented cells: a cell of type c(u) carries its own profile plus a leaked share from its neighbours,
  Y_ug ≈ (1 − ε_u) E_{c(u)g} + Σ_{v ∈ nb(u)} ε_uv E_{c(v)g}. The leakage weights ε come from the segmentation
  geometry (shared boundary, distance) or are estimated, and they play the role of π in the leakage operator A.
- Contrast z: niche label (in vs out), a continuous position (distance to the lesion edge, local T-cell
  density), or a patient-level group (TB vs control). Continuous z uses the single-group path
  (`associate_unmix`, benchmarking.md 0.8), so that benchmark's result carries over directly.

## 3. What has to change (and what does not)

**Unchanged:** reference profiles B from a single-cell atlas, programs G, reallocation, the unmixing estimator
and its interval logic, contributions (Shapley split), robustness readouts.

**Changes:**
1. **The replicate is the patient, not the unit.** Spots and cells of one section are strongly correlated. The
   bootstrap and the calibration resample patients (sections nested inside), with spatial block resampling
   inside a section only when one patient is analysed. Resampling spots would give intervals that are far too
   narrow. This is the most important change and the easiest to get wrong.
2. **Scale and sparsity.** 10^3–10^5 units per section with few counts each. Fractions from a fast step
   (PALM-lite, or RCTD / cell2location fractions passed in, as `fractions=` already allows for bulk), and a
   sparse solver for the unmixing (A is block-structured by section). Full NUTS only for niche pools.
3. **Gene panels.** CosMx 1k/6k and Xenium 5k panels cover only part of each program. A coverage rule per
   program (as `min_program_genes`) and a check that the reported program is not driven by 2–3 panel genes.
   CosMx WTX and Visium are whole-transcriptome.
4. **Platform gap.** Probe-based Visium (FFPE) and imaging panels have per-gene capture efficiencies that
   differ from scRNA-seq. `correct_platform` applies with the same limit as in bulk: the average composition
   is not identifiable from the spatial data alone (method_spec, platform note). A matched scRNA-seq or a
   ground-truth section fixes it.
5. **Segmentation leakage (single-cell platforms).** A neighbour-based leakage operator (section 2). Correcting
   transcripts one by one is a separate problem with its own tools (MisTIC, FastReseg, Baysor, proseg,
   ResolVI); the claim here is narrower: per-type program changes are unbiased after accounting for leakage
   at the level the analysis is done.
6. **Observational contrasts.** Niche comparisons are not randomised (and pathologist ROI choice is not random).
   Report adjusted associations with `robustness_value`, not causal effects.

## 4. Competitors

- **C-SIDE** (Cable et al., Nat Methods 2022; R package spacexr, built on RCTD): cell-type-specific
  differential expression along spatial covariates (niche, proximity to a cell type, a continuous position),
  accounting for mixed spots. The main competitor; it answers almost the same question at gene level.
  Where program2cell should differ: program-level effects with spillover unmixing across types, calibrated
  intervals checked against a known answer, contributions, and the composition-vs-activity split.
- **Niche-level and cell-type-specific spatially variable gene methods** (e.g. Niche-DE, Celina, STANCE, spVC,
  CTSV, ctSVG): related questions; a 2026 benchmark found C-SIDE among the more conservative and scalable
  options (PMC13112436).
- **Two-step practice:** RCTD / cell2location fractions, then per-type pseudobulk or a regression on fractions,
  then a t-test or limma; the spatial version of InstaPrism + t-test.
- **Segmentation correction** (MisTIC, FastReseg, cellAdmix-type corrections, Mitchel et al., Nat Genet 2026)
  is a preprocessing step, not a competitor; the benchmark should include "correct first, then naive
  per-type DE" as a baseline.
- SpatialDecon (GeoMx): fractions only.

A literature search before starting is needed; this list is from a quick search on 2026-11-03.

## 5. Benchmark with a known answer

**Pseudo-spots from single-cell spatial data.** Take CosMx WTX or Xenium sections with segmented, typed cells.
- Split the cells of each type in each region into two halves (as McFarland, §3.9 of research_plan): one half
  gives the truth (per-type program activity per niche), the other builds the test units.
- Visium-like units: sum cells in 55 µm grid squares (keeps real neighbourhood structure and real mixing).
- Niche pools: sum by niche × section.
- Segmentation stress: re-assign a share of each cell's transcripts to touching neighbours (planted leakage),
  and, where available, compare against a nuclear-only segmentation.
- Planted effects: optional, by mixing treated and control cells as in the McFarland gradient, so that the
  truth is exact.

**Metrics:** the same as bulk (coverage of 90% intervals, false calls on unchanged pairs, power, rank,
bias), computed at the patient level. Plus: how much naive per-cell DE on segmented cells is biased by
leakage (a figure that motivates the method by itself).

**Splits and freeze:** sections (and patients) split into development and held-out before any run, as for
freeze v2.

## 6. Candidate datasets (to check; found 2026-11-03, not yet downloaded)

TB-relevant:
- Multicenter Visium atlas of human TB and non-tuberculous mycobacterial granulomas: 38 samples, discovery
  32 / validation 6, lung and lymph node, integrated with HLCA v2 (bioRxiv 2025.10.17.683002). The best first
  real dataset: many patients, niche annotations likely, a built-in validation cohort.
- Visium v2 (CytAssist) plus matched scRNA-seq from the same tissue, osteopontin (SPP1) macrophages (mBio
  2025, mbio.01559-25): matched reference for the platform gap.
- FFPE Visium of TB lung resections, TB myofibroblasts and SPP1+ macrophages (J Exp Med 2026, e20251067).
- Single-cell and spatial three-layer granuloma architecture, 19 lesions (bioRxiv 2024.07.15.603490).
- Lung and omentum TB granulomas (Theranostics 2024, v14p6185).
- CosMx (protein) + GeoMx + scRNA-seq, TREM2+ macrophages (bioRxiv 2025.07.15.664002): GeoMx ROIs are a direct
  fit; the CosMx part is protein, not RNA.

For the known-answer benchmark (not TB): public CosMx WTX or Xenium 5k sections of lung or lymph node with
cell segmentation; to be chosen.

No human TB granuloma Xenium or CosMx RNA dataset was found in the quick search.

## 7. Steps

1. Literature search (C-SIDE successors, niche DE, 2025–26) and dataset check (licences, annotations,
   number of patients). Decide Visium-first or GeoMx-first.
2. Feasibility on real data: niche-pooled program2cell on the multicenter TB Visium atlas (core vs rim vs
   normal lung; macrophage and T-cell interferon programs). No claims; checks scale, platform gap and
   interval width at the patient level.
3. Pseudo-spot benchmark (section 5), development sections only; build the patient-level bootstrap and the
   neighbour leakage operator; compare with C-SIDE and the two-step practice.
4. Freeze the spatial method, run held-out sections once.
5. Scope decision: a section of a later paper or a separate paper.

**Talk (2026-10-30 outline):** one "future directions" line, for example: "TB granuloma niches: which cell
type drives the IFN-γ program at the lesion edge". No results are claimed.

## Sources

- C-SIDE: Cable et al., Nat Methods 2022 (PMC10463137); spacexr: https://github.com/dmcable/spacexr
- Benchmark of cell-type-specific spatially variable gene methods: https://pmc.ncbi.nlm.nih.gov/articles/PMC13112436/
- MisTIC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12724677/; FastReseg: https://www.biorxiv.org/content/10.1101/2024.12.05.627051
- Segmentation errors: https://www.biorxiv.org/content/10.1101/2025.01.02.631135
- TB Visium atlas: https://www.biorxiv.org/content/10.1101/2025.10.17.683002v1.full
- mBio 2025: https://journals.asm.org/doi/10.1128/mbio.01559-25
- J Exp Med: https://rupress.org/jem/article/223/3/e20251067/281353
- Three-layer granulomas: https://www.biorxiv.org/content/10.1101/2024.07.15.603490
- Theranostics 2024: https://www.thno.org/v14p6185.htm
- TREM2+ macrophages: https://www.biorxiv.org/content/10.1101/2025.07.15.664002v2
