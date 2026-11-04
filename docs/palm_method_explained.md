# program2cell and its PALM model, explained simply

*What the method does, how it works, and what the evidence says (status 2026-11-04, method freeze v2).
Plain words first, then the precise version, then where the simple story breaks down. The formal
definitions are in `method_spec.md` and `model_dag.md`; this file is the explanation.*

**Names.** **program2cell** is the tool (the package, formerly palm-bulk). **PALM** is the Bayesian model at
its core, which estimates each sample's cell mix. The per-cell-type changes come from calibrated steps built
around that model (section 4).

---

## 1. The problem in one picture

A bulk RNA-seq sample is a smoothie. You blend many fruits (cell types) and taste only the whole glass (the
total reads of each gene). Two things can change the taste:

- **The recipe changed.** There is more mango than before (more monocytes, fewer T cells).
- **The fruit changed.** The mangoes themselves got sweeter (monocytes switched on an interferon program).

A plain bulk analysis tastes "sweeter" and stops there. program2cell answers four questions, from bulk
counts plus a single-cell atlas:

1. **How much of each fruit:** cell-type fractions per sample.
2. **How each fruit changed:** for each cell type and known gene program (PROGENy pathways such as
   interferon, TNF-α/NF-κB, hypoxia), the change between groups (TB vs healthy, treated vs control), with a
   90% range.
3. **Who is responsible:** each cell type's share of the bulk change, split into "more cells" and "cells
   changed".
4. **How sure, and what this design can resolve:** every answer has an interval that was checked against
   a known answer, plus the smallest change this study could detect.

---

## 2. The building blocks

### 2.1 What a cell type looks like: the reference `B`

From the single-cell atlas we know, for each cell type, what share of its reads goes to each gene: `B[c, g]`.
It is built from pseudo-bulks per donor (summing one type's cells in one donor), averaged across donors.

**Where the simple story breaks:** `B` is the *average* cell of that type in the atlas, not a "resting" cell.
An atlas contains activated cells too. That is fine for group comparisons (the baseline cancels), but it has
two consequences:
- use healthy donors of the atlas for a disease cohort, so the baseline is a known state;
- merge activation-state clusters ("IFN-stimulated monocytes") into their parent type. Otherwise a cell that
  switches a program on is counted as *more cells of another type* instead of *cells changed*.

### 2.2 What a program does: `G`

A program is a list of genes with weights (`G[p, g]`, from PROGENy). A positive weight means "goes up when
the program is on", a negative one "goes down", and the size says how strongly. PROGENy's weights are
continuous; each program is rescaled so its median |weight| is 1, which keeps `delta` on a comparable scale.
The same weights are used in every cell type.

### 2.3 How active a program is: `delta`

`delta[i, c, p]` says how strongly program `p` is switched on in cell type `c` of patient `i`, relative to
`B`. Every patient has their own value:

> profile of cell type c in patient i = B × exp(sum over programs of delta × program weights), rescaled to
> add up to 1.

With `delta = 1` for interferon in monocytes, an interferon gene with weight 1 is multiplied by e ≈ 2.7 in
monocytes. Multiplying (on the log scale) matches how regulation works: programs double or halve genes.

### 2.4 Mixing and noise

> expected share of gene g in sample i = Σ over cell types of (fraction of reads from c) × (profile of c)

The fractions are **read fractions**: a monocyte makes more RNA than a T cell, so read fractions are
converted to cell fractions with each type's RNA content. Counts scatter around the expected value as
negative binomial noise, with one dispersion per gene.

**Where it breaks:** RNA content is the mean UMIs per cell in the atlas. That mixes true RNA content with
capture efficiency (worst for neutrophils), carries no uncertainty, and is the same in TB and healthy. If
activated cells make more RNA, that can only show up as "more cells". Blood counts would fix this (v3 plan).

---

## 3. Groups: how a disease enters

For each patient with disease status `D` (0 healthy, 1 TB):
- **Composition:** the cell mix shifts with disease (`gamma`).
- **Activity:** each program in each cell type shifts with disease (`beta`).

`beta` per cell type and program is the main answer. The contributions (section 4, stage E) combine it with
the fractions.

**One quantity, two estimates.** PALM has its own parameter for `beta`, but it is used only inside the model.
The reported value, written `β̂` where the difference matters, comes from stage D (section 4), which is
checked against a known answer.

**Known covariates** (age, sex, site, batch, HIV status) enter both the model and the unmixing, so a
difference in them is not read as disease. Paired designs (the same patient twice) and weighting for
non-randomised groups (IPW / AIPW) are supported. Disease does not have to be yes/no: an ordered or continuous
measure goes through the path of section 6; an ordered TB axis (uninfected, latent, active, on treatment) is
in the v3 plan.

---

## 4. How the answer is computed: five stages

The first version of this method (October 2026) read everything off one Bayesian model. Benchmarks with a
known answer showed that its per-cell effects were biased and its intervals too narrow when cell types
spill into each other. The frozen method keeps the model for what it does well, the cell mix, and computes
the per-cell changes with steps that are checked against a known answer.

### Stage 0 (optional, real bulk): platform correction

Bulk and single-cell protocols capture genes differently: a gene can look twice as high in bulk for purely
technical reasons. `correct_platform` estimates one offset per gene from the study's own bulk and removes it.
One offset for all patients, so a disease difference is not absorbed.

**Where it breaks:** the study's *average* composition cannot be identified from the bulk alone, so fraction
accuracy stays limited on real bulk. In a TB-only cohort, an offset estimated from TB samples could absorb a
signal shared by every patient.

### Stage A: the cell mix (the PALM model)

PALM is a Bayesian model of the whole smoothie (sections 2–3). It is fitted in two steps: each sample is
summarised around its best fit, then NUTS samples the population-level parameters. It returns 50 plausible
cell mixes per patient (posterior draws), so the uncertainty about the mix is carried forward. PALM-lite is a
fast approximation for large studies.

### Stage B: split each patient's reads between cell types (reallocation)

Each gene's reads are shared out between cell types in proportion to fraction × reference profile, and each
type's share is scored on the `delta` scale (`estimate_delta`). This is the first step of
BayesPrism/InstaPrism with a fixed reference.

**Where it breaks, and why the next two stages exist:** the split uses the *unchanged* profiles. So a cell
type that switched a program on keeps only part of its change (attenuation), and the rest leaks into the
other cell types (spillover). Every deconvolve-then-test workflow has this problem; it is what makes
InstaPrism + t-test call changes in cell types that did not change.

### Stage C: calibrate on atlas donors with a known answer

We build synthetic bulk samples from atlas donors that were not used for the reference: half of each donor's
cells make a bulk with known fractions, the other half give the true activity. Running stage B on these shows
how much each type's activity is attenuated, and the leftover errors set the width of the 90% ranges. Because
the ranges come from errors measured against a truth, they are honest by construction, not by assumption.

### Stage D: undo the spillover across patients (the effects router)

The spillover follows a known pattern: how much of type k's change appears in type c depends on each
patient's fractions and the reference. Across many patients with different mixes, that pattern can be
inverted: **unmixing** solves for the per-cell changes that, after spilling, explain what reallocation saw.

Two estimates are available for each program:
- **unmixed:** less biased, but its range is wide when the cell mix barely varies between patients;
- **calibrated reallocation:** stable, with a range widened to the spillover error simulated for this study.

The frozen rule: use the unmixed estimate where its range is narrower; but if a cell type's unmixed value
falls outside its calibrated range, keep the calibrated one (the guard). A design check with about 10
simulated cohorts, built from real reference counts with this study's fractions and design, supplies the
spillover error.

### Stage E: the readouts

| Answer | What it means |
|---|---|
| Per-cell change `beta` ± 90% range, call | Did this cell type change this program between groups? Called when the range excludes 0 |
| Contributions | Each cell type's share of the bulk change, split exactly into "more cells" and "cells changed" (Shapley rule: average over both orders, like splitting a bill fairly) |
| Shared response | When every type responds (interferon), the program-level change instead of a per-type split |
| Robustness | How strong an unmeasured confounder would have to be to explain the change away |
| Design limit | The smallest per-type change this design can resolve |

---

## 5. Why some intervals are long, and why that is right

A long range is the method telling you that this design cannot separate this cell type's change from its
neighbours'. It happens when the type is rare, when its fraction barely varies between patients, or when its
program genes look the same as another type's. A short range in that situation would be a guess presented as
a measurement: InstaPrism + t-test gives short ranges there and calls changes that did not happen. The
design-limit readout says in advance how large a change must be to show up.

**Is it a Bayesian method?** For the cell mix, yes. The per-cell change is a calibrated, spillover-aware
estimate, because that was both more accurate and more honest on benchmarks with a known answer than the
model's own posterior.

---

## 6. Cohorts with one group only (e.g. TB patients only)

Without a comparison group there is no disease effect. The question becomes: does a cell type's program
follow a clinical measure `z` (bacterial load, severity, cavitation) across patients?
- Correlating each patient's per-type activity with `z` inherits spillover: in the first McFarland gradient
  test, 63% of unchanged pairs were called.
- `associate_unmix` puts `z` into the unmixing instead of the 0/1 group, which cut that to 21%. The target
  is ≤ 10%, so this path is **not yet validated**; a 10-split benchmark is queued (benchmarking.md §0.8).

---

## 7. How it is tested

**Datasets with a known answer** (benchmarking.md): pooled cancer cell lines with real drug responses
(McFarland; Tahoe-100M, 4 drugs), stimulated blood cells (Kang IFN-β; Parse, 5 cytokines; Zhang), and
COVID-19/sepsis blood with paired bulk and single-cell data (COMBAT). The truth per cell type comes from
held-back cells.

**Competitors,** each followed by the usual test: CIBERSORT, InstaPrism, NNLS reallocation, a naive bulk
score; TOAST, TCA and cellGeometry where affordable.

**Scores:** coverage of the 90% ranges (target ≥ 0.90), false calls on unchanged pairs, share of true
effects called, correlation with the true effects.

**Pre-registration.** The method was frozen (tag `method-freeze-v2`) before the held-out splits 20–29 were
run, once, with no retuning. Any later change becomes v3 with new held-out splits.

---

## 8. What the evidence says

Cell-line results (McFarland, Tahoe) and blood-cell results (Kang, Parse, Zhang, COMBAT) should be read
separately: PROGENy's programs were learned largely from cell lines, so the cell-line benchmarks test the
spillover machinery more than the programs. For blood, Kang's recall is the realistic expectation.

**Development** (splits 0–2):
- Coverage ≥ 0.93 on every dataset.
- The v2 routing rule raised the share of true effects called (Kang 0.23 → 0.48, Tahoe trametinib 0.33 → 0.89,
  McFarland with a composition shift 0.67 → 1.00) and lowered false calls (Zhang 0.038 → 0.008). It costs
  some accuracy where every type responds (Tahoe RMC-6236, r 0.80 → 0.64).
- COMBAT, real bulk: InstaPrism + t-test flags 59–67% of cell type × program boxes that did not change;
  program2cell flags 0–1%.

**Held-out (v2, splits 20–29, run once; heldout_v2_results.md):**
- The ranges covered the truth on all 13 datasets (0.93–1.00) and false calls on unchanged pairs stayed at or
  below 8%; on real COMBAT bulk, 0–0.6%. Deconvolution + t-test flagged up to 90% of unchanged pairs (cell lines), 50 percentage points more on average.
- Accuracy and power met their targets on 5 of 13 datasets (McFarland, three Parse cytokines). Blood: Kang
  found 40% of true effects; real bulk: almost no calls. Two weak Tahoe drugs were poor for every method, and
  worse for program2cell than for the best comparator.
- In short: honest on unseen data; power is the next problem (v3_plan.md).

---

## 9. Limits to keep in mind

- **A cell type missing from the reference gives confidently wrong answers.** Its signal is absorbed by the
  types that are present, and there is no detector for it yet. For whole-blood TB samples, the reference must
  include neutrophils.
- **Neutrophils are not yet validated.** Every benchmark is cell lines or PBMC (COMBAT's single-cell arm is
  PBMC), so no result so far could show a granulocyte failure. A whole-blood benchmark comes before any
  whole-blood TB claim.
- **Whole blood has contaminants** that vary per sample: globin and platelet reads. Mask them before fitting.
- **Broad responses are the hard case.** When every cell type responds (interferon in TB), the per-type
  split is least accurate, and the method may report the program-level change instead.
- **Power depends on the design.** With little variation in the cell mix between patients, rare types get
  wide ranges (section 5).
- **Real bulk loses accuracy** through the platform gap (stage 0).
- **Programs are known and the same in every cell type.** The method measures known programs; it does not
  discover new ones.
- **Association, not causation.** "Cells changed" means "the cells responded" only in a randomised or
  well-adjusted design; the robustness readout says how fragile a call is.

---

## 10. The whole method in five sentences

1. Each cell type has a reference profile from a healthy atlas, and each gene program is a known set of up
   and down genes.
2. PALM, a Bayesian model of the bulk as a mixture of those profiles, estimates each patient's cell mix with
   its uncertainty.
3. Each patient's reads are split between cell types, and the split's bias is measured on atlas donors with
   a known answer, which also sets honest 90% ranges.
4. Across patients, the spillover between cell types is undone where the design allows it, giving each cell
   type's change between groups.
5. The changes and the cell mix together say who drives a program's change in bulk: more cells, or cells
   changed.
