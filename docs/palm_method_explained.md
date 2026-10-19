# PALM-bulk, explained simply

*What the method does, how it works, and what has been built and checked so far (status 2026-10-19).
Written in the Feynman style: plain words first, then the precise version, then where the simple story
breaks down.*

---

## 1. The problem in one picture

A bulk RNA-seq sample is a smoothie. You blend many fruits (cell types) and measure only the taste of the
whole glass (the total reads of each gene). Two things can change the taste:

- **The recipe changed.** There is more mango than before (more monocytes, fewer T cells).
- **The fruit changed.** The mangoes themselves got sweeter (monocytes switched on an interferon program).

A plain bulk analysis tastes "sweeter" and stops there. It cannot say whether you added mango or the mango
got riper. PALM tries to answer both, for every sample, and to say how sure it is.

Concretely, from bulk counts plus a single-cell reference it estimates:

1. **Cell-type fractions:** how much of each fruit is in the glass.
2. **Program activity inside each cell type:** how "ripe" each fruit is, for known gene programs
   such as interferon response, TNF-α/NF-κB or hypoxia (PROGENy, Hallmark, CollecTRI).
3. **Attribution:** of the total interferon signal in the glass, what share comes from each fruit.
4. **For treated vs control:** how much of a program's change comes from cells responding (intrinsic) and
   how much from the cell mix shifting (composition).

Every answer comes with an interval, not only a number.

---

## 2. The building blocks

### 2.1 What each cell type normally looks like: the reference `B`

From single-cell data we know, for each cell type, what share of its reads goes to each gene. Call it
`B[c, g]`: a cell type's resting "flavour profile". We build it from pseudo-bulks per donor (summing cells of
one type from one donor) and then average across donors, so one unusual donor cannot dominate.

### 2.2 What a program does: the dictionary `G`

A program is a list of genes with weights. Weight +1 means "this gene goes up when the program is on";
−1 means "goes down". This is `G[p, g]`, taken from public resources.

### 2.3 How active a program is in a cell type: `delta`

This is the key number. `delta[c, p]` says how strongly program `p` is switched on in cell type `c`,
**relative to that cell type's reference state**. The model turns it into a new flavour profile:

> new profile of cell type c = reference profile × exp(sum over programs of delta × program weights),
> then rescaled so the shares add up to 1.

In words: if interferon has `delta = 1` in monocytes, every interferon gene with weight 1 is multiplied by
e (about 2.7) in monocytes. Genes with negative weight are divided. Genes outside the program stay where they
are, apart from the rescaling.

Why multiply on the log scale? Because gene regulation acts like that: a program doubles or halves genes. It
does not add a fixed number of reads.

**Where the simple story breaks:** the rescaling means that when a program pushes some genes up, every other
gene's *share* goes down slightly. That is why gene selection includes "anchor" genes (section 4): stable
genes stop a strong program from making everything else look repressed.

### 2.4 Mixing: the bulk sample

> expected share of gene g in sample i = sum over cell types of (fraction of reads from c) × (profile of c in sample i)

The fractions here are **read fractions**: the share of the sample's reads that come from each cell type.
A monocyte has more RNA than a T cell, so read fractions are not cell fractions. We convert using each cell
type's average RNA content from the reference.

### 2.5 Noise: negative binomial counts

Real counts scatter more than coin flips would predict (biological variation between samples), so counts
are negative binomial around the expected value. Each gene gets its own dispersion, partly pooled across
genes ("gene-wise dispersion"), because some genes are simply noisier than others.

### 2.6 Platform bias: `m`

Single-cell and bulk protocols do not capture every gene equally well. A gene can look twice as high in
bulk for purely technical reasons. A per-gene scale `m[g]`, shared by all samples, absorbs this. Without
it, the model would explain a technical shift with a biological program.

---

## 3. Treatments, covariates and the decomposition

### 3.1 How a treatment enters

For each sample, with treatment indicator `T` (0 or 1):

- **Composition:** `fractions = softmax(a + gamma × T + sample noise)`. `gamma` is how much the
  treatment shifts the mix.
- **Activity:** `delta = kappa + beta × T + sample noise`. `kappa` is the baseline activity of each
  program in each cell type; `beta` is how much the treatment changes it.

Known covariates such as batch, sex or age shift both parts. Effects are then reported for a sample at the
average covariate value. This matters: in simulations with a batch linked to treatment, the coverage of
treatment-effect intervals rose from 0.25–0.44 without covariates to 0.81–1.00 with them.

### 3.2 Sparse priors: the horseshoe

Most programs are not active in most cell types. `kappa` and `beta` get a *horseshoe* prior, which says
"most of these are near zero, but a few can be large". It shrinks noise towards zero without crushing real
signals. Think of a strict editor who deletes weak claims but leaves strong ones alone.

### 3.3 Attribution: switch one program off and see what is lost

To ask "how much of sample i's interferon signal comes from monocytes", the model runs a small thought
experiment:

1. Compute monocytes' program signal with everything as estimated.
2. Compute it again with only interferon switched off in monocytes.
3. The difference, weighted by the monocyte fraction, is monocytes' contribution.

Do this for every cell type. Each one's share of the total is its attribution. Because each draw from the
posterior gives a slightly different answer, the shares come with intervals.

**Where it breaks:** if a program's genes look the same in every cell type, the bulk cannot tell who
carries it. Attribution intervals then become wide. That is the honest answer, not a failure.

### 3.4 Intrinsic vs composition: a fair split

The total change of a program between treated and control has two causes that act together. Fractions
change (composition) and per-cell activity changes (intrinsic). Splitting a product of two changes is
ambiguous: whichever you change first "gets" the interaction term. PALM uses the Shapley rule, the same idea
as splitting a restaurant bill fairly: average over both orders. The split is then **exact**
(intrinsic + composition = total, to machine precision) and symmetric.

---

## 4. Which genes go into the model

Using all 20,000 genes would be slow and add noise. PALM picks about 1,500 genes from four groups, each
serving one job:

| Group | Share | Job |
|---|---|---|
| Program genes (top weights, taken in turn per program) | 45% | Measure program activity and attribution |
| Cell-type markers (taken in turn per type) | 30% | Pin down fractions, so programs are not confused with composition |
| Highly variable genes in the reference | 10% | Catch cell-state variation that markers miss |
| Anchors (well expressed, stable across types, in no program) | 15% | Hold the rescaling steady |

**Where the simple story breaks:** the four shares are a hand-set rule. A principled alternative
(`gene_selection="fisher"`) asks of every gene: *how much would adding you sharpen our estimates?* Each
gene's answer depends on two things. The first is how strongly its expected count moves when a fraction or
a program shift moves. The second is how noisy it is (its dispersion). Adding genes one at a time to
maximise the total information (the determinant of the Fisher information, a "D-optimal design") recovers
the four roles without fixing their shares:
program genes inform the shifts, markers inform the fractions, and stable genes become valuable because
they separate a program change from a composition change. Noisy genes are down-weighted automatically.
Selection stops when the next gene would add almost nothing, so the number of genes comes from the data.
In a first simulation it chose about 1,040 genes instead of the fixed 1,500. Whether this improves
accuracy is being tested in the gene-budget study; until then the four groups stay the default.

Genes must be expressed in the reference (a gene absent from the baseline cannot be "induced") and
detected in at least 20% of bulk samples. The second filter is lenient, so genes switched on in one arm only
are kept.

**Many cell types.** Up to about 20 cell types, each gets its own program activity. Beyond that the data
cannot support so many unknowns. Either estimate fine fractions first and programs on a coarser reference,
or let subtypes of one lineage share program activity (`program_groups="lineage"`).

---

## 5. How the model is fitted

### 5.1 Bayesian sampling (NUTS)

We do not want one best guess; we want the range of answers compatible with the data. NUTS (the No-U-Turn
Sampler) explores that range like a ball rolling over a landscape whose height is how poorly the parameters
fit the data. It spends time in proportion to how plausible each region is. The collection of visited
points (draws) is the posterior: medians give estimates, quantiles give intervals.

### 5.2 Warm start with variational inference

Rolling the ball from a random spot wastes most of the run finding the valley. PALM first runs a quick
approximate fit (variational inference) to locate the valley and its rough width, then starts NUTS there.
This was about 2.5 times more efficient per gradient, with the same accuracy, and allows a shorter warmup.

### 5.3 Precision: float64 for real data

With hundreds of samples and about 1,500 genes, the total log-likelihood is around 10 million. In 32-bit
floats, rounding errors of several units then appear in exactly the differences NUTS uses to accept steps.
At realistic size this shrank the step size 330-fold and raised the fraction error by about 70%. Real data
should run in 64-bit.

### 5.4 Checking the sampler worked (new)

A sampler can look fine and still be wrong if it got stuck. PALM now:

- runs **2 chains by default**, started at different points. If they disagree, something is wrong;
- reports **split R-hat**: how much chains (and halves of chains) disagree. It should be below 1.01;
- reports **effective sample size (ESS)**: how many independent draws the correlated draws are worth.
  It should be at least 100 per chain;
- checks fractions, program activity, baseline and treatment effects, the intrinsic/composition parts and
  dispersion, and warns when any check fails.

Before this change, most runs used one chain, and one chain cannot see two chains disagreeing.

---

## 6. The code base after the restructure

Three layers, matching the two-paper plan:

- **Core** (shared by both papers): the equations of section 2 written once, so the same code runs on
  numpy (truth tables), JAX (the sampler) and PyTorch (training the fast network in Paper 2). A test checks
  that all three give the same numbers and that PyTorch can take gradients through them. The core also
  holds reference building, gene selection, covariates and the simulators.
- **Bayesian inference** (Paper 1): priors, sampler, warm start, diagnostics, result tables.
- **Comparison** (new): competing methods behind one interface, and scoring against truth.

### 6.1 One ruler for everyone: `estimate_delta`

To compare methods, everything must be measured on the same scale. `estimate_delta` takes any observed
cell-type profile and asks: "what `delta` would turn this cell type's reference profile into this one?"
It answers with the same equation the model uses, solved directly by Newton steps on a concave likelihood.

It is used in two places:

- **Ground truth on real data:** pool the real cells of one type in one sample, compute that pseudo-bulk,
  and run `estimate_delta`. That is the true program activity, by the model's own definition. The same
  approach gives Paper 2's training labels.
- **Competitors:** BayesPrism, CIBERSORTx and BLADE output a per-sample expression profile for each cell type.
  Running `estimate_delta` on those puts them on PALM's scale. The conventional route (decoupler ULM, the
  usual "deconvolve then PROGENy") is reported alongside it.

On simulated profiles drawn from the model, `estimate_delta` recovers the true shifts to within 0.05.

---

## 7. The competitors and how they are scored

| Method | What it gives | How it becomes comparable |
|---|---|---|
| Naive two-step | NNLS fractions; one program shift per bulk sample | Same shift copied to every cell type, so it cannot say who carries a program |
| Reallocation | NNLS fractions; reads split between cell types in proportion to fraction × reference | Programs per cell type with `estimate_delta` (the first step of BayesPrism, with a fixed reference) |
| BayesPrism | Fractions; per-type expression per sample | `estimate_delta` or ULM |
| CIBERSORTx | Fractions; high-resolution per-type expression | Same (not run yet: needs Docker and a token) |
| BLADE | Fractions; per-type expression per sample | Same |
| TOAST, TCA | Gene-level treatment effect per cell type | Projected onto program weights to give per-type program effects |

**Scores** (each method only on what it outputs):

- **Fractions:** error and correlation with the truth.
- **Activity:** rank correlation across samples, per cell type and program ("does it go up in the right
  samples?"). This metric also works for paired real data, where absolute values are not trustworthy.
  Rank correlation across cell types within a sample ("does it name the right carrier?").
- **Effects:** correlation with the true `beta`. AUROC for telling responding cell types from the rest
  (0.5 means it cannot tell).
- **Intervals:** coverage of the 90% intervals, for methods that give intervals (only PALM).

---

## 8. What the evidence says so far

All of this is on **simulated data**. No real-data validation exists yet.

**Calibration** (calibration benchmarks): 90% intervals cover 91–96% of true values with the current
defaults. Gene-wise dispersion raised coverage on noisy genes from 0.88 to 0.94.

**First head-to-head** (2 simulated studies, 24 samples, 5 cell types, 4 programs):

| | Fraction error | Activity rank corr. | Effect corr. | Finds responding cells (AUROC) |
|---|---|---|---|---|
| PALM | 0.011 | 0.76 | 0.99 | 1.00 |
| Reallocation | 0.048 | 0.45 | 0.48 | 0.82 |
| Naive two-step | 0.048 | 0.34 | 0.05 | 0.50 |

PALM's 90% intervals covered 89% of true activities and 95% of true effects.

**All methods on one simple study** (12 samples, 4 cell types, 3 programs; a check that every part works,
not a benchmark):

| | Fraction error | Activity rank corr. | Names the right carrier | Effect corr. | AUROC |
|---|---|---|---|---|---|
| PALM | 0.014 | 0.71 | 0.84 | 0.96 | 1.00 |
| BayesPrism | 0.051 | 0.50 | 0.47 | 0.40 | 0.69 |
| Reallocation | 0.067 | 0.52 | 0.58 | 0.30 | 0.64 |
| BLADE | 0.118 | 0.51 | 0.23 | 0.33 | 0.72 |
| Naive two-step | 0.067 | 0.44 | none | 0.03 | 0.50 |
| TOAST | (NNLS fractions) | none | none | 0.10 | 0.53 |
| TCA | (NNLS fractions) | none | none | −0.36 | 0.39 |

Every deconvolution method scored worse at naming the carrier when its per-type expression went through
the usual decoupler ULM instead of `estimate_delta`. TOAST and TCA are weak with 12 samples, as expected
for gene-level cell-type-specific tests. BLADE's fractions were poor with both input recipes tried; whether
that is BLADE or the inputs is not settled. CIBERSORTx has not been run.

**An important warning from the new diagnostics.** In that head-to-head, PALM's chains had not converged
(R-hat up to 1.27, ESS as low as 3) with 200 warmup and 200 sampling steps, although the estimates scored
well. With 500 warmup and 500 sampling steps on the simple study, R-hat fell to at most 1.024 and the
lowest ESS rose to 116: much better, still short of the targets (1.01 and 200). Before the PBMC benchmark
(E2), we need to settle how long warmup must be. The current defaults were chosen from runs that could not
detect this problem.

---

## 9. Limits to keep in mind

- **A cell type missing from the reference gives confidently wrong answers.** Its signal is absorbed by the
  cell types that are present, and the intervals do not widen. The built-in reference check is weak.
  The reference must cover the tissue.
- **The reference is treated as known.** Donor-to-donor variation in the baseline profiles is not
  carried into the intervals.
- **Causal reading needs a causal design.** "Intrinsic effect" means "the cells responded" only when
  treatment is randomised or confounders are adjusted for. Bulk data cannot tell which cell type *caused*
  another to respond.
- **Programs must be right.** The model measures known programs; it does not discover new ones.
- **Compute.** Fit time grows with samples × genes × cell types × programs. Hundreds of samples need a GPU
  or a long CPU run.

---

## 10. The whole method in five sentences

1. Each cell type has a known resting profile, and each gene program is a known set of up and down genes.
2. In each sample, every cell type's profile is its resting profile pushed along each program by an unknown
   amount (`delta`), and the bulk sample is a mixture of those profiles with unknown fractions.
3. A Bayesian model, with noise per gene, a platform correction and sparse priors, finds all plausible values
   of the fractions and pushes at once, so every answer has an interval.
4. From those values it computes who carries each program (switch-one-off attribution) and, for treatments,
   an exact fair split of each program change into "cells responded" and "cell mix changed".
5. The same equations, written once, serve as the ruler for ground truth, for scoring competing methods, and
   later for training a fast network for all public bulk data.
