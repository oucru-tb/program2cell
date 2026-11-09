# Paper section plan: false associations from per-cell-type scores

Status: plan written 2026-11-09, before any held-out association result. Development evidence is from
splits 0-9 (benchmarking.md 0.8, `benchmarks/dev/single_group_summary.txt`). Held-out splits 50-59 are unused.

## 1. Question and claim

A common analysis correlates each sample's per-cell-type pathway score, taken from deconvolved bulk, with a
sample-level variable: a clinical measure, severity, or another cell type's fraction. BayesPrism's embedding
tutorial does this ("correlation with clinical covariates", "correlation between omega and theta_env"), and
so does the lab workflow (InstaPrism -> per-type score -> Spearman -> BH).

Deconvolution leaks one cell type's signal into the others. So when one type truly follows Z, the unchanged
types follow Z too, and correlation calls them.

- **Hypothesis H1 (primary):** PALM's unmixing along Z (`associate_unmix`) makes fewer false associations
  than the standard workflow.
- **Hypothesis H2 (guardrail):** PALM does not buy this by losing the true associations. Ranking of true vs
  null pairs is non-inferior.

The section reports the result of both, whichever way they come out.

## 2. Development evidence (splits 0-9; not the paper's result)

False calls are pairs with no true association called at alpha = 0.1.

| Dataset | InstaPrism + correlation | PALM activity + correlation | PALM unmixing | Calls per split, InstaPrism / unmix |
|---|---|---|---|---|
| McFarland gradient, idasanutlin | 0.68 | 0.62 | **0.16** | 32 / 14 |
| McFarland gradient, trametinib | 0.64 | 0.73 | **0.17** | 34 / 19 |
| COMBAT COVID severity, raw bulk | 0.68 | 0.39 | **0.24** | 34 / 8 |

AUROC (true vs null pairs), the H2 metric:

| Dataset | InstaPrism | PALM unmixing |
|---|---|---|
| McFarland idasanutlin | 0.82 | 0.88 |
| McFarland trametinib | 0.98 | 0.95 |
| COMBAT raw bulk | 0.50 | 0.52 |
| COMBAT single-cell pseudobulk (ceiling) | 0.71 | 0.54 |

What this means for the claim:

- **H1 is supported on development data** in every dataset, by about 3-4x.
- **H2 is mixed.** Unmixing is more conservative: about half as many calls, and power falls to 0.06 on COMBAT
  raw bulk. Ranking is better on idasanutlin, slightly worse on trametinib, tied on COMBAT raw bulk, and worse
  on the COMBAT ceiling.
- So the defensible headline is **"fewer false associations at equal or lower power"**. It is not "better in
  every way".
- The gain comes from **unmixing**, not from PALM's activity estimates. PALM activity correlated with Z is as
  leaky as InstaPrism (0.62-0.73), so the section must name the unmixing path.
- The absolute target (false calls <= 0.1) is missed: 0.16-0.24. It must be stated as a limitation, not hidden.

## 3. Design

**Datasets.** All have truth per sample and per cell type.

1. McFarland gradient mixtures (`MIX_VARIANT=gradient`), idasanutlin and trametinib.
   - Each pseudo-patient has z ~ U(0, 1), the share of drug-treated cells.
   - The TP53-mutant lines (idasanutlin) and the non-responder lines (trametinib) are null cell types fixed
     by biology.
2. COMBAT, COVID-19 patients only, with Z = severity (1-4). This is real bulk with same-sample single-cell
   truth. Bulk arms:
   - raw (primary);
   - real_self, offsets estimated on the COVID bulk alone (the PTB / TB-only situation);
   - sc, single-cell pseudobulk (ceiling).

**Methods** (same samples, same truth, same alpha):

| Arm | Pipeline | Role |
|---|---|---|
| Standard workflow | InstaPrism -> per-type activity -> correlation with Z -> p < 0.1 | comparator (primary) |
| Lab workflow | InstaPrism -> per-type GSVA -> Spearman with Z -> BH 0.1 | comparator (what labs run) |
| PALM activity + correlation | `program_deconvolution` without T -> `associate` | shows the gain is not from activity |
| PALM unmixing | `associate_unmix` (z rescaled to 0-1) | the method |

`associate_unmix` is used exactly as it is in the code on 2026-11-09, with no tuning; it is outside freeze v4.
The lab-workflow arm (GSVA + Spearman) is to add to `single_group.py`. It reuses the GSVA runner in
`compare/`.

**Splits.** Held-out 50-59, run once, after the held-out v4 lanes (40-49) finish.

**Null definitions** (fixed in `single_group.py`):

- Positives: true change over the Z range >= 0.1.
- Nulls: true change < 0.03.
- Pairs in between are not scored.
- The permutation null (Z shuffled) is reported for every arm.

**Secondary null N2: program-composition association (descriptive).**

- In the McFarland gradient mixtures, fractions are drawn independently of the biology. So a null line's true
  activity has no association with any other line's fraction.
- For each null (type A, program) and each other type B, compute the correlation of the estimated activity
  with B's fraction.
- Report the spread of correlations and the share with p < 0.1, for InstaPrism and for PALM activity.
- PALM unmixing with Z = B's fraction is tried as well. If it is ill-posed, because the fraction already
  enters the model, the plan says so and that cell is reported as "not applicable".
- Not part of the decision rule.

## 4. Pre-specified decision rule (per dataset, then pooled)

Differences are PALM unmixing minus the standard workflow. Intervals are 90%, from a two-stage cluster
bootstrap (splits, then patients), as in heldout_v4_compare.py.

| Claim | Metric | Pass if |
|---|---|---|
| H1 fewer false associations | false-call rate difference | upper CI < 0 |
| H2 ranking not lost | AUROC difference | lower CI > -0.05 |
| Honesty, absolute | PALM false-call rate; permutation false-call rate | reported against 0.1; permutation <= 0.1 |
| Interval | coverage of PALM's change interval | reported against 0.85 |

**Wording follows the rule:**

- H1 and H2 pass: "PALM reduces false cell-type associations without losing ranking."
- H1 passes, H2 fails: "PALM reduces false associations but is more conservative." Report power.
- H1 fails: the section becomes a negative result on unmixing along Z, and the claim is dropped.

COMBAT raw bulk is reported with the McFarland datasets, not dropped when it is the weakest.

## 5. Display item (one figure, 3 panels)

- **a.** False-call rate per arm and dataset, dots with 90% CI, with a dashed line at 0.1.
- **b.** Power vs false-call rate per arm. This shows the trade-off honestly.
- **c.** Worked example: idasanutlin gradient. Estimated association with Z per cell line:
  - the standard workflow gives the TP53-mutant lines the same slope as the responders;
  - PALM unmixing keeps it in the intact lines.

## 6. Placement and text

- Article only; in a Brief Communication it shrinks to panel a in the supplement.
- Results section title: "Correlating deconvolved cell-type scores with a clinical variable makes false
  associations".
- Discussion: link to the PTB use (gestational age as Z). Cite BayesPrism embedding learning as an example of
  the correlation analysis, not as a failing method: its correlation is on a learned malignant program, and
  the same leakage risk applies.

## 7. Compute and order

1. Wait for the held-out v4 lanes a-e.
2. Add the GSVA + Spearman arm and the N2 computation to `single_group.py`.
   - Check both on split 0 (development) only.
   - Do not inspect any split 50-59 output while coding.
3. Run splits 50-59 on 4 lanes. About 3-4 h, as for the development run.
4. Score with the rule above, unchanged, and write `docs/association_results.md` with every dataset, pass or
   fail.
