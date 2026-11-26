# Held-out results of freeze v5 (splits 30-39): the Paper 1 numbers

Pre-registered in `docs/method_spec.md`, "Method freeze v5" (tag `method-freeze-v5`, commit `bef2d8f`). Run
2026-11-16 to 2026-11-17 with `scripts/experiments/heldout_v5_chain.sh` (lanes a-e); result commit `436ed12`.
Written up 2026-11-23 from the saved outputs; nothing was rerun for this document.

Files: `benchmarks/dev/heldout_v5_decision.txt` (decision), `heldout_v5_transport.txt`, `practice_v5_heldout.txt`
(current practice), `combat_v5_summary.txt` (COMBAT).

## 1. Decision

**v5 passes and is the Paper 1 method.** Its estimates are identical to freeze v2 (C0); only the calibrated-route
interval differs (size-matched simulated leakage error, `sim_widen="sized"`).

| Check | Rule | Result | |
|---|---|---|---|
| C0 | estimates identical to v2 (<= 1e-8) | max difference 0 | pass |
| S1 | every dataset: coverage >= 0.85, null-call rate <= 0.10 | lowest coverage 0.897 (Tahoe gemcitabine), 0.898 (McFarland idasanutlin); highest null calls 0.074 (McFarland trametinib) | pass |
| S2 | coverage difference lower CI >= -0.03; null-call difference upper CI <= +0.02 | -0.016; +0.008 | pass |
| S3 | wrong-sign calls per split (COMBAT), v5 <= v2 + 0.2 | 0.3 vs 0.1 (raw), 0.1 vs 0.0 (offset) | pass |
| E (primary) | power difference lower CI > 0 | +0.047 [+0.010, +0.097] | superior |

Differences v5 - v2, 13 datasets [95% cluster bootstrap CI]: coverage -0.009 [-0.016, -0.003], null calls +0.003
[+0.001, +0.008], power +0.047 [+0.010, +0.097], width -0.044 [-0.062, -0.025]. The gain is in PBMC (power +0.084
[+0.024, +0.163]); cell lines are unchanged (+0.004). Study-level sign-flip p 0.125 (5 studies, 3 nonzero).

## 2. Per dataset (freeze v5, means over 10 splits)

| Dataset | Coverage | r | Within-program r | Null calls | Power | Width |
|---|---|---|---|---|---|---|
| McFarland idasanutlin | 0.898 | 0.901 | 0.316 | 0.055 | 0.437 | 0.218 |
| McFarland trametinib | 0.938 | 0.915 | 0.797 | 0.074 | 0.774 | 0.172 |
| Tahoe encorafenib | 0.911 | 0.444 | 0.115 | 0.028 | 0.267 | 0.178 |
| Tahoe gemcitabine | 0.897 | 0.536 | 0.497 | 0.045 | 0 | 0.124 |
| Tahoe RMC-6236 | 0.967 | 0.718 | 0.064 | 0 | 0.694 | 0.270 |
| Tahoe trametinib | 0.986 | 0.842 | 0.661 | 0.008 | 0.786 | 0.288 |
| Kang IFN-beta | 0.989 | 0.874 | 0.738 | 0 | 0.292 | 0.627 |
| Zhang paclitaxel | 0.993 | 0.817 | - | 0.037 | 0 | 0.097 |
| Parse GM-CSF | 1.000 | 0.806 | 0.951 | 0.004 | 0.298 | 0.226 |
| Parse IFN-beta | 1.000 | 0.906 | 0.586 | 0.003 | 0.105 | 0.373 |
| Parse IFN-gamma | 1.000 | 0.892 | 0.940 | 0.011 | 0 | 0.238 |
| Parse IL-4 | 1.000 | 0.752 | 0.385 | 0.003 | 0.133 | 0.293 |
| Parse TNF | 0.996 | 0.836 | - | 0.011 | 0.100 | 0.130 |

Power: pairs with a true change >= 0.1 called with the right sign. Null calls: pairs with |true change| < 0.05
called. Nominal level 0.90.

## 3. PALM vs current practice (splits 30-39)

Current practice: deconvolution, a per-cell-type program score, a t-test, BH q < 0.1. Means over 13 datasets x 10
splits; power and responder AUROC on the known program only.

| Pipeline | Coverage | Null-call rate | Power | Responder AUROC |
|---|---|---|---|---|
| **PALM (freeze v5)** | **0.967** | **0.023** | 0.651 | **0.800** |
| InstaPrism + score | 0.407 | 0.538 | 0.963 | 0.781 |
| CIBERSORT + score | 0.422 | 0.526 | 0.963 | 0.763 |
| NNLS reallocation + score | 0.263 | 0.415 | 0.845 | 0.731 |
| Bulk score (no deconvolution) | 0.372 | 0.582 | 0.963 | 0.500 |
| InstaPrism fractions + PALM module | 0.968 | 0.015 | 0.567 | 0.815 |
| CIBERSORT fractions + PALM module | 0.966 | 0.026 | 0.673 | 0.787 |

- PALM minus InstaPrism + score: null-call rate -0.516 [-0.571, -0.462], coverage +0.560 [+0.515, +0.603].
- Current practice calls a change in about half of the cell types that did not change: 0.31-0.36 of null pairs in
  PBMC, 0.68-0.84 in cell lines. Its power is high because it calls nearly everything.
- The PALM module turns any pipeline's fractions into calibrated effects (null calls 0.015-0.026).
- TOAST and TCA (cell-type-specific DE, then a per-type ULM of the gene log fold changes on the program; BH q < 0.1),
  4 datasets (McFarland x2, Kang, Zhang), run 2026-11-23 with `heldout_v5_csde.sh`. Means over those 4 datasets:

  | Pipeline | Null-call rate | Power | PALM minus it: null calls | power | AUROC (McFarland ida only) |
  |---|---|---|---|---|---|
  | PALM (freeze v5) | 0.042 | 0.75 | | | |
  | TCA + ULM | 0.246 | 0.87 | -0.204 [-0.248, -0.160] | -0.117 [-0.183, -0.058] | -0.022 [-0.100, +0.033] |
  | TOAST + ULM | 0.121 | 0.97 | -0.079 [-0.116, -0.045] | -0.217 [-0.317, -0.117] | +0.100 [0.000, +0.189] |

  PALM makes fewer null calls than both, as on 40-49 (-0.111 TOAST, -0.206 TCA), at lower power (power over 3
  datasets; Zhang has no known program). TOAST and TCA give no intervals, so no coverage. The PALM module rows on
  TOAST and TCA are identical (both reuse the same fractions): coverage 0.89-0.98, null calls 0.015-0.093.

## 4. Real bulk: COMBAT sepsis vs healthy (splits 30-39)

125 whole-blood bulk samples with paired PBMC single-cell data from the same people; truth from the single-cell
data. Per-split means; calls are counts of 49 pairs.

| Bulk | Within-program r | r | Calls | Null calls | Wrong sign | Coverage |
|---|---|---|---|---|---|---|
| Single-cell pseudobulk | 0.78 | 0.63 | 1.5 | 0.5 | 0 | 0.97 |
| Real bulk, raw | 0.50 | 0.05 | 0.8 | 0.1 | 0.3 | 0.93 |
| Real bulk, paired platform offset | 0.62 | 0.43 | 1.8 | 0.7 | 0.1 | 0.98 |
| Real bulk, self offset | 0.27 | 0.25 | 2.1 | 0.5 | 0.1 | 0.98 |

Raw real bulk loses the attribution across cell types (r 0.05); a platform offset restores it (0.25-0.43).
Coverage holds in every setting (0.93-0.98).

## 4b. Cell-type fraction accuracy (splits 30-39, descriptive)

From the dev_bench outputs, row `model` (`benchmarks/dev/fraction_joint_v5_heldout.txt`, made by
`scripts/experiments/fraction_joint_score.py`) and PALM-lite recomputed on the same splits
(`fraction_lite_v5_heldout.txt`). RMSE over samples x types; r = mean per-type Pearson correlation over samples. The
v5 protocol ran the PALM joint NB fit only on McFarland, Kang and Zhang; Tahoe and Parse effects used PALM-lite
fractions. The joint NB fit on Tahoe and Parse 30-39 was added for this table only
(`scripts/experiments/fraction_joint_chain.sh`, 2026-11-23 to 2026-11-25, output in `benchmarks/dev/fraction_joint/`,
kept apart so it cannot change a rerun of the v5 effects). Its sampler diagnostics: R-hat <= 1.049, minimum ESS 49
(Tahoe trametinib), divergences 0-8 per dataset on Parse and 17-58 per dataset (10 splits) on Tahoe.

| Dataset | PALM joint NB | PALM-lite | InstaPrism | CIBERSORT | NNLS |
|---|---|---|---|---|---|
| Kang | **0.020** (r 0.75) | 0.047 (0.66) | 0.053 (0.61) | 0.073 (0.51) | 0.102 (0.43) |
| Zhang | 0.061 (0.91) | **0.049** (0.94) | 0.088 (0.83) | 0.090 (0.85) | 0.091 (0.84) |
| McFarland idasanutlin | 0.026 (0.94) | 0.025 (0.98) | 0.023 (0.98) | 0.023 (0.97) | 0.034 (0.95) |
| McFarland trametinib | 0.019 (0.98) | 0.026 (0.98) | **0.010** (1.00) | 0.017 (0.99) | 0.053 (0.90) |
| Tahoe (4 drugs) | **0.014-0.016** (0.98) | 0.014-0.026 (0.95-0.98) | 0.015-0.025 (0.95-0.98) | 0.018-0.031 (0.94-0.97) | 0.023-0.040 (0.89-0.95) |
| Parse (5 cytokines) | **0.049-0.087** (0.78-0.88) | 0.110-0.160 (0.67-0.75) | 0.087-0.122 (0.65-0.81) | 0.089-0.123 (0.65-0.77) | 0.128-0.159 (0.54-0.63) |

- The PALM joint NB fit gives the most accurate fractions on 11 of 13 datasets. PBMC: Kang 0.020 vs 0.053
  InstaPrism; Parse 0.049-0.087 vs 0.087-0.122 (about 40% lower error); Zhang 0.061 vs 0.088 (PALM-lite 0.049 is
  lower still). Tahoe: 0.014-0.016 vs 0.015-0.025.
- McFarland is the exception: InstaPrism is slightly better on both drugs (0.023 vs 0.026, 0.010 vs 0.019).
- PALM-lite, which the v5 protocol used for Tahoe and Parse effects, is worse than InstaPrism on Parse (0.110-0.160).
  Effects stay calibrated regardless (Parse coverage 0.996-1.000, null calls <= 0.011), because the effect
  calibration and leakage widening absorb the fraction error.
- PALM's fraction intervals are too narrow: 90% intervals cover 0.44-0.75 (joint NB; 0.44-0.60 in PBMC). Fraction
  uncertainty is not a claim of the paper.

## 5. Transport (descriptive; 9 perturbations never used in development)

v5 - v2: coverage -0.011 [-0.017, -0.004], null calls +0.002 [0.000, +0.005], power +0.036 [0.000, +0.106]. Every
transport dataset has v5 coverage >= 0.92 and null calls <= 0.067 (Tahoe everolimus).

## 6. Deviations

- An interim look at 08:50 on 2026-11-17 (lanes a and e finished; `heldout_v5_interim_20261117_0850.txt`), not in
  the protocol. Nothing was changed after it.
- Fraction table (4b): the first version (2026-11-23) averaged all joint NB output rows, including the stage-1 mode;
  corrected the same day to the `model` row (McFarland idasanutlin 0.024 -> 0.026, trametinib 0.018 -> 0.019).
- `fraction_joint_chain.sh` was edited while running (second Parse lane added 2026-11-23 16:30); three lanes
  ended with a bash error after their last split instead of a DONE line. All 90 fits finished (10 per dataset).
- `practice_vs_palm.py summary`: reads only its own row files (2026-11-23; the folder also holds another
  experiment's `*__palm_v2.json`). Reporting change only.
