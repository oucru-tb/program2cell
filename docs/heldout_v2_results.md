# Held-out v2 results (method freeze v2; splits 20–29, run once)

Run 2026-11-03 12:18 to 2026-11-04 11:49, four lanes, one pass, no retuning. Method: tag `method-freeze-v2`
(cda0a01), `program_deconvolution` defaults (`causal_loop` estimator `auto_v2`). Protocol: method_spec.md,
"Method freeze v2". Raw outputs: `benchmarks/dev/heldout_v2_out/` (report, causal_loop, practice, combat);
per-split rows in `benchmarks/dev/causal_loop/`, `practice/`, `combat_real/`. Report script:
`scripts/experiments/heldout_v2_report.py` (sections 5a–5c, results by family / route / regime, were added on
2026-11-04 before the run finished, as reporting only).

**Summary.**
- **Calibration and specificity hold on every dataset.** 90% interval coverage 0.93–1.00 (target ≥ 0.85) and
  false calls on unchanged pairs ≤ 0.08 (target ≤ 0.10) on all 13 datasets, and on COMBAT real bulk
  (coverage 0.97–1.00, false calls 0.000–0.006).
- **Accuracy and power do not meet the pre-set targets on 8 of 13 datasets.** All five targets are met on 5
  (McFarland idasanutlin and trametinib; Parse IFN-β, IFN-γ, TNF-α). The misses are correlation with the true
  effects (r < 0.9: Kang, Zhang, Parse GM-CSF and IL-4, all four Tahoe drugs) and, on Kang and three Tahoe drugs,
  the share of true effects called.
- **Against current practice** (deconvolution + t-test), PALM's intervals cover the truth far more often and it
  makes far fewer false calls; current practice calls more true effects because it calls nearly everything; the
  two correlate equally well with the truth.

## 1. PALM per dataset against the pre-set targets

Targets (set before the run): coverage ≥ 0.85, r ≥ 0.9, responder AUROC ≥ 0.95, true effects called ≥ 0.9, false
calls on unchanged pairs ≤ 0.1. Means over 10 splits; `diff` = calibrated reallocation without the unmixing
step or the simulation widening, for reference (for programs routed to the calibrated estimate, PALM's point
estimate equals `diff`'s; only the interval is widened).

| Dataset | Family | Coverage | r | False calls (unchanged) | True effects called | AUROC | Verdict | diff: coverage / r / called |
|---|---|---|---|---|---|---|---|---|
| McFarland idasanutlin | cell lines | 0.96 | 0.92 | 0.049 | 0.95 | 1.00 | **pass** | 0.61 / 0.88 / 0.15 |
| McFarland trametinib | cell lines | 0.96 | 0.92 | 0.076 | 1.00 | — | **pass** | 0.67 / 0.86 / 0.08 |
| Tahoe trametinib | cell lines | 0.99 | 0.81 | 0.006 | 0.97 | — | r | 0.67 / 0.78 / 0.00 |
| Tahoe RMC-6236 | cell lines | 0.98 | 0.72 | 0.011 | 0.78 | — | r, called | 0.66 / 0.69 / 0.00 |
| Tahoe gemcitabine | cell lines | 0.93 | 0.54 | 0.049 | 0.00 | 0.83 | r, called, AUROC | 0.82 / 0.66 / 0.00 |
| Tahoe encorafenib | cell lines | 0.95 | 0.42 | 0.024 | 0.40 | 0.70 | r, called, AUROC | 0.79 / 0.45 / 0.00 |
| Kang IFN-β | blood | 1.00 | 0.82 | 0.000 | 0.40 | — | r, called | 0.98 / 0.94 / 0.28 |
| Zhang | blood | 1.00 | 0.85 | 0.004 | — | — | r | 0.98 / 0.85 / — |
| Parse IFN-γ | blood | 1.00 | 0.91 | 0.000 | — | — | **pass** | 0.96 / 0.91 / — |
| Parse IFN-β | blood | 1.00 | 0.90 | 0.003 | — | — | **pass** | 0.96 / 0.90 / — |
| Parse TNF-α | blood | 1.00 | 0.94 | 0.000 | — | — | **pass** | 0.97 / 0.94 / — |
| Parse GM-CSF | blood | 1.00 | 0.85 | 0.000 | — | — | r | 0.96 / 0.85 / — |
| Parse IL-4 | blood | 1.00 | 0.69 | 0.000 | — | — | r | 0.94 / 0.69 / — |

Notes:
- "—": not defined (no known responder set, or no true effect above threshold).
- McFarland idasanutlin, known-answer pairs only: the p53 program was called in the 3 non-responder lines in 57%
  of cases (0, 1, 2 or 3 of 3 per split). These lines show a small measured p53 induction (0.5–0.7 log2 FC); over
  all unchanged pairs the false-call rate is 0.049.
- Against `diff`, PALM raises coverage everywhere (the simulation widening) and calls true effects that `diff`
  misses (McFarland, Tahoe RMC-6236 and trametinib; the unmixing). Its r is equal on Parse and Zhang, where
  almost nothing was unmixed, and lower on Kang (0.82 vs 0.94) and Tahoe gemcitabine (0.54 vs 0.66). Because
  calibrated-route point estimates are identical, the r loss comes from the unmixed programs: on Kang the
  interferon program (JAK-STAT), where unmixing raised the share of true effects called (0.40 vs 0.28) but its
  pair error was about ten times that of the calibrated route (section 4).

## 2. Against current practice and on other tools' fractions

Paired PALM minus comparator over dataset × split (95% bootstrap CI; n = number of pairs).

| PALM minus | Coverage | False calls (unchanged) | True effects called | r | AUROC |
|---|---|---|---|---|---|
| CIBERSORT + t-test | +0.55 [+0.50, +0.60] | −0.50 [−0.55, −0.45] | −0.33 [−0.43, −0.25] | −0.02 [−0.05, +0.02] | +0.09 [+0.01, +0.17] |
| InstaPrism + t-test | +0.56 [+0.51, +0.61] | −0.50 [−0.55, −0.44] | −0.33 [−0.43, −0.25] | −0.03 [−0.06, +0.01] | +0.06 [+0.00, +0.13] |
| reallocation + t-test | +0.71 [+0.68, +0.74] | −0.39 [−0.44, −0.34] | −0.21 [−0.31, −0.11] | +0.06 [−0.04, +0.17] | +0.10 [+0.01, +0.20] |
| naive bulk score + t-test | +0.60 [+0.55, +0.64] | −0.56 [−0.62, −0.50] | −0.32 [−0.41, −0.24] | +0.06 [+0.02, +0.09] | +0.34 [+0.28, +0.40] |
| TOAST | — | −0.13 [−0.18, −0.09] | −0.18 [−0.29, −0.06] | +0.24 [+0.16, +0.32] | +0.17 [+0.12, +0.21] |
| TCA | — | −0.23 [−0.27, −0.18] | −0.10 [−0.18, −0.04] | +0.28 [+0.19, +0.38] | +0.03 [+0.00, +0.08] |
| cellGeometry + t-test | +0.63 [+0.50, +0.76] | −0.55 [−0.68, −0.42] | −0.09 [−0.19, −0.01] | +0.06 [−0.01, +0.11] | 0.00 |

- Current practice calls 70–90% of unchanged cell type × program pairs in the cell-line data (e.g. InstaPrism
  + t-test: McFarland 0.90 / 0.74, Tahoe 0.68–0.82) and 20–50% in Parse, so its higher share of true effects
  called is not usable as a discovery rate. Its point estimates correlate as well as PALM's (r difference not
  significant); on Kang and the Tahoe drugs they correlate better (e.g. Kang 0.97 vs 0.82).
- **PALM's effects module on other tools' fractions** gives nearly the same results as on PALM's own fractions
  (coverage +0.00 to +0.03; r within ±0.04 except vs TOAST/TCA/reallocation fractions); PALM's own fractions call
  more true effects than InstaPrism's (+0.12 [+0.06, +0.17]).
- cellGeometry failed on Kang in 8 of 10 splits (R error in `cellMarkers`, as in development); its rows cover the
  remaining splits.

## 3. COMBAT real bulk (125 paired bulk + single-cell; 10 splits)

| Contrast | Bulk | PALM coverage / r / false calls / calls | InstaPrism + t-test (BH) coverage / r / false calls / calls | PALM fraction r |
|---|---|---|---|---|
| sepsis | raw | 0.97 / 0.08 / 0.000 / 0.0 | 0.31 / −0.04 / 0.64 / 28.8 | 0.60 |
| sepsis | paired offsets | 1.00 / 0.49 / 0.000 / 0.5 | 0.21 / 0.29 / 0.81 / 38.3 | 0.70 |
| sepsis | study offsets | 0.99 / 0.33 / 0.000 / 1.1 | 0.36 / 0.40 / 0.52 / 24.6 | 0.39 |
| sepsis | single-cell bulk | 0.98 / 0.63 / 0.006 / 0.6 | 0.60 / 0.46 / 0.22 / 12.1 | 0.91 |
| severe COVID | raw | 0.99 / 0.20 / 0.000 / 0.0 | 0.32 / 0.02 / 0.56 / 27.5 | 0.09 |
| severe COVID | paired offsets | 0.98 / 0.31 / 0.000 / 0.0 | 0.17 / 0.35 / 0.91 / 43.6 | 0.61 |
| severe COVID | study offsets | 0.97 / 0.66 / 0.000 / 0.4 | 0.38 / 0.45 / 0.50 / 24.4 | 0.31 |
| severe COVID | single-cell bulk | 0.97 / 0.70 / 0.000 / 0.7 | 0.64 / 0.60 / 0.28 / 16.0 | 0.88 |

On real bulk PALM is specific (no false calls) but finds almost nothing (≤ 1 call per split); InstaPrism +
t-test flags half or more of the unchanged pairs. Accuracy on real bulk is low for both and depends on the
platform correction; fraction accuracy on raw real bulk is poor (r 0.09 for severe COVID).

## 4. By family, route and response regime (reporting added 2026-11-04)

| | Coverage | r | True effects called | False calls |
|---|---|---|---|---|
| Blood cells (7 datasets) | 1.00 | 0.84 | 0.40 | 0.001 |
| Cell lines (6 datasets) | 0.96 | 0.73 | 0.69 | 0.036 |

- **By route** (cell type × program pairs): blood — calibrated 2,932 pairs, coverage 1.00, RMSE 0.03; unmixed
  78 pairs, coverage 0.97, RMSE 0.27. Cell lines — calibrated 1,527, coverage 0.96; unmixed 1,833, coverage 0.96.
  Coverage holds on both routes; in blood the unmixed route is rare and much less precise. (Which types the guard
  switched back is not recorded in v2.)
- **When every cell type responds** (truth |β| > 0.1 in all types): Kang 116 pairs, coverage 0.99 with wide
  intervals (mean width 1.0); Tahoe RMC-6236 90 pairs, coverage 0.90; Tahoe trametinib 90 pairs, 0.99. Coverage
  holds; in blood the price is width, i.e. power.

## 5. Reading

1. **What the frozen method delivers on unseen data:** honest intervals and very few false calls, across cell
   lines, stimulated blood cells and real whole-blood bulk. This is the main claim and it held on every dataset.
2. **What it does not deliver:** the accuracy (r ≥ 0.9) and power targets outside McFarland and three Parse
   cytokines. Kang (blood, broad interferon response) calls 40% of true effects; Tahoe encorafenib and
   gemcitabine are weak for every method (best comparator r 0.61 and 0.75, InstaPrism + t-test) — small or
   diffuse responses at the doses used — and PALM is clearly worse than the best comparator there (0.42, 0.54). On real bulk the method is specific but has almost no power.
3. **Current practice is not a viable alternative at the same specificity:** it calls most unchanged pairs. Its
   point estimates are as accurate as PALM's on average and better on Kang and Tahoe, which shows where PALM's
   calibration costs accuracy (cell lines and broad responses).
4. These results go into the v3 plan (docs/v3_plan.md): power in blood (candidates 5, 9), real bulk (6, 7), broad
   responses (9), and reporting by family, route and regime (A2–A4).

No change to the method follows from this run under v2; any change is v3, with new held-out splits (30–39).
