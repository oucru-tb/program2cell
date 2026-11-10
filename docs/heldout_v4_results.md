# Held-out results, method freeze v4 (splits 40–49)

Run 2026-11-09 to 2026-11-10 under the protocol in `docs/method_spec.md`, "Method freeze v4" (tag
`method-freeze-v4`, amendment commit `2d16f18`). The decision script `scripts/experiments/heldout_v4_compare.py`
ran unchanged. Nothing was rerun or retuned after the results were seen. Raw outputs are in
`benchmarks/dev/heldout_v4_out/`: `compare.txt`, `transport.txt`, `causal_summary.txt`, `practice_v2.txt`,
`practice_v4g.txt` and `combat_summary.txt`.

## Decision (pre-registered)

| Criterion | Rule | Result | |
|---|---|---|---|
| S1 | every dataset: gated v4 coverage ≥ 0.85 and null-call rate ≤ 0.10 | null calls 0.121 (McFarland idasanutlin), 0.107 (McFarland trametinib) | **FAIL** |
| S2 | coverage difference lower CI ≥ −0.03; null-call difference upper CI ≤ +0.02 | coverage −0.010 (pass); null calls +0.027 | **FAIL** |
| S3 | COMBAT wrong-sign calls per split, gated v4 ≤ v2 + 0.2 | 0.0 vs 0.0 (raw and offset-corrected) | pass |
| E (primary) | r difference, gated v4 − v2 | +0.030 [−0.012, +0.081] | non-inferior |

**Outcome: freeze v2 (`estimator="auto"`) stays the Paper 1 method. Gated v4 is reported as secondary.**

On the same splits, freeze v2 meets the S1 thresholds on every dataset: null calls at most 0.060, coverage at
least 0.93.

## Gated v4 vs freeze v2, per dataset (means over 10 splits)

| Dataset | Estimator | Coverage | r | Within-program r | Null calls | Power | Width | v4 chosen |
|---|---|---|---|---|---|---|---|---|
| McFarland idasanutlin | v2 | 0.940 | 0.903 | 0.354 | 0.041 | 0.441 | 0.248 | |
| | v4g | 0.912 | 0.856 | 0.662 | **0.121** | 0.450 | 0.245 | 10/10 |
| McFarland trametinib | v2 | 0.957 | 0.912 | 0.716 | 0.060 | 0.776 | 0.176 | |
| | v4g | 0.919 | 0.897 | 0.593 | **0.107** | 0.829 | 0.150 | 10/10 |
| Tahoe encorafenib | v2 | 0.940 | 0.433 | 0.070 | 0.021 | 0.033 | 0.205 | |
| | v4g | 0.973 | 0.607 | 0.397 | 0.025 | 0.517 | 0.199 | 10/10 |
| Tahoe gemcitabine | v2 | 0.930 | 0.510 | 0.490 | 0.037 | 0.000 | 0.141 | |
| | v4g | 0.970 | 0.729 | 0.796 | 0.059 | 0.250 | 0.137 | 10/10 |
| Tahoe RMC-6236 | v2 | 0.981 | 0.733 | 0.336 | 0.014 | 0.740 | 0.302 | |
| | v4g | 0.994 | 0.840 | 0.613 | 0.008 | 0.716 | 0.290 | 8/10 |
| Tahoe trametinib | v2 | 0.984 | 0.851 | 0.684 | 0.006 | 0.770 | 0.300 | |
| | v4g | 0.989 | 0.880 | 0.620 | 0.006 | 0.704 | 0.321 | 5/10 |
| Kang IFN-β (paired) | v2 | 0.993 | 0.826 | 0.714 | 0.000 | 0.263 | 0.706 | |
| | v4g | 0.996 | 0.848 | 0.735 | 0.000 | 0.143 | 0.790 | 0/10 |
| Zhang (null) | v2 | 0.996 | 0.700 | – | 0.005 | – | 0.121 | |
| | v4g | 0.993 | 0.607 | – | 0.012 | – | 0.137 | 1/10 |
| Parse GM-CSF, IFN-β, IFN-γ, IL-4, TNF | v2 = v4g | 0.996–1.000 | 0.779–0.919 | 0.694–0.967 | 0.000 | 0.025–0.100 | 0.137–0.465 | 0/10 each |

Pooled differences, gated v4 − v2 [95% cluster bootstrap CI]:

| | All (13) | Cell lines (6) | PBMC (7) |
|---|---|---|---|
| Coverage | +0.002 [−0.010, +0.014] | +0.004 [−0.022, +0.027] | +0.000 [−0.003, +0.003] |
| r | +0.030 [−0.012, +0.081] | +0.078 [+0.001, +0.157] | −0.010 [−0.044, +0.011] |
| Within-program r | +0.096 [−0.005, +0.215] | +0.172 [+0.007, +0.324] | +0.004 [+0.000, +0.019] |
| Null calls | +0.012 [+0.000, +0.027] | +0.024 [+0.001, +0.053] | +0.001 [+0.000, +0.004] |
| Power | +0.045 [−0.027, +0.145] | +0.117 [−0.022, +0.295] | −0.017 [−0.054, +0.000] |

Study-level sensitivity (descriptive): McFarland −0.031, Tahoe +0.132, Kang +0.022, Zhang −0.093, Parse 0.000.
The study-weighted mean is +0.006, and the exact one-sided sign-flip p is 0.44 over 5 studies. The gain of gated
v4 rests on Tahoe.

**Reading:**
- The gate behaved as on development: v4 was chosen on McFarland and Tahoe, never on Parse, and once on Zhang.
- Where it switched on, gated v4 raised accuracy and power on Tahoe:
  - encorafenib power 0.03 → 0.52;
  - gemcitabine power 0.00 → 0.25.
- It also called more null pairs on McFarland, above the 0.10 limit. That is the failure the safety criteria
  were written to catch.

## PALM (freeze v2) vs current practice

Current practice means deconvolution, then a per-type score, then a t-test with BH q < 0.1. Differences are
PALM − tool, paired over dataset × split [95% bootstrap CI]. Positive coverage and AUROC differences favour
PALM; negative differences in null calls and power mean PALM is lower.

| Tool | Coverage | Null-call rate | Power (true changes ≥ 0.1) | Responder AUROC |
|---|---|---|---|---|
| InstaPrism | +0.579 [+0.533, +0.624] | −0.529 [−0.583, −0.476] | −0.418 [−0.518, −0.324] | +0.063 [+0.007, +0.130] |
| CIBERSORT | +0.556 [+0.510, +0.603] | −0.521 [−0.572, −0.469] | −0.418 [−0.518, −0.324] | +0.078 [+0.011, +0.154] |
| NNLS | +0.707 [+0.678, +0.735] | −0.409 [−0.460, −0.357] | −0.301 [−0.416, −0.191] | +0.106 [+0.023, +0.199] |
| Bulk score | +0.604 [+0.558, +0.648] | −0.566 [−0.626, −0.505] | −0.418 [−0.518, −0.324] | +0.335 [+0.278, +0.387] |
| TOAST (4 datasets) | – | −0.111 [−0.146, −0.075] | −0.281 [−0.422, −0.150] | +0.133 [+0.078, +0.189] |
| TCA (4 datasets) | – | −0.206 [−0.250, −0.165] | −0.150 [−0.225, −0.083] | +0.000 |

Per dataset, PALM's null-call rate is 0.00–0.06. With InstaPrism or CIBERSORT and a t-test it is 0.25–0.88,
except 0.00 on Zhang and Parse TNF.

PALM finds fewer real changes:
- Tahoe encorafenib: 0.05 (tools 1.00);
- Tahoe gemcitabine: 0.00 (tools 0.94);
- Kang: 0.22 (tools 1.00).

The tools' near-complete power comes with calling most unchanged pairs. PALM ranks responders better (AUROC
higher than every tool except TCA).

## COMBAT sepsis vs healthy (real bulk, splits 40–49)

| Bulk | Estimator | Within-program r | r | Calls | Null calls | Wrong sign | Coverage |
|---|---|---|---|---|---|---|---|
| Single-cell pseudobulk | v2 | 0.62 | 0.53 | 1.0 | 0.4 | 0.1 | 0.98 |
| Raw real bulk | v2 | 0.52 | 0.03 | 0.0 | 0.0 | 0.0 | 0.97 |
| Paired-offset real bulk | v2 | 0.62 | 0.48 | 0.3 | 0.1 | 0.0 | 1.00 |
| Self-offset real bulk | v2 | 0.45 | 0.30 | 1.3 | 0.1 | 0.3 | 0.98 |

The gate chose v2 on all 40 COMBAT runs, so gated v4 equals v2 here. On self-offset bulk, v4 stacked alone was
better (r 0.44 vs 0.30), but the gate did not pick it.

The real-bulk conclusion is unchanged from development:
- raw bulk loses the attribution (r 0.03);
- platform offsets restore it (0.30–0.48);
- no estimator makes wrong-sign calls.

## Transport check (descriptive; 9 perturbations never run in development)

Gated v4 − v2 over 9 datasets:
- r: +0.052 [+0.012, +0.101];
- null calls: +0.002 [−0.003, +0.010];
- coverage: +0.004 [−0.002, +0.013].

Per dataset:
- v4 was chosen on Tahoe afatinib, alpelisib and paclitaxel (10/10) and everolimus (6/10), never on bortezomib or
  Parse.
- Freeze v2 coverage is 0.95–1.00 and null calls 0.00–0.085 on every transport dataset.

Parse IL-10 has 9 of 10 splits: only Donor4 has IL-10 samples, and split 41's test donors exclude it.

This covers new perturbations of two benchmark studies, not new studies. Transport to a new laboratory or
tissue stays untested.

## Deviations from the protocol (none affects the method or the decision rule)

1. **The chain script was edited while lanes a–d ran** (the amendment added lane e). Bash read the edited file at
   old offsets, and each lane exited with a syntax error after its dataset loop.
   - All dataset outputs were complete; they were checked file by file per split.
   - Lane d's trailing COMBAT step (`combat_v4.py 40-49`) did not run, so it was run by hand with the same
     command.
2. **Parse IL-10 split 41 has no treated samples.**
   - `causal_loop.py run` now skips a split that lacks either arm instead of crashing.
   - This is a runner change only; IL-10 is reported on 9 splits.
3. **Reporting:** `practice_vs_palm.py summary heldout4` was run with PALM_EST=v4g as the protocol states, and
   also with PALM_EST=auto_v2, because the decision makes freeze v2 the method. The table above uses auto_v2.

## Figures

`docs/figures/heldout/heldout_v4/` (`scripts/experiments/heldout_figures.py 40-49 heldout_v4`, PALM_EST=auto_v2):
- `calling`: null calls and power, PALM vs 6 practice pipelines, 13 datasets;
- `accuracy`: r and coverage, freeze v2, gated v4 and no correction;
- `transport`: the same two panels for the 9 transport perturbations;
- `heatmap`: the summary heatmap.

## Consequences

- **Paper 1:** method = freeze v2. The headline claim is "far fewer wrong-cell calls than deconvolution + t-test,
  with honest intervals." Its conservative side must be shown: lower power on small or shared effects (Tahoe
  encorafenib and gemcitabine, Kang).
- **Gated v4:** secondary. It is reported as a more powerful option on cell-line designs, with its McFarland
  null-call excess stated.
- **Package default:** `program_deconvolution` still defaults to `estimator="v4gated"` (set at the freeze). To
  match the decision it should go back to `"auto"`. That is a code change pending the user's go-ahead.
- **Splits 30–39** stay unused, in reserve.
