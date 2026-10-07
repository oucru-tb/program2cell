# Program2Cell

**Cell-state attribution of bulk-derived gene programs.** Program2Cell finds
co-expression programs in bulk RNA-seq and estimates which atlas cell states
best explain each program. It is intended for program interpretation, not cell
proportion deconvolution.

## Installation

```bash
pip install .
```

Module discovery uses [py-hdWGCNA](https://github.com/omicverse/py-hdWGCNA)
directly. Install it as a package dependency (included by `pip install .`).
AnnData is consumed using its standard `X`, `obs`, `obs_names`, and `var_names`
attributes. Plotting is optional: `pip install ".[plot]"`.

## Quick start

```python
from program2cell import Program2Cell

model = Program2Cell(state_col="cell_state", n_bootstrap=100, random_state=7)
model.fit(atlas, bulk)  # atlas: AnnData; bulk: genes x samples DataFrame or AnnData
results = model.run()

results.attribution
#       module              state  contribution
# 0  turquoise        IFN_Monocyte          0.44

results.ci       # bootstrap mean and 95% interval for each module/state
results.metrics  # reconstruction R2, cosine similarity, attribution entropy
results.modules  # weighted genes comprising each bulk-derived module

model.plot_module("turquoise", result=results)
```

Bulk DataFrames must have genes in rows and samples in columns. AnnData matrices
are interpreted using `var_names` as genes and `obs` as observations. The atlas
must contain `atlas.obs["cell_state"]`. Expression values must be finite and
non-negative; raw counts and common non-negative normalized/log-transformed
matrices are accepted.

## Analysis

Atlas cells are averaged within each state to produce pseudobulk profiles.
Gene-state specificity is calculated as within-gene expression share multiplied
by a smoothed inverse-state-frequency weight, then normalized per state. Bulk
modules are found from a signed, soft-thresholded co-expression network and
hierarchical clustering. Within each module, gene weights are normalized
`|kME| * intramodular connectivity`.

For each module, a weighted non-negative least-squares fit constrained to sum
to one estimates cell-state contributions. Gene bootstrap resampling supplies
percentile confidence intervals. Module quality includes unweighted
reconstruction R2, cosine similarity, and Shannon attribution entropy.

Module discovery runs py-hdWGCNA's `setup_for_wgcna`,
`test_soft_powers`, `construct_network`, `module_eigengenes`, and
`module_connectivity` steps on an AnnData view of the bulk samples. No
single-cell metacell step is applied to bulk observations. Program2Cell derives
within-module connectivity from the returned TOM and uses it with absolute kME
to construct normalized gene weights. Module discovery requires at least four
bulk samples.

## Benchmark helpers

```python
from program2cell.benchmark import simulate_bulk, evaluate_attribution

synthetic_bulk, truth = simulate_bulk(
    model.pseudobulk_,
    {"IFN_Monocyte": 0.6, "Activated_NK": 0.4},
    n_samples=20,
    noise=0.1,
)
scores = evaluate_attribution(predicted_contributions, truth)
```

## Limitations

Attribution is conditional on the selected atlas, input gene identifiers, and
the bulk programs found from the available samples. It is an interpretation
framework, not proof of causal cellular origin. Very similar cell-state
profiles can be difficult to distinguish and should be interpreted with the
confidence intervals and reconstruction metrics.
