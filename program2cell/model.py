"""Public Program2Cell estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .attribution import fit_simplex
from .bootstrap import bootstrap_attribution
from .dictionary import build_dictionary
from .hdwgcna import discover_modules
from .metrics import calculate_metrics
from .preprocessing import prepare_inputs
from .result import Program2CellResult


class Program2Cell:
    """Attribute bulk co-expression programs to atlas-defined cell states.

    Parameters
    ----------
    state_col:
        AnnData ``obs`` column defining atlas cell states.
    min_cells:
        Minimum number of atlas cells required to retain a state.
    n_bootstrap:
        Number of gene-resampling replicates used for confidence intervals.
    random_state:
        Seed for reproducible bootstrap estimates.
    """

    def __init__(
        self,
        state_col: str = "cell_state",
        *,
        min_cells: int = 10,
        n_bootstrap: int = 100,
        random_state: int | None = 0,
    ):
        if not state_col:
            raise ValueError("state_col must be a non-empty column name.")
        if min_cells < 1:
            raise ValueError("min_cells must be at least 1.")
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be at least 1.")
        self.state_col = state_col
        self.min_cells = min_cells
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self._fitted = False

    def fit(self, atlas, bulk) -> "Program2Cell":
        """Validate inputs, build the cell-state dictionary, and discover modules."""
        atlas_expr, states, bulk_expr = prepare_inputs(atlas, bulk, self.state_col)
        dictionary, pseudobulk = build_dictionary(
            atlas_expr, states, min_cells=self.min_cells
        )
        modules, module_details = discover_modules(bulk_expr)
        if hasattr(self, "results_"):
            del self.results_
        self.dictionary_ = dictionary
        self.pseudobulk_ = pseudobulk
        self.modules_ = modules
        self.module_details_ = module_details
        self.bulk_ = bulk_expr
        self._fitted = True
        return self

    def run(self) -> Program2CellResult:
        """Attribute every discovered bulk program and estimate uncertainty."""
        if not self._fitted:
            raise RuntimeError("Call fit(atlas, bulk) before run().")
        attribution_rows: list[dict] = []
        ci_rows: list[dict] = []
        metric_rows: list[dict] = []
        program_rows: list[dict] = []
        weights_rows: list[dict] = []
        shared_dictionary_genes = self.dictionary_.index

        for module, full_weights in self.modules_.items():
            genes = full_weights.index.intersection(shared_dictionary_genes, sort=False)
            if len(genes) < 2:
                raise ValueError(
                    f"Module {module!r} has fewer than two genes represented in the atlas."
                )
            weights = full_weights.loc[genes].to_numpy(dtype=float)
            weights = weights / weights.sum()
            dictionary = self.dictionary_.loc[genes].to_numpy(dtype=float)
            program = weights.copy()
            alpha = fit_simplex(program, dictionary, weights)
            reconstruction = dictionary @ alpha
            metrics = calculate_metrics(program, reconstruction, alpha)
            bootstrap = bootstrap_attribution(
                program,
                dictionary,
                weights,
                n_bootstrap=self.n_bootstrap,
                random_state=self._module_seed(module),
            )
            for state, contribution in zip(self.dictionary_.columns, alpha):
                attribution_rows.append(
                    {"module": module, "state": state, "contribution": float(contribution)}
                )
            for index, state in enumerate(self.dictionary_.columns):
                ci_rows.append(
                    {
                        "module": module,
                        "state": state,
                        "mean": float(bootstrap[:, index].mean()),
                        "lower": float(np.quantile(bootstrap[:, index], 0.025)),
                        "upper": float(np.quantile(bootstrap[:, index], 0.975)),
                    }
                )
            metric_rows.append({"module": module, **metrics})
            program_rows.extend(
                {"module": module, "gene": gene, "weight": float(weight)}
                for gene, weight in zip(genes, weights)
            )
            weights_rows.extend(
                {"module": module, "gene": gene, "gene_weight": float(weight)}
                for gene, weight in zip(genes, weights)
            )

        modules_table = pd.DataFrame(program_rows)
        attribution = pd.DataFrame(attribution_rows)
        ci = pd.DataFrame(ci_rows)
        metrics = pd.DataFrame(metric_rows).set_index("module")
        gene_weights = pd.DataFrame(weights_rows)
        result = Program2CellResult(
            modules=modules_table,
            attribution=attribution,
            ci=ci,
            metrics=metrics,
            dictionary=self.dictionary_.copy(),
            gene_weights=gene_weights,
        )
        self.results_ = result
        return result

    def _module_seed(self, module: str) -> int | None:
        if self.random_state is None:
            return None
        # Stable across Python processes (unlike hash()).
        offset = sum((index + 1) * ord(char) for index, char in enumerate(module))
        return (int(self.random_state) + offset) % (2**32)

    def plot_module(self, module: str, *, result: Program2CellResult | None = None, ax=None):
        """Plot state contributions for one module and return the Matplotlib axis."""
        if result is None:
            result = getattr(self, "results_", None)
        if result is None:
            raise RuntimeError("Run the model first or pass a result to plot_module().")
        selected = result.attribution.loc[result.attribution["module"] == module]
        if selected.empty:
            raise KeyError(f"Unknown module {module!r}.")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install program2cell[plot] to use plot_module().") from exc
        if ax is None:
            _, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(selected))))
        selected = selected.sort_values("contribution")
        ax.barh(selected["state"], selected["contribution"])
        ax.set(xlabel="Contribution", ylabel="Cell state", title=f"{module} attribution")
        ax.set_xlim(0, 1)
        return ax

    def fit_run(self, atlas, bulk) -> Program2CellResult:
        """Convenience method equivalent to ``fit(...).run()``."""
        result = self.fit(atlas, bulk).run()
        self.results_ = result
        return result
