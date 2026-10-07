import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from program2cell.hdwgcna import discover_modules


def test_runs_py_hdwgcna_and_returns_weighted_modules(monkeypatch):
    rng = np.random.default_rng(12)
    samples = 12
    first_signal = rng.normal(0, 3, samples)
    second_signal = rng.normal(0, 3, samples)
    first = 10 + first_signal + rng.normal(0, 0.1, (12, samples))
    second = 10 + second_signal + rng.normal(0, 0.1, (12, samples))
    expression = pd.DataFrame(
        np.vstack([first, second]),
        index=[f"g{index}" for index in range(24)],
        columns=[f"s{index}" for index in range(samples)],
    )

    calls = []

    class FakeAnnData:
        def __init__(self, X, obs, var):
            self.X = X
            self.obs = obs
            self.var = var

    class FakeHDWGCNA:
        def __init__(self, adata):
            self.adata = adata

        def setup_for_wgcna(self, **kwargs):
            calls.append(("setup_for_wgcna", kwargs))
            self.selected_genes = kwargs["genes_use"]
            return self

        def test_soft_powers(self, **kwargs):
            calls.append(("test_soft_powers", kwargs))
            return self

        def construct_network(self, **kwargs):
            calls.append(("construct_network", kwargs))
            gene_names = self.selected_genes
            tom = np.eye(len(gene_names))
            tom[:12, :12] = 1.0
            tom[12:, 12:] = 1.0
            labels = ["M1"] * 12 + ["M2"] * 12
            colors = ["turquoise"] * 12 + ["blue"] * 12
            assignments = pd.DataFrame(
                {
                    "gene_name": gene_names,
                    "module": labels,
                    "color": colors,
                    "kME_M1": [0.9] * 12 + [0.1] * 12,
                    "kME_M2": [0.1] * 12 + [0.9] * 12,
                }
            )
            self.adata.uns = {
                "hdWGCNA": {
                    "program2cell": {
                        "modules_df": assignments,
                        "dat_expr_genes": gene_names,
                        "TOM": tom,
                    }
                }
            }
            return self

        def module_eigengenes(self, **kwargs):
            calls.append(("module_eigengenes", kwargs))
            return self

        def module_connectivity(self, **kwargs):
            calls.append(("module_connectivity", kwargs))
            return self

    anndata_module = ModuleType("anndata")
    anndata_module.AnnData = FakeAnnData
    hdwgcna_module = ModuleType("py_hdWGCNA")
    hdwgcna_module.HDWGCNA = FakeHDWGCNA
    monkeypatch.setitem(sys.modules, "anndata", anndata_module)
    monkeypatch.setitem(sys.modules, "py_hdWGCNA", hdwgcna_module)

    modules, details = discover_modules(expression, min_module_size=5)

    assert set(modules) == {"turquoise", "blue"}
    assert all(np.isclose(weights.sum(), 1.0) for weights in modules.values())
    assert details["gene"].nunique() == 24
    assert [name for name, _ in calls] == [
        "setup_for_wgcna",
        "test_soft_powers",
        "construct_network",
        "module_eigengenes",
        "module_connectivity",
    ]
    assert calls[2][1]["minModuleSize"] == 5


def test_missing_py_hdwgcna_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "anndata", None)
    monkeypatch.setitem(sys.modules, "py_hdWGCNA", None)
    bulk = pd.DataFrame(
        np.arange(40, dtype=float).reshape(5, 8) + 1,
        index=[f"g{i}" for i in range(5)],
    )
    with pytest.raises(ImportError, match="py-hdwgcna and anndata"):
        discover_modules(bulk, min_module_size=2)
