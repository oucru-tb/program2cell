"""Bulk co-expression module discovery using the py-hdWGCNA package."""

from __future__ import annotations

import numpy as np
import pandas as pd


def discover_modules(
    bulk: pd.DataFrame,
    *,
    min_module_size: int = 5,
    max_genes: int = 2500,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Run py-hdWGCNA on bulk samples and return weighted gene modules.

    Bulk samples are represented as AnnData observations and genes as variables,
    which lets py-hdWGCNA run its native setup, soft-power, network, eigengene,
    and connectivity pipeline without the single-cell metacell workflow.
    """
    if bulk.shape[1] < 4:
        raise ValueError("At least four bulk samples are required for module discovery.")
    if min_module_size < 2 or max_genes < min_module_size:
        raise ValueError("Invalid module discovery settings.")

    variances = bulk.var(axis=1, ddof=1)
    variable = variances[variances > 0].sort_values(ascending=False)
    selected_genes = variable.index[:max_genes].tolist()
    if len(selected_genes) < min_module_size:
        raise ValueError("Too few variable shared genes to discover co-expression modules.")

    try:
        import anndata as ad
        from py_hdWGCNA import HDWGCNA
    except ImportError as exc:
        raise ImportError(
            "Module discovery requires py-hdwgcna and anndata. "
            "Install Program2Cell with its declared dependencies."
        ) from exc

    bulk_adata = ad.AnnData(
        X=bulk.loc[selected_genes].T.to_numpy(dtype=np.float64),
        obs=pd.DataFrame(index=bulk.columns),
        var=pd.DataFrame(index=pd.Index(selected_genes, name="gene")),
    )
    wgcna_name = "program2cell"
    hdw = HDWGCNA(bulk_adata)
    (
        hdw.setup_for_wgcna(
            gene_select="custom",
            genes_use=selected_genes,
            wgcna_name=wgcna_name,
        )
        .test_soft_powers(
            power_range=list(range(1, 11)),
            network_type="signed",
            cor_method="pearson",
            wgcna_name=wgcna_name,
        )
        .construct_network(
            network_type="signed",
            tom_type="signed",
            minModuleSize=min_module_size,
            mergeCutHeight=0.2,
            wgcna_name=wgcna_name,
        )
        .module_eigengenes(
            harmonize=False,
            wgcna_name=wgcna_name,
        )
        .module_connectivity(
            cor_method="pearson",
            sparse=False,
            wgcna_name=wgcna_name,
        )
    )

    try:
        wgcna = hdw.adata.uns["hdWGCNA"][wgcna_name]
        assignments = wgcna["modules_df"].copy()
        genes = list(wgcna["dat_expr_genes"])
        tom = np.asarray(wgcna["TOM"], dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "py-hdWGCNA completed without the expected module assignments and TOM."
        ) from exc

    if len(genes) != len(assignments) or tom.shape != (len(genes), len(genes)):
        raise RuntimeError("py-hdWGCNA returned inconsistent gene assignment/TOM dimensions.")
    required_columns = {"gene_name", "module", "color"}
    if not required_columns.issubset(assignments.columns):
        missing = sorted(required_columns.difference(assignments.columns))
        raise RuntimeError(f"py-hdWGCNA module assignments are missing columns: {missing}")

    gene_positions = {gene: index for index, gene in enumerate(genes)}
    module_weights: dict[str, pd.Series] = {}
    detail_rows: list[dict] = []
    non_grey = assignments.loc[assignments["color"] != "grey"]
    for color, color_rows in non_grey.groupby("color", sort=False):
        module_label = str(color_rows["module"].iloc[0])
        kme_column = f"kME_{module_label}"
        if kme_column not in color_rows.columns:
            raise RuntimeError(
                f"py-hdWGCNA did not return eigengene connectivity column {kme_column!r}."
            )
        module_genes = color_rows["gene_name"].astype(str).tolist()
        positions = np.array([gene_positions[gene] for gene in module_genes], dtype=int)
        module_tom = tom[np.ix_(positions, positions)].copy()
        np.fill_diagonal(module_tom, 0.0)
        intramodular_connectivity = module_tom.sum(axis=1)
        membership = color_rows[kme_column].to_numpy(dtype=float)
        weights = np.abs(membership) * intramodular_connectivity
        weights[~np.isfinite(weights)] = 0.0
        if weights.sum() <= 0:
            raise RuntimeError(f"py-hdWGCNA produced no valid gene weights for module {color!r}.")
        weights /= weights.sum()
        module_weights[str(color)] = pd.Series(
            weights, index=module_genes, name="gene_weight"
        )
        detail_rows.extend(
            {
                "module": str(color),
                "gene": gene,
                "kME": float(kme),
                "connectivity": float(connectivity),
                "gene_weight": float(weight),
            }
            for gene, kme, connectivity, weight in zip(
                module_genes, membership, intramodular_connectivity, weights
            )
        )

    if not module_weights:
        raise ValueError(
            "py-hdWGCNA did not identify any non-grey modules; provide more bulk "
            "samples or a richer expression matrix."
        )
    return module_weights, pd.DataFrame(detail_rows)
