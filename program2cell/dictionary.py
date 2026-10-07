"""Atlas-derived cell-state specificity dictionary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pseudobulk import pseudobulk_by_state


def build_dictionary(
    expression: pd.DataFrame,
    states: pd.Series,
    *,
    min_cells: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a non-negative TF-IDF specificity dictionary.

    Returns the normalized dictionary and the raw state pseudobulk, each with
    genes in rows and cell states in columns.
    """
    pseudobulk = pseudobulk_by_state(expression, states, min_cells=min_cells)
    values = pseudobulk.to_numpy(dtype=float)
    gene_totals = values.sum(axis=1, keepdims=True)
    tf = np.divide(values, gene_totals, out=np.zeros_like(values), where=gene_totals > 0)
    number_of_states = values.shape[1]
    detected_states = (values > 0).sum(axis=1)
    idf = np.log((number_of_states + 1.0) / (detected_states + 1.0)) + 1.0
    weighted = tf * idf[:, None]
    keep = weighted.sum(axis=1) > 0
    weighted = weighted[keep]
    genes = pseudobulk.index[keep]
    # Unit L1 columns make coefficients interpretable as mixture proportions.
    column_sums = weighted.sum(axis=0, keepdims=True)
    weighted = np.divide(weighted, column_sums, out=np.zeros_like(weighted), where=column_sums > 0)
    if (column_sums <= 0).any():
        empty = pseudobulk.columns[np.flatnonzero(column_sums.ravel() <= 0)].tolist()
        raise ValueError(f"Cell states have no informative expression: {empty}")
    dictionary = pd.DataFrame(weighted, index=genes, columns=pseudobulk.columns)
    return dictionary, pseudobulk
