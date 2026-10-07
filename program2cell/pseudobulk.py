"""Cell-state pseudobulk generation."""

from __future__ import annotations

import pandas as pd


def pseudobulk_by_state(
    expression: pd.DataFrame,
    states: pd.Series,
    *,
    min_cells: int = 1,
) -> pd.DataFrame:
    """Compute mean expression per state (genes x cell states)."""
    if min_cells < 1:
        raise ValueError("min_cells must be at least 1.")
    if not expression.columns.equals(states.index):
        raise ValueError("State labels must be aligned to expression columns.")
    counts = states.value_counts()
    retained = counts[counts >= min_cells].index
    if len(retained) == 0:
        raise ValueError(f"No cell state contains at least {min_cells} cells.")
    selected = expression.loc[:, states.isin(retained)]
    labels = states.loc[selected.columns]
    result = selected.T.groupby(labels, sort=True).mean().T
    if result.shape[1] < 2:
        raise ValueError("At least two cell states are required for attribution.")
    return result
