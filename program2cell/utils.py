"""Shared input validation and expression-matrix conversion helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


def _as_frame(matrix: Any, *, genes_in_rows: bool, name: str) -> pd.DataFrame:
    """Convert a DataFrame or AnnData-like object to genes-by-observations."""
    if isinstance(matrix, pd.DataFrame):
        frame = matrix.copy()
        if not genes_in_rows:
            frame = frame.T
    elif hasattr(matrix, "X") and hasattr(matrix, "var_names"):
        values = matrix.X
        if sparse.issparse(values):
            values = values.toarray()
        frame = pd.DataFrame(
            np.asarray(values),
            index=getattr(matrix, "obs_names", None),
            columns=matrix.var_names,
        ).T
    else:
        raise TypeError(f"{name} must be a pandas DataFrame or AnnData-like object.")

    if frame.empty:
        raise ValueError(f"{name} must contain at least one gene and observation.")
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    if (frame.index == "").any() or (frame.columns == "").any():
        raise ValueError(f"{name} gene and observation identifiers must be non-empty.")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} contains duplicate observation names.")
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} expression values must be numeric.") from exc
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite expression values.")
    if (values < 0).any():
        raise ValueError(f"{name} must contain non-negative expression values.")

    # Duplicate gene identifiers are combined instead of silently dropping data.
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0, sort=False).mean()
    return frame


def bulk_to_frame(bulk: Any) -> pd.DataFrame:
    """Return bulk expression as genes x samples."""
    return _as_frame(bulk, genes_in_rows=True, name="bulk")


def atlas_to_frame(atlas: Any, state_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return single-cell expression as genes x cells and aligned cell states."""
    if not hasattr(atlas, "obs") or not hasattr(atlas, "X") or not hasattr(atlas, "var_names"):
        raise TypeError("atlas must be an AnnData-like object with X, obs, and var_names.")
    if state_col not in atlas.obs:
        raise ValueError(f"atlas.obs does not contain the required column {state_col!r}.")
    states = pd.Series(atlas.obs[state_col].to_numpy(), index=atlas.obs_names, name=state_col)
    if states.isna().any():
        raise ValueError(f"atlas.obs[{state_col!r}] contains missing cell states.")
    frame = _as_frame(atlas, genes_in_rows=True, name="atlas")
    if len(states) != frame.shape[1]:
        raise ValueError("Atlas state labels are not aligned with expression cells.")
    states.index = frame.columns
    states = states.astype(str)
    if not states.str.len().gt(0).all():
        raise ValueError("Atlas cell states must be non-empty strings.")
    return frame, states


def normalize_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    """Normalize non-negative vectors across rows, rejecting empty vectors."""
    sums = values.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        raise ValueError(f"{name} contains a vector with no positive mass.")
    return values / sums
