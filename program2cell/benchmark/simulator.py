"""Pseudo-bulk mixtures and controlled noise simulations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_bulk(
    pseudobulk: pd.DataFrame,
    contributions: dict[str, float] | pd.Series,
    *,
    n_samples: int = 1,
    noise: float = 0.0,
    random_state: int | None = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate genes x samples pseudo-bulk mixtures with optional noise.

    Noise is Gaussian with standard deviation ``noise`` times the mean absolute
    mixture expression. Generated values are clipped to zero to preserve the
    non-negative expression contract.
    """
    if n_samples < 1 or noise < 0:
        raise ValueError("n_samples must be positive and noise must be non-negative.")
    weights = pd.Series(contributions, dtype=float)
    if weights.empty or not np.isfinite(weights.to_numpy()).all() or (weights < 0).any():
        raise ValueError("Contributions must be finite, non-negative, and non-empty.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Ground-truth contributions must sum to one.")
    missing = weights.index.difference(pseudobulk.columns)
    if len(missing):
        raise ValueError(f"Unknown pseudobulk states: {missing.tolist()}")
    mixture = pseudobulk.loc[:, weights.index].to_numpy(dtype=float) @ weights.to_numpy()
    rng = np.random.default_rng(random_state)
    scale = float(np.mean(np.abs(mixture)))
    values = np.repeat(mixture[:, None], n_samples, axis=1)
    if noise:
        values += rng.normal(0.0, noise * scale, size=values.shape)
    values = np.clip(values, 0.0, None)
    samples = [f"sample_{index + 1}" for index in range(n_samples)]
    return (
        pd.DataFrame(values, index=pseudobulk.index, columns=samples),
        weights.rename("contribution"),
    )
