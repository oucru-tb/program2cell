"""Gene-resampling uncertainty estimates."""

from __future__ import annotations

import numpy as np

from .attribution import fit_simplex


def bootstrap_attribution(
    program: np.ndarray,
    dictionary: np.ndarray,
    gene_weights: np.ndarray,
    *,
    n_bootstrap: int = 100,
    random_state: int | None = None,
) -> np.ndarray:
    """Resample program genes with replacement and refit state contributions."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    y = np.asarray(program, dtype=float)
    rng = np.random.default_rng(random_state)
    estimates = np.empty((n_bootstrap, dictionary.shape[1]), dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(y), size=len(y))
        estimates[index] = fit_simplex(
            y[sampled],
            dictionary[sampled],
            gene_weights[sampled],
        )
    return estimates
