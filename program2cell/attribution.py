"""Non-negative simplex-constrained program attribution."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def fit_simplex(
    program: np.ndarray,
    dictionary: np.ndarray,
    gene_weights: np.ndarray,
) -> np.ndarray:
    """Fit state proportions by weighted least squares on the probability simplex."""
    y = np.asarray(program, dtype=float)
    d = np.asarray(dictionary, dtype=float)
    weights = np.asarray(gene_weights, dtype=float)
    if y.ndim != 1 or d.ndim != 2 or d.shape[0] != y.size:
        raise ValueError("Program and dictionary dimensions are inconsistent.")
    if weights.shape != y.shape:
        raise ValueError("Gene weights must match the program vector.")
    if d.shape[1] < 1:
        raise ValueError("The dictionary must contain at least one cell state.")
    if not np.isfinite(y).all() or not np.isfinite(d).all() or not np.isfinite(weights).all():
        raise ValueError("Attribution inputs must be finite.")
    if (y < 0).any() or (d < 0).any() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Attribution inputs and gene weights must be non-negative.")

    root_weights = np.sqrt(weights / weights.sum())
    weighted_d = d * root_weights[:, None]
    weighted_y = y * root_weights
    initial = np.full(d.shape[1], 1.0 / d.shape[1])

    def objective(alpha):
        residual = weighted_y - weighted_d @ alpha
        return float(residual @ residual)

    def gradient(alpha):
        return 2.0 * weighted_d.T @ (weighted_d @ alpha - weighted_y)

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * d.shape[1],
        constraints=[{"type": "eq", "fun": lambda alpha: alpha.sum() - 1.0,
                      "jac": lambda alpha: np.ones_like(alpha)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Simplex attribution optimization failed: {result.message}")
    alpha = np.clip(result.x, 0.0, 1.0)
    return alpha / alpha.sum()
