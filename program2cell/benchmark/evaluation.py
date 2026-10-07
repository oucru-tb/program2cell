"""Comparison of estimated and known state contributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def evaluate_attribution(predicted, truth) -> dict[str, float]:
    """Return RMSE, MAE, and Pearson correlation for aligned state proportions."""
    predicted = pd.Series(predicted, dtype=float)
    truth = pd.Series(truth, dtype=float)
    states = truth.index.union(predicted.index)
    p = predicted.reindex(states, fill_value=0.0).to_numpy()
    y = truth.reindex(states, fill_value=0.0).to_numpy()
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Predicted and true contributions must be finite.")
    correlation = float(pearsonr(y, p).statistic) if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
        "correlation": correlation,
    }
