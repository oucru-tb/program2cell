"""Reconstruction and attribution quality metrics."""

from __future__ import annotations

import numpy as np


def calculate_metrics(program, reconstructed, contribution) -> dict[str, float]:
    """Calculate weighted-independent reconstruction metrics for a module."""
    y = np.asarray(program, dtype=float)
    estimate = np.asarray(reconstructed, dtype=float)
    alpha = np.asarray(contribution, dtype=float)
    residual = y - estimate
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(residual @ residual) / total if total > 0 else float("nan")
    denominator = float(np.linalg.norm(y) * np.linalg.norm(estimate))
    cosine = float(y @ estimate) / denominator if denominator > 0 else float("nan")
    positive = alpha[alpha > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return {
        "r2": r_squared,
        "cosine_similarity": cosine,
        "attribution_entropy": entropy,
    }
