"""Plotting helpers."""

from __future__ import annotations


def plot_attribution(result, module: str, *, ax=None):
    """Plot a result object's cell-state contributions for a module."""
    selected = result.attribution.loc[result.attribution["module"] == module]
    if selected.empty:
        raise KeyError(f"Unknown module {module!r}.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install program2cell[plot] to use plotting helpers.") from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(selected))))
    selected = selected.sort_values("contribution")
    ax.barh(selected["state"], selected["contribution"])
    ax.set(xlabel="Contribution", ylabel="Cell state", title=f"{module} attribution")
    ax.set_xlim(0, 1)
    return ax
