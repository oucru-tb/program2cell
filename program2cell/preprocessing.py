"""Input preparation and gene alignment."""

from __future__ import annotations

from .utils import atlas_to_frame, bulk_to_frame


def prepare_inputs(atlas, bulk, state_col: str):
    """Validate both inputs and retain the genes shared by atlas and bulk."""
    atlas_expr, states = atlas_to_frame(atlas, state_col)
    bulk_expr = bulk_to_frame(bulk)
    shared = atlas_expr.index.intersection(bulk_expr.index, sort=False)
    if len(shared) < 2:
        raise ValueError("Atlas and bulk must share at least two gene identifiers.")
    atlas_expr = atlas_expr.loc[shared]
    bulk_expr = bulk_expr.loc[shared]
    if (atlas_expr.sum(axis=0) == 0).all():
        raise ValueError("The atlas has no positive expression in the shared genes.")
    if (bulk_expr.sum(axis=0) == 0).any():
        raise ValueError("Each bulk sample must have positive expression in shared genes.")
    states = states.loc[atlas_expr.columns]
    return atlas_expr, states, bulk_expr
