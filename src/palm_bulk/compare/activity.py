"""Turn other methods' outputs into program activity and treatment effects comparable with PALM's.

Two routes:
  * 'delta' (default): palm_bulk.estimate_delta on each per-sample per-cell-type profile, i.e. the
    model's own definition. Values are comparable with the truth and with PALM.
  * 'ulm': decoupler's univariate linear model on log-CPM of each profile, the usual
    "deconvolve, then PROGENy" pipeline. A different scale: compare by rank.
"""
import numpy as np
import pandas as pd

from ..core.scores import estimate_delta


def activity_from_expression(type_expr, inputs, how="delta", programs=None, scale_to=1e6):
    """type_expr: (N, C, G) non-negative per-type expression on inputs.genes (nan rows allowed for
    cell types a method could not estimate; genes a method dropped, i.e. nan everywhere, are left
    out). Returns ((N, C, P) activity, scale)."""
    X = np.asarray(type_expr, float)
    keep = ~np.isnan(X).all((0, 1))                                             # genes the method returned
    X = X[..., keep]
    B, Gm, genes = inputs.B[:, keep], inputs.Gmat[:, keep], list(np.asarray(inputs.genes)[keep])
    bad = ~np.isfinite(X).all(-1) | (np.nan_to_num(X).sum(-1) <= 0)
    X = np.where(bad[..., None], 0.0, np.clip(np.nan_to_num(X), 0, None))
    if how == "delta":
        act = estimate_delta(X, B, Gm, scale_to=scale_to)
        return np.where(bad[..., None], np.nan, act), "delta"
    if how == "ulm":
        try:
            import decoupler as dc
        except ImportError as e:
            raise ImportError("activity='ulm' needs decoupler: pip install 'palm-bulk[progeny]'") from e
        W = programs if programs is not None else pd.DataFrame(Gm, index=inputs.programs, columns=genes)
        W = W.reindex(index=inputs.programs, columns=genes).fillna(0.0)
        net = W.stack().rename("weight").reset_index()
        net.columns = ["source", "target", "weight"]
        net = net[net.weight != 0]
        N, C, _ = X.shape
        act = np.full((N, C, len(inputs.programs)), np.nan)
        for c in range(C):
            ok = ~bad[:, c]
            if not ok.any():
                continue
            cpm = X[ok, c] / X[ok, c].sum(-1, keepdims=True) * 1e6
            df = pd.DataFrame(np.log1p(cpm), columns=genes)
            est, _ = dc.mt.ulm(df, net, tmin=3)
            act[ok, c] = est.reindex(columns=inputs.programs).values
        return act, "score"
    raise ValueError("how must be 'delta' or 'ulm'")


def effects_from_activity(activity, treatment):
    """(C, P) difference in mean activity, treated minus control (the model's beta in expectation)."""
    T = np.asarray(treatment, float)
    return np.nanmean(activity[T == 1], 0) - np.nanmean(activity[T == 0], 0)


def effects_from_logfc(lfc, inputs):
    """Per-cell-type gene log-fold-changes (C, G) on inputs.genes -> (C, P) program effects, by least
    squares of lfc_cg on [1, G_pg] across genes. The intercept absorbs the renormalisation of the
    profile, as in the model (log E shifts by delta.G minus a per-cell-type constant)."""
    lfc = np.asarray(lfc, float)
    A = np.column_stack([np.ones(inputs.Gmat.shape[1]), inputs.Gmat.T])                  # (G, 1+P)
    out = np.full((lfc.shape[0], inputs.Gmat.shape[0]), np.nan)
    for c in range(lfc.shape[0]):
        ok = np.isfinite(lfc[c])
        if ok.sum() > A.shape[1]:
            out[c] = np.linalg.lstsq(A[ok], lfc[c, ok], rcond=None)[0][1:]
    return out
