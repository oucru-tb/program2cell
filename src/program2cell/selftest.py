"""In-study self-tests that need no ground truth (v4 development, idea register 3 and 4).

Both replace the programs with test programs whose answer is known by construction and rerun the whole
analysis (gene selection included) on the same bulk and reference:

- null programs (null_programs): each real program's gene weights moved onto genes outside every real
  program, matched gene by gene on mean bulk expression (bins of log CPM), and not specific to one cell type.
  No program is active there, so their call rate is an in-study false-call rate. Caveat: genes outside the
  programs can still respond to the treatment (the off-program response); the rate is then an upper bound,
  and the truth (where there is one) shows by how much.
- marker programs (marker_programs): one program per cell type from genes expressed (almost) only in that
  type in the reference. Their bulk signal is close to pure composition, so an activity change reported for
  *another* type is leakage: a change in abundance read as a change in activity.

Usage: for each test program table, prepare_inputs(bulk, ref, table, ...) and a calibration on the new
inputs' genes and programs, run the analysis, then null_summary / marker_summary.
"""
import numpy as np
import pandas as pd


def _ref_shares(ref, genes):
    gi = {g: i for i, g in enumerate(ref.genes)}
    B = np.asarray(ref.B, float)[:, [gi[g] for g in genes]]
    return B / B.sum(1, keepdims=True)


def null_programs(programs, bulk, ref, n_sets=3, n_bins=20, max_specificity=1.0, seed=0):
    """n_sets program tables (DataFrames, programs x genes): each real program's nonzero weights moved, gene by
    gene, to a random gene of the same bulk-expression bin that is in the bulk and the reference, outside every
    real program, and less than e^max_specificity-fold above the runner-up cell type (weights and signs kept;
    no gene used twice within a program)."""
    rng = np.random.default_rng(seed)
    inb = set(bulk.index)
    genes = [g for g in ref.genes if g in inb]
    cpm = bulk.reindex(genes).values.astype(float)
    m = np.log((cpm / cpm.sum(0, keepdims=True)).mean(1) + 1e-9)
    B = _ref_shares(ref, genes)
    Bs = np.sort(B, 0)
    spec = np.log(Bs[-1] + 1e-12) - np.log(Bs[-2] + 1e-12)
    P = programs.reindex(columns=genes).fillna(0.0)
    inprog = (P.abs() > 0).any(axis=0).values
    expressed = m > np.quantile(m, 0.25)
    free = ~inprog & (spec < max_specificity) & expressed
    edges = np.quantile(m[expressed], np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(m, edges)
    pools = {b: np.flatnonzero(free & (bins == b)) for b in np.unique(bins)}
    out = []
    for s in range(n_sets):
        rows = {}
        for name, row in P.iterrows():
            w = np.zeros(len(genes))
            for g in np.flatnonzero(row.values):
                near = sorted(pools, key=lambda b: abs(b - bins[g]))
                for b in near:                          # same bin, else the nearest with an unused free gene
                    cand = pools[b][w[pools[b]] == 0]
                    if len(cand):
                        w[rng.choice(cand)] = row.values[g]
                        break
            rows["null%d_%s" % (s, name)] = w
        out.append(pd.DataFrame(rows, index=genes).T)
    return out


def marker_programs(ref, bulk=None, n_genes=30, min_specificity=2.0):
    """Program table (cell types x genes) of marker programs: per type the n_genes genes (in the bulk, if
    given) expressed above the type's median with the highest log(B_k / max_{c != k} B_c), at least
    min_specificity (e^2 ~ 7-fold), weight 1. Program names 'marker_<type>'."""
    inb = None if bulk is None else set(bulk.index)
    genes = [g for g in ref.genes if inb is None or g in inb]
    B = _ref_shares(ref, genes)
    rows = {}
    for k, c in enumerate(ref.cell_types):
        other = np.delete(B, k, 0).max(0)
        spec = np.log(B[k] + 1e-12) - np.log(other + 1e-12)
        spec[B[k] <= np.median(B[k][B[k] > 0])] = -np.inf
        top = np.argsort(-spec)[:n_genes]
        top = top[spec[top] >= min_specificity]
        if len(top) >= 5:
            w = np.zeros(len(genes))
            w[top] = 1.0
            rows["marker_%s" % c] = w
    return pd.DataFrame(rows, index=genes).T


def marker_owner(program_names, cell_types):
    """Index of the owning cell type of each marker program."""
    return np.array([list(cell_types).index(n[len("marker_"):]) for n in program_names])


def null_summary(calls, effects):
    """In-study false-call rate and mean |effect| on null programs (all cell types x programs)."""
    calls, effects = np.asarray(calls, bool), np.asarray(effects, float)
    return dict(null_call_rate=float(calls.mean()), null_mean_abs=float(np.abs(effects).mean()))


def marker_summary(calls, effects, owner):
    """Leakage: calls and mean |effect| for marker programs in cell types other than their owner
    (effects (C, P_marker), owner (P_marker,) the owning type of each program)."""
    calls, effects = np.asarray(calls, bool), np.asarray(effects, float)
    C, P = effects.shape
    off = np.ones((C, P), bool)
    off[np.asarray(owner), np.arange(P)] = False
    return dict(leak_call_rate=float(calls[off].mean()), leak_mean_abs=float(np.abs(effects[off]).mean()),
                own_call_rate=float(calls[~off].mean()), own_mean_abs=float(np.abs(effects[~off]).mean()))
