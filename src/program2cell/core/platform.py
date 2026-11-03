"""Platform gap between real bulk RNA-seq and a single-cell reference (docs/benchmarking.md, COMBAT).

Real bulk differs from single-cell pseudobulk of the same sample mostly by a per-gene factor (capture, 3' bias,
ambient RNA, total vs polyA RNA): on COMBAT, per-gene offsets explain 89% of the gap, with split-half r 0.998.
Left in, the gap moves every sample alike, so it biases fractions and program activity but not much the
treatment contrast; with programs it still costs most of the attribution (COMBAT sepsis r 0.58 -> 0.12).

platform_offsets: per-gene log2 offsets estimated from the study bulk alone (no paired data needed), by
               alternating PALM-lite fractions on the offset-adjusted reference with
               b_g += median over samples of log2(observed / predicted CPM), on genes expressed in the bulk.
               One offset per gene for every sample, so a treatment difference is not absorbed (it moves the
               median only through the treated share of samples).
correct_platform: the bulk divided by 2^offset per gene, each sample's depth kept; run PALM on the result.

What it cannot fix: the study's MEAN composition. An offset on one type's marker genes and a different average
fraction of that type explain the bulk equally well, so the study-estimated offsets keep the average fractions the
uncorrected bulk implies (simulation: mean fraction bias unchanged, per-gene gap to the true platform cut >3x;
COMBAT: fraction r with the single-cell truth 0.25-0.40). What it fixes is the per-gene shape that reallocation and
the program estimates use (COMBAT effect r 0.12 -> 0.52 sepsis, 0.21 -> 0.64 severe). For fractions, pass offsets
from a few paired bulk + single-cell samples, or anchor the composition with measured counts (e.g. blood counts).
"""
import numpy as np
import pandas as pd

from .realloc import em_fractions


def platform_offsets(bulk, ref, iters=4, min_cpm=10, clip=8.0):
    """bulk: genes x samples DataFrame of raw counts; ref: Reference. Returns a Series of log2 offsets
    (bulk / reference, per gene) on the genes shared with the reference; 0 for genes below min_cpm mean CPM."""
    ri = {g: i for i, g in enumerate(ref.genes)}
    genes = [g for g in bulk.index if g in ri]
    Bm = np.asarray(ref.B, float)[:, [ri[g] for g in genes]] + 1e-9
    V = bulk.loc[genes].values.T.astype(float)                              # N x G
    cpm = V / V.sum(1, keepdims=True) * 1e6
    expr = cpm.mean(0) > min_cpm
    b = np.zeros(len(genes))
    for _ in range(iters):
        Bb = Bm * np.exp2(b)[None]
        Bb = Bb / Bb.sum(1, keepdims=True)
        pi = em_fractions(V[:, expr], Bb[:, expr])
        q = pi @ Bb
        q = q / q.sum(1, keepdims=True) * 1e6
        step = np.median(np.log2(cpm + 1) - np.log2(q + 1), 0)
        b = np.where(expr, b + step, b)
    return pd.Series(np.clip(b, -clip, clip), index=genes)


def correct_platform(bulk, ref, offsets=None, **kw):
    """Bulk counts on the reference's platform: bulk / 2^offset per gene (offsets from platform_offsets unless
    given, e.g. from paired samples), rounded, each sample's total kept. Only genes shared with the reference."""
    off = platform_offsets(bulk, ref, **kw) if offsets is None else pd.Series(offsets)
    Y = bulk.loc[off.index].astype(float)
    c = Y.div(np.exp2(off.values), axis=0)
    return (c / c.sum(axis=0) * Y.sum(axis=0)).round()
