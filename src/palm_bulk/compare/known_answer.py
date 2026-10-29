"""Metrics for perturbation benchmarks with a known answer, and statistics for comparing methods.

Known-answer metrics (effects beta_hat (C, P) of one contrast):
  responder_auroc      the treatment acts on known (cell type, program) pairs only (e.g. idasanutlin -> p53 in
                       TP53-wild-type lines): AUROC of the signed effect for responders vs non-responders
  leakage_fpr          share of known non-responder pairs that are called (interval excludes 0)
  responder_tpr        share of responder pairs called with the right sign
  program_specificity  per cell type, is the stimulus' expected program the top program (top-1 accuracy), and
                       AUROC of expected vs other programs (e.g. IFN-beta -> JAK-STAT, IFN-gamma -> IFN-gamma resp.)
  decomposition_accuracy   estimated intrinsic / composition split vs the truth (causal.decompose on true
                       fractions and true activity): correlation, sign agreement, error of the intrinsic share

Method comparison over splits and datasets:
  paired_test          paired difference of a metric between two methods across splits: mean, bootstrap CI,
                       Wilcoxon signed-rank and paired t p-values
  holm                 Holm step-down adjustment of p-values (several metrics or datasets)
  random_effects       DerSimonian-Laird pooled difference across datasets with its CI and heterogeneity
"""
import numpy as np
from scipy import stats


def _auroc(score, label):
    score, label = np.asarray(score, float).ravel(), np.asarray(label, bool).ravel()
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    if label.all() or not label.any():
        return np.nan
    r = stats.rankdata(score)
    n1 = label.sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (~label).sum())


def responder_auroc(effects, responders, sign=1):
    """AUROC of sign * effect for responder (C, P) pairs (bool mask) vs the rest."""
    return _auroc(sign * np.asarray(effects, float), responders)


def leakage_fpr(calls, responders):
    """Share of non-responder pairs that are called."""
    calls, responders = np.asarray(calls, bool), np.asarray(responders, bool)
    return float(calls[~responders].mean()) if (~responders).any() else np.nan


def responder_tpr(effects, calls, responders, sign=1):
    """Share of responder pairs called with the expected sign."""
    e, calls, r = np.asarray(effects, float), np.asarray(calls, bool), np.asarray(responders, bool)
    return float((calls & (np.sign(e) == sign))[r].mean()) if r.any() else np.nan


def program_specificity(effects, expected, types=None, sign=1):
    """effects (C, P); expected: bool (P,) of the programs the stimulus should move. Per cell type in
    `types` (default all): top-1 hit (largest sign*effect is an expected program) and AUROC.
    Returns dict(top1, auroc) averaged over types."""
    e = sign * np.asarray(effects, float)
    expected = np.asarray(expected, bool)
    types = range(e.shape[0]) if types is None else types
    top1 = [float(expected[np.nanargmax(e[c])]) for c in types]
    au = [_auroc(e[c], expected) for c in types]
    return dict(top1=float(np.mean(top1)), auroc=float(np.nanmean(au)))


def decomposition_accuracy(est, truth):
    """est, truth: dicts from causal.decompose. Correlation and sign agreement of I and K over (c, p), and the
    mean absolute error of the intrinsic share over programs."""
    out = {}
    for k in ("I", "K"):
        a, b = np.asarray(est[k]).ravel(), np.asarray(truth[k]).ravel()
        big = np.abs(b) > 0.05 * np.abs(b).max() if np.abs(b).max() > 0 else np.zeros_like(b, bool)
        out[k + "_r"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
        out[k + "_sign"] = float((np.sign(a[big]) == np.sign(b[big])).mean()) if big.any() else np.nan
    out["share_mae"] = float(np.nanmean(np.abs(np.asarray(est["share_intrinsic"]) - np.asarray(truth["share_intrinsic"]))))
    return out


# ---- comparing methods -----------------------------------------------------------------------------------
def paired_test(a, b, n_boot=5000, level=0.95, seed=0):
    """Paired comparison of one metric between methods A and B over the same splits (arrays, nan-aware).
    Returns dict(n, diff = mean(A - B), lo, hi (bootstrap CI), p_wilcoxon, p_t, win_rate)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = len(d)
    if n < 2:
        return dict(n=n, diff=float(d.mean()) if n else np.nan, lo=np.nan, hi=np.nan, p_wilcoxon=np.nan,
                    p_t=np.nan, win_rate=np.nan)
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, n, (n_boot, n))].mean(1)
    alpha = (1 - level) / 2
    try:
        pw = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        pw = np.nan
    return dict(n=n, diff=float(d.mean()), lo=float(np.quantile(bm, alpha)), hi=float(np.quantile(bm, 1 - alpha)),
                p_wilcoxon=pw, p_t=float(stats.ttest_1samp(d, 0).pvalue), win_rate=float((d > 0).mean()))


def holm(pvals):
    """Holm-adjusted p-values (same order as given; nan kept)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        out[j] = run
    return out


def random_effects(diffs, ses, level=0.95):
    """DerSimonian-Laird random-effects pooling of per-dataset differences (e.g. paired_test diffs) with
    standard errors. Returns dict(pooled, lo, hi, p, tau2, I2)."""
    y, s = np.asarray(diffs, float), np.asarray(ses, float)
    w = 1 / s ** 2
    fixed = (w * y).sum() / w.sum()
    Q = (w * (y - fixed) ** 2).sum()
    k = len(y)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    wr = 1 / (s ** 2 + tau2)
    mu = (wr * y).sum() / wr.sum()
    se = np.sqrt(1 / wr.sum())
    z = stats.norm.ppf(0.5 + level / 2)
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 and k > 1 else 0.0
    return dict(pooled=float(mu), lo=float(mu - z * se), hi=float(mu + z * se),
                p=float(2 * stats.norm.sf(abs(mu / se))), tau2=float(tau2), I2=float(I2))
