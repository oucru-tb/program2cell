"""Score MethodResults against ground truth.

Truth dict (arrays aligned to data.inputs: samples, cell_types, programs):
  pi     (N, C)    read fractions on the modelled genes
  delta  (N, C, P) per-sample per-cell-type program activity (model scale)
  beta   (C, P)    treatment shift of activity (optional)
Simulations give these exactly; for real pseudo-bulk they come from the pooled cells
(per-type pseudo-bulk -> palm_bulk.estimate_delta).

Metrics are chosen so methods on another scale ('score') are still comparable by rank:
  fractions:  rmse, pearson (pooled), per-cell-type correlation across samples
  activity:   spearman across samples per (cell type, program), averaged (the paired-cohort metric);
              spearman across cell types within (sample, program) ("which cell type carries it");
              mae, coverage on the model scale only
  effects:    pearson with true beta; AUROC for telling responding cell types (|beta| > 0.1) from the
              rest, i.e. "which cell types respond" (0.5 = cannot tell); sign agreement where
              |beta| > 0.1; mae and coverage on the model scale
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata

EFFECT_MIN = 0.1        # |true beta| above which the effect's sign counts


def _corr(f, a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) <= 1e-9 * (1 + np.abs(a[ok]).max()) or np.ptp(b[ok]) <= 1e-12:
        return np.nan
    return float(f(a[ok], b[ok])[0])


def _nanmean(x):
    x = np.asarray(x, float)
    return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan


def auroc(score, positive):
    """Probability that a random positive scores above a random negative (ties count half)."""
    score, positive = np.asarray(score, float).ravel(), np.asarray(positive, bool).ravel()
    ok = np.isfinite(score)
    score, positive = score[ok], positive[ok]
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _coverage(iv, truth):
    lo, hi = iv
    ok = np.isfinite(truth)
    return float(((truth >= lo) & (truth <= hi))[ok].mean())


def score(result, truth, activity_var_min=1e-4):
    """One flat dict of metrics for one method (nan where the method has no such output)."""
    r, out = result, dict(method=result.method, runtime=result.runtime)
    if r.fractions is not None:
        f, t = np.asarray(r.fractions), truth["pi"]
        out["frac_rmse"] = float(np.sqrt(np.nanmean((f - t) ** 2)))
        out["frac_pearson"] = _corr(pearsonr, f.ravel(), t.ravel())
        out["frac_r_per_type"] = _nanmean([_corr(pearsonr, f[:, c], t[:, c]) for c in range(t.shape[1])])
        if "fractions" in r.intervals:
            out["frac_cov90"] = _coverage(r.intervals["fractions"], t)
    if r.activity is not None and "delta" in truth:
        a, t = np.asarray(r.activity), truth["delta"]
        N, C, P = t.shape
        across = [_corr(spearmanr, a[:, c, p], t[:, c, p]) for c in range(C) for p in range(P)
                  if np.nanvar(t[:, c, p]) > activity_var_min]
        within = [_corr(spearmanr, a[i, :, p], t[i, :, p]) for i in range(N) for p in range(P)]
        out["act_spearman_across_samples"] = _nanmean(across)
        out["act_spearman_across_types"] = _nanmean(within)        # nan if a method cannot rank cell types
        if r.activity_scale == "delta":
            out["act_mae"] = float(np.nanmean(np.abs(a - t)))
            out["act_pearson"] = _corr(pearsonr, a.ravel(), t.ravel())
            if "activity" in r.intervals:
                out["act_cov90"] = _coverage(r.intervals["activity"], t)
    if r.effects is not None and truth.get("beta") is not None:
        e, t = np.asarray(r.effects), truth["beta"]
        out["eff_pearson"] = _corr(pearsonr, e.ravel(), t.ravel())
        big = np.abs(t) > EFFECT_MIN
        out["eff_auroc"] = auroc(np.abs(e), big)
        out["eff_sign_agree"] = float((np.sign(e[big]) == np.sign(t[big])).mean()) if big.any() else np.nan
        if r.effects_scale == "delta":
            out["eff_mae"] = float(np.nanmean(np.abs(e - t)))
            if "effects" in r.intervals:
                out["eff_cov90"] = _coverage(r.intervals["effects"], t)
    return out
