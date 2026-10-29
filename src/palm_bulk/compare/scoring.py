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
              the same across-sample spearman split into a shared part (mean over cell types, per
              sample and program) and a cell-type-specific part (deviation from that mean): a method
              that copies one bulk shift to every cell type scores on the shared part only (its
              specific part is constant: nan); act_specific_share is the truth's share of
              across-sample variance that is cell-type specific (how much the specific metric matters);
              mae, coverage on the model scale only
              leakage diagnostics (per cell type and program, regress the estimate across samples on the
              true activity of every cell type for that program): act_gain, the coefficient on its own
              type (1 = no attenuation; model scale only), and act_leakage, the R^2 the other types add
              beyond its own (0 = no contamination; scale-free);
              act_prog_leakage: the same across programs within a cell type, the R^2 the other
              programs' truth adds beyond the program's own (overlapping gene sets, e.g. IFN-alpha vs
              IFN-gamma, show up here; nan with fewer than P + 5 samples);
              act_coupling: |corr(estimate, true fraction) - corr(truth, true fraction)| per type and
              program (activity that moves with composition when the truth does not);
              act_iscore: interval score of the 90% intervals (width + 20 x miss distance; lower is
              better, rewards calibration and sharpness together)
  effects:    pearson with true beta; AUROC for telling responding cell types (|beta| > 0.1) from the
              rest, i.e. "which cell types respond" (0.5 = cannot tell); sign agreement where
              |beta| > 0.1; mae and coverage on the model scale;
              eff_localise: for programs with a true effect, the share where the cell type with the
              largest |estimate| is the one with the largest |true effect| (ties share the credit);
              eff_iscore; among effects whose 90% interval excludes 0 (eff_n_called of them), eff_fsr
              the share with the wrong sign (false sign rate) and eff_fdr the share that is not a
              meaningful effect (wrong sign or true |beta| <= 0.1)
See docs/benchmarking.md for what each metric shows.
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


def _interval_score(iv, truth, level=0.9):
    lo, hi = iv
    a = 1 - level
    s = (hi - lo) + 2 / a * (lo - truth) * (truth < lo) + 2 / a * (truth - hi) * (truth > hi)
    return float(np.nanmean(np.where(np.isfinite(truth), s, np.nan)))


def _leakage(a, t, var_min):
    """(gain, leakage) averaged over (c, p): regress a[:, c, p] on t[:, :, p] (all types) across samples."""
    N, C, P = t.shape
    gains, leaks = [], []
    for p in range(P):
        for c in range(C):
            y = a[:, c, p]
            ok = np.isfinite(y) & np.isfinite(t[:, :, p]).all(1)
            if ok.sum() < C + 3 or np.nanvar(t[ok, c, p]) <= var_min or np.ptp(y[ok]) <= 1e-12:
                continue
            y, X = y[ok], t[ok, :, p]
            tss = ((y - y.mean()) ** 2).sum()
            X1 = np.column_stack([np.ones(len(y)), X])
            coef = np.linalg.lstsq(X1, y, rcond=None)[0]
            r2_full = 1 - ((y - X1 @ coef) ** 2).sum() / tss
            Xo = np.column_stack([np.ones(len(y)), X[:, c]])
            r2_own = 1 - ((y - Xo @ np.linalg.lstsq(Xo, y, rcond=None)[0]) ** 2).sum() / tss
            gains.append(coef[1 + c])
            leaks.append(max(r2_full - r2_own, 0.0))
    return _nanmean(gains), _nanmean(leaks)


def _program_leakage(a, t, var_min):
    """Mean over (c, p) of the R^2 that the other programs' truth (same cell type) adds beyond program p's
    own truth, regressing a[:, c, p] across samples; needs more samples than programs (else nan)."""
    N, C, P = t.shape
    leaks = []
    for c in range(C):
        for p in range(P):
            y = a[:, c, p]
            ok = np.isfinite(y) & np.isfinite(t[:, c, :]).all(1)
            if P < 2 or ok.sum() < P + 5 or np.nanvar(t[ok, c, p]) <= var_min or np.ptp(y[ok]) <= 1e-12:
                continue
            y, X = y[ok], t[ok, c, :]
            tss = ((y - y.mean()) ** 2).sum()
            X1 = np.column_stack([np.ones(len(y)), X])
            r2_full = 1 - ((y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]) ** 2).sum() / tss
            Xo = X1[:, [0, 1 + p]]
            r2_own = 1 - ((y - Xo @ np.linalg.lstsq(Xo, y, rcond=None)[0]) ** 2).sum() / tss
            leaks.append(max(r2_full - r2_own, 0.0))
    return _nanmean(leaks)


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
        a_sh, t_sh = a.mean(1), t.mean(1)                            # (N, P) shared over cell types
        a_sp, t_sp = a - a_sh[:, None], t - t_sh[:, None]            # (N, C, P) cell-type specific
        out["act_spearman_shared"] = _nanmean([_corr(spearmanr, a_sh[:, p], t_sh[:, p]) for p in range(P)
                                               if np.nanvar(t_sh[:, p]) > activity_var_min])
        out["act_spearman_specific"] = _nanmean([_corr(spearmanr, a_sp[:, c, p], t_sp[:, c, p])
                                                 for c in range(C) for p in range(P)
                                                 if np.nanvar(t_sp[:, c, p]) > activity_var_min])
        v_sp, v_tot = np.nanvar(t_sp, 0).sum(), np.nanvar(t, 0).sum()
        out["act_specific_share"] = float(v_sp / v_tot) if v_tot > 0 else np.nan
        gain, out["act_leakage"] = _leakage(a, t, activity_var_min)
        out["act_prog_leakage"] = _program_leakage(a, t, activity_var_min)
        if "pi" in truth:
            pt = truth["pi"]
            out["act_coupling"] = _nanmean([abs(_corr(spearmanr, a[:, c, p], pt[:, c])
                                                - _corr(spearmanr, t[:, c, p], pt[:, c]))
                                            for c in range(C) for p in range(P)
                                            if np.nanvar(t[:, c, p]) > activity_var_min])
        if r.activity_scale == "delta":
            out["act_mae"] = float(np.nanmean(np.abs(a - t)))
            out["act_pearson"] = _corr(pearsonr, a.ravel(), t.ravel())
            out["act_gain"] = gain
            if "activity" in r.intervals:
                out["act_cov90"] = _coverage(r.intervals["activity"], t)
                out["act_iscore"] = _interval_score(r.intervals["activity"], t)
    if r.effects is not None and truth.get("beta") is not None:
        e, t = np.asarray(r.effects), truth["beta"]
        out["eff_pearson"] = _corr(pearsonr, e.ravel(), t.ravel())
        big = np.abs(t) > EFFECT_MIN
        out["eff_auroc"] = auroc(np.abs(e), big)
        out["eff_sign_agree"] = float((np.sign(e[big]) == np.sign(t[big])).mean()) if big.any() else np.nan
        loc = []
        for p in range(t.shape[1]):
            if np.abs(t[:, p]).max() > EFFECT_MIN and np.isfinite(e[:, p]).all():
                m = np.abs(e[:, p])
                top = np.flatnonzero(m >= m.max() - 1e-12 * (1 + m.max()))
                loc.append(float(np.argmax(np.abs(t[:, p])) in top) / len(top))
        out["eff_localise"] = _nanmean(loc)
        if r.effects_scale == "delta":
            out["eff_mae"] = float(np.nanmean(np.abs(e - t)))
            if "effects" in r.intervals:
                out["eff_cov90"] = _coverage(r.intervals["effects"], t)
                out["eff_iscore"] = _interval_score(r.intervals["effects"], t)
                lo, hi = r.intervals["effects"]
                called = ((lo > 0) | (hi < 0)) & np.isfinite(t)
                wrong_sign = np.sign(e) != np.sign(t)
                out["eff_n_called"] = int(called.sum())
                out["eff_fsr"] = float(wrong_sign[called].mean()) if called.any() else np.nan
                out["eff_fdr"] = float((wrong_sign | (np.abs(t) <= EFFECT_MIN))[called].mean()) if called.any() else np.nan
    return out
