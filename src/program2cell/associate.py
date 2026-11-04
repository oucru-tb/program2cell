"""Association of per-cell-type program activity with a patient-level measure, for single-group cohorts
(e.g. TB patients only; docs/benchmarking.md 0.8). Opt-in, post-freeze-v2, not yet chosen on benchmarks.

Without a comparison group there is no treatment effect. PALM still gives each patient's calibrated activity
per cell type and program; this module asks whether that activity follows a clinical measure z (bacterial load,
cavitation, severity, time on treatment) within the group:

  slope_cp   least-squares slope of activity_icp on z_i (optionally with covariates), per (cell type, program)
  interval   patient bootstrap (resampling patients, refitting the slope), percentile, at `level`
  p          permutation of z across patients (two-sided, on |slope|); the null keeps every patient's activity
             and its cross-type structure, so spillover between cell types cannot create calls by itself
  call       p < 1 - level (same rule as the effect intervals), or BH q < 1 - level with adjust="bh"

Measurement error in each patient's activity enters through the residuals (it is independent across patients
given the fractions); a calibration bias shared by all patients moves the level, not the slope.
Works on any activity array, so competitors' per-type activity can be scored the same way.

associate_unmix: the same question answered by PALM's unmixing (core.unmix) with z, rescaled to 0-1, in place of
the 0/1 treatment. Correlating per-patient activity with z inherits spillover: a responding type's change leaks
into the other types' activity and rises with z too (first McFarland gradient split: 63% of null pairs called).
Unmixing models that leakage across patients, as in the two-group path. Effect = change over the observed z
range; interval = patient bootstrap (+ prediction-error variance), t quantile; call = interval excludes 0.
"""
import numpy as np


def _slopes(A, z, X=None):
    """A (N, M), z (N,), X (N, k) or None -> slope of each column of A on z, adjusting for X."""
    Z = np.column_stack([np.ones(len(z)), z] + ([X] if X is not None else []))
    coef, *_ = np.linalg.lstsq(Z, A, rcond=None)
    return coef[1]


def _bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def associate(activity, z, covariates=None, level=0.9, n_boot=500, n_perm=999, adjust=None, seed=0,
              cell_types=None, programs=None):
    """activity (N, C, P) per-patient activity (e.g. DeconvolutionResult.activity); z (N,) patient measure.
    Returns dict(slope, lo, hi, p, calls (C, P), z_scale, and a long table via ['table'])."""
    A = np.asarray(activity, float)
    N, C, P = A.shape
    z = np.asarray(z, float)
    ok = np.isfinite(z) & np.isfinite(A.reshape(N, -1)).all(1)
    A, z = A[ok].reshape(ok.sum(), C * P), z[ok]
    X = None if covariates is None else np.asarray(covariates, float).reshape(len(ok), -1)[ok]
    n = len(z)
    if n < 4 or np.std(z) == 0:
        raise ValueError("need at least 4 patients with varying z")
    rng = np.random.default_rng(seed)
    b = _slopes(A, z, X)
    boot = np.empty((n_boot, C * P))
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        while np.std(z[idx]) == 0:
            idx = rng.integers(0, n, n)
        boot[k] = _slopes(A[idx], z[idx], None if X is None else X[idx])
    a = (1 - level) / 2
    lo, hi = np.quantile(boot, a, 0), np.quantile(boot, 1 - a, 0)
    perm = np.empty((n_perm, C * P))
    for k in range(n_perm):
        perm[k] = _slopes(A, rng.permutation(z), X)
    p = (1 + (np.abs(perm) >= np.abs(b)[None]).sum(0)) / (n_perm + 1)
    alpha = 1 - level
    calls = (_bh(p) if adjust == "bh" else p) < alpha
    sh = (C, P)
    out = dict(slope=b.reshape(sh), lo=lo.reshape(sh), hi=hi.reshape(sh), p=p.reshape(sh),
               calls=calls.reshape(sh), z_scale=float(np.ptp(z)), n=int(n))
    if cell_types is not None and programs is not None:
        import pandas as pd
        out["table"] = pd.DataFrame(dict(cell_type=np.repeat(cell_types, P), program=np.tile(programs, C),
                                         slope=b, lo=lo, hi=hi, p=p, called=calls))
    return out


def associate_unmix(inputs, fractions, z, covariates=None, groups=None, level=0.9, n_boot=200, seed=0):
    """Per (cell type, program) change of activity over the observed range of z, by unmixing.
    inputs: prepare_inputs output (Y, B, Gmat); fractions (N, C) read fractions (e.g. PALM-lite or posterior
    median); z (N,). Returns dict(change, lo, hi, calls (C, P) over the z range, slope = change / range(z))."""
    from scipy.stats import t as tdist
    from .core.realloc import reallocate
    from .core import unmix as um
    z = np.asarray(z, float)
    if np.ptp(z) == 0:
        raise ValueError("z does not vary")
    zs = (z - z.min()) / np.ptp(z)
    Y, B, G = np.asarray(inputs.Y, float), np.asarray(inputs.B, float), np.asarray(inputs.Gmat, float)
    pi = np.asarray(fractions, float)
    raw = reallocate(Y, pi, B, G)
    A = um.leakage_operator(pi, B, G, diagonal=False)
    r = um.unmix_effects(raw, A, zs, covariates, groups, level=level, n_boot=n_boot, pi=pi, B=B, G=G, seed=seed,
                         pev=True)
    q = tdist.ppf(0.5 + level / 2, max(r["n_units"] - 1, 1))
    ch, se = r["effects"], r["se"]
    lo, hi = ch - q * se, ch + q * se
    return dict(change=ch, lo=lo, hi=hi, se=se, calls=(lo > 0) | (hi < 0), slope=ch / np.ptp(z),
                z_scale=float(np.ptp(z)), n=int(len(z)))
