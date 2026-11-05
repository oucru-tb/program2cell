"""Program deconvolution on any fractions: the frozen PALM pipeline (docs/method_spec.md).

    fractions (PALM, PALM-lite or any external method)
      -> posterior reallocation (core.realloc.reallocate)
      -> calibration on reference donors (core.calibrate)
      -> calibrated activity with conformal intervals, treatment effects with intervals and calls

How fraction error enters the intervals depends on what the fraction source provides:
  fraction draws (PALM posterior)        calibration with the true fractions; the spread over draws is added
  a fraction function (PALM-lite, NNLS)  calibration with the engine's own fractions on the calibration bulks
  neither (an external point estimate)   calibration with PALM-lite fractions as a proxy for the error
"""
from dataclasses import dataclass, field
import numpy as np

from .core.realloc import reallocate, em_fractions
from .core.calibrate import Calibrator


@dataclass
class DeconvolutionResult:
    fractions: np.ndarray                 # (N, C) read fractions on the modelled genes
    activity: np.ndarray                  # (N, C, P) calibrated program activity
    activity_interval: tuple              # (lo, hi), each (N, C, P)
    raw_activity: np.ndarray              # (N, C, P) reallocated activity before calibration
    effects: np.ndarray = None            # (C, P) treated minus control
    effects_interval: tuple = None
    calls: np.ndarray = None              # (C, P) bool: interval excludes 0
    robustness: np.ndarray = None         # (C, P) robustness value to unmeasured confounding (causal.py)
    decomposition: dict = None            # intrinsic / composition split of bulk program effects (causal.py)
    cell_types: list = None
    programs: list = None
    info: dict = field(default_factory=dict)

    def table(self, what="effects"):
        """Long table of effects (with interval and call) or of activity (sample, cell type, program)."""
        import pandas as pd
        if what == "effects":
            if self.effects is None:
                raise ValueError("no treatment was given")
            lo, hi = self.effects_interval
            C, P = self.effects.shape
            tab = pd.DataFrame(dict(cell_type=np.repeat(self.cell_types, P), program=np.tile(self.programs, C),
                                    effect=self.effects.ravel(), lo=lo.ravel(), hi=hi.ravel(),
                                    called=self.calls.ravel()))
            if self.robustness is not None:
                tab["robustness"] = self.robustness.ravel()
            if self.decomposition is not None:
                tab["intrinsic"] = self.decomposition["I"].ravel()
                tab["composition"] = self.decomposition["K"].ravel()
            return tab
        lo, hi = self.activity_interval
        N, C, P = self.activity.shape
        return pd.DataFrame(dict(sample=np.repeat(np.arange(N), C * P), cell_type=np.tile(np.repeat(self.cell_types, P), N),
                                 program=np.tile(self.programs, N * C), activity=self.activity.ravel(),
                                 lo=lo.ravel(), hi=hi.ravel()))


# unmixing options on by default: valid intervals on the perturbation benchmarks (method_spec 4.5b); route per
# program by the narrower simulation-calibrated interval with the per-type guard (freeze v2, method_spec 4.8)
UNMIX_DEFAULTS = dict(pev=True, misfit=True, choose="width_guard")


def program_deconvolution(inputs, calibration, fractions=None, fraction_draws=None, engine=None, frac_fn=None,
                          treatment=None, effect_interval="lk", own=True, level=0.9, n_draws=50, seed=0,
                          covariates=None, groups=None, estimator="auto", decompose=False, n_sim=10, direct_kw=None,
                          sim_widen="abs", unmix_kw=None, v4_kw=None, **fit_kw):
    """Per-sample per-cell-type program activity and treatment effects from bulk counts and fractions.

    inputs: prepared inputs (program2cell.prepare_inputs): Y, B, Gmat, cell_types, programs, T.
    calibration: a CalibrationSet on the same genes, cell types and programs
        (core.calibrate.calibration_from_table or build_calibration_set; reference donors only).
    Fractions, one of:
        fractions (N, C)        read fractions on the modelled genes from any method (cell fractions must be
                                converted first; see compare.methods.to_modelled_read_fractions)
        fraction_draws (S,N,C)  posterior draws (e.g. a PALM fit)
        engine="palm"           fit PALM (run_palm_fast, joint NB) on `inputs`; extra keywords go to it
        engine="lite"           PALM-lite (multinomial EM)
    frac_fn(Y, B) -> pi: the method itself, if it can be rerun; then its error is calibrated directly.
    treatment: (N,) 0/1; default inputs.T. effect_interval: one of Calibrator.INTERVALS.
    Causal options (program2cell.causal): estimator 'diff' (randomised), 'ols' (covariates and/or groups, e.g.
    donor ids for paired samples), 'ipw' or 'aipw' (doubly robust); covariates (N, k); groups (N,).
    decompose=True also splits each bulk program effect into intrinsic and composition parts per cell type.
    estimator 'unmix' (core.unmix): cell-type-specific effects from how the bulk effect varies with the fractions
    across samples (reallocation alone leaks a type's shift into all types); intervals are cluster-bootstrap
    intervals widened by the studentised error of n_sim simulated cohorts with this design (design_check).
    'auto': per program, 'unmix' where its interval is narrower than calibrated reallocation's (unmix_kw choose;
    freeze v1: where its simulated error is lower), else the calibrated effects (with covariates / groups:
    'ols'); sim_widen=True also widens those programs'
    intervals to the simulated error of rescaled reallocation (False: the calibrated intervals as they are;
    "abs" (default since freeze v2): widen to the simulated level-quantile of the absolute leakage error per type and program (not studentised:
    leakage bias does not grow with the study's noise, and the simulated cohorts are less noisy than real ones)
    plus the study's contrast noise in quadrature, so well-separated types are not widened like poorly separated ones;
    "replace": the simulated-error interval alone, i.e. the study's own (group-bootstrap) contrast standard error
    times the simulated studentised error quantile of rescaled reallocation, which carries its leakage bias).
    Default (the causal setting, freeze v2 2026-11-03, method_spec 4.8): estimator="auto", sim_widen="abs" and
    UNMIX_DEFAULTS (pev=True, misfit=True, choose="width_guard"). Freeze v1: sim_widen=True, choose="rmse".
    estimator="diff" is the fast setting (seconds instead of minutes).
    unmix_kw ('unmix', 'auto'): updates UNMIX_DEFAULTS; pev=True adds the prediction-error variance of the shrunk type deviations;
    separate=True / "program" gives the REML penalty groups (core.unmix._fit_joint); misfit=True measures the
    treatment response outside the programs on this bulk (offprogram_tau) and adds the error it causes
    (misfit_sd) to the unmixed intervals, which then use the t quantile alone, while the design check simulates
    cohorts with that response (choice of engine, widening of calibrated programs); per_pair=True ('auto'):
    choose the engine per (type, program) instead of per program (a type keeping little of its own shift is
    better left calibrated even where unmixing wins for the program). plant="observed": the design check plants
    this study's shared effect plus type deviations of its observed spread (core.unmix.design_check);
    plant="shared": the shared-response model's common effect plus deviations of the unmixed effects' spread.
    choose ('auto'): "rmse" (default) unmixes a program where its simulated RMSE is lower than rescaled
    reallocation's; "width" / "width_guard": where its interval (bootstrap, misfit- and simulation-widened) is narrower than the
    calibrated one after sim_widen, i.e. the route with more power at its simulation-calibrated coverage (for a
    response shared by all types the RMSEs nearly tie and the RMSE choice flips between splits); "width_guard"
    keeps the calibrated result for a type whose unmixed effect lies outside its calibrated interval.
    estimator 'direct' (core.direct): maximum likelihood of the treatment effects on the bulk counts given the
    fractions (no reallocation), cluster-bootstrap intervals; direct_kw go to direct_effects (e.g. lam).
    estimator 'v4' (core.gls; development, opt-in, not frozen): GLS unmixing with each sample's leakage
    operator, simulation-calibrated with the study's off-program response and a stated per-sample fraction
    error; v4_kw go to core.gls.v4_effects: frac_sd (0 for assembled bulk with PALM-lite fractions;
    core.gls.FRAC_SD_REAL_BULK for real bulk; "estimate" for fractions from elsewhere), min_effect (call
    floor), n_sim. out.info["v4"] holds the shared response and type deviations.
    """
    Y, B, G = inputs.Y, inputs.B, inputs.Gmat
    info = {}
    if engine == "palm":
        from .bayes.twostage import run_palm_fast
        kw = dict(likelihood="nb", activity="common", use_m=False)
        kw.update(fit_kw)
        res = run_palm_fast(inputs, **kw)
        z = np.asarray(res.draws["pi"])
        fraction_draws = z[np.linspace(0, len(z) - 1, min(n_draws, len(z))).astype(int)]
        info["engine"] = "palm (run_palm_fast, %s)" % kw
    elif engine == "lite":
        fractions, frac_fn = em_fractions(Y, B), em_fractions
        info["engine"] = "lite (multinomial EM)"
    elif engine is not None:
        raise ValueError("engine must be 'palm', 'lite' or None")
    if fraction_draws is not None:
        fraction_draws = np.asarray(fraction_draws, float)
        fractions = np.median(fraction_draws, 0) if fractions is None else fractions
    if fractions is None:
        raise ValueError("give fractions, fraction_draws or an engine")
    fractions = np.asarray(fractions, float)

    if fraction_draws is not None:
        mode, cal_fn = "true fractions + posterior draws", None
    elif frac_fn is not None:
        mode, cal_fn = "engine's own fractions", frac_fn
    else:
        mode, cal_fn = "PALM-lite proxy for fraction error", em_fractions
    info.update(calibration=mode, n_calibration=len(calibration), effect_interval=effect_interval)

    raw = reallocate(Y, fractions, B, G)
    raw_draws = reallocate(Y, fraction_draws, B, G) if fraction_draws is not None else None
    cal = Calibrator.from_set(calibration, frac_fn=cal_fn, own=own, level=level, seed=seed)
    act, act_iv = cal.activity(raw, raw_draws)
    out = DeconvolutionResult(fractions=fractions, activity=act, activity_interval=act_iv, raw_activity=raw,
                              cell_types=list(inputs.cell_types), programs=list(inputs.programs), info=info)
    T = inputs.T if treatment is None else treatment
    if T is not None and len(np.unique(np.asarray(T))) == 2:
        from . import causal
        mode = estimator
        if mode == "v4" and covariates is not None:
            raise ValueError("estimator 'v4' takes groups, not covariates")
        if estimator in ("diff", "unmix", "auto", "direct", "v4") and (covariates is not None or groups is not None):
            estimator = "ols"
        elif estimator in ("unmix", "auto", "direct", "v4"):
            estimator = "diff"
        w = causal.estimator_weights(T, covariates, groups, estimator)
        eff, iv, parts = cal.effects(raw, dh_draws=raw_draws, interval=effect_interval, weights=w)
        if mode == "direct":
            from .core.direct import direct_effects
            zd = None if fraction_draws is None else np.asarray(fraction_draws)[
                np.linspace(0, len(fraction_draws) - 1, min(20, len(fraction_draws))).astype(int)]
            d = direct_effects(Y, fractions, B, G, T, covariates, groups, level=level, pi_draws=zd, seed=seed,
                               **(direct_kw or {}))
            eff, iv = d["effects"], (d["lo"], d["hi"])
            out.info["direct"] = dict(n_units=d["n_units"])
        if mode == "v4":
            from .core.gls import v4_effects
            v = v4_effects(Y, fractions, B, G, T, calibration, groups=groups, level=level, seed=seed,
                           **(v4_kw or {}))
            eff, iv = v["effects"], (v["lo"], v["hi"])
            out.info["v4"] = {k: (x.tolist() if isinstance(x, np.ndarray) else x) for k, x in v.items()
                              if k not in ("effects", "lo", "hi")}
        if mode in ("unmix", "auto"):
            eff, iv = _unmix_effects(out, inputs, calibration, fractions, fraction_draws, raw, raw_draws, T,
                                     covariates, groups, level, seed, n_sim, mode, eff, iv, sim_widen,
                                     **{**UNMIX_DEFAULTS, **(unmix_kw or {})})
        out.effects, out.effects_interval = eff, iv
        out.calls = (iv[0] > 0) | (iv[1] < 0)
        if mode == "v4":
            out.calls = np.asarray(out.info["v4"]["calls"], bool)
        out.robustness = causal.robustness_value(eff, iv[0], iv[1], causal.degrees_of_freedom(T, covariates, groups),
                                                 level=level)
        out.info["effect_variance_parts"] = {k: float(np.mean(v)) for k, v in parts.items()}
        out.info["estimator"] = estimator if mode not in ("unmix", "auto", "direct", "v4") else mode
        if decompose:
            out.decomposition = causal.decompose(fractions, act, B, G, T, covariates, groups, estimator,
                                                 level=level, fraction_draws=fraction_draws, seed=seed)
    return out


def _tau2_upper(d, s2, alpha):
    """Upper 1 - alpha confidence bound of the between-type variance tau^2 of deviations d with sampling
    variances s2 (Q-profile, as in random-effects meta-analysis): the tau^2 at which sum d^2 / (s2 + tau^2)
    equals the alpha quantile of chi^2 with len(d) - 1 df."""
    from scipy.stats import chi2
    from scipy.optimize import brentq
    target = chi2.ppf(alpha, max(len(d) - 1, 1))
    Q = lambda t2: float(np.sum(d ** 2 / (s2 + t2))) - target
    if Q(0.0) <= 0:
        return 0.0
    hi = max(float(np.max(d ** 2)), 1e-12)
    while Q(hi) > 0:
        hi *= 4
    return brentq(Q, 0.0, hi)


def _unmix_effects(out, inputs, calibration, fractions, fraction_draws, raw, raw_draws, T, X, groups, level, seed,
                   n_sim, mode, eff_cal, iv_cal, sim_widen=True, pev=False, separate=False, misfit=False,
                   per_pair=False, plant="prior", shared=False, margin=0.1, choose="rmse", pool=False,
                   pool_alpha=0.1):
    """Effects by unmixing (core.unmix) for every program ('unmix') or where the design check favours it
    ('auto'); the others keep the calibrated effects. Intervals: unmixed, bootstrap half-width widened to
    max(t, simulated studentised error quantile) standard errors; calibrated, the calibrated interval widened
    to at least the simulated studentised error quantile of rescaled reallocation (its leakage bias).
    shared=True: also out.info["shared_response"] (shared_response).
    pool=True (v3 candidate, opt-in; docs/v3_plan.md item 9): per program, a Wald test of the types' deviations
    from the shared-response fit (bootstrap covariance, its diagonal raised to the deviation se with shrinkage
    and misfit); where it does not reject at pool_alpha, every type gets the shared effect, interval
    +- q sqrt(common se^2 + misfit^2 + tau^2) with tau^2 the between-type spread beyond noise. Motivation: a
    response shared by all types (Kang interferon) unmixed type by type flips a rare type's sign (NK -0.28,
    truth +0.26, held-out v2). The width rule and the guard then compare this (pooled or unmixed) estimate with
    the calibrated one as before.
    pool="partial": no test; every type is pulled towards the shared effect by w = tau^2 / (tau^2 + s^2) (empirical
    Bayes), interval from the upper confidence bound of tau^2 (the all-or-nothing test pooled a real between-type
    spread of tiny effects on McFarland and Zhang and its interval missed it: false calls, dev splits 20-29)."""
    from scipy.stats import t as tdist
    from .core import unmix as um
    B, G = inputs.B, inputs.Gmat
    T = np.asarray(T)
    A = um.leakage_operator(fractions, B, G, diagonal=False)
    dd = Ad = zd = None
    if fraction_draws is not None:
        zd = np.asarray(fraction_draws)[np.linspace(0, len(fraction_draws) - 1, min(10, len(fraction_draws))).astype(int)]
        dd = np.asarray(raw_draws)[np.linspace(0, len(raw_draws) - 1, len(zd)).astype(int)]
        Ad = np.stack([um.leakage_operator(z, B, G, diagonal=False) for z in zd])
    r = um.unmix_effects(raw, A, T, X, groups, level=level, dhat_draws=dd, A_draws=Ad, pi=fractions, B=B, G=G,
                         pi_draws=zd, seed=seed, pev=pev, separate=separate,
                         misfit_tau="auto" if misfit else None, Y=inputs.Y,
                         shared_fit=shared or plant == "shared" or pool)
    chk = um.design_check(calibration, fractions, T, X, groups, n_sim=n_sim, level=level, seed=seed,
                          effect_hint=r["effects"] if plant == "shared" else eff_cal,
                          offprog_sd=r["tau"] if misfit else 0.0, plant=plant,
                          shared_hint=r.get("common") if plant == "shared" else None)
    q = tdist.ppf(0.5 + level / 2, max(r["n_units"] - 1, 1))
    q = q if misfit else max(chk["z_q"], q)
    C, P = r["effects"].shape
    eff_u, se_u = r["effects"], r["se"]
    pooled, p_het = np.zeros(P, bool), np.full(P, np.nan)
    if pool and "common" in r:
        from scipy.stats import chi2
        d = r["effects"] - r["common"][None]
        Db = r["boot"] - r["common_boot"][:, None]
        for p in range(P):
            S = np.atleast_2d(np.cov(Db[:, :, p], rowvar=False))
            sd = np.sqrt(np.clip(np.diag(S), 1e-24, None))
            k = np.maximum(r["deviation_se"][:, p] / sd, 1.0)          # raise to the full deviation se
            S = S * np.outer(k, k)
            p_het[p] = chi2.sf(float(d[:, p] @ np.linalg.pinv(S) @ d[:, p]), max(C - 1, 1))
        tau2 = np.clip((d ** 2).mean(0) - (r["deviation_se"] ** 2).mean(0), 0, None)
        mis = 0.0 if r["misfit_sd"] is None else np.asarray(r["misfit_sd"])
        if pool == "partial":
            # empirical-Bayes partial pooling: each type moves towards the shared effect by
            # w = tau^2 / (tau^2 + s^2); the interval uses the upper confidence bound of tau^2 (Q-profile,
            # 1 - pool_alpha), so a between-type spread that few types cannot estimate is not taken as zero
            s2 = r["deviation_se"] ** 2
            tau2_hi = np.array([_tau2_upper(d[:, p], s2[:, p], pool_alpha) for p in range(P)])
            w = tau2[None] / (tau2[None] + s2)
            w_hi = tau2_hi[None] / (tau2_hi[None] + s2)
            eff_u = r["common"][None] + w * d
            se_u = np.sqrt(r["common_se"][None] ** 2 + mis ** 2 + w_hi * s2 + ((w_hi - w) * d) ** 2)
            pooled = (w < 0.5).mean(0) > 0.5            # recorded: programs mostly pulled to the shared effect
        else:
            pooled = p_het > pool_alpha
            se_pool = np.sqrt(r["common_se"][None] ** 2 + mis ** 2 + tau2[None])
            eff_u = np.where(pooled[None], r["common"][None], r["effects"])
            se_u = np.where(pooled[None], se_pool, r["se"])
    lo_u, hi_u = eff_u - q * se_u, eff_u + q * se_u
    # conditioning of the unmixing design per program (v3 reporting item 4b): the leakage maps stacked over samples
    Ap = np.stack([A[:, :, p, :, p] for p in range(P)], 0) if A.ndim == 5 else np.moveaxis(A, 3, 0)
    cond = [float(np.linalg.cond(Ap[p].reshape(-1, C))) for p in range(P)]
    # calibrated programs: widen to the simulated studentised error of rescaled reallocation (at least)
    h = (iv_cal[1] - iv_cal[0]) / 2
    if sim_widen:
        se_r = np.abs(chk["slope"])[None] * um.contrast_se(raw, T, np.random.default_rng(seed), groups=groups)
        h_sim = chk["z_q_realloc"][None] * se_r
        if sim_widen == "abs":              # leakage error in absolute units per type, plus this study's noise
            q_t = tdist.ppf(0.5 + level / 2, max(r["n_units"] - 1, 1))
            h_sim = np.sqrt(chk["q_abs_realloc_cp"] ** 2 + (q_t * se_r) ** 2)
        h = h_sim if sim_widen == "replace" else np.maximum(h, h_sim)
    guard = np.zeros((C, P), bool)
    if mode == "unmix":
        use = np.ones((C, P), bool)
    elif choose in ("width", "width_guard"):    # the narrower of the two simulation-calibrated intervals
        hu = (hi_u - lo_u) / 2
        use = hu < h if per_pair else np.broadcast_to((hu.mean(0) < h.mean(0))[None], (C, P))
        if choose == "width_guard":         # a type whose unmixed effect falls outside its (valid, wide)
            # calibrated interval keeps the calibrated result: unmixing has failed for it (e.g. Kang NK NF-kB
            # unmixed -1.34 [-1.96, -0.73], truth +0.15, calibrated [-0.49, 0.72])
            ok = np.abs(eff_u - eff_cal) <= h
            guard = use & ~ok
            use = use & ok
    else:                                   # simulated RMSE, per program, or per (type, program) with per_pair
        use = chk["use_unmix_cp"] if per_pair else np.broadcast_to(chk["use_unmix"][None], (C, P))
    eff = np.where(use, eff_u, eff_cal)
    iv = (np.where(use, lo_u, eff_cal - h), np.where(use, hi_u, eff_cal + h))
    out.info["unmix"] = dict(programs=[p for p, u in zip(inputs.programs, use.any(0)) if u], pairs=use.tolist(),
                             rmse_unmix=chk["rmse_unmix"].tolist(), rmse_realloc=chk["rmse_realloc"].tolist(),
                             z_q=chk["z_q"], shrinkage=r["lam"][:, 1].tolist(), tau=r["tau"],
                             misfit_sd=None if r["misfit_sd"] is None else r["misfit_sd"].tolist(),
                             guard=guard.tolist(), condition=cond,
                             pooled=[p for p, u in zip(inputs.programs, pooled & use.any(0)) if u],
                             p_heterogeneity=p_het.tolist())
    if shared:
        out.info["shared_response"] = shared_response(r, eff, iv, use, q, tdist.ppf(0.5 + level / 2, max(r["n_units"] - 1, 1)),
                                                      inputs.cell_types, inputs.programs, margin)
    return eff, iv


def shared_response(r, eff, iv, use, q_dev, q_common, cell_types, programs, margin=0.1):
    """Shared vs cell-type-specific response per program, from unmix_effects(shared_fit=True) r and the final
    effects eff, intervals iv and unmixed pairs use (C, P) of _unmix_effects.
    common: the effect if every type responded alike (shared-response model; precise even where the types
    cannot be told apart), interval common +- q_common se, the se adding in quadrature the lack of fit of the
    shared model |common - mean_c eff_c| (its value depends on the model's weighting when the types differ).
    deviation_c = eff_c - common: unmixed pairs, interval
    +- q_dev se (bootstrapped jointly with the common effect); calibrated pairs, the half-widths of the effect's
    interval and of the common interval in quadrature.
    Per type: 'differs' (deviation interval clear of +-margin/2), 'same' (it lies within +-margin), else
    'unclear'. The common interval adds the between-type heterogeneity tau^2 / C (random effects).
    Per program: 'type-specific' (some type differs), 'shared' (common changed, every type same), 'shared?'
    (common changed, no type differs, some unclear: consistent with a shared response, not resolved per type),
    'none' (common unchanged, every type same) or 'unclear'."""
    c = r["common"]
    # between-type heterogeneity (random-effects, DerSimonian-Laird): variance of the types' effects beyond
    # their own uncertainty; when the types differ the common value depends on the weighting, so its
    # interval carries tau^2 / C (equal weights) on top of the shared model's se and lack of fit
    se_c = (iv[1] - iv[0]) / 2 / q_dev
    tau2 = np.clip(eff.var(0, ddof=1) - (se_c ** 2).mean(0), 0, None) if eff.shape[0] > 1 else 0.0
    cs = np.sqrt(r["common_se"] ** 2 + (c - eff.mean(0)) ** 2 + tau2 / eff.shape[0])
    d = eff - c[None]
    h = np.where(use, q_dev * r["deviation_se"], np.sqrt(((iv[1] - iv[0]) / 2) ** 2 + (q_common * cs[None]) ** 2))
    clo, chi, dlo, dhi = c - q_common * cs, c + q_common * cs, d - h, d + h
    # 'differs' needs the deviation interval clear of +-margin/2 (a relevant difference, not just non-zero)
    lab = np.where((dlo > margin / 2) | (dhi < -margin / 2), "differs",
                   np.where((dlo > -margin) & (dhi < margin), "same", "unclear"))
    moved = (clo > 0) | (chi < 0)
    pat = []
    for p in range(len(programs)):
        lp = lab[:, p]
        pat.append("type-specific" if (lp == "differs").any() else
                   ("shared" if (lp == "same").all() else "shared?") if moved[p] else
                   "none" if (lp == "same").all() else "unclear")
    # program-level call (second tier): a net change across the cell types, reported where no type is shown to
    # differ (pattern shared / shared?) and the common interval is clear of +-margin; per-type attribution
    # stays with the per-type calls. Benchmarks 2026-11-02 (splits 0-2): Kang 10 calls (JAK-STAT, NFkB, TNFa
    # every split), every type truly moved the same way; idasanutlin none; trametinib MAPK 3 (LNCaP flat).
    net = ["up" if (pat[p] in ("shared", "shared?") and clo[p] > margin) else
           "down" if (pat[p] in ("shared", "shared?") and chi[p] < -margin) else "" for p in range(len(programs))]
    return dict(cell_types=list(cell_types), programs=list(programs), margin=margin,
                common=c.tolist(), common_lo=clo.tolist(), common_hi=chi.tolist(),
                deviation=d.tolist(), deviation_lo=dlo.tolist(), deviation_hi=dhi.tolist(),
                type_label=lab.tolist(), pattern=pat, net_change=net)
