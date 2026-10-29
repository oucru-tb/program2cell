"""Calibration of reallocated activity, conformal intervals and effect intervals (docs/method_spec.md, section 4).

Reallocation attenuates a cell type's own program shift and leaks the other types' shifts into it. The
calibration set measures both on reference specimens, where the truth is known:

  build_calibration_set   per-specimen per-type counts of reference donors (never test donors) -> K folds; for
                          fold k the profiles B_k come from the other donors, each specimen's counts are thinned
                          50/50: half 1 summed over types is a synthetic bulk with known read fractions, half 2
                          gives the truth d = estimate_delta(half 2, B_k).
  CalibrationSet.dhat     reallocated activity of the synthetic bulks, with the true fractions or with an
                          engine's own fractions (frac_fn), so fraction error can be inside the calibration.
  Calibrator              per (cell type, program) ridge map d ~ 1 + dhat (own type, or all types), out-of-fold
                          residuals, split-conformal activity intervals, and effect intervals:
                            normal   z * sqrt(bootstrap + residual/n + fraction-draw variance)
                            conf     conformal quantile of random pseudo-contrasts
                            strat    conformal quantile of pseudo-contrasts split by one type's true activity
                            lk       conf + sum_c' |L_cc'p| |beta_c'p| (leakage-aware; L from the calibration set)
                            lk+      as lk with |beta| + z se
                            lka      leakage-aware, asymmetric: other types' leakage widens both sides
                                     (sum_c'!=c |L_cc'p| |beta_c'p|); the own type's attenuation L_ccp only
                                     extends the side away from zero, to the de-attenuated estimate
                                     beta / (1 - L_ccp), so real effects keep their power
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import norm

from .scores import estimate_delta
from .realloc import reallocate


# ---- calibration set ------------------------------------------------------------------------------------
def _profiles(counts, donors, keep, floor=1e-6):
    """(C, G) reference profiles from the specimens `keep`: per donor and type sum, normalise, average over
    donors (as make_benchmark builds the reference)."""
    C, G = counts.shape[1:]
    B = np.zeros((C, G))
    for c in range(C):
        prof = []
        for d in np.unique(donors[keep]):
            s = counts[keep & (donors == d), c].sum(0)
            if s.sum() > 0:
                prof.append(s / s.sum())
        if not prof:
            raise ValueError("no reference counts for cell type %d" % c)
        B[c] = np.mean(prof, 0)
    B = B + floor
    return B / B.sum(1, keepdims=True)


@dataclass
class CalibrationSet:
    """Synthetic bulks with known truth, in folds by donor. Y[k] (n_k, G), pi[k] (n_k, C), d[k] (n_k, C, P),
    B[k] (C, G) profiles without fold k's donors, donors[k] (n_k,)."""
    Y: list
    pi: list
    d: list
    B: list
    donors: list
    G: np.ndarray
    planted: list = None          # planted[k] (n_k,) bool: specimen is a planted copy (plant_shifts)
    counts: list = None           # counts[k] (n_k, C, G) per-type counts (for design simulations, core.unmix)
    info: dict = field(default_factory=dict)

    def __len__(self):
        return sum(len(x) for x in self.Y)

    def is_planted(self):
        """(n,) bool over all folds, in the order of dhat()."""
        if self.planted is None:
            return np.zeros(len(self), bool)
        return np.concatenate([np.asarray(x, bool) for x in self.planted])

    def dhat(self, frac_fn=None):
        """(dhat, d, donors, fold) over all folds: reallocated activity of the synthetic bulks with the true
        fractions (frac_fn None) or with frac_fn(Y, B) -> pi, an engine's own estimate."""
        dh, d, dn, fk = [], [], [], []
        for k, (Y, pi, dd, B, don) in enumerate(zip(self.Y, self.pi, self.d, self.B, self.donors)):
            p = pi if frac_fn is None else frac_fn(Y, B)
            dh.append(reallocate(Y, p, B, self.G))
            d.append(dd); dn.append(np.asarray(don)); fk.append(np.full(len(Y), k))
        return np.concatenate(dh), np.concatenate(d), np.concatenate(dn), np.concatenate(fk)


def plant_shifts(counts, G, n_plant=3, scale=1.0, prob=0.5, max_log=6.0, seed=0):
    """Planted copies of per-specimen per-type counts (S, C, G) with known program shifts, so that the
    calibration set spans perturbation-sized effects (between-donor spread in reference specimens is 3-20x
    smaller than real drug or cytokine effects, so a control-only set cannot identify the attenuation and
    leakage map where it matters).

    Each copy picks one program p; every type independently (probability `prob`) gets a shift
    delta ~ U(-scale, scale) along it, i.e. its counts are redrawn as Poisson(x_g e^{delta G_pg}), rescaled
    to the type's original total (the generative model of estimate_delta). The truth is still measured by
    estimate_delta on the held-out half, so the clip of delta G_pg at +-max_log does not bias it.
    Independent shifts across types identify the leakage between them.
    Returns (S * n_plant, C, G) counts and the (S * n_plant, C, P) planted deltas."""
    counts = np.asarray(counts, float)
    G = np.asarray(G, float)
    rng = np.random.default_rng(seed + 7)
    S, C, _ = counts.shape
    P = G.shape[0]
    out, planted = [], []
    for _ in range(n_plant):
        delta = np.zeros((S, C, P))
        p = rng.integers(P, size=S)
        on = rng.random((S, C)) < prob
        delta[np.arange(S)[:, None], np.arange(C)[None], p[:, None]] = on * rng.uniform(-scale, scale, (S, C))
        logf = np.clip(np.einsum("scp,pg->scg", delta, G), -max_log, max_log)
        mu = counts * np.exp(logf)
        mu *= counts.sum(-1, keepdims=True) / np.maximum(mu.sum(-1, keepdims=True), 1e-12)
        out.append(rng.poisson(mu).astype(float))
        planted.append(delta)
    return np.concatenate(out), np.concatenate(planted)


def build_calibration_set(counts, donors, G, n_folds=5, seed=0, profile_counts=None, profile_donors=None,
                          n_plant=0, plant_scale=1.0, plant_prob=0.5):
    """CalibrationSet from per-specimen per-type counts (S, C, G) on the modelled genes of reference donors
    `donors` (S,). Folds are by donor (at most n_folds, at least 2 donors needed). Profiles B_k use
    `profile_counts` / `profile_donors` (default: the same specimens) without fold k's donors.
    n_plant > 0 adds n_plant copies of every specimen with planted program shifts (plant_shifts; same donor,
    so the same fold); profiles are always built from the unplanted counts."""
    counts = np.asarray(counts, float)
    donors = np.asarray(donors).astype(str)
    pc = counts if profile_counts is None else np.asarray(profile_counts, float)
    pdn = donors if profile_donors is None else np.asarray(profile_donors).astype(str)
    n_real = len(counts)
    if n_plant > 0:
        extra, _ = plant_shifts(counts, G, n_plant=n_plant, scale=plant_scale, prob=plant_prob, seed=seed)
        counts = np.concatenate([counts, extra])
        donors = np.concatenate([donors, np.tile(donors, n_plant)])
    planted = np.arange(len(counts)) >= n_real
    rng = np.random.default_rng(seed)
    ud = rng.permutation(np.unique(donors))
    if len(ud) < 2:
        raise ValueError("calibration needs specimens from at least 2 donors")
    kf = min(n_folds, len(ud))
    fold_of = {d: i % kf for i, d in enumerate(ud)}
    out = dict(Y=[], pi=[], d=[], B=[], donors=[], planted=[], counts=[])
    for k in range(kf):
        inside = np.array([fold_of[d] == k for d in donors])
        B = _profiles(pc, pdn, ~np.isin(pdn, [d for d in ud if fold_of[d] == k]))
        tc = counts[inside]
        half = rng.binomial(np.rint(tc).astype(np.int64), 0.5).astype(float)
        tot = half.sum(-1)
        out["Y"].append(half.sum(1))
        out["pi"].append(tot / tot.sum(1, keepdims=True))
        out["d"].append(estimate_delta(tc - half, B, G))
        out["B"].append(B)
        out["donors"].append(donors[inside])
        out["planted"].append(planted[inside])
        out["counts"].append(tc)
    return CalibrationSet(G=np.asarray(G, float), info=dict(n_folds=kf, n_donors=len(ud), n_real=n_real,
                                                            n_plant=n_plant, plant_scale=plant_scale), **out)


def calibration_from_table(table, exclude_donors, cell_types, genes, G, min_cells=20, ref_filter=None,
                           all_specimens=False, max_donors=300, n_folds=5, seed=0, n_plant=0, plant_scale=1.0,
                           plant_prob=0.5):
    """CalibrationSet from a PseudobulkTable: reference donors (not in exclude_donors, the bulk donors) whose
    specimens have >= min_cells cells of every type; one specimen per donor at random (all_specimens=True
    keeps all, for small studies), at most max_donors donors. Profiles use every reference row with >= 10
    cells (as make_benchmark). ref_filter: optional (column, value), e.g. ("label", "ctrl").
    n_plant, plant_scale, plant_prob: planted program shifts (build_calibration_set)."""
    from .pseudobulk import _csr
    obs = table.obs
    if ref_filter is not None:
        obs = obs[obs[ref_filter[0]] == ref_filter[1]]
    obs = obs[~obs["donor_id"].astype(str).isin(set(map(str, exclude_donors))) & obs["cell_type"].isin(cell_types)]
    X = _csr(table.counts)
    pos = {k: i for i, k in enumerate(table.obs.index)}
    gi = [list(table.genes).index(g) for g in genes]

    def spec_counts(specs, min_rows=0):
        out = np.zeros((len(specs), len(cell_types), len(gi)))
        for i, s in enumerate(specs):
            rows = obs[(obs["specimen"] == s) & (obs["n_cells"] >= min_rows)]
            for j, c in enumerate(cell_types):
                r = rows[rows["cell_type"] == c]
                if len(r):
                    out[i, j] = np.asarray(X[[pos[q] for q in r.index]].sum(0)).ravel()[gi]
        return out

    wide = obs.pivot_table(index="specimen", columns="cell_type", values="n_cells", aggfunc="sum",
                           fill_value=0).reindex(columns=cell_types, fill_value=0)
    spec = obs.drop_duplicates("specimen").set_index("specimen")
    ok = (wide >= min_cells).all(1)
    elig = spec.loc[ok[ok].index]
    rng = np.random.default_rng(seed + 100)
    ud = rng.permutation(np.array(sorted(map(str, elig["donor_id"].unique())), dtype=object))[:max_donors]
    specs, owners = [], []
    for d in ud:
        own = np.array(elig.index[elig["donor_id"].astype(str) == d], dtype=object)
        pick = list(own) if all_specimens else [rng.choice(own)]
        specs += pick; owners += [d] * len(pick)
    allspec = list(spec.index)
    prof = spec_counts(allspec, min_rows=10)
    return build_calibration_set(spec_counts(specs), owners, G, n_folds=n_folds, seed=seed,
                                 profile_counts=prof, profile_donors=spec.loc[allspec, "donor_id"].astype(str).values,
                                 n_plant=n_plant, plant_scale=plant_scale, plant_prob=plant_prob)


# ---- calibrator ---------------------------------------------------------------------------------------------
def contrast_weights(T):
    """(N,) weights of the difference of means: +1/n1 for treated, -1/n0 for controls."""
    T = np.asarray(T)
    return np.where(T == 1, 1.0 / max((T == 1).sum(), 1), -1.0 / max((T == 0).sum(), 1))


def contrast(T):
    """Treated-minus-control mean over samples: (..., N, C, P) -> (..., C, P)."""
    T = np.asarray(T)
    return lambda a: a[..., T == 1, :, :].mean(-3) - a[..., T == 0, :, :].mean(-3)


class Calibrator:
    """Per (c, p) ridge map from reallocated activity to the truth scale, fitted on a calibration set.
    own=True: d_cp ~ 1 + dhat_cp (default); own=False: d_cp ~ 1 + dhat_.p (all types).
    Own-type slopes are shrunk toward 1 (no recalibration) with the weight of `prior_n` calibration specimens
    and floored at `min_slope`, so a small calibration set cannot reverse a ranking (on Kang, 11 specimens
    gave one slope of -1.2 without this).
    noise: (n,) bool, the specimens whose residuals measure noise (conformal quantiles, residual variance);
    default all. With planted shifts (plant_shifts) only the real specimens: the planted ones fit the map and
    the leakage matrix, whose share of the error the lk intervals add explicitly."""
    INTERVALS = ("normal", "conf", "strat", "lk", "lk+", "lka")

    def __init__(self, dhat, d, donors, folds, own=True, ridge=1e-3, level=0.9, n_boot=200, n_rep=400, seed=0,
                 prior_n=5.0, min_slope=0.05, noise=None):
        self.dhat, self.d = np.asarray(dhat, float), np.asarray(d, float)
        self.donors, self.folds = np.asarray(donors).astype(str), np.asarray(folds)
        self.own, self.ridge, self.level = own, ridge, level
        self.prior_n, self.min_slope = prior_n, min_slope
        self.n_boot, self.n_rep = n_boot, n_rep
        self.rng = np.random.default_rng(seed)
        self.z = norm.ppf(0.5 + level / 2)
        self.W = self._fit(self.dhat, self.d)
        self.resid = np.zeros_like(self.d)
        for k in np.unique(self.folds):
            tr = self.folds != k
            self.resid[~tr] = self.d[~tr] - self._predict(self._fit(self.dhat[tr], self.d[tr]), self.dhat[~tr])
        self.noise = np.ones(len(self.d), bool) if noise is None else np.asarray(noise, bool)
        n = int(self.noise.sum())
        self.q = np.quantile(np.abs(self.resid[self.noise]), min(1.0, np.ceil((n + 1) * level) / n), axis=0)

    @classmethod
    def from_set(cls, cal, frac_fn=None, **kw):
        dh, d, dn, fk = cal.dhat(frac_fn)
        kw.setdefault("noise", ~cal.is_planted())
        return cls(dh, d, dn, fk, **kw)

    def _x(self, dh, c, p):
        Xp = dh[:, c:c + 1, p] if self.own else dh[:, :, p]
        return np.column_stack([np.ones(len(dh)), Xp])

    def _fit(self, dh, d):
        W = {}
        for c in range(d.shape[1]):
            for p in range(d.shape[2]):
                if self.own:                       # slope shrunk toward 1, floored; intercept by least squares
                    x, y = dh[:, c, p], d[:, c, p]
                    xc = x - x.mean()
                    sxx = (xc ** 2).sum()
                    lam = self.prior_n * sxx / max(len(x), 1) + self.ridge
                    slope = max(((xc * (y - y.mean())).sum() + lam) / (sxx + lam), self.min_slope)
                    W[c, p] = np.array([y.mean() - slope * x.mean(), slope])
                    continue
                Xm = self._x(dh, c, p)
                R = self.ridge * np.eye(Xm.shape[1]); R[0, 0] = 0
                W[c, p] = np.linalg.solve(Xm.T @ Xm + R, Xm.T @ d[:, c, p])
        return W

    def _predict(self, W, dh):
        dh = np.asarray(dh, float)
        flat = dh.reshape(-1, *dh.shape[-2:])
        out = np.zeros(flat.shape)
        for (c, p), w in W.items():
            out[:, c, p] = self._x(flat, c, p) @ w
        return out.reshape(dh.shape)

    def predict(self, dh):
        """Calibrated activity of reallocated activity (..., N, C, P)."""
        return self._predict(self.W, dh)

    def activity(self, dh, dh_draws=None):
        """Point (N, C, P) and conformal interval; fraction-draw spread (if draws) added in quadrature."""
        act = self.predict(dh)
        h = self.q[None]
        if dh_draws is not None:
            a = self.predict(dh_draws)
            h = np.sqrt(h ** 2 + ((np.quantile(a, 0.5 + self.level / 2, 0) - np.quantile(a, 0.5 - self.level / 2, 0))
                                  / 2) ** 2)
        return act, (act - h, act + h)

    def leak_matrix(self):
        """(C, C, P): slope of the out-of-fold residual of (c, p) on the true activity of type c'."""
        C, P = self.d.shape[1:]
        L = np.zeros((C, C, P))
        for p in range(P):
            D = np.column_stack([np.ones(len(self.d)), self.d[:, :, p] - self.d[:, :, p].mean(0)])
            R = 1e-3 * np.eye(C + 1); R[0, 0] = 0
            L[:, :, p] = np.linalg.solve(D.T @ D + R, D.T @ self.resid[:, :, p])[1:].T
        return L

    def _contrast_q(self, sw2, strat):
        res, d = self.resid[self.noise], self.d[self.noise]
        n = len(res)
        C, P = res.shape[1:]
        vals = np.zeros((self.n_rep, C, P))
        m = n // 3 if strat else n // 2
        for r in range(self.n_rep):
            if not strat:
                idx = self.rng.permutation(n)
                vals[r] = res[idx[:m]].mean(0) - res[idx[m:2 * m]].mean(0)
            else:
                cs = r % C
                sub = self.rng.choice(n, size=2 * m, replace=False)
                for p in range(P):
                    o = sub[np.argsort(d[sub, cs, p])]
                    vals[r, :, p] = res[o[m:]].mean(0)[:, p] - res[o[:m]].mean(0)[:, p]
        scale = np.sqrt(sw2 / (2 / m))                  # sum of squared estimator weights vs the pseudo-contrast's
        lev = min(1.0, np.ceil((self.n_rep + 1) * self.level) / self.n_rep)
        return np.quantile(np.abs(vals), lev, axis=0) * scale

    def effects(self, dh, T=None, dh_draws=None, interval="lk", weights=None):
        """Treatment effects (C, P) of calibrated activity, with an interval (lo, hi) of type `interval` (see
        module doc) and the parts of its variance. The effect estimator is any estimator linear in the
        outcome, beta = sum_n w_n activity_n: by default the difference of means of T (contrast_weights);
        pass `weights` (N,) for covariate adjustment, paired designs, IPW or AIPW (palm_bulk.causal)."""
        if interval not in self.INTERVALS:
            raise ValueError("interval must be one of %s" % (self.INTERVALS,))
        w = contrast_weights(T) if weights is None else np.asarray(weights, float)
        des = lambda a: np.einsum("n,...ncp->...cp", w, a)
        sw2 = float((w ** 2).sum())
        eff = des(self.predict(dh))
        ud = np.unique(self.donors)
        boot = []
        for _ in range(self.n_boot):
            pick = self.rng.choice(ud, len(ud))
            idx = np.concatenate([np.flatnonzero(self.donors == x) for x in pick])
            boot.append(des(self._predict(self._fit(self.dhat[idx], self.d[idx]), dh)))
        v_boot = np.var(boot, 0)
        v_draw = np.var(des(self.predict(dh_draws)), 0) if dh_draws is not None else np.zeros_like(eff)
        v_res = self.resid[self.noise].var(0) * sw2
        if interval == "normal":
            h = self.z * np.sqrt(v_boot + v_res + v_draw)
        else:
            qe = self._contrast_q(sw2, strat=(interval == "strat"))
            h = np.sqrt(qe ** 2 + self.z ** 2 * (v_boot + v_draw))
            L = np.abs(self.leak_matrix()) if interval.startswith("lk") else None
            if interval in ("lk", "lk+"):
                size = np.abs(eff)
                if interval == "lk+":
                    size = size + self.z * np.sqrt(v_boot + v_res + v_draw)
                h = h + np.einsum("ckp,kp->cp", L, size)
            elif interval == "lka":
                own = np.clip(np.einsum("ccp->cp", L), 0, 0.8)
                h = h + np.einsum("ckp,kp->cp", L, np.abs(eff)) - own * np.abs(eff)      # other types only
                ext = own / (1 - own) * np.abs(eff)
                lo = eff - h - np.where(eff < 0, ext, 0)
                hi = eff + h + np.where(eff > 0, ext, 0)
                return eff, (lo, hi), dict(boot=v_boot, residual=v_res, draws=v_draw)
        return eff, (eff - h, eff + h), dict(boot=v_boot, residual=v_res, draws=v_draw)
