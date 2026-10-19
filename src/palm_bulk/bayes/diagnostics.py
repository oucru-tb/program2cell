"""Posterior summaries (intervals, coverage) and sampler convergence checks."""
import numpy as np
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

# Vehtari et al. (2021): split R-hat below 1.01 and at least ~100 effective draws per chain.
RHAT_MAX = 1.01
ESS_PER_CHAIN_MIN = 100


# ----- summaries of posterior draws (draw axis = 0) --------------------------
def interval(x, level=0.9):
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return np.nanquantile(x, lo, axis=0), np.nanquantile(x, hi, axis=0)


def coverage(samples, truth, level=0.9):
    lo, hi = interval(samples, level)
    ok = (truth >= lo) & (truth <= hi)
    valid = ~np.isnan(truth)
    return float(ok[valid].mean())


def mean_width(samples, level=0.9):
    lo, hi = interval(samples, level)
    return float(np.nanmean(hi - lo))


# ----- convergence ------------------------------------------------------------
def convergence(by_chain):
    """Split R-hat and effective sample size per quantity.
    by_chain: {name: array (chains, draws, ...)}. Works with one chain (split R-hat compares its
    halves), but several chains are needed to detect chains stuck in different modes.
    Returns {name: dict(rhat_max, ess_min, ess_median, n)}; R-hat ignores constant elements."""
    out = {}
    for name, x in by_chain.items():
        x = np.asarray(x, float)
        if x.ndim < 2 or x.shape[1] < 4:
            continue
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        flat = flat[..., np.isfinite(flat).all((0, 1)) & (flat.std((0, 1)) > 0)]
        if flat.shape[-1] == 0:
            continue
        rhat = np.asarray(split_gelman_rubin(flat))
        ess = np.asarray(effective_sample_size(flat), float)
        # numpyro's autocorrelation estimate can give ESS <= 0 for very short chains; such values are
        # meaningless, so they count as invalid (reported) instead of entering the minimum
        invalid = ~(ess > 0)
        ess = np.where(invalid, np.nan, ess)
        out[name] = dict(rhat_max=float(np.nanmax(rhat)),
                         ess_min=float(np.nanmin(ess)) if (~invalid).any() else float("nan"),
                         ess_median=float(np.nanmedian(ess)) if (~invalid).any() else float("nan"),
                         n=int(flat.shape[-1]), n_ess_invalid=int(invalid.sum()))
    return out


def convergence_problems(conv, chains, divergences=0, rhat_max=RHAT_MAX, ess_per_chain=ESS_PER_CHAIN_MIN):
    """Human-readable list of failed checks (empty if all pass)."""
    msgs = []
    bad_r = {k: round(v["rhat_max"], 3) for k, v in conv.items() if v["rhat_max"] > rhat_max}
    bad_e = {k: round(v["ess_min"]) for k, v in conv.items() if v["ess_min"] < ess_per_chain * chains}
    bad_i = {k: v["n_ess_invalid"] for k, v in conv.items() if v.get("n_ess_invalid")}
    if bad_r:
        msgs.append("R-hat above %.2f: %s" % (rhat_max, bad_r))
    if bad_e:
        msgs.append("effective sample size below %d: %s" % (ess_per_chain * chains, bad_e))
    if bad_i:
        msgs.append("effective sample size not estimable (chains too short) for some elements: %s" % bad_i)
    if divergences:
        msgs.append("%d divergent transitions" % divergences)
    if chains < 2:
        msgs.append("only one chain: R-hat cannot detect chains stuck in different modes (use chains >= 2)")
    return msgs
