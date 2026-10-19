"""S1: synthetic recovery and calibration.
Setting 'identifiable': program genes have cell-type-specific baselines (contrast=1).
Setting 'shared'      : program genes have identical baselines in all cell types (contrast=0).
Baseline B0: oracle read fractions x prior-mean program shift (attribution = the prior).
"""
import sys, time, json
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
from palm_bulk import attribution, shares, interval, coverage, cell_profiles, provenance
from palm_bulk.core.sim import make_reference, simulate

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
TAU, THETA = 0.7, 40.0


# Legacy simple model (flat prior scale, no treatment), kept here so S1 stays reproducible.
def palm_model_simple(Y, L, B, G, tau, theta):
    N, Gn = Y.shape
    C, P = B.shape[0], G.shape[0]
    eta = numpyro.sample("eta", dist.Normal(0.0, 1.5).expand([N, C]).to_event(2))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([N, C, P]).to_event(3))
    pi = numpyro.deterministic("pi", jax.nn.softmax(eta, axis=-1))
    delta = numpyro.deterministic("delta", tau * z)
    E = cell_profiles(B, G, delta)
    mu = L[:, None] * jnp.einsum("ic,icg->ig", pi, E)
    numpyro.sample("Y", dist.NegativeBinomial2(mu, theta).to_event(2), obs=Y)


def fit(Y, L, B, G, tau=0.7, theta=40.0, num_warmup=400, num_samples=400, seed=0):
    kernel = NUTS(palm_model_simple, init_strategy=init_to_median, target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1, progress_bar=False)
    Bj, Gj = jnp.array(B, jnp.float32), jnp.array(G, jnp.float32)
    mcmc.run(jax.random.PRNGKey(seed), jnp.array(Y, jnp.float32), jnp.array(L, jnp.float32), Bj, Gj, tau, theta)
    s = mcmc.get_samples()
    Cicp, S = jax.vmap(lambda p, d: attribution(Bj, Gj, p, d))(s["pi"], s["delta"])
    return dict(pi=np.array(s["pi"]), delta=np.array(s["delta"]), Cicp=np.array(Cicp), S=np.array(S),
                A=np.array(shares(Cicp)),
                divergences=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))


def b0_attribution(B, G, pi, active):
    """Prior-only: delta set to its prior-mean pattern (active mask x 0.8 x 1.1) in every sample."""
    dmean = active * 0.8 * 1.1
    d = jnp.broadcast_to(jnp.array(dmean), (pi.shape[0],) + dmean.shape)
    Cicp, _ = attribution(jnp.array(B), jnp.array(G), jnp.array(pi), d)
    return np.array(shares(Cicp, 1e-9))


def evaluate(setting, contrast, seed0):
    out = dict(l1_palm=[], l1_b0=[], cov_A=[], width_A=[], cov_pi=[], cov_delta=[],
               pi_rmse=[], S_corr=[], div=0, pts=[])
    for r in range(REPS):
        rng = np.random.default_rng(seed0 + r)
        B, G = make_reference(rng, contrast=contrast)
        sim = simulate(rng, B, G, N=N, tau=TAU, theta=THETA)
        res = fit(sim["Y"], sim["L"], B, G, tau=TAU, theta=THETA,
                  num_warmup=300, num_samples=300, seed=r)
        out["div"] += max(res["divergences"], 0)
        A_true = np.array(shares(jnp.array(sim["Cicp"]), 1e-9))           # (N,C,P)
        A_hat = np.nanmedian(res["A"], axis=0)
        A_b0 = b0_attribution(B, G, sim["pi"], sim["active"])
        on = (sim["delta"] > 0).any(axis=1) & (sim["S"] > 0)                # (N,P) program on
        for i in range(N):
            for p in range(G.shape[0]):
                if not on[i, p]:
                    continue
                out["l1_palm"].append(np.abs(A_hat[i, :, p] - A_true[i, :, p]).sum())
                out["l1_b0"].append(np.abs(A_b0[i, :, p] - A_true[i, :, p]).sum())
                lo, hi = interval(res["A"][:, i, :, p])
                out["cov_A"].extend(((A_true[i, :, p] >= lo) & (A_true[i, :, p] <= hi)).tolist())
                out["width_A"].extend((hi - lo).tolist())
                out["pts"].extend(zip(A_true[i, :, p].tolist(), A_hat[i, :, p].tolist(),
                                      lo.tolist(), hi.tolist()))
        out["cov_pi"].append(coverage(res["pi"], sim["pi"]))
        out["cov_delta"].append(coverage(res["delta"], sim["delta"]))
        out["pi_rmse"].append(np.sqrt(((res["pi"].mean(0) - sim["pi"]) ** 2).mean()))
        out["S_corr"].append(np.corrcoef(np.median(res["S"], 0).ravel(), sim["S"].ravel())[0, 1])
    return out


results = {}
t0 = time.time()
for name, contrast, seed in [("identifiable", 1.0, 100), ("shared", 0.0, 200)]:
    results[name] = evaluate(name, contrast, seed)
    print(name, "done %.0fs" % (time.time() - t0), flush=True)

summary = {}
for k, o in results.items():
    summary[k] = dict(
        n_sample_program_pairs=len(o["l1_palm"]),
        attribution_L1_PALM=float(np.mean(o["l1_palm"])),
        attribution_L1_prior_only_B0=float(np.mean(o["l1_b0"])),
        attribution_90pct_coverage=float(np.mean(o["cov_A"])),
        attribution_mean_90pct_width=float(np.mean(o["width_A"])),
        pi_90pct_coverage=float(np.mean(o["cov_pi"])),
        delta_90pct_coverage=float(np.mean(o["cov_delta"])),
        pi_RMSE=float(np.mean(o["pi_rmse"])),
        program_activity_corr=float(np.mean(o["S_corr"])),
        divergences=o["div"])
summary["provenance"] = provenance(reps=REPS, N=N, tau=TAU, theta=THETA)
json.dump(summary, open("s1_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for a, (k, title) in zip(ax[:2], [("identifiable", "Identifiable: cell-type-specific baselines"),
                                  ("shared", "Shared program: equal baselines")]):
    pts = np.array(results[k]["pts"])
    if len(pts):
        idx = np.random.default_rng(0).choice(len(pts), size=min(250, len(pts)), replace=False)
        t, m, lo, hi = pts[idx].T
        a.errorbar(t, m, yerr=[np.clip(m - lo, 0, None), np.clip(hi - m, 0, None)], fmt="o",
                   ms=3, alpha=0.5, elinewidth=0.6)
    a.plot([0, 1], [0, 1], "k--", lw=1)
    a.set_xlim(-0.1, 1.1); a.set_ylim(-0.4, 1.4)
    a.set_xlabel("true attribution share"); a.set_ylabel("posterior median (90% interval)")
    a.set_title(title, fontsize=10)
labels = ["identifiable", "shared"]
x = np.arange(2)
ax[2].bar(x - 0.18, [summary[k]["attribution_L1_PALM"] for k in labels], 0.36, label="PALM (posterior median)")
ax[2].bar(x + 0.18, [summary[k]["attribution_L1_prior_only_B0"] for k in labels], 0.36, label="B0: prior-only")
ax[2].set_xticks(x); ax[2].set_xticklabels(labels)
ax[2].set_ylabel("attribution L1 error (lower is better)"); ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig("s1_results.png", dpi=130)
print("total %.0fs" % (time.time() - t0))
