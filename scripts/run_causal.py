"""Causal S1: randomized treatment T shifts composition (gamma) and per-cell-type
program activity (beta). Compare PALM's intrinsic/composition decomposition with
  - naive bulk score difference (total effect only)
  - two-step: NNLS fractions + reference baseline program score (non-oracle)
Usage: python run_causal.py [reps] [N]
"""
import sys, time, json
import numpy as np
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from palm_bulk import fit, interval, coverage, provenance
from palm_bulk.core.sim import make_reference, simulate_causal

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24


def two_step(Y, T, B, G):
    """Non-oracle baseline. Returns total, composition, intrinsic (per program) and
    prior-only intrinsic shares (per cell type, program)."""
    frac = Y / Y.sum(1, keepdims=True)
    pi_hat = np.array([nnls(B.T, f)[0] for f in frac])
    pi_hat /= pi_hat.sum(1, keepdims=True)
    score = frac @ G.T                                         # (N,P) bulk program score
    total = score[T == 1].mean(0) - score[T == 0].mean(0)
    s0 = B @ G.T                                               # (C,P) reference baseline score
    dpi = pi_hat[T == 1].mean(0) - pi_hat[T == 0].mean(0)
    comp = (dpi[:, None] * s0).sum(0)
    intr = total - comp
    w = pi_hat.mean(0)[:, None] * s0
    shares0 = w / w.sum(0, keepdims=True)
    return total, comp, intr, shares0


rows = dict(tot_palm=[], tot_naive=[], I_palm=[], I_two=[], K_palm=[], K_two=[],
            I_true=[], K_true=[], I_lo=[], I_hi=[], K_lo=[], K_hi=[],
            l1_palm=[], l1_b0=[], cov_share=[], cov_Icp=[], cov_Kcp=[],
            cov_beta=[], cov_gamma=[], div=0)
t0 = time.time()
for r in range(REPS):
    rng = np.random.default_rng(500 + r)
    B, G = make_reference(rng, contrast=1.0)
    sim = simulate_causal(rng, B, G, N=N)
    res = fit(sim["Y"], sim["L"], sim["T"], B, G, theta=sim["theta"], num_warmup=250, num_samples=250, seed=r)
    rows["div"] += res["divergences"]
    tr, dec = sim["truth"], res["dec"]
    tot_n, comp_n, intr_n, shares0 = two_step(sim["Y"], sim["T"], B, G)
    rows["cov_beta"].append(coverage(res["beta"], sim["beta"]))
    rows["cov_gamma"].append(coverage(res["gamma"], sim["gamma"]))
    rows["cov_Icp"].append(coverage(dec["I"], tr["I"]))
    rows["cov_Kcp"].append(coverage(dec["K"], tr["K"]))
    I_tot_true, K_tot_true = tr["I"].sum(0), tr["K"].sum(0)
    I_draw, K_draw = dec["I"].sum(1), dec["K"].sum(1)           # (draws,P)
    Ilo, Ihi = interval(I_draw); Klo, Khi = interval(K_draw)
    A_true = tr["I"] / tr["I"].sum(0, keepdims=True)
    A_draw = dec["I"] / dec["I"].sum(1, keepdims=True)
    A_hat = np.nanmedian(A_draw, 0)
    lo, hi = interval(A_draw)
    for p in range(G.shape[0]):
        rows["tot_palm"].append(abs(np.median(dec["total"][:, p]) - tr["total"][p]))
        rows["tot_naive"].append(abs(tot_n[p] - tr["total"][p]))
        rows["I_palm"].append(np.median(I_draw[:, p])); rows["I_two"].append(intr_n[p])
        rows["K_palm"].append(np.median(K_draw[:, p])); rows["K_two"].append(comp_n[p])
        rows["I_true"].append(I_tot_true[p]); rows["K_true"].append(K_tot_true[p])
        rows["I_lo"].append(Ilo[p]); rows["I_hi"].append(Ihi[p])
        rows["K_lo"].append(Klo[p]); rows["K_hi"].append(Khi[p])
        rows["l1_palm"].append(np.abs(A_hat[:, p] - A_true[:, p]).sum())
        rows["l1_b0"].append(np.abs(shares0[:, p] - A_true[:, p]).sum())
        rows["cov_share"].extend(((A_true[:, p] >= lo[:, p]) & (A_true[:, p] <= hi[:, p])).tolist())
    print("rep", r, "done %.0fs" % (time.time() - t0), flush=True)

R = {k: np.array(v) for k, v in rows.items() if k != "div"}
mae = lambda est, tru: float(np.mean(np.abs(est - tru)))
summary = dict(
    programs_evaluated=int(len(R["I_true"])),
    total_effect_MAE_palm=float(R["tot_palm"].mean()),
    total_effect_MAE_naive_bulk=float(R["tot_naive"].mean()),
    intrinsic_total_MAE_palm=mae(R["I_palm"], R["I_true"]),
    intrinsic_total_MAE_two_step=mae(R["I_two"], R["I_true"]),
    composition_total_MAE_palm=mae(R["K_palm"], R["K_true"]),
    composition_total_MAE_two_step=mae(R["K_two"], R["K_true"]),
    intrinsic_share_L1_palm=float(R["l1_palm"].mean()),
    intrinsic_share_L1_B0_reference_prior=float(R["l1_b0"].mean()),
    intrinsic_share_90pct_coverage=float(np.mean(rows["cov_share"])),
    I_cp_90pct_coverage=float(np.mean(rows["cov_Icp"])),
    K_cp_90pct_coverage=float(np.mean(rows["cov_Kcp"])),
    beta_90pct_coverage=float(np.mean(rows["cov_beta"])),
    gamma_90pct_coverage=float(np.mean(rows["cov_gamma"])),
    divergences=rows["div"], reps=REPS, N_per_rep=N, provenance=provenance(reps=REPS, N=N))
json.dump(summary, open("causal_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for a, (key, title) in zip(ax[:2], [("I", "Intrinsic effect (per program)"), ("K", "Composition effect (per program)")]):
    t = R[key + "_true"]
    est, two = R[key + "_palm"], R[key + "_two"]
    lo, hi = R[key + "_lo"], R[key + "_hi"]
    a.errorbar(t, est, yerr=[np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)], fmt="o", ms=4,
               label="PALM (90% interval)")
    a.plot(t, two, "x", color="tab:orange", label="two-step")
    lim = [min(t.min(), est.min(), two.min()) - 0.01, max(t.max(), est.max(), two.max()) + 0.01]
    a.plot(lim, lim, "k--", lw=1); a.set_xlabel("true"); a.set_ylabel("estimated"); a.set_title(title, fontsize=10)
    a.legend(fontsize=8)
ax[2].bar([0, 1], [summary["intrinsic_share_L1_palm"], summary["intrinsic_share_L1_B0_reference_prior"]],
          color=["tab:blue", "tab:orange"])
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["PALM", "B0: reference prior"])
ax[2].set_ylabel("intrinsic-share L1 error (lower is better)")
plt.tight_layout(); plt.savefig("causal_results.png", dpi=130)
print("total %.0fs" % (time.time() - t0))
