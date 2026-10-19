"""Robustness: platform bias and a hidden (missing) cell type.
Settings (3 reps x N=24 each by default):
  well_specified   : model matches simulator
  platform_no_m    : hidden gene-wise platform bias (sd 0.3), model has no m_g
  platform_with_m  : same data, model fits m_g
  hidden_cell      : simulator has C+1 cell types, model's reference lacks the last one
Usage: python scripts/run_misspec.py [reps] [N]
"""
import sys, json, time
import numpy as np
import palm_bulk as pb
from palm_bulk import sim

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
SETTINGS = {
    "well_specified": dict(platform_sd=0.0, hide=False, use_m=False),
    "platform_no_m": dict(platform_sd=0.3, hide=False, use_m=False),
    "platform_with_m": dict(platform_sd=0.3, hide=False, use_m=True),
    "hidden_cell": dict(platform_sd=0.0, hide=True, use_m=False),
}
out, t0 = {}, time.time()
for name, cfg in SETTINGS.items():
    m = dict(I_mae=[], K_mae=[], I_cov=[], K_cov=[], beta_cov=[], l1=[], width=[], div=0)
    for r in range(REPS):
        rng = np.random.default_rng(900 + r)
        Ctrue = 5 if cfg["hide"] else 4
        B, G = sim.make_reference(rng, C=Ctrue, contrast=1.0)
        s = sim.simulate_causal(rng, B, G, N=N, platform_sd=cfg["platform_sd"])
        Bm = B[:-1] / B[:-1].sum(1, keepdims=True) if cfg["hide"] else B
        res = pb.fit(s["Y"], s["L"], s["T"], Bm, G, theta=s["theta"], use_m=cfg["use_m"],
                     num_warmup=250, num_samples=250, seed=r)
        m["div"] += res["divergences"]
        tr, dec = s["truth"], res["dec"]
        I_t, K_t = tr["I"].sum(0), tr["K"].sum(0)               # per program, full truth
        I_d, K_d = dec["I"].sum(1), dec["K"].sum(1)
        m["I_mae"] += np.abs(np.median(I_d, 0) - I_t).tolist()
        m["K_mae"] += np.abs(np.median(K_d, 0) - K_t).tolist()
        m["I_cov"].append(pb.coverage(I_d, I_t)); m["K_cov"].append(pb.coverage(K_d, K_t))
        m["width"].append(pb.mean_width(I_d))
        if not cfg["hide"]:
            m["beta_cov"].append(pb.coverage(res["beta"], s["beta"]))
            At = tr["I"] / tr["I"].sum(0, keepdims=True)
            Ad = res["dec"]["I"] / res["dec"]["I"].sum(1, keepdims=True)
            m["l1"] += np.abs(np.nanmedian(Ad, 0) - At).sum(0).tolist()
        print(name, "rep", r, "%.0fs" % (time.time() - t0), flush=True)
    out[name] = dict(intrinsic_total_MAE=float(np.mean(m["I_mae"])),
                     composition_total_MAE=float(np.mean(m["K_mae"])),
                     intrinsic_total_90cov=float(np.mean(m["I_cov"])),
                     composition_total_90cov=float(np.mean(m["K_cov"])),
                     intrinsic_total_mean_90width=float(np.mean(m["width"])),
                     intrinsic_share_L1=float(np.mean(m["l1"])) if m["l1"] else None,
                     beta_90cov=float(np.mean(m["beta_cov"])) if m["beta_cov"] else None,
                     divergences=m["div"])
out["provenance"] = pb.provenance(reps=REPS, N=N, settings=SETTINGS)
json.dump(out, open("misspec_summary.json", "w"), indent=2)
print(json.dumps(out, indent=2))
