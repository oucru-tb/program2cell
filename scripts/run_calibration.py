"""Calibration and efficiency check for fit options, on simulated data with known truth.

For each replicate dataset, fits one configuration and records:
  * coverage of 50% and 90% credible intervals (should be ~0.5 and ~0.9) for read fractions pi,
    treatment effects beta, and the intrinsic / composition / total program effects;
  * error of posterior medians (MAE);
  * sampler efficiency: effective sample size per 1000 gradient evaluations, wall time.
Data come from palm_bulk.sim (not the model prior), so this checks calibration under a
realistic simulator rather than exact simulation-based calibration.

Usage:
  python run_calibration.py CONFIG [reps] [out.json]      run one configuration
  python run_calibration.py --summary file1.json ...        print a comparison table
Configs: baseline, reparam, svi (init from VI, warmup 150), float64, gene_disp, shared_on_overdisp,
         covariates, no_covariates (the last four use harder data, see SCENARIOS)."""
import sys, time, json

# Settings are explicit (not package defaults) so saved results stay reproducible. "baseline" is the
# original sampler; the package defaults (init='svi', dispersion='gene') were chosen from these runs.
BASE = dict(init="median", dispersion="shared", reparam=False)   # the original sampler
CONFIGS = {
    "baseline": dict(scenario="standard", fit=dict(BASE)),
    "reparam": dict(scenario="standard", fit=dict(BASE, reparam=True)),
    "svi": dict(scenario="standard", fit=dict(BASE, init="svi", svi_steps=2000), warmup=150),
    "float64": dict(scenario="standard", fit=dict(BASE), precision="float64"),
    "gene_disp": dict(scenario="overdispersed", fit=dict(BASE, dispersion="gene")),
    "shared_on_overdisp": dict(scenario="overdispersed", fit=dict(BASE)),
    "covariates": dict(scenario="confounded", fit=dict(BASE), use_X=True),
    "no_covariates": dict(scenario="confounded", fit=dict(BASE)),
    "defaults": dict(scenario="overdispersed", fit={}, warmup=200),   # current package defaults
}
SCENARIOS = {
    "standard": dict(),
    "overdispersed": dict(theta_sd=0.6),            # gene-wise dispersion in the data
    "confounded": dict(confounding=0.8),            # batch linked to treatment, shifts both levels
}
N, C, P, GN, WARMUP, SAMPLES, CHAINS = 24, 4, 3, 300, 250, 250, 2


def run(name, reps, out):
    cfg = CONFIGS[name]
    import palm_bulk as pb
    pb.configure_compute("cpu", parallel_chains=1, precision=cfg.get("precision", "float32"))
    import numpy as np
    from numpyro.diagnostics import effective_sample_size
    from palm_bulk import sim

    def cover(draws, truth, level):
        lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2], axis=0)
        return float(((truth >= lo) & (truth <= hi)).mean())

    def ess_per_kgrad(x, grads):
        e = np.asarray(effective_sample_size(x.reshape(CHAINS, -1, int(np.prod(x.shape[1:])))))
        return float(np.median(e) / grads * 1000), float(np.min(e) / grads * 1000)

    rows = []
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        B, G = sim.make_reference(rng, C=C, P=P, Gn=GN, genes_per_prog=30)
        s = sim.simulate_causal(rng, B, G, N=N, platform_sd=0.2, **SCENARIOS[cfg["scenario"]])
        kw = dict(use_m=True, num_warmup=cfg.get("warmup", WARMUP), num_samples=SAMPLES, chains=CHAINS,
                  chain_method="sequential", seed=r, **cfg["fit"])
        if cfg.get("use_X"):
            kw["X"] = s["X"]
        t = time.time()
        res = pb.fit(s["Y"], s["Y"].sum(1), s["T"], B, G, **kw)
        el = time.time() - t
        grads = res["mean_steps"] * (kw["num_warmup"] + SAMPLES) * CHAINS
        tr, dec = s["truth"], res["dec"]
        row = dict(rep=r, time=el, mean_steps=res["mean_steps"], divergences=res["divergences"],
                   rhat_max=res.get("rhat_max"))
        for key, draws, truth in [("pi", res["pi"], s["pi"]), ("beta", res["beta"], s["beta"]),
                                  ("I", dec["I"], tr["I"]), ("K", dec["K"], tr["K"]),
                                  ("total", dec["total"], tr["total"])]:
            row[key + "_cov50"] = cover(draws, truth, 0.5)
            row[key + "_cov90"] = cover(draws, truth, 0.9)
            row[key + "_mae"] = float(np.abs(np.median(draws, 0) - truth).mean())
        row["ess_pi_med"], row["ess_pi_min"] = ess_per_kgrad(res["pi"], grads)
        IK = np.concatenate([dec["I"].reshape(len(dec["I"]), -1), dec["K"].reshape(len(dec["K"]), -1)], 1)
        row["ess_IK_med"], row["ess_IK_min"] = ess_per_kgrad(IK, grads)
        rows.append(row)
        print(name, "rep", r, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    json.dump(dict(config=name, settings=cfg, rows=rows,
                   provenance=pb.provenance(config=name, reps=reps, N=N, C=C, P=P, genes=GN, chains=CHAINS)),
              open(out, "w"), indent=1)


def summary(files):
    import numpy as np
    keys = ["pi_cov90", "pi_cov50", "beta_cov90", "I_cov90", "K_cov90", "total_cov90",
            "pi_mae", "I_mae", "K_mae", "ess_pi_med", "ess_pi_min", "ess_IK_min", "mean_steps", "time", "divergences"]
    data = [json.load(open(f)) for f in files]
    print("| metric | " + " | ".join(d["config"] for d in data) + " |")
    print("|---" * (len(data) + 1) + "|")
    for k in keys:
        vals = [np.mean([r[k] for r in d["rows"]]) for d in data]
        print("| %s | " % k + " | ".join(("%.3g" % v) for v in vals) + " |")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2:])
    else:
        name = sys.argv[1]
        run(name, int(sys.argv[2]) if len(sys.argv) > 2 else 8,
            sys.argv[3] if len(sys.argv) > 3 else "calib_%s.json" % name)
