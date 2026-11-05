"""Fraction-error propagation in the v4 base estimator (v4 development, 2026-11-05; follows fraction_error.py,
where per-sample fraction noise broke the GLS intervals on McFarland: coverage 0.50).

Per dataset x split, fractions as in fraction_error.py (true / lite / rand / sys), GLS unmixing + off-program
calibration (offprog_sd = tau) with the simulated cohorts' fractions perturbed by frac_sd:
  none - frac_sd = 0 (the current v4 base)
  est  - frac_sd estimated from the data (gls_headroom.estimate_frac_sd: misfit matching)
  orac - (rand only) frac_sd = the noise actually added
Usage: python frac_prop.py DATASETS SPLITS -> benchmarks/dev/frac_prop/{ds}__s{split}.json
"""
import json, os, sys
import numpy as np
from scipy.special import softmax

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import gls_headroom as gh
from fraction_error import score, perturb_rand
from program2cell.core.realloc import reallocate, em_fractions
from program2cell.core import unmix as um

ANCHOR = os.environ.get("FRAC_ANCHOR", "excess")
OUT = os.path.join(ROOT, "benchmarks", "dev", "frac_prop" + ("" if ANCHOR == "absolute" else "_" + ANCHOR))


def run(datasets, splits):
    os.makedirs(OUT, exist_ok=True)
    for dsn in datasets:
        for split in splits:
            f = os.path.join(OUT, "%s__s%d.json" % (dsn, split))
            if os.path.exists(f):
                continue
            pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
            beta = np.asarray(beta)
            Y, Bm, G = np.asarray(inp.Y, float), np.asarray(inp.B, float), np.asarray(inp.Gmat, float)
            rng = np.random.default_rng(split + 7)                       # same perturbations as fraction_error
            pt = np.asarray(truth["pi"], float)
            fr = dict(true=pt, lite=em_fractions(Y, Bm))
            fr["rand"], s_r, r_r = perturb_rand(pt, rng)
            fr["sys"] = softmax(np.log(np.clip(pt, 1e-4, None)) + rng.normal(0, 0.5, pt.shape[1])[None], 1)
            paired = dsn in ("kang", "zhang")
            out = dict(dataset=dsn, split=split, beta=beta.tolist(), rand_sd=float(s_r), rand_r=r_r, anchor=ANCHOR)
            for k, pi in fr.items():
                raw = reallocate(inp.Y, pi, inp.B, inp.Gmat)
                A = um.leakage_operator(pi, inp.B, inp.Gmat, diagonal=False)
                b, se, s2 = gh.gls_unmix(np.asarray(raw), np.asarray(A), T, donors if paired else None)
                tau = float(um.offprogram_tau(Y, pi, Bm, G, T, b)[0])
                s_est, info = gh.estimate_frac_sd(cal, pi, Y, T, inp.B, inp.Gmat, b, offprog_sd=tau, seed=split,
                                                     anchor=ANCHOR)
                out["frac_sd_" + k] = s_est
                out["misfit_" + k] = info
                levels = dict(none=0.0, est=s_est)
                if k == "rand":
                    levels["orac"] = float(s_r)
                for lv, s in levels.items():
                    eff, lo, hi, kk = gh.gls_calibrated(cal, pi, T, inp.B, inp.Gmat, b, seed=split, offprog_sd=tau,
                                                        frac_sd=s)
                    sc = out["%s_%s" % (k, lv)] = score(beta, eff, lo, hi)
                    print("%s s%d %-4s %-4s frac_sd %.2f cov %.2f r %.2f within %.2f null %.2f power %.2f width %.3f"
                          % (dsn, split, k, lv, s, sc["cov"], sc["r"], sc["within_r"] or np.nan, sc["null_calls"],
                             sc["power"], sc["width"]), flush=True)
            json.dump(out, open(f, "w"))


if __name__ == "__main__":
    run(sys.argv[1].split(","), cl.db._splits(sys.argv[2]))
