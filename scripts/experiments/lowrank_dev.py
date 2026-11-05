"""Low-rank type x program deviations (v4 development, idea register 1) on the v4 base estimator.

Per dataset x split: GLS unmixing with PALM-lite fractions, off-program-calibrated (gls_calibrated with
offprog_sd = tau), deviations from the shared response left full (k=None) or reduced to rank 1 / 2
(gls_headroom.lowrank_deviation), the same transform applied inside the calibration simulations.
Prediction (methods review): gain on cell lines (design limit 0.04-0.05), little on Kang (0.19).
Usage: python lowrank_dev.py DATASETS SPLITS -> benchmarks/dev/lowrank/{ds}__s{split}.json
"""
import json, os, sys
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import gls_headroom as gh
from fraction_error import score
from program2cell.core.realloc import reallocate, em_fractions
from program2cell.core import unmix as um

OUT = os.path.join(ROOT, "benchmarks", "dev", "lowrank")


def run(datasets, splits, ranks=(None, 1, 2)):
    os.makedirs(OUT, exist_ok=True)
    for dsn in datasets:
        for split in splits:
            f = os.path.join(OUT, "%s__s%d.json" % (dsn, split))
            if os.path.exists(f):
                continue
            pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
            beta = np.asarray(beta)
            Y, B, G = np.asarray(inp.Y, float), np.asarray(inp.B, float), np.asarray(inp.Gmat, float)
            pi = em_fractions(Y, B)
            raw = reallocate(inp.Y, pi, inp.B, inp.Gmat)
            A = um.leakage_operator(pi, inp.B, inp.Gmat, diagonal=False)
            paired = dsn in ("kang", "zhang")
            b, se, s2 = gh.gls_unmix(np.asarray(raw), np.asarray(A), T, donors if paired else None)
            tau = float(um.offprogram_tau(Y, pi, B, G, T, b)[0])
            out = dict(dataset=dsn, split=split, beta=beta.tolist(), tau=tau, programs=list(inp.programs),
                       cell_types=list(inp.cell_types))
            for k in ranks:
                post = (lambda bb, ss, pf, k=k: gh.lowrank_deviation(bb, ss, pf.mean(0), k))
                bs = post(b, se, pi)
                eff, lo, hi, kk = gh.gls_calibrated(cal, pi, T, inp.B, inp.Gmat, bs, seed=split, offprog_sd=tau,
                                                    post=post if k is not None else None)
                name = "full" if k is None else "rank%d" % k
                out[name] = dict(eff=eff.tolist(), lo=lo.tolist(), hi=hi.tolist(), **score(beta, eff, lo, hi))
                s = out[name]
                print("%s s%d %-5s cov %.2f r %.2f within %.2f null %.2f power %.2f width %.3f" % (
                    dsn, split, name, s["cov"], s["r"], s["within_r"] or np.nan, s["null_calls"], s["power"],
                    s["width"]), flush=True)
            json.dump(out, open(f, "w"))


if __name__ == "__main__":
    run(sys.argv[1].split(","), cl.db._splits(sys.argv[2]))
