"""Why does gls_calibrated undercover on McFarland? Variants: study fractions (palm draws vs PALM-lite) x
off-program response in the simulation (0 vs offprogram_tau)."""
import json, os, sys
import numpy as np
ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import gls_headroom as gh
from program2cell.core.realloc import reallocate, em_fractions
from program2cell.core import unmix as um

OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
for dsn in sys.argv[1].split(","):
    for split in cl.db._splits(sys.argv[2]):
        pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
        beta = np.asarray(beta)
        Y, B, G = np.asarray(inp.Y, float), np.asarray(inp.B, float), np.asarray(inp.Gmat, float)
        pis = dict(lite=em_fractions(Y, B))
        src = {}
        if "fraction_draws" in src:
            pis["palm"] = np.median(src["fraction_draws"], 0)
        out = dict(beta=beta.tolist(), cell_types=list(inp.cell_types), programs=list(inp.programs))
        for fk, pi in pis.items():
            raw = reallocate(inp.Y, pi, inp.B, inp.Gmat)
            A = um.leakage_operator(pi, inp.B, inp.Gmat, diagonal=False)
            b, se, s2 = gh.gls_unmix(np.asarray(raw), np.asarray(A), T, None)
            tau = float(um.offprogram_tau(Y, pi, B, G, T, b)[0])
            for ok, pm in (("plant1", 1.0), ("plant2", 2.0)):
                eff, lo, hi, k = gh.gls_calibrated(cal, pi, T, inp.B, inp.Gmat, b, seed=split, offprog_sd=tau, plant_mult=pm)
                out["%s_%s" % (fk, ok)] = dict(eff=eff.tolist(), lo=lo.tolist(), hi=hi.tolist(), tau=tau)
                cov = ((beta >= lo) & (beta <= hi)).mean()
                calls = (lo > 0) | (hi < 0)
                print("%s s%d %s %s tau %.3f  cov %.2f  fpr_null %.2f  r %.2f  width %.3f" % (
                    dsn, split, fk, ok, tau, cov, calls[np.abs(beta) < 0.05].mean(),
                    np.corrcoef(eff.ravel(), beta.ravel())[0, 1], (hi - lo).mean()), flush=True)
        json.dump(out, open(os.path.join(OUT, "%s__s%d.json" % (dsn, split)), "w"))
