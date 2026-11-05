"""Composition measured with error (v4 development, idea register 2): how much cross-type error does fraction
error alone induce, and do the intervals know about it?

Per dataset x split, the same study analysed with fractions:
  true   - the truth's read fractions (no fraction error)
  lite   - PALM-lite estimates (what the pipeline uses)
  rand   - true, perturbed per sample: softmax(log pi + N(0, s^2)), s set so the per-type correlation with the
           truth across samples is ~0.5 (COMBAT real bulk: 0.09-0.70)
  sys    - true, perturbed per type: one log-scale bias N(0, 0.5^2) per type shared by all samples
           (a platform gap; correlation unchanged, levels wrong)
Estimators: v2 effect step (program_deconvolution, estimator auto, given fractions) and the v4 base
(gls_unmix + gls_calibrated with offprog_sd = tau).
Scores: cov, r, within-program r, null calls, power. Usage: python fraction_error.py DATASETS SPLITS
-> benchmarks/dev/fraction_error/{ds}__s{split}.json
"""
import json, os, sys
import numpy as np
from scipy.special import softmax

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import gls_headroom as gh
from program2cell.core.realloc import reallocate, em_fractions
from program2cell.core import unmix as um

OUT = os.path.join(ROOT, "benchmarks", "dev", "fraction_error")


def score(b, e, lo, hi):
    c = (lo > 0) | (hi < 0)
    big, null = np.abs(b) >= 0.1, np.abs(b) < 0.05
    ws = [np.corrcoef(e[:, p], b[:, p])[0, 1] for p in range(b.shape[1]) if b[:, p].std() >= 0.05]
    return dict(cov=float(((b >= lo) & (b <= hi)).mean()), r=float(np.corrcoef(e.ravel(), b.ravel())[0, 1]),
                rmse=float(np.sqrt(((e - b) ** 2).mean())),
                power=float((c & big & (np.sign(e) == np.sign(b))).sum() / max(big.sum(), 1)),
                null_calls=float((c & null).sum() / max(null.sum(), 1)),
                within_r=float(np.mean(ws)) if ws else None, width=float((hi - lo).mean()))


def perturb_rand(pi, rng, target_r=0.5):
    lp = np.log(np.clip(pi, 1e-4, None))
    for s in np.linspace(0.05, 3, 60):
        z = softmax(lp + rng.normal(0, s, pi.shape), 1)
        r = np.nanmean([np.corrcoef(z[:, c], pi[:, c])[0, 1] for c in range(pi.shape[1])])
        if r <= target_r:
            return z, s, float(r)
    return z, s, float(r)


def run(datasets, splits):
    os.makedirs(OUT, exist_ok=True)
    for dsn in datasets:
        for split in splits:
            f = os.path.join(OUT, "%s__s%d.json" % (dsn, split))
            if os.path.exists(f):
                continue
            pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, split)
            beta = np.asarray(beta)
            rng = np.random.default_rng(split + 7)
            pt = np.asarray(truth["pi"], float)
            fr = dict(true=pt, lite=em_fractions(np.asarray(inp.Y, float), np.asarray(inp.B, float)))
            fr["rand"], s_r, r_r = perturb_rand(pt, rng)
            fr["sys"] = softmax(np.log(np.clip(pt, 1e-4, None)) + rng.normal(0, 0.5, pt.shape[1])[None], 1)
            out = dict(dataset=dsn, split=split, beta=beta.tolist(), rand_sd=float(s_r), rand_r=r_r,
                       frac_r={k: float(np.nanmean([np.corrcoef(v[:, c], pt[:, c])[0, 1] for c in range(pt.shape[1])]))
                               for k, v in fr.items()},
                       frac_mae={k: float(np.abs(v - pt).mean()) for k, v in fr.items()})
            paired = dsn in ("kang", "zhang")
            for k, pi in fr.items():
                kw = dict(estimator="auto")
                if paired:
                    kw["groups"] = donors
                r = pb.program_deconvolution(inp, cal, fractions=pi, seed=split, **kw)
                lo, hi = (np.asarray(x) for x in r.effects_interval)
                out["v2_" + k] = score(beta, np.asarray(r.effects), lo, hi)
                raw = reallocate(inp.Y, pi, inp.B, inp.Gmat)
                A = um.leakage_operator(pi, inp.B, inp.Gmat, diagonal=False)
                b, se, s2 = gh.gls_unmix(np.asarray(raw), np.asarray(A), T, donors if paired else None)
                tau = float(um.offprogram_tau(np.asarray(inp.Y, float), pi, np.asarray(inp.B, float),
                                              np.asarray(inp.Gmat, float), T, b)[0])
                eff, lo, hi, kk = gh.gls_calibrated(cal, pi, T, inp.B, inp.Gmat, b, seed=split, offprog_sd=tau)
                out["gls_" + k] = score(beta, eff, lo, hi)
                for est in ("v2_", "gls_"):
                    s = out[est + k]
                    print("%s s%d %-4s %-4s frac r %.2f  cov %.2f r %.2f within %.2f null %.2f power %.2f" % (
                        dsn, split, est[:-1], k, out["frac_r"][k], s["cov"], s["r"], s["within_r"] or np.nan,
                        s["null_calls"], s["power"]), flush=True)
            json.dump(out, open(f, "w"))


if __name__ == "__main__":
    run(sys.argv[1].split(","), cl.db._splits(sys.argv[2]))
