"""PALM freeze v2 (estimator="auto") on InstaPrism's fractions, with the per-type arrays kept (2026-11-12).

practice_vs_palm.py's palm_module rows hold scores only, run with the v1 settings; this reruns the PALM module on
the same InstaPrism fractions with freeze v2 and saves effects, intervals, calls and the fractions, for the talk
figure InstaPrism + PALM vs PALM (talk_figures.fig_ida_fractions).
SOURCE=truth (2026-11-12, plan W2/W6 oracle): the true read fractions (truth["pi"]) instead of InstaPrism's, to
test whether the fraction step causes the leakage residue on non-responding lines.
Usage: [SOURCE=instaprism|truth] python insta_palm_v2.py DATASET SPLITS   (e.g. mcf_ida 40-49)
Writes benchmarks/dev/practice/<ds>__s<k>__<SOURCE>__palm_v2.json. Restart-safe.
"""
import json, os, sys, time
import numpy as np

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts", "experiments"))
import causal_loop as cl
import practice_vs_palm as pv


SOURCE = os.environ.get("SOURCE", "instaprism")


def main(dsn, splits):
    for k in splits:
        f = os.path.join(pv.OUT, "%s__s%d__%s__palm_v2.json" % (dsn, k, SOURCE))
        if os.path.exists(f):
            continue
        t0 = time.time()
        pb, inp, T, beta, truth, src, fsrc, cal, donors = cl._setup(dsn, k)
        pi = np.clip(np.asarray(truth["pi"], float) if SOURCE == "truth" else
                     np.load(os.path.join(pv.DEV, dsn, "s%d__instaprism.npz" % k))["model__fractions"], 1e-6, None)
        pi = pi / pi.sum(1, keepdims=True)
        r = pb.program_deconvolution(inp, cal, fractions=pi, seed=k, **cl.ESTIMATORS["auto_v2"])
        lo, hi = r.effects_interval
        s = cl.score(r.effects, lo, hi, r.calls, beta, list(inp.cell_types), list(inp.programs), cl.KNOWN.get(dsn))
        s.update(dataset=dsn, split=k, method=SOURCE, kind="palm_v2", seconds=time.time() - t0,
                 arrays=dict(effects=r.effects.tolist(), lo=lo.tolist(), hi=hi.tolist(), calls=r.calls.tolist(),
                             beta=beta.tolist(), cell_types=list(inp.cell_types), programs=list(inp.programs),
                             fractions=pi.tolist(), fractions_palm=np.asarray(r.fractions, float).tolist()))
        json.dump(s, open(f, "w"))
        print("[%s] %s s%d %s + PALM v2: cov %.2f fpr_null %.2f (%.0fs)" % (
            time.strftime("%H:%M"), dsn, k, SOURCE, s["cov"], s["fpr_null"], s["seconds"]), flush=True)


if __name__ == "__main__":
    a0, a1 = map(int, sys.argv[2].split("-"))
    main(sys.argv[1], range(a0, a1 + 1))
