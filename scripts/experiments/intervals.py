"""Effect-interval choice on real data with the packaged pipeline (palm_bulk.program_deconvolution).

For each dataset x prior x split: PALM posterior fractions (fitted once with PROGENy-7, moved to the prior's
genes) and PALM-lite, each through program_deconvolution with every effect interval type; scored against
the truth (thinned held-out half) with dev_bench's metrics. One json per (dataset, prior, split) in
benchmarks/dev/intervals/. Restart-safe.
Usage: python intervals.py run DATASETS PRIORS SPLITS      e.g. run kang,pbmc,o1k progeny7,hallmark 0-9
       python intervals.py summary
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dev_bench as db

OUT = os.path.join(ROOT, "benchmarks", "dev", "intervals")
KINDS = ("normal", "strat", "lk", "lka")


def run(datasets, priors, splits):
    import palm_bulk as pb
    from palm_bulk.core import PseudobulkTable
    from palm_bulk.compare.base import MethodResult
    os.makedirs(OUT, exist_ok=True)
    for dsn in datasets:
        ds = db.DATASETS[dsn]
        table = db._table(ds)
        prog7 = db._programs(pb, "progeny7")
        for prior in priors:
            prog = db._programs(pb, prior)
            for split in splits:
                f = os.path.join(OUT, "%s__%s__s%d.json" % (dsn, prior, split))
                z7 = os.path.join(ROOT, "benchmarks", "dev", dsn, "s%d__joint_nb.npz" % split)
                if os.path.exists(f) or not os.path.exists(z7):
                    continue
                t0 = time.time()
                b = db._benchmark(table, ds, split)
                inp = pb.prepare_inputs(b.bulk, b.reference, prog, treatment=b.treatment, max_genes=1000)
                inp7 = inp if prior == "progeny7" else pb.prepare_inputs(b.bulk, b.reference, prog7,
                                                                         treatment=b.treatment, max_genes=1000)
                truth = b.truth(inp)
                truth["beta"] = pb.core.contrast(inp.T)(truth["delta"])
                mass = lambda genes: b.reference.B[:, [b.reference.genes.index(g) for g in genes]].sum(1)
                ratio = mass(inp.genes) / mass(inp7.genes)
                z = np.load(z7)["pi"]
                z = z[np.linspace(0, len(z) - 1, 50).astype(int)] * ratio
                z = z / z.sum(-1, keepdims=True)
                cal = pb.calibration_from_table(table, set(b.meta["donor_id"].astype(str)), inp.cell_types, inp.genes,
                                                inp.Gmat, ref_filter=ds.get("ref_filter"),
                                                all_specimens=ds.get("all_test_specimens", False), seed=split)
                rows = []
                for src, kw in (("palm", dict(fraction_draws=z)), ("lite", dict(engine="lite"))):
                    for kind in KINDS:
                        r = pb.program_deconvolution(inp, cal, effect_interval=kind, seed=split, estimator="diff", **kw)
                        s = db._score(MethodResult(src, fractions=r.fractions, activity=r.activity, effects=r.effects,
                                                   intervals=dict(activity=r.activity_interval,
                                                                  effects=r.effects_interval)),
                                      truth, dict(pi=truth["pi"], delta=truth["delta"]))
                        s.update(dataset=dsn, prior=prior, split=split, source=src, interval=kind,
                                 n_cal=len(cal), calibration=r.info["calibration"])
                        rows.append(s)
                with open(f, "w") as fh:
                    json.dump(rows, fh, indent=1, default=float)
                print("[%s] %s %s split %d done (%d calibration specimens, %.0fs)" % (
                    time.strftime("%H:%M"), dsn, prior, split, len(cal), time.time() - t0), flush=True)


def summary():
    rows = [r for f in sorted(glob.glob(os.path.join(OUT, "*.json"))) for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    cols = dict(act_spearman_across_samples="rho", act_cov90="act_cov", eff_pearson="eff_r", eff_cov90="eff_cov",
                eff_iscore="eff_is", eff_fsr="fsr", eff_fdr="fdr", eff_n_called="calls", eff_localise="local")
    t = df.groupby(["dataset", "prior", "source", "interval"], sort=False)[list(cols)].mean().rename(columns=cols)
    t.insert(0, "n", df.groupby(["dataset", "prior", "source", "interval"], sort=False).split.nunique())
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(t.round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(sys.argv[2].split(","), sys.argv[3].split(","), db._splits(sys.argv[4]))
