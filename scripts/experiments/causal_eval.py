"""Causal evaluation on paired perturbation datasets (Kang, Zhang) with the packaged pipeline.

Per dataset x prior x split and fraction source (PALM posterior from the PROGENy-7 fit, PALM-lite, CIBERSORT):
  - estimator 'auto' (PALM only): unmixing where the design check favours it (core.unmix, method_spec 4.5)
  - effects with the paired estimator ('ols', groups = donor) and with the difference of means ('diff'),
    each scored against the truth effect computed with the same estimator weights on the true activity
  - the intrinsic / composition decomposition (causal.decompose) vs the truth decomposition on the true
    fractions and true activity (compare.known_answer.decomposition_accuracy)
  - robustness values of the called effects
One json per (dataset, prior, split) in benchmarks/dev/causal/. Restart-safe.
Usage: python causal_eval.py run DATASETS PRIORS SPLITS   |   python causal_eval.py summary
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd

ROOT = "C:/Users/duy.pham/Documents/dev/program2cell"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dev_bench as db

OUT = os.path.join(ROOT, "benchmarks", "dev", "causal")
# known answers (McFarland): which (cell type, program) pairs the drug moves, and in which direction; the
# program's name in each prior (Hallmark has P53_PATHWAY; the dev_bench Hallmark set has no MAPK/KRAS program)
KNOWN = {"mcf_ida": dict(program=("p53", "P53_PATHWAY"), types=["LNCAPCLONEFGC", "RCC10RGB", "SNU1079"], sign=1),
         "mcf_tram": dict(program=("MAPK",), types=None, sign=-1)}


def run(datasets, priors, splits):
    import palm_bulk as pb
    from palm_bulk import causal
    from palm_bulk.core import PseudobulkTable
    from palm_bulk.compare.base import MethodResult
    from palm_bulk.compare import known_answer as ka
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
                T = np.asarray(b.treatment)
                donors = b.meta["donor_id"].astype(str).values
                truth = b.truth(inp)
                mass = lambda genes: b.reference.B[:, [b.reference.genes.index(g) for g in genes]].sum(1)
                ratio = mass(inp.genes) / mass(inp7.genes)
                to_prior = lambda p: (p * ratio) / (p * ratio).sum(-1, keepdims=True)
                z = np.load(z7)["pi"]
                z = to_prior(z[np.linspace(0, len(z) - 1, 50).astype(int)])
                sources = [("palm", dict(fraction_draws=z)), ("lite", dict(engine="lite"))]
                cib = os.path.join(ROOT, "benchmarks", "dev", dsn, "s%d__cibersort.npz" % split)
                if os.path.exists(cib):
                    sources.append(("cibersort", dict(fractions=to_prior(np.load(cib)["model__fractions"]))))
                cal = pb.calibration_from_table(table, set(donors), inp.cell_types, inp.genes, inp.Gmat,
                                                ref_filter=ds.get("ref_filter"),
                                                all_specimens=ds.get("all_test_specimens", False), seed=split)
                rows = []
                for est, grp in (("ols", donors), ("diff", None), ("auto", None)):
                    w = causal.estimator_weights(T, groups=grp, estimator="diff" if est == "auto" else est)
                    tr = dict(pi=truth["pi"], delta=truth["delta"], beta=np.einsum("n,ncp->cp", w, truth["delta"]))
                    dec_true = causal.decompose(truth["pi"], truth["delta"], inp.B, inp.Gmat, T, groups=grp,
                                                estimator="diff" if est == "auto" else est, n_boot=20)
                    for src, kw in sources:
                        if est == "auto" and src != "palm":          # unmixing with PALM fractions only (cost)
                            continue
                        r = pb.program_deconvolution(inp, cal, groups=grp, estimator=est, decompose=True,
                                                     seed=split, unmix_kw=dict(pev=False, misfit=False), **kw)
                        s = db._score(MethodResult(src, fractions=r.fractions, activity=r.activity,
                                                   effects=r.effects, intervals=dict(activity=r.activity_interval,
                                                                                     effects=r.effects_interval)),
                                      tr, dict(pi=truth["pi"], delta=truth["delta"]))
                        s.update(ka.decomposition_accuracy(r.decomposition, dec_true))
                        kp = [q for q in KNOWN.get(dsn, {}).get("program", ()) if q in list(inp.programs)]
                        if kp:
                            kn = KNOWN[dsn]
                            resp = np.zeros(r.effects.shape, bool)
                            rows_ = [i for i, c in enumerate(inp.cell_types) if kn["types"] is None or c in kn["types"]]
                            j = list(inp.programs).index(kp[0])
                            resp[rows_, j] = True
                            raw = np.einsum("n,ncp->cp", w, r.raw_activity)
                            # localisation within the known program: responder types vs the other types
                            # (only meaningful when some types do not respond)
                            s.update(known_auroc_prog=ka.responder_auroc(r.effects[:, j], resp[:, j], kn["sign"]),
                                     known_auroc_prog_raw=ka.responder_auroc(raw[:, j], resp[:, j], kn["sign"]),
                                     # calls among the known non-responder types of the known program (other
                                     # programs may really respond, so known_fpr over all pairs overstates)
                                     known_fpr_prog=ka.leakage_fpr(r.calls[:, j], resp[:, j]))
                            s.update(known_auroc=ka.responder_auroc(r.effects, resp, kn["sign"]),
                                     known_tpr=ka.responder_tpr(r.effects, r.calls, resp, kn["sign"]),
                                     known_fpr=ka.leakage_fpr(r.calls, resp))
                        lo, hi = r.effects_interval
                        s.update(dataset=dsn, prior=prior, split=split, source=src, estimator=est,
                                 eff_width=float(np.mean(hi - lo)),
                                 rv_called=float(np.nanmedian(r.robustness[r.calls])) if r.calls.any() else np.nan,
                                 true_share_intrinsic=float(np.nanmean(dec_true["share_intrinsic"])),
                                 est_share_intrinsic=float(np.nanmean(r.decomposition["share_intrinsic"])),
                                 n_cal=len(cal),
                                 unmixed=",".join(r.info.get("unmix", {}).get("programs", [])))
                        rows.append(s)
                with open(f, "w") as fh:
                    json.dump(rows, fh, indent=1, default=float)
                print("[%s] %s %s split %d done (%.0fs)" % (time.strftime("%H:%M"), dsn, prior, split,
                                                          time.time() - t0), flush=True)


def summary():
    rows = [r for f in sorted(glob.glob(os.path.join(OUT, "*.json"))) for r in json.load(open(f))]
    df = pd.DataFrame(rows)
    cols = dict(frac_rmse="frac", act_spearman_across_samples="rho", eff_pearson="eff_r", eff_cov90="eff_cov",
                eff_iscore="eff_is", eff_width="width", eff_fsr="fsr", eff_n_called="calls", rv_called="rv",
                I_r="I_r", K_r="K_r", K_sign="K_sign", share_mae="share_err", true_share_intrinsic="share_true",
                est_share_intrinsic="share_est", known_auroc="k_auroc", known_auroc_prog="k_auroc_p",
                known_auroc_prog_raw="k_auroc_p_raw", known_tpr="k_tpr", known_fpr="k_fpr",
                known_fpr_prog="k_fpr_p")
    key = ["dataset", "prior", "estimator", "source"]
    t = df.groupby(key, sort=False)[[c for c in cols if c in df]].mean().rename(columns=cols)
    t.insert(0, "n", df.groupby(key, sort=False).split.nunique())
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(t.round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(sys.argv[2].split(","), sys.argv[3].split(","), db._splits(sys.argv[4]))
